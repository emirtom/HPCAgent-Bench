# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""LDA XC potential and energy in one fused pass: each program streams a strided set of tiles, writes vxc and
keeps a per-lane energy accumulator, then adds one partial into exc (one atomic per program)."""

import torch
import triton
import triton.language as tl

AX = tl.constexpr(0.9847450218426965)  # (3/pi)^(1/3), Slater-exchange coefficient
GAMMA = tl.constexpr(-0.1423)  # Perdew-Zunger correlation, rs >= 1
B1 = tl.constexpr(1.0529)
B2 = tl.constexpr(0.3334)
A = tl.constexpr(0.0311)  # Perdew-Zunger correlation, rs < 1
B = tl.constexpr(-0.0480)
C = tl.constexpr(0.0020)
D = tl.constexpr(-0.0116)
RS_FACTOR = tl.constexpr(3.0 / (4.0 * 3.141592653589793))

PROGRAMS_PER_SM = 8


def get_configs():
    # Streaming kernel: wide tiles with 4-8 warps keep enough loads in flight to saturate DRAM.
    return [
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ]


# reset_to_zero: exc is accumulated into, so each autotune trial starts from zero.
@triton.autotune(configs=get_configs(), key=["n"], cache_results=True, reset_to_zero=["exc"])
@triton.jit
def lda_xc_kernel(rho, vxc, exc, n, DVOL: tl.float64, COMPUTE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK)
    energy = tl.zeros((BLOCK,), dtype=COMPUTE)
    for start in range(pid * BLOCK, n, num_programs * BLOCK):
        offs = start + lanes
        valid = offs < n
        density = tl.load(rho + offs, mask=valid, other=1.0).to(COMPUTE)
        # The floor goes through an add: a float literal in a comparison is rounded to fp32 first.
        density = tl.maximum(density, tl.zeros_like(density) + 1.0e-12)
        # Cube roots through exp/log keep the kernel in core triton.language (density > 0 after the floor).
        rs = tl.exp(tl.log(RS_FACTOR / density) * (1.0 / 3.0))
        n13 = tl.exp(tl.log(density) * (1.0 / 3.0))
        sqrt_rs = tl.sqrt(rs)
        ln_rs = tl.log(rs)
        denom = 1.0 + B1 * sqrt_rs + B2 * rs
        eps_c_ge1 = GAMMA / denom
        v_c_ge1 = eps_c_ge1 * (1.0 + (7.0 / 6.0) * B1 * sqrt_rs + (4.0 / 3.0) * B2 * rs) / denom
        eps_c_lt1 = A * ln_rs + B + C * rs * ln_rs + D * rs
        v_c_lt1 = A * ln_rs + (B - A / 3.0) + (2.0 / 3.0) * C * rs * ln_rs + (2.0 * D - C) / 3.0 * rs
        high_density = rs < 1.0
        eps_c = tl.where(high_density, eps_c_lt1, eps_c_ge1)
        v_c = tl.where(high_density, v_c_lt1, v_c_ge1)
        tl.store(vxc + offs, (-AX * n13 + v_c).to(vxc.dtype.element_ty), mask=valid)
        energy += tl.where(valid, (-0.75 * AX * n13 + eps_c) * density, 0.0)
    tl.atomic_add(exc, (tl.sum(energy, axis=0) * DVOL).to(exc.dtype.element_ty), sem="relaxed")


def sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def kernel(dvol, rho, vxc, exc):
    n = rho.numel()
    assert rho.is_contiguous() and vxc.is_contiguous()
    # fp64 runs compute in fp64; every narrower precision computes and accumulates in fp32.
    compute = tl.float64 if rho.dtype == torch.float64 else tl.float32
    # dvol is annotated tl.float64: an unannotated Python float reaches the kernel as fp32.
    exc.zero_()
    sms = sm_count(rho.device.index or 0)
    grid = lambda meta: (max(1, min(triton.cdiv(n, meta["BLOCK"]), sms * PROGRAMS_PER_SM)),)
    lda_xc_kernel[grid](rho, vxc, exc, n, float(dvol), compute)
