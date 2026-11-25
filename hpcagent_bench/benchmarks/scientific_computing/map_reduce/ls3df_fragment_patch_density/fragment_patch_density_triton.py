# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""LS3DF fragment density patch: one pass over psi_frag. Each program reads a tile of fragment voxels with all
k orbitals (contiguous rows, so the loads coalesce), reduces |psi|^2 over k in registers, and scatter-adds
alpha * density into the periodic rho grid. Fragments overlap under the wrap, so the scatter is atomic."""

import torch
import triton
import triton.language as tl


def get_configs():
    # Bandwidth-bound on psi_frag: tiles of 4-16 KiB per program with 4-8 warps.
    return [
        triton.Config({"BLOCK_V": 64}, num_warps=4),
        triton.Config({"BLOCK_V": 128}, num_warps=8),
        triton.Config({"BLOCK_V": 32}, num_warps=4),
        triton.Config({"BLOCK_V": 128}, num_warps=4),
    ]


# reset_to_zero: rho is accumulated into, so each autotune trial starts from zero.
@triton.autotune(configs=get_configs(), key=["nvox", "k", "N", "Lb"], cache_results=True, reset_to_zero=["rho"])
@triton.jit
def patch_density_kernel(
    offsets,
    alpha,
    psi,
    rho,
    nvox,
    k,
    N,
    Lb,
    offset_row_stride,
    offset_col_stride,
    alpha_stride,
    COMPUTE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    voxels = tl.program_id(0).to(tl.int64) * BLOCK_V + tl.arange(0, BLOCK_V)
    valid = voxels < nvox
    orbitals = tl.arange(0, BLOCK_K)
    psi_tile = tl.load(
        psi + voxels[:, None] * k + orbitals[None, :],
        mask=valid[:, None] & (orbitals[None, :] < k),
        other=0.0,
    ).to(COMPUTE)
    density = tl.sum(psi_tile * psi_tile, axis=1)

    box = Lb * Lb * Lb
    fragment = voxels // box
    local = voxels % box
    bx = local // (Lb * Lb)
    by = (local // Lb) % Lb
    bz = local % Lb
    corner = offsets + fragment * offset_row_stride
    ox = tl.load(corner, mask=valid, other=0)
    oy = tl.load(corner + offset_col_stride, mask=valid, other=0)
    oz = tl.load(corner + 2 * offset_col_stride, mask=valid, other=0)
    # Python's % is non-negative; C-style % keeps the dividend's sign, so fold negatives back.
    x = ((ox + bx) % N + N) % N
    y = ((oy + by) % N + N) % N
    z = ((oz + bz) % N + N) % N
    weight = tl.load(alpha + fragment * alpha_stride, mask=valid, other=0.0).to(COMPUTE)
    tl.atomic_add(rho + (x * N + y) * N + z, (weight * density).to(rho.dtype.element_ty), mask=valid, sem="relaxed")


def kernel(offsets, alpha, psi_frag, rho, N, Lb):
    N = int(N)
    Lb = int(Lb)
    nfrag = psi_frag.shape[0]
    k = psi_frag.shape[-1]
    assert psi_frag.is_contiguous() and rho.is_contiguous()
    rho.zero_()
    nvox = nfrag * Lb * Lb * Lb
    if nvox == 0 or k == 0:
        return
    # fp64 runs compute in fp64; every narrower precision computes and accumulates in fp32. Atomic adds in
    # fp16/bf16 would round every partial, so this kernel expects rho in fp32 or fp64.
    compute = tl.float64 if psi_frag.dtype == torch.float64 else tl.float32
    grid = lambda meta: (triton.cdiv(nvox, meta["BLOCK_V"]),)
    patch_density_kernel[grid](
        offsets,
        alpha,
        psi_frag,
        rho,
        nvox,
        k,
        N,
        Lb,
        offsets.stride(0),
        offsets.stride(1),
        alpha.stride(0),
        compute,
        triton.next_power_of_2(k),
    )
