# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Triton kernel for fdtd_2d: TMAX steps of the 2-D Yee scheme. Each step sets the source row ey[0, :] = fict[t],
# updates ey (rows 1 ..) and ex (columns 1 ..) from the old hz, then hz (all but the last row and column) from the
# new ex and ey.
#
# Two launches per step: the E launch writes ey and ex in place (each reads only hz besides its own element), the
# H launch then writes hz in place (it reads only ex and ey besides its own element). Each program owns a BI x BJ
# tile (j contiguous).
#
# Precision (hpcagent_bench/docs/numerical_validation.md): fp64 and fp32 compute in themselves; fp16, bf16 and
# both fp8 formats load into fp32, compute there and round to the storage format when a field is written. The
# Courant coefficients reach the kernels in a tensor of the accumulation dtype, so an fp64 run keeps them in fp64.
"""2-D FDTD (Yee) time stepping, one E launch and one H launch per step."""

import torch
import triton
import triton.language as tl


def fdtd_configs():
    return [
        triton.Config({"BI": 8, "BJ": 128}, num_warps=4),
        triton.Config({"BI": 16, "BJ": 64}, num_warps=4),
        triton.Config({"BI": 4, "BJ": 256}, num_warps=4),
        triton.Config({"BI": 16, "BJ": 128}, num_warps=8),
    ]


@triton.autotune(configs=fdtd_configs(), key=["NX", "NY"], cache_results=True, restore_value=["ex", "ey"])
@triton.jit(do_not_specialize=["t"])
def e_step(ex, ey, hz, fict, coeff, t, NX, NY, ACC: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr):
    """ey[0, :] = fict[t]; ey[1:, :] -= cy * (hz[1:] - hz[:-1]); ex[:, 1:] -= cx * (hz[:, 1:] - hz[:, :-1])."""
    i = tl.program_id(1) * BI + tl.arange(0, BI)
    j = tl.program_id(0) * BJ + tl.arange(0, BJ)
    inside = (i[:, None] < NX) & (j[None, :] < NY)
    offset = i[:, None].to(tl.int64) * NY + j[None, :]
    cy = tl.load(coeff)
    cx = tl.load(coeff + 1)
    source = tl.load(fict + t).to(ACC)
    h = tl.load(hz + offset, mask=inside).to(ACC)
    h_north = tl.load(hz + offset - NY, mask=inside & (i[:, None] > 0)).to(ACC)
    h_west = tl.load(hz + offset - 1, mask=inside & (j[None, :] > 0)).to(ACC)
    e_y = tl.load(ey + offset, mask=inside).to(ACC)
    e_x = tl.load(ex + offset, mask=inside).to(ACC)
    e_y = tl.where(i[:, None] > 0, e_y - cy * (h - h_north), source)
    e_x = tl.where(j[None, :] > 0, e_x - cx * (h - h_west), e_x)
    tl.store(ey + offset, e_y.to(ey.dtype.element_ty), mask=inside)
    tl.store(ex + offset, e_x.to(ex.dtype.element_ty), mask=inside & (j[None, :] > 0))


@triton.autotune(configs=fdtd_configs(), key=["NX", "NY"], cache_results=True, restore_value=["hz"])
@triton.jit
def h_step(ex, ey, hz, coeff, NX, NY, ACC: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr):
    """hz[:-1, :-1] -= ch * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])."""
    i = tl.program_id(1) * BI + tl.arange(0, BI)
    j = tl.program_id(0) * BJ + tl.arange(0, BJ)
    mask = (i[:, None] < NX - 1) & (j[None, :] < NY - 1)
    offset = i[:, None].to(tl.int64) * NY + j[None, :]
    ch = tl.load(coeff + 2)
    e_x = tl.load(ex + offset, mask=mask).to(ACC)
    e_x_east = tl.load(ex + offset + 1, mask=mask).to(ACC)
    e_y = tl.load(ey + offset, mask=mask).to(ACC)
    e_y_south = tl.load(ey + offset + NY, mask=mask).to(ACC)
    h = tl.load(hz + offset, mask=mask).to(ACC)
    h = h - ch * (e_x_east - e_x + e_y_south - e_y)
    tl.store(hz + offset, h.to(hz.dtype.element_ty), mask=mask)


# The last coefficient tensor, keyed by (courants, dtype, device): building it costs a host-to-device copy per call.
COEFFICIENTS: dict[tuple[float, float, float, torch.dtype, torch.device], torch.Tensor] = {}


def coefficients(cy, cx, ch, dtype, device):
    key = (float(cy), float(cx), float(ch), dtype, device)
    if key not in COEFFICIENTS:
        COEFFICIENTS.clear()
        COEFFICIENTS[key] = torch.tensor([float(cy), float(cx), float(ch)], dtype=dtype, device=device)
    return COEFFICIENTS[key]


def kernel(TMAX, ex, ey, hz, fict, ey_courant=0.5, ex_courant=0.5, hz_courant=0.7):
    NX, NY = int(ex.shape[0]), int(ex.shape[1])
    acc = torch.float64 if ex.dtype == torch.float64 else torch.float32
    coeff = coefficients(ey_courant, ex_courant, hz_courant, acc, ex.device)
    ACC = tl.float64 if acc == torch.float64 else tl.float32

    def grid(meta):
        return (triton.cdiv(NY, meta["BJ"]), triton.cdiv(NX, meta["BI"]))

    for t in range(int(TMAX)):
        e_step[grid](ex, ey, hz, fict, coeff, t, NX, NY, ACC=ACC)
        h_step[grid](ex, ey, hz, coeff, NX, NY, ACC=ACC)
