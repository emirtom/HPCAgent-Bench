# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Triton kernel for fdtd_2d: TMAX steps of the 2-D Yee scheme. Each step sets the source row ey[0, :] = fict[t],
# updates ey (rows 1 ..) and ex (columns 1 ..) from the old hz, then hz (all but the last row and column) from the
# new ex and ey.
#
# One fused launch per step: each program owns a BI x BJ tile (j contiguous) and computes the new ey and ex of its
# tile, the new ex one column east and the new ey one row south (recomputed from the old fields, as the neighbour
# tile computes them), and from those the new hz. A step thus reads each field once and writes it once, instead of
# the five reads and three writes of separate E and H launches, and takes one launch instead of two. Neighbour
# tiles still read the old fields, so the steps ping-pong between the arguments and three temporaries; an even
# number of steps ends in the arguments, and an odd TMAX runs its first step as the in-place E and H launches.
#
# Precision (hpcagent_bench/docs/numerical_validation.md): fp64 and fp32 compute in themselves; fp16, bf16 and
# both fp8 formats load into fp32, compute there and round to the storage format when a field is written. The
# Courant coefficients are compile-time constants (CY, CX, CH): a constexpr float in arithmetic takes the tensor
# operand's type with the Python value, so an fp64 run multiplies by the exact doubles (a float passed as a runtime
# argument would arrive as fp32). Nothing is kept between calls.
"""2-D FDTD (Yee) time stepping, one fused E and H launch per step."""

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
def e_step(
    ex,
    ey,
    hz,
    fict,
    t,
    NX,
    NY,
    CY: tl.constexpr,
    CX: tl.constexpr,
    CH: tl.constexpr,
    ACC: tl.constexpr,
    BI: tl.constexpr,
    BJ: tl.constexpr,
):
    """ey[0, :] = fict[t]; ey[1:, :] -= cy * (hz[1:] - hz[:-1]); ex[:, 1:] -= cx * (hz[:, 1:] - hz[:, :-1])."""
    i = tl.program_id(1) * BI + tl.arange(0, BI)
    j = tl.program_id(0) * BJ + tl.arange(0, BJ)
    inside = (i[:, None] < NX) & (j[None, :] < NY)
    offset = i[:, None].to(tl.int64) * NY + j[None, :]
    source = tl.load(fict + t).to(ACC)
    h = tl.load(hz + offset, mask=inside).to(ACC)
    h_north = tl.load(hz + offset - NY, mask=inside & (i[:, None] > 0)).to(ACC)
    h_west = tl.load(hz + offset - 1, mask=inside & (j[None, :] > 0)).to(ACC)
    e_y = tl.load(ey + offset, mask=inside).to(ACC)
    e_x = tl.load(ex + offset, mask=inside).to(ACC)
    e_y = tl.where(i[:, None] > 0, e_y - CY * (h - h_north), source)
    e_x = tl.where(j[None, :] > 0, e_x - CX * (h - h_west), e_x)
    tl.store(ey + offset, e_y.to(ey.dtype.element_ty), mask=inside)
    tl.store(ex + offset, e_x.to(ex.dtype.element_ty), mask=inside & (j[None, :] > 0))


@triton.autotune(configs=fdtd_configs(), key=["NX", "NY"], cache_results=True, restore_value=["hz"])
@triton.jit
def h_step(
    ex,
    ey,
    hz,
    NX,
    NY,
    CY: tl.constexpr,
    CX: tl.constexpr,
    CH: tl.constexpr,
    ACC: tl.constexpr,
    BI: tl.constexpr,
    BJ: tl.constexpr,
):
    """hz[:-1, :-1] -= ch * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])."""
    i = tl.program_id(1) * BI + tl.arange(0, BI)
    j = tl.program_id(0) * BJ + tl.arange(0, BJ)
    mask = (i[:, None] < NX - 1) & (j[None, :] < NY - 1)
    offset = i[:, None].to(tl.int64) * NY + j[None, :]
    e_x = tl.load(ex + offset, mask=mask).to(ACC)
    e_x_east = tl.load(ex + offset + 1, mask=mask).to(ACC)
    e_y = tl.load(ey + offset, mask=mask).to(ACC)
    e_y_south = tl.load(ey + offset + NY, mask=mask).to(ACC)
    h = tl.load(hz + offset, mask=mask).to(ACC)
    h = h - CH * (e_x_east - e_x + e_y_south - e_y)
    tl.store(hz + offset, h.to(hz.dtype.element_ty), mask=mask)


@triton.autotune(configs=fdtd_configs(), key=["NX", "NY"], cache_results=True)
@triton.jit(do_not_specialize=["t"])
def fused_step(
    ex0,
    ey0,
    hz0,
    ex1,
    ey1,
    hz1,
    fict,
    t,
    NX,
    NY,
    CY: tl.constexpr,
    CX: tl.constexpr,
    CH: tl.constexpr,
    ACC: tl.constexpr,
    BI: tl.constexpr,
    BJ: tl.constexpr,
):
    """(ex1, ey1, hz1) = one step from (ex0, ey0, hz0), written out of place."""
    i = tl.program_id(1) * BI + tl.arange(0, BI)
    j = tl.program_id(0) * BJ + tl.arange(0, BJ)
    row = i[:, None]
    col = j[None, :]
    inside = (row < NX) & (col < NY)
    h_mask = (row < NX - 1) & (col < NY - 1)
    offset = row.to(tl.int64) * NY + col
    source = tl.load(fict + t).to(ACC)
    h = tl.load(hz0 + offset, mask=inside).to(ACC)
    h_north = tl.load(hz0 + offset - NY, mask=inside & (row > 0)).to(ACC)
    h_west = tl.load(hz0 + offset - 1, mask=inside & (col > 0)).to(ACC)
    h_south = tl.load(hz0 + offset + NY, mask=h_mask).to(ACC)
    h_east = tl.load(hz0 + offset + 1, mask=h_mask).to(ACC)
    # The new E fields, rounded to the storage format as the reference stores them.
    e_y = tl.load(ey0 + offset, mask=inside).to(ACC)
    e_y = tl.where(row > 0, (e_y - CY * (h - h_north)).to(ey1.dtype.element_ty).to(ACC), source)
    e_x = tl.load(ex0 + offset, mask=inside).to(ACC)
    e_x = tl.where(col > 0, (e_x - CX * (h - h_west)).to(ex1.dtype.element_ty).to(ACC), e_x)
    e_y_south = tl.load(ey0 + offset + NY, mask=h_mask).to(ACC)
    e_y_south = (e_y_south - CY * (h_south - h)).to(ey1.dtype.element_ty).to(ACC)
    e_x_east = tl.load(ex0 + offset + 1, mask=h_mask).to(ACC)
    e_x_east = (e_x_east - CX * (h_east - h)).to(ex1.dtype.element_ty).to(ACC)
    h_new = tl.where(h_mask, h - CH * (e_x_east - e_x + e_y_south - e_y), h)
    tl.store(ey1 + offset, e_y.to(ey1.dtype.element_ty), mask=inside)
    tl.store(ex1 + offset, e_x.to(ex1.dtype.element_ty), mask=inside)
    tl.store(hz1 + offset, h_new.to(hz1.dtype.element_ty), mask=inside)


def kernel(TMAX, ex, ey, hz, fict, ey_courant=0.5, ex_courant=0.5, hz_courant=0.7):
    NX, NY = int(ex.shape[0]), int(ex.shape[1])
    acc = torch.float64 if ex.dtype == torch.float64 else torch.float32
    courants = {"CY": float(ey_courant), "CX": float(ex_courant), "CH": float(hz_courant)}
    ACC = tl.float64 if acc == torch.float64 else tl.float32

    def grid(meta):
        return (triton.cdiv(NY, meta["BJ"]), triton.cdiv(NX, meta["BI"]))

    steps = int(TMAX)
    first = 0
    if steps % 2:
        e_step[grid](ex, ey, hz, fict, 0, NX, NY, **courants, ACC=ACC)
        h_step[grid](ex, ey, hz, NX, NY, **courants, ACC=ACC)
        first = 1
    if first == steps:
        return
    temps = torch.empty((3, NX, NY), dtype=ex.dtype, device=ex.device)
    fields = ((ex, ey, hz), (temps[0], temps[1], temps[2]))
    # A fused launch reads one set and writes the other, so its autotune trials need no restore.
    for t in range(first, steps):
        source, target = fields[(t - first) % 2], fields[(t - first + 1) % 2]
        fused_step[grid](*source, *target, fict, t, NX, NY, **courants, ACC=ACC)
