# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Triton kernel for hdiff: the gt4py horizontal diffusion. For each output point (i, j, k) it needs the Laplacian
# of in_field at the centre (i + 2, j + 2) and its four horizontal neighbours, the two flux limiters in x and y on
# both faces, and coeff; out = centre - coeff * (flx_east - flx_west + fly_north - fly_south).
#
# One fused launch: each program owns a BLOCK-wide segment of one output row out[i, :, :] flattened over (j, k)
# (k contiguous, so no lane idles when K is not a power of two). It loads the 13 in_field values of the radius-2
# diamond around each point (neighbours come through L1/L2) and recomputes the five Laplacians in registers, so
# lap, flx and fly are never written to memory.
#
# Precision (hpcagent_bench/docs/numerical_validation.md): fp64 and fp32 compute in themselves; fp16, bf16 and
# both fp8 formats load into fp32, compute there and round to the storage format when out_field is written.
"""Fused horizontal diffusion (Laplacian, flux limiters and update) in one launch."""

import torch
import triton
import triton.language as tl


def hdiff_configs():
    return [
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=2),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ]


@triton.jit
def ld(centre, di: tl.constexpr, dj: tl.constexpr, row, col, mask, ACC: tl.constexpr):
    """in_field at (i + 2 + di, j + 2 + dj) of every lane; ``row`` and ``col`` are the strides of i and j."""
    return tl.load(centre + di * row + dj * col, mask=mask).to(ACC)


@triton.autotune(configs=hdiff_configs(), key=["I", "J", "K"], cache_results=True)
@triton.jit
def hdiff_kernel(in_field, out_field, coeff, I, J, K, ACC: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(1)
    s = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = s < J * K
    j = s // K
    k = s - j * K
    col = K  # stride of j in in_field
    row = (J + 4) * K  # stride of i in in_field
    centre = in_field + (i + 2).to(tl.int64) * row + (j + 2) * col + k

    # The radius-2 diamond around the centre: (di, dj) offsets in (i, j).
    c = ld(centre, 0, 0, row, col, mask, ACC)
    n = ld(centre, 1, 0, row, col, mask, ACC)
    s_ = ld(centre, -1, 0, row, col, mask, ACC)
    e = ld(centre, 0, 1, row, col, mask, ACC)
    w = ld(centre, 0, -1, row, col, mask, ACC)
    nn = ld(centre, 2, 0, row, col, mask, ACC)
    ss = ld(centre, -2, 0, row, col, mask, ACC)
    ee = ld(centre, 0, 2, row, col, mask, ACC)
    ww = ld(centre, 0, -2, row, col, mask, ACC)
    ne = ld(centre, 1, 1, row, col, mask, ACC)
    nw = ld(centre, 1, -1, row, col, mask, ACC)
    se = ld(centre, -1, 1, row, col, mask, ACC)
    sw = ld(centre, -1, -1, row, col, mask, ACC)
    # The reference's order: 4 * x - (x[i + 1] + x[i - 1] + x[j + 1] + x[j - 1]).
    lap_c = 4.0 * c - (n + s_ + e + w)
    lap_n = 4.0 * n - (nn + c + ne + nw)
    lap_s = 4.0 * s_ - (c + ss + se + sw)
    lap_e = 4.0 * e - (ne + se + ee + c)
    lap_w = 4.0 * w - (nw + sw + c + ww)

    # x fluxes on the faces (i + 1/2) and (i - 1/2); zero where the flux runs down the gradient of in_field.
    res = lap_n - lap_c
    flx_hi = tl.where(res * (n - c) > 0.0, 0.0, res)
    res = lap_c - lap_s
    flx_lo = tl.where(res * (c - s_) > 0.0, 0.0, res)
    # y fluxes on the faces (j + 1/2) and (j - 1/2).
    res = lap_e - lap_c
    fly_hi = tl.where(res * (e - c) > 0.0, 0.0, res)
    res = lap_c - lap_w
    fly_lo = tl.where(res * (c - w) > 0.0, 0.0, res)

    out_at = i.to(tl.int64) * J * K + s
    cf = tl.load(coeff + out_at, mask=mask).to(ACC)
    out = c - cf * (flx_hi - flx_lo + fly_hi - fly_lo)
    tl.store(out_field + out_at, out.to(out_field.dtype.element_ty), mask=mask)


def hdiff(in_field, out_field, coeff, I, J, K):
    I, J, K = int(I), int(J), int(K)
    acc = tl.float64 if in_field.dtype == torch.float64 else tl.float32

    def grid(meta):
        return (triton.cdiv(J * K, meta["BLOCK"]), I)

    hdiff_kernel[grid](in_field, out_field, coeff, I, J, K, ACC=acc)
