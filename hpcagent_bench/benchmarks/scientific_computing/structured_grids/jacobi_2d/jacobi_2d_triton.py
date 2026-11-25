# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Triton kernel for jacobi_2d: TSTEPS iterations of two 5-point averages, B from A and then A from B, over the
# interior 1 .. N - 2 of both axes; the boundary of both arrays is a Dirichlet value and never changes.
#
# One launch per half-step: each program owns a BI x BJ tile of the interior (j contiguous), loads the five
# shifted tiles of the source and writes the destination. A half-step reads only the other array, so updating it
# in place is race free.
#
# Precision (hpcagent_bench/docs/numerical_validation.md): fp64 and fp32 compute in themselves; fp16, bf16 and
# both fp8 formats load into fp32, compute there and round to the storage format when the half-step is written.
"""Two-array 5-point Jacobi sweep in two dimensions, one launch per half-step."""

import torch
import triton
import triton.language as tl


def jacobi_configs():
    return [
        triton.Config({"BI": 8, "BJ": 128}, num_warps=4),
        triton.Config({"BI": 16, "BJ": 64}, num_warps=4),
        triton.Config({"BI": 4, "BJ": 256}, num_warps=4),
        triton.Config({"BI": 16, "BJ": 128}, num_warps=8),
    ]


@triton.autotune(configs=jacobi_configs(), key=["N"], cache_results=True)
@triton.jit
def half_step(src, dst, N, ACC: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr):
    """dst[i, j] = 0.2 * (src[i, j] + src[i, j - 1] + src[i, j + 1] + src[i + 1, j] + src[i - 1, j])."""
    i = 1 + tl.program_id(1) * BI + tl.arange(0, BI)
    j = 1 + tl.program_id(0) * BJ + tl.arange(0, BJ)
    mask = (i[:, None] < N - 1) & (j[None, :] < N - 1)
    at = src + i[:, None].to(tl.int64) * N + j[None, :]
    centre = tl.load(at, mask=mask).to(ACC)
    west = tl.load(at - 1, mask=mask).to(ACC)
    east = tl.load(at + 1, mask=mask).to(ACC)
    south = tl.load(at + N, mask=mask).to(ACC)
    north = tl.load(at - N, mask=mask).to(ACC)
    out = 0.2 * (centre + west + east + south + north)
    tl.store(dst + i[:, None].to(tl.int64) * N + j[None, :], out.to(dst.dtype.element_ty), mask=mask)


def kernel(TSTEPS, A, B):
    N = int(A.shape[0])
    acc = tl.float64 if A.dtype == torch.float64 else tl.float32
    if N < 3:
        return

    def grid(meta):
        return (triton.cdiv(N - 2, meta["BJ"]), triton.cdiv(N - 2, meta["BI"]))

    # Autotune trials rerun a half-step, which only overwrites the interior with the same function of the other
    # array: repeating it is idempotent, so no restore is needed.
    for step in range(int(TSTEPS)):
        half_step[grid](A, B, N, ACC=acc)
        half_step[grid](B, A, N, ACC=acc)
