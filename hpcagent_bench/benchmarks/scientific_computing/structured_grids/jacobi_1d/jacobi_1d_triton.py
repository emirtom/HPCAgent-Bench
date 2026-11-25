# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Triton kernel for jacobi_1d: TSTEPS - 1 iterations of two 3-point averages, B from A and then A from B,
# over the interior 1 .. N - 2; the end points of both arrays are Dirichlet values and never change.
#
# One launch per half-step: each program owns a BLOCK-wide segment of the interior, loads the three shifted
# segments of the source and writes the destination. A half-step reads only the other array, so updating it in
# place is race free.
#
# Precision (hpcagent_bench/docs/numerical_validation.md): fp64 and fp32 compute in themselves; fp16, bf16 and
# both fp8 formats load into fp32, compute there and round to the storage format when the half-step is written.
"""Two-array 3-point Jacobi sweep in one dimension, one launch per half-step."""

import torch
import triton
import triton.language as tl


def jacobi_configs():
    return [
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 4096}, num_warps=8),
    ]


@triton.autotune(configs=jacobi_configs(), key=["N"], cache_results=True)
@triton.jit
def half_step(src, dst, N, ACC: tl.constexpr, BLOCK: tl.constexpr):
    """dst[i] = 0.33333 * (src[i - 1] + src[i] + src[i + 1]) for 1 <= i <= N - 2."""
    i = 1 + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N - 1
    left = tl.load(src + i - 1, mask=mask).to(ACC)
    centre = tl.load(src + i, mask=mask).to(ACC)
    right = tl.load(src + i + 1, mask=mask).to(ACC)
    tl.store(dst + i, (0.33333 * (left + centre + right)).to(dst.dtype.element_ty), mask=mask)


def kernel(TSTEPS, A, B):
    N = int(A.shape[0])
    acc = tl.float64 if A.dtype == torch.float64 else tl.float32
    if N < 3:
        return

    def grid(meta):
        return (triton.cdiv(N - 2, meta["BLOCK"]),)

    # Autotune trials rerun a half-step, which only overwrites the interior with the same function of the other
    # array: repeating it is idempotent, so no restore is needed.
    for step in range(1, int(TSTEPS)):
        half_step[grid](A, B, N, ACC=acc)
        half_step[grid](B, A, N, ACC=acc)
