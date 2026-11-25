# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Triton kernel for conv_3d: a dense (2R + 1)^3-tap box stencil with edge-replicated ("edge" padding) boundaries,
# out[i, j, k] = sum over (di, dj, dk) of w[di, dj, dk] * in[clamp(i + di), clamp(j + dj), clamp(k + dk)].
#
# One launch: each program owns a BJ x BK tile of one i plane (k contiguous) and sums the taps in the reference's
# order (di, then dj, then dk), each tap a clamped shifted load of the tile (neighbours come through L1/L2) times
# one weight. R is a compile-time constant, so the tap loops unroll.
#
# Precision (hpcagent_bench/docs/numerical_validation.md): fp64 and fp32 compute and accumulate in themselves;
# fp16, bf16 and both fp8 formats load into fp32, accumulate there and round to the storage format only when
# out_grid is written.
"""Dense 3-D box stencil with edge-replicated boundaries, one tap at a time."""

import torch
import triton
import triton.language as tl


def conv_configs():
    return [
        triton.Config({"BJ": 8, "BK": 64}, num_warps=4),
        triton.Config({"BJ": 16, "BK": 32}, num_warps=4),
        triton.Config({"BJ": 4, "BK": 128}, num_warps=4),
        triton.Config({"BJ": 16, "BK": 64}, num_warps=8),
    ]


@triton.autotune(configs=conv_configs(), key=["N"], cache_results=True)
@triton.jit
def conv_kernel(in_grid, out_grid, w_box, N, R: tl.constexpr, ACC: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr):
    k = tl.program_id(0) * BK + tl.arange(0, BK)
    j = tl.program_id(1) * BJ + tl.arange(0, BJ)
    i = tl.program_id(2)
    mask = (j[:, None] < N) & (k[None, :] < N)
    W: tl.constexpr = 2 * R + 1
    acc = tl.zeros((BJ, BK), dtype=ACC)
    for di in tl.static_range(W):
        ii = tl.minimum(tl.maximum(i + di - R, 0), N - 1).to(tl.int64) * N * N
        for dj in tl.static_range(W):
            jj = tl.minimum(tl.maximum(j + dj - R, 0), N - 1) * N
            for dk in tl.static_range(W):
                kk = tl.minimum(tl.maximum(k + dk - R, 0), N - 1)
                w = tl.load(w_box + (di * W + dj) * W + dk).to(ACC)
                x = tl.load(in_grid + ii + jj[:, None] + kk[None, :], mask=mask).to(ACC)
                acc = acc + w * x
    out_at = i.to(tl.int64) * N * N + j[:, None] * N + k[None, :]
    tl.store(out_grid + out_at, acc.to(out_grid.dtype.element_ty), mask=mask)


def conv_3d(in_grid, out_grid, w_box, N, R):
    N, R = int(N), int(R)
    acc = tl.float64 if in_grid.dtype == torch.float64 else tl.float32

    def grid(meta):
        return (triton.cdiv(N, meta["BK"]), triton.cdiv(N, meta["BJ"]), N)

    conv_kernel[grid](in_grid, out_grid, w_box, N, R=R, ACC=acc)
