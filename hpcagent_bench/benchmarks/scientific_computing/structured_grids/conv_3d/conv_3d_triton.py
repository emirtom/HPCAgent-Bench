# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Triton kernel for conv_3d: a dense (2R + 1)^3-tap box stencil with edge-replicated ("edge" padding) boundaries,
# out[i, j, k] = sum over (di, dj, dk) of w[di, dj, dk] * in[clamp(i + di), clamp(j + dj), clamp(k + dk)].
#
# R = 2 (every preset): each program owns a BJ x BK tile (k contiguous) of an I_CHUNK-deep column of i planes and
# marches along i. It loads the 25 clamped (dj, dk) shifts of each input plane p once and adds each into the five
# outputs p - 2 .. p + 2 that use plane p, held in five rotating register accumulators; output p - 2 is complete
# after plane p and is stored. That is 25 loads per input plane instead of 125 per output plane, and each output
# still sums its taps in the reference's order (di, then dj, then dk). Other R take the plain kernel: one output
# plane per program, one clamped load per tap.
#
# Precision (hpcagent_bench/docs/numerical_validation.md): fp64 and fp32 compute and accumulate in themselves;
# fp16, bf16 and both fp8 formats load into fp32, accumulate there and round to the storage format only when
# out_grid is written.
"""Dense 3-D box stencil with edge-replicated boundaries, marching along i with rotating accumulators."""

import torch
import triton
import triton.language as tl


def march_configs():
    return [
        triton.Config({"BJ": 8, "BK": 64, "I_CHUNK": 16}, num_warps=4),
        triton.Config({"BJ": 4, "BK": 64, "I_CHUNK": 16}, num_warps=2),
        triton.Config({"BJ": 4, "BK": 64, "I_CHUNK": 8}, num_warps=2),
        triton.Config({"BJ": 8, "BK": 64, "I_CHUNK": 32}, num_warps=4),
    ]


def plain_configs():
    return [
        triton.Config({"BJ": 8, "BK": 64}, num_warps=4),
        triton.Config({"BJ": 16, "BK": 32}, num_warps=4),
        triton.Config({"BJ": 4, "BK": 128}, num_warps=4),
        triton.Config({"BJ": 16, "BK": 64}, num_warps=8),
    ]


@triton.jit
def add_plane(acc, x, w_box, di: tl.constexpr, dj: tl.constexpr, dk: tl.constexpr, ACC: tl.constexpr):
    """acc + w[di, dj, dk] * x, the weight read from the 5 x 5 x 5 box."""
    return acc + tl.load(w_box + (di * 5 + dj) * 5 + dk).to(ACC) * x


@triton.autotune(configs=march_configs(), key=["N"], cache_results=True)
@triton.jit
def march_kernel(
    in_grid, out_grid, w_box, N, ACC: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr, I_CHUNK: tl.constexpr
):
    k = tl.program_id(0) * BK + tl.arange(0, BK)
    j = tl.program_id(1) * BJ + tl.arange(0, BJ)
    i0 = tl.program_id(2) * I_CHUNK
    mask = (j[:, None] < N) & (k[None, :] < N)
    out_tile = j[:, None] * N + k[None, :]
    plane = N.to(tl.int64) * N
    # q0 .. q4 accumulate outputs p - 2 .. p + 2 while plane p is read.
    q0 = tl.zeros((BJ, BK), dtype=ACC)
    q1 = tl.zeros((BJ, BK), dtype=ACC)
    q2 = tl.zeros((BJ, BK), dtype=ACC)
    q3 = tl.zeros((BJ, BK), dtype=ACC)
    i_end = tl.minimum(i0 + I_CHUNK, N)
    for p in range(i0 - 2, i_end + 2):
        q4 = tl.zeros((BJ, BK), dtype=ACC)
        base = in_grid + tl.minimum(tl.maximum(p, 0), N - 1) * plane
        for dj in tl.static_range(5):
            jj = tl.minimum(tl.maximum(j + dj - 2, 0), N - 1) * N
            for dk in tl.static_range(5):
                kk = tl.minimum(tl.maximum(k + dk - 2, 0), N - 1)
                x = tl.load(base + jj[:, None] + kk[None, :], mask=mask).to(ACC)
                # Output o = p + 2 - di takes this plane with weight index di.
                q0 = add_plane(q0, x, w_box, 4, dj, dk, ACC)
                q1 = add_plane(q1, x, w_box, 3, dj, dk, ACC)
                q2 = add_plane(q2, x, w_box, 2, dj, dk, ACC)
                q3 = add_plane(q3, x, w_box, 1, dj, dk, ACC)
                q4 = add_plane(q4, x, w_box, 0, dj, dk, ACC)
        done = p - 2
        if done >= i0:
            tl.store(out_grid + done * plane + out_tile, q0.to(out_grid.dtype.element_ty), mask=mask)
        q0, q1, q2, q3 = q1, q2, q3, q4


@triton.autotune(configs=plain_configs(), key=["N"], cache_results=True)
@triton.jit
def plain_kernel(in_grid, out_grid, w_box, N, R: tl.constexpr, ACC: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr):
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
    if R == 2:

        def march_grid(meta):
            return (triton.cdiv(N, meta["BK"]), triton.cdiv(N, meta["BJ"]), triton.cdiv(N, meta["I_CHUNK"]))

        march_kernel[march_grid](in_grid, out_grid, w_box, N, ACC=acc)
        return

    def grid(meta):
        return (triton.cdiv(N, meta["BK"]), triton.cdiv(N, meta["BJ"]), N)

    plain_kernel[grid](in_grid, out_grid, w_box, N, R=R, ACC=acc)
