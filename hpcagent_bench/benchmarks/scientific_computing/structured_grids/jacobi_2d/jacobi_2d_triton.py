# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Triton kernel for jacobi_2d: TSTEPS iterations of two 5-point averages, B from A and then A from B, over the
# interior 1 .. N - 2 of both axes; the boundary of both arrays is a Dirichlet value and never changes.
#
# Temporal blocking: one launch advances up to STEPS iterations. Each program loads a BH x BW tile of the A state
# (j contiguous) with a halo of HALO = 2 * STEPS on every side, then runs the half-steps on the whole tile in
# registers; the valid region shrinks by one point per side and half-step, so after STEPS iterations the middle
# (BH - 2 HALO) x (BW - 2 HALO) points are exact and are written. Neighbours are exchanged through the program's
# own two scratch tiles (one for the A state, one for the B state, L1-resident) between CTA barriers; two tiles
# need one barrier per half-step. B's boundary never changes, so the programs read it from B. Launches ping-pong
# between A and temporaries (a launch must not overwrite a halo another program still reads), and the last one
# writes A and, for the final half-step, B's interior. 50 iterations take 13 launches instead of 100.
#
# Precision (hpcagent_bench/docs/numerical_validation.md): fp64 and fp32 compute in themselves; fp16, bf16 and
# both fp8 formats load into fp32 and compute there. Every half-step's value is rounded to the storage format in
# registers, as the reference rounds it when it stores the array, so blocking does not change the result.
"""Temporally blocked two-array 5-point Jacobi sweep in two dimensions."""

import torch
import triton
import triton.language as tl

# Iterations per launch; the halo is 2 * STEPS points on each side of a tile.
STEPS = 4
HALO = 2 * STEPS


def jacobi_configs():
    return [
        triton.Config({"BH": 64, "BW": 128}, num_warps=8),
        triton.Config({"BH": 32, "BW": 128}, num_warps=4),
        triton.Config({"BH": 64, "BW": 64}, num_warps=4),
        triton.Config({"BH": 32, "BW": 64}, num_warps=4),
    ]


@triton.jit
def neighbour_sum(row, centre, r, c, BH: tl.constexpr, BW: tl.constexpr):
    """centre + west + east + south + north of the tile stored at ``row`` (the reference's order)."""
    at = row + r[:, None] * BW + c[None, :]
    west = tl.load(at - 1, mask=(c > 0)[None, :], other=0.0)
    east = tl.load(at + 1, mask=(c < BW - 1)[None, :], other=0.0)
    south = tl.load(at + BW, mask=(r < BH - 1)[:, None], other=0.0)
    north = tl.load(at - BW, mask=(r > 0)[:, None], other=0.0)
    return centre + west + east + south + north


@triton.autotune(configs=jacobi_configs(), key=["N"], cache_results=True)
@triton.jit(do_not_specialize=["steps"])
def blocked_steps(
    src,
    dst,
    B,
    scratch,
    N,
    steps,
    WRITE_B: tl.constexpr,
    ACC: tl.constexpr,
    HALO: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
):
    """dst = the A state ``steps`` (<= HALO / 2) iterations after src; with WRITE_B, also B's last half-step."""
    IH: tl.constexpr = BH - 2 * HALO
    IW: tl.constexpr = BW - 2 * HALO
    pid = tl.program_id(1) * tl.num_programs(0) + tl.program_id(0)
    r = tl.arange(0, BH)
    c = tl.arange(0, BW)
    i = tl.program_id(1) * IH - HALO + r
    j = tl.program_id(0) * IW - HALO + c
    inside = ((i >= 0) & (i < N))[:, None] & ((j >= 0) & (j < N))[None, :]
    offset = i[:, None].to(tl.int64) * N + j[None, :]
    edge = (((i == 0) | (i == N - 1))[:, None] | ((j == 0) | (j == N - 1))[None, :]) & inside
    a = tl.load(src + offset, mask=inside, other=0.0).to(ACC)
    b_edge = tl.load(B + offset, mask=edge, other=0.0).to(ACC)
    b = a
    row_a = scratch + pid.to(tl.int64) * (2 * BH * BW)
    row_b = row_a + BH * BW
    lanes = r[:, None] * BW + c[None, :]
    for t in range(steps):
        tl.store(row_a + lanes, a)
        tl.debug_barrier()
        b = (0.2 * neighbour_sum(row_a, a, r, c, BH, BW)).to(B.dtype.element_ty).to(ACC)
        b = tl.where(edge, b_edge, b)
        tl.store(row_b + lanes, b)
        tl.debug_barrier()
        a_new = (0.2 * neighbour_sum(row_b, b, r, c, BH, BW)).to(dst.dtype.element_ty).to(ACC)
        a = tl.where(edge, a, a_new)
    owned = (((r >= HALO) & (r < HALO + IH))[:, None] & ((c >= HALO) & (c < HALO + IW))[None, :]) & inside
    tl.store(dst + offset, a.to(dst.dtype.element_ty), mask=owned)
    if WRITE_B:
        tl.store(B + offset, b.to(B.dtype.element_ty), mask=owned & ~edge)


def kernel(TSTEPS, A, B):
    N = int(A.shape[0])
    iterations = int(TSTEPS)
    if N < 3 or iterations < 1:
        return
    acc = torch.float64 if A.dtype == torch.float64 else torch.float32
    ACC = tl.float64 if acc == torch.float64 else tl.float32
    # Launches k = 0 .. L - 1 read X_k and write X_{k + 1}, X_0 = X_L = A, the rest alternating temporaries; at
    # least two launches, so no launch reads and writes A. One iteration splits no further: it starts from a copy.
    launches = max(2, triton.cdiv(iterations, STEPS)) if iterations > 1 else 1
    chunks = [iterations // launches + (k < iterations % launches) for k in range(launches)]
    temps = torch.empty((min(launches - 1, 2) if launches > 1 else 1, N, N), dtype=A.dtype, device=A.device)
    buffers = [A] + [temps[k % 2] for k in range(launches - 1)] + [A]
    if launches == 1:
        temps[0].copy_(A)
        buffers[0] = temps[0]

    def tiles(bh, bw):
        return triton.cdiv(N, bh - 2 * HALO), triton.cdiv(N, bw - 2 * HALO)

    def grid(meta):
        rows, cols = tiles(meta["BH"], meta["BW"])
        return (cols, rows)

    def scratch_size(config):
        rows, cols = tiles(config.kwargs["BH"], config.kwargs["BW"])
        return 2 * config.kwargs["BH"] * config.kwargs["BW"] * rows * cols

    size = max(scratch_size(config) for config in jacobi_configs())
    scratch = torch.empty(size, dtype=acc, device=A.device)
    # A launch reads X_k and writes X_{k + 1} != X_k (and B only from X_k), so autotune trials need no restore.
    for k, steps in enumerate(chunks):
        blocked_steps[grid](
            buffers[k], buffers[k + 1], B, scratch, N, steps, WRITE_B=k == launches - 1, ACC=ACC, HALO=HALO
        )
