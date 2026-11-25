# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Triton kernel for heat_3d: TSTEPS explicit 7-point heat steps, B from A and then A from B, over the interior
# 1 .. N - 2 of all three axes; the boundary of both arrays is a Dirichlet value and never changes.
#
# Temporal blocking as a 2.5-D wavefront: one launch advances up to STEPS iterations (STAGES = 2 * steps
# half-steps). Each program owns a BJ x BK tile in (j, k) (k contiguous) with a halo of HALO = 2 * STEPS on every
# side, and an IC-deep chunk of i planes with STAGES planes of halo above and below. It marches along i: step p
# loads plane p of the A state (stage 0), and stage s = 1 .. STAGES computes its plane p - s from the three planes
# p - s - 1 .. p - s + 1 of stage s - 1. Each stage keeps its last three planes in a ring of the program's scratch
# (a global buffer; reads hit L1/L2), with a CTA barrier after every stage. Odd stages are B states and take B's
# boundary values from B, even stages are A states and keep the A boundary of stage s - 2. The valid (j, k) region
# shrinks by one point per side and stage, so the middle (BJ - 2 HALO) x (BK - 2 HALO) points of the last stage
# are exact and are written. Launches ping-pong between A and temporaries (a launch must not overwrite a halo
# another program still reads), the last one writes A and, from its last B stage, B's interior. 20 iterations
# take 10 launches instead of 40.
#
# Precision (hpcagent_bench/docs/numerical_validation.md): fp64 and fp32 compute in themselves; fp16, bf16 and
# both fp8 formats load into fp32 and compute there. Every half-step's value is rounded to the storage format in
# registers, as the reference rounds it when it stores the array, so blocking does not change the result. alpha
# is a compile-time constant (ALPHA): a constexpr float in arithmetic takes the tensor operand's type with the
# Python value, so an fp64 run multiplies by the exact double (a float passed as a runtime argument would arrive
# as fp32). Nothing is kept between calls.
"""Temporally blocked (2.5-D wavefront) explicit 7-point heat equation in three dimensions."""

import torch
import triton
import triton.language as tl

# Iterations per launch; the (j, k) halo is 2 * STEPS points on each side of a tile.
STEPS = 2
HALO = 2 * STEPS


def heat_configs():
    return [
        triton.Config({"BJ": 16, "BK": 64, "IC": 32}, num_warps=4),
        triton.Config({"BJ": 32, "BK": 64, "IC": 32}, num_warps=8),
        triton.Config({"BJ": 16, "BK": 64, "IC": 16}, num_warps=4),
        triton.Config({"BJ": 32, "BK": 32, "IC": 32}, num_warps=4),
    ]


@triton.autotune(configs=heat_configs(), key=["N"], cache_results=True)
@triton.jit
def blocked_steps(
    src,
    dst,
    B,
    scratch,
    N,
    ALPHA: tl.constexpr,
    STAGES: tl.constexpr,
    WRITE_B: tl.constexpr,
    ACC: tl.constexpr,
    HALO: tl.constexpr,
    BJ: tl.constexpr,
    BK: tl.constexpr,
    IC: tl.constexpr,
):
    """dst = the A state STAGES / 2 (<= HALO / 2) iterations after src; with WRITE_B, also B's last half-step."""
    IJ: tl.constexpr = BJ - 2 * HALO
    IK: tl.constexpr = BK - 2 * HALO
    TILE: tl.constexpr = BJ * BK
    r = tl.arange(0, BJ)
    c = tl.arange(0, BK)
    j = tl.program_id(1) * IJ - HALO + r
    k = tl.program_id(0) * IK - HALO + c
    in_jk = ((j >= 0) & (j < N))[:, None] & ((k >= 0) & (k < N))[None, :]
    edge_jk = ((j == 0) | (j == N - 1))[:, None] | ((k == 0) | (k == N - 1))[None, :]
    owned = in_jk & (((r >= HALO) & (r < HALO + IJ))[:, None] & ((c >= HALO) & (c < HALO + IK))[None, :])
    jk = j[:, None] * N + k[None, :]
    lanes = r[:, None] * BK + c[None, :]
    has_west = (c > 0)[None, :]
    has_east = (c < BK - 1)[None, :]
    has_north = (r > 0)[:, None]
    has_south = (r < BJ - 1)[:, None]
    plane = N.to(tl.int64) * N
    pid = (tl.program_id(2) * tl.num_programs(1) + tl.program_id(1)) * tl.num_programs(0) + tl.program_id(0)
    rings = scratch + pid.to(tl.int64) * ((STAGES + 1) * 3 * TILE)
    i0 = tl.program_id(2) * IC
    i1 = tl.minimum(i0 + IC, N)
    p_lo = tl.maximum(i0 - STAGES, 0)
    p_hi = tl.minimum(i1 + STAGES, N)
    for p in range(p_lo, p_hi + STAGES):
        # Stage 0: plane p of the A state.
        if p < p_hi:
            x = tl.load(src + p * plane + jk, mask=in_jk, other=0.0).to(ACC)
            tl.store(rings + (p % 3) * TILE + lanes, x)
        tl.debug_barrier()
        for s in tl.static_range(1, STAGES + 1):
            q = p - s
            if (q >= tl.maximum(i0 - STAGES + s, 0)) & (q < tl.minimum(i1 + STAGES - s, N)):
                prev = rings + (s - 1) * 3 * TILE
                here = prev + (q % 3) * TILE + lanes
                centre = tl.load(here)
                if s % 2 == 1:
                    fixed = tl.load(B + q * plane + jk, mask=in_jk & (edge_jk | (q == 0) | (q == N - 1)), other=0.0)
                    fixed = fixed.to(ACC)
                else:
                    fixed = tl.load(rings + (s - 2) * 3 * TILE + (q % 3) * TILE + lanes)
                if (q > 0) & (q < N - 1):
                    up = tl.load(prev + ((q + 1) % 3) * TILE + lanes)
                    down = tl.load(prev + ((q + 2) % 3) * TILE + lanes)
                    north = tl.load(here - BK, mask=has_north, other=0.0)
                    south = tl.load(here + BK, mask=has_south, other=0.0)
                    west = tl.load(here - 1, mask=has_west, other=0.0)
                    east = tl.load(here + 1, mask=has_east, other=0.0)
                    two_c = 2.0 * centre
                    value = (
                        ALPHA * (up - two_c + down)
                        + ALPHA * (south - two_c + north)
                        + ALPHA * (east - two_c + west)
                        + centre
                    )
                    value = value.to(dst.dtype.element_ty).to(ACC)
                    value = tl.where(edge_jk, fixed, value)
                else:
                    value = fixed
                tl.store(rings + s * 3 * TILE + (q % 3) * TILE + lanes, value)
                if s == STAGES:
                    if q >= i0:
                        if q < i1:
                            tl.store(dst + q * plane + jk, value.to(dst.dtype.element_ty), mask=owned)
                if WRITE_B:
                    if s == STAGES - 1:
                        if (q >= i0) & (q < i1) & (q > 0) & (q < N - 1):
                            tl.store(B + q * plane + jk, value.to(B.dtype.element_ty), mask=owned & ~edge_jk)
            tl.debug_barrier()


def kernel(TSTEPS, A, B, alpha=0.125):
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
    temps = torch.empty((min(launches - 1, 2) if launches > 1 else 1, N, N, N), dtype=A.dtype, device=A.device)
    buffers = [A] + [temps[k % 2] for k in range(launches - 1)] + [A]
    if launches == 1:
        temps[0].copy_(A)
        buffers[0] = temps[0]

    def tiles(config):
        bj, bk, ic = config["BJ"], config["BK"], config["IC"]
        return triton.cdiv(N, bk - 2 * HALO), triton.cdiv(N, bj - 2 * HALO), triton.cdiv(N, ic)

    def grid(meta):
        return tiles(meta)

    def scratch_size(config):
        ks, js, chunks_i = tiles(config.kwargs)
        return ks * js * chunks_i * (2 * STEPS + 1) * 3 * config.kwargs["BJ"] * config.kwargs["BK"]

    scratch = torch.empty(max(scratch_size(config) for config in heat_configs()), dtype=acc, device=A.device)
    # A launch reads X_k and writes X_{k + 1} != X_k (and B only from X_k), so autotune trials need no restore.
    for k, steps in enumerate(chunks):
        blocked_steps[grid](
            buffers[k],
            buffers[k + 1],
            B,
            scratch,
            N,
            ALPHA=float(alpha),
            STAGES=2 * steps,
            WRITE_B=k == launches - 1,
            ACC=ACC,
            HALO=HALO,
        )
