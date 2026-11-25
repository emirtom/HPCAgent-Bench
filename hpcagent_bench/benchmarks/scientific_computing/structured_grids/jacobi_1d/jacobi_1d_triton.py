# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Triton kernel for jacobi_1d: TSTEPS - 1 iterations of two 3-point averages, B from A and then A from B,
# over the interior 1 .. N - 2; the end points of both arrays are Dirichlet values and never change.
#
# Temporal blocking: one launch advances up to STEPS iterations. Each program loads a BLOCK-wide tile of the A
# state with a halo of HALO = 2 * STEPS on each side, then runs the half-steps on the whole tile in registers;
# the valid region shrinks by one point per half-step, so after STEPS iterations the middle BLOCK - 2 * HALO
# points are exact and are written. Neighbours are exchanged through the program's own two scratch rows (one for
# the A state, one for the B state) between CTA barriers; two rows need one barrier per half-step. The scratch is
# an ordinary global buffer (2 values per grid point in all): reads hit L1/L2, but at M (52 MB of scratch) it does
# not fit in L2 and its writes go back to DRAM.
# B's end values never change, so the programs read them from B. Launches ping-pong between A and temporaries
# (a launch must not overwrite a halo another program still reads), the last one writes A and, for the final
# half-step, B. A 124-iteration run takes 4 launches instead of 248.
#
# Precision (hpcagent_bench/docs/numerical_validation.md): fp64 and fp32 compute in themselves; fp16, bf16 and
# both fp8 formats load into fp32 and compute there. Every half-step's value is rounded to the storage format in
# registers, as the reference rounds it when it stores the array, so blocking does not change the result.
"""Temporally blocked two-array 3-point Jacobi sweep in one dimension."""

import torch
import triton
import triton.language as tl

# Iterations per launch; the halo is 2 * STEPS points on each side of a tile.
STEPS = 32
HALO = 2 * STEPS


def jacobi_configs():
    return [
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 4096}, num_warps=8),
        triton.Config({"BLOCK": 2048}, num_warps=4),
    ]


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
    BLOCK: tl.constexpr,
):
    """dst = the A state ``steps`` (<= HALO / 2) iterations after src; with WRITE_B, also B's last half-step."""
    pid = tl.program_id(0)
    INNER: tl.constexpr = BLOCK - 2 * HALO
    lane = tl.arange(0, BLOCK)
    idx = pid * INNER - HALO + lane
    inside = (idx >= 0) & (idx < N)
    a = tl.load(src + idx, mask=inside, other=0.0).to(ACC)
    b = a
    b_first = tl.load(B).to(ACC)
    b_last = tl.load(B + N - 1).to(ACC)
    is_first = idx == 0
    is_last = idx == N - 1
    row_a = scratch + pid.to(tl.int64) * (2 * BLOCK)
    row_b = row_a + BLOCK
    has_left = lane > 0
    has_right = lane < BLOCK - 1
    for t in range(steps):
        tl.store(row_a + lane, a)
        tl.debug_barrier()
        left = tl.load(row_a + lane - 1, mask=has_left, other=0.0)
        right = tl.load(row_a + lane + 1, mask=has_right, other=0.0)
        b = (0.33333 * (left + a + right)).to(B.dtype.element_ty).to(ACC)
        b = tl.where(is_first, b_first, tl.where(is_last, b_last, b))
        tl.store(row_b + lane, b)
        tl.debug_barrier()
        left = tl.load(row_b + lane - 1, mask=has_left, other=0.0)
        right = tl.load(row_b + lane + 1, mask=has_right, other=0.0)
        a_new = (0.33333 * (left + b + right)).to(dst.dtype.element_ty).to(ACC)
        a = tl.where(is_first | is_last, a, a_new)
    owned = (lane >= HALO) & (lane < HALO + INNER) & inside
    tl.store(dst + idx, a.to(dst.dtype.element_ty), mask=owned)
    if WRITE_B:
        tl.store(B + idx, b.to(B.dtype.element_ty), mask=owned & (idx > 0) & (idx < N - 1))


def kernel(TSTEPS, A, B):
    N = int(A.shape[0])
    iterations = int(TSTEPS) - 1
    if N < 3 or iterations < 1:
        return
    acc = torch.float64 if A.dtype == torch.float64 else torch.float32
    ACC = tl.float64 if acc == torch.float64 else tl.float32
    # Launches k = 0 .. L - 1 read X_k and write X_{k + 1}, X_0 = X_L = A, the rest alternating temporaries; at
    # least two launches, so no launch reads and writes A. One iteration splits no further: it starts from a copy.
    launches = max(2, triton.cdiv(iterations, STEPS)) if iterations > 1 else 1
    chunks = [iterations // launches + (k < iterations % launches) for k in range(launches)]
    temps = torch.empty((min(launches - 1, 2) if launches > 1 else 1, N), dtype=A.dtype, device=A.device)
    buffers = [A] + [temps[k % 2] for k in range(launches - 1)] + [A]
    if launches == 1:
        temps[0].copy_(A)
        buffers[0] = temps[0]
    programs = max(triton.cdiv(N, c.kwargs["BLOCK"] - 2 * HALO) * c.kwargs["BLOCK"] for c in jacobi_configs())
    scratch = torch.empty(2 * programs, dtype=acc, device=A.device)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK"] - 2 * HALO),)

    # A launch reads X_k and writes X_{k + 1} != X_k (and B only from X_k), so autotune trials need no restore.
    for k, steps in enumerate(chunks):
        blocked_steps[grid](
            buffers[k], buffers[k + 1], B, scratch, N, steps, WRITE_B=k == launches - 1, ACC=ACC, HALO=HALO
        )
