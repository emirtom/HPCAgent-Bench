# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Triton kernel for heat_3d: TSTEPS explicit 7-point heat steps, B from A and then A from B, over the interior
# 1 .. N - 2 of all three axes; the boundary of both arrays is a Dirichlet value and never changes.
#
# One launch per half-step: each program owns a BJ x BK tile of one interior i plane (k contiguous), loads the
# seven shifted tiles of the source and writes the destination. A half-step reads only the other array, so
# updating it in place is race free.
#
# Precision (hpcagent_bench/docs/numerical_validation.md): fp64 and fp32 compute in themselves; fp16, bf16 and
# both fp8 formats load into fp32, compute there and round to the storage format when the half-step is written.
# alpha reaches the kernel in a one-element tensor of the accumulation dtype, so an fp64 run keeps it in fp64.
"""Two-array explicit 7-point heat equation in three dimensions, one launch per half-step."""

import torch
import triton
import triton.language as tl


def heat_configs():
    return [
        triton.Config({"BJ": 8, "BK": 32}, num_warps=4),
        triton.Config({"BJ": 4, "BK": 64}, num_warps=4),
        triton.Config({"BJ": 16, "BK": 32}, num_warps=4),
        triton.Config({"BJ": 8, "BK": 64}, num_warps=4),
    ]


@triton.autotune(configs=heat_configs(), key=["N"], cache_results=True)
@triton.jit
def half_step(src, dst, alpha_ptr, N, ACC: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr):
    """dst = alpha * (d2_i + d2_j + d2_k of src) + src at the interior, in the reference's order."""
    k = 1 + tl.program_id(0) * BK + tl.arange(0, BK)
    j = 1 + tl.program_id(1) * BJ + tl.arange(0, BJ)
    i = 1 + tl.program_id(2)
    alpha = tl.load(alpha_ptr)
    mask = (j[:, None] < N - 1) & (k[None, :] < N - 1)
    plane = N * N
    offset = i.to(tl.int64) * plane + j[:, None] * N + k[None, :]
    at = src + offset
    c = tl.load(at, mask=mask).to(ACC)
    two_c = 2.0 * c
    di = alpha * (tl.load(at + plane, mask=mask).to(ACC) - two_c + tl.load(at - plane, mask=mask).to(ACC))
    dj = alpha * (tl.load(at + N, mask=mask).to(ACC) - two_c + tl.load(at - N, mask=mask).to(ACC))
    dk = alpha * (tl.load(at + 1, mask=mask).to(ACC) - two_c + tl.load(at - 1, mask=mask).to(ACC))
    tl.store(dst + offset, (di + dj + dk + c).to(dst.dtype.element_ty), mask=mask)


# The last alpha tensor, keyed by (alpha, dtype, device): building it costs a host-to-device copy per call.
ALPHA: dict[tuple[float, torch.dtype, torch.device], torch.Tensor] = {}


def alpha_tensor(alpha, dtype, device):
    key = (float(alpha), dtype, device)
    if key not in ALPHA:
        ALPHA.clear()
        ALPHA[key] = torch.tensor([float(alpha)], dtype=dtype, device=device)
    return ALPHA[key]


def kernel(TSTEPS, A, B, alpha=0.125):
    N = int(A.shape[0])
    if N < 3:
        return
    acc = torch.float64 if A.dtype == torch.float64 else torch.float32
    alpha_t = alpha_tensor(alpha, acc, A.device)
    ACC = tl.float64 if acc == torch.float64 else tl.float32

    def grid(meta):
        return (triton.cdiv(N - 2, meta["BK"]), triton.cdiv(N - 2, meta["BJ"]), N - 2)

    # Autotune trials rerun a half-step, which only overwrites the interior with the same function of the other
    # array: repeating it is idempotent, so no restore is needed.
    for step in range(int(TSTEPS)):
        half_step[grid](A, B, alpha_t, N, ACC=ACC)
        half_step[grid](B, A, alpha_t, N, ACC=ACC)
