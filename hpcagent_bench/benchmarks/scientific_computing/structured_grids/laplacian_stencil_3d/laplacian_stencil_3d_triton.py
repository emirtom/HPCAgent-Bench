# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Triton baseline for laplacian_stencil_3d: 8th-order periodic 3-D Laplacian of a batch of k
# wavefunctions psi[x, y, z, k] (k innermost), then the per-state kinetic energy
# ekin[j] = -1/2 sum_r psi[r, j] * lap[r, j].
#
# Pass 1 (laplacian_kernel): one program per (x, y, z tile); the tile is [BLOCK_Z, BLOCK_K] with k contiguous,
# so every one of the 25 stencil loads is a coalesced row segment. Wraparound indices reproduce np.roll.
# Pass 2 (ekin_partial_kernel): each program reduces a [BLOCK_S, BLOCK_K] slab of psi * lap over space into one
# partial row; torch sums the partial rows. No atomics, so the result is deterministic.
import torch
import triton
import triton.language as tl


def laplacian_configs():
    return [
        triton.Config({"BLOCK_Z": 16}, num_warps=4),
        triton.Config({"BLOCK_Z": 32}, num_warps=4),
        triton.Config({"BLOCK_Z": 8}, num_warps=2),
        triton.Config({"BLOCK_Z": 64}, num_warps=8),
    ]


def ekin_configs():
    return [
        triton.Config({"BLOCK_S": 256}, num_warps=4),
        triton.Config({"BLOCK_S": 512}, num_warps=8),
        triton.Config({"BLOCK_S": 128}, num_warps=4),
        triton.Config({"BLOCK_S": 1024}, num_warps=8),
    ]


@triton.jit
def periodic(i, n):
    # Offsets stay within one period (|m| <= 4 < n), so a single correction suffices.
    i = tl.where(i >= n, i - n, i)
    return tl.where(i < 0, i + n, i)


@triton.autotune(configs=laplacian_configs(), key=["N", "K"], cache_results=True)
@triton.jit
def laplacian_kernel(psi, lap, coeff, inv_h2, N, K, BLOCK_Z: tl.constexpr, BLOCK_K: tl.constexpr):
    """coeff = [3 * c0, w1, w2, w3, w4] in psi's dtype; inv_h2 is a one-element tensor."""
    zt = tl.program_id(0)
    y = tl.program_id(1)
    x = tl.program_id(2)
    z = zt * BLOCK_Z + tl.arange(0, BLOCK_Z)
    k = tl.arange(0, BLOCK_K)
    mask = (z < N)[:, None] & (k < K)[None, :]
    kk = k[None, :]
    row = N * K  # stride of y
    plane = N * row  # stride of x

    base = x.to(tl.int64) * plane + y * row
    centre = tl.load(psi + base + z[:, None] * K + kk, mask=mask, other=0.0)
    acc = tl.load(coeff) * centre
    # Same association as the reference: axis 0, 1, 2; radius 1..4; acc + w * (psi[i - m] + psi[i + m]).
    for m in tl.static_range(1, 5):
        w = tl.load(coeff + m)
        lo = periodic(x - m, N).to(tl.int64) * plane + y * row + z[:, None] * K + kk
        hi = periodic(x + m, N).to(tl.int64) * plane + y * row + z[:, None] * K + kk
        acc = acc + w * (tl.load(psi + lo, mask=mask, other=0.0) + tl.load(psi + hi, mask=mask, other=0.0))
    for m in tl.static_range(1, 5):
        w = tl.load(coeff + m)
        lo = x.to(tl.int64) * plane + periodic(y - m, N) * row + z[:, None] * K + kk
        hi = x.to(tl.int64) * plane + periodic(y + m, N) * row + z[:, None] * K + kk
        acc = acc + w * (tl.load(psi + lo, mask=mask, other=0.0) + tl.load(psi + hi, mask=mask, other=0.0))
    for m in tl.static_range(1, 5):
        w = tl.load(coeff + m)
        lo = base + periodic(z - m, N)[:, None] * K + kk
        hi = base + periodic(z + m, N)[:, None] * K + kk
        acc = acc + w * (tl.load(psi + lo, mask=mask, other=0.0) + tl.load(psi + hi, mask=mask, other=0.0))
    tl.store(lap + base + z[:, None] * K + kk, tl.load(inv_h2) * acc, mask=mask)


@triton.autotune(configs=ekin_configs(), key=["S", "K"], cache_results=True)
@triton.jit
def ekin_partial_kernel(psi, lap, partial, S, K, BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr):
    """partial[p, j] = sum over the p-th block of BLOCK_S grid points of psi[:, j] * lap[:, j]."""
    p = tl.program_id(0)
    s = p * BLOCK_S + tl.arange(0, BLOCK_S)
    k = tl.arange(0, BLOCK_K)
    mask = (s < S)[:, None] & (k < K)[None, :]
    offs = s.to(tl.int64)[:, None] * K + k[None, :]
    a = tl.load(psi + offs, mask=mask, other=0.0)
    b = tl.load(lap + offs, mask=mask, other=0.0)
    tl.store(partial + p * K + k, tl.sum(a * b, axis=0), mask=k < K)


def kernel(inv_h2, psi, lap, ekin):
    N = int(psi.shape[0])
    K = int(psi.shape[3])
    S = N * N * N
    block_k = triton.next_power_of_2(K)
    # Same literals as the reference (_C0, _CW), stored in psi's dtype.
    coeff = torch.tensor(
        [3.0 * (-205.0 / 72.0), 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0],
        dtype=psi.dtype,
        device=psi.device,
    )
    inv = torch.tensor([float(inv_h2)], dtype=psi.dtype, device=psi.device)

    laplacian_kernel[lambda meta: (triton.cdiv(N, meta["BLOCK_Z"]), N, N)](psi, lap, coeff, inv, N, K, BLOCK_K=block_k)

    # The partial buffer is sized for the smallest BLOCK_S so every config fits.
    partial = torch.empty((triton.cdiv(S, 128), K), dtype=psi.dtype, device=psi.device)
    blocks = {}

    def ekin_grid(meta):
        blocks["n"] = triton.cdiv(S, meta["BLOCK_S"])
        return (blocks["n"],)

    ekin_partial_kernel[ekin_grid](psi, lap, partial, S, K, BLOCK_K=block_k)
    ekin.copy_(-0.5 * partial[: blocks["n"]].sum(dim=0))
