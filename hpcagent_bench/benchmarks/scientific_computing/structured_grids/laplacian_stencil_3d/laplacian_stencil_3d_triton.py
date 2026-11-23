# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Triton kernel for laplacian_stencil_3d: 8th-order periodic 3-D Laplacian of a batch of k
# wavefunctions psi[x, y, z, k] (k innermost), fused with the per-state kinetic energy
# ekin[j] = -1/2 sum_r psi[r, j] * lap[r, j].
#
# Each (x, y) row psi[x, y, :, :] is one contiguous run of L = N * K values, and a z shift by m is a shift by
# m * K along it, periodic modulo L. One program owns a BLOCK-wide segment of that run at fixed y and marches
# along x over X_CHUNK planes. The nine x-plane segments x-4 .. x+4 rotate through registers, so each step loads
# one new x plane instead of nine; y and z neighbours come through L1/L2. The flat segment avoids padding K to a
# power of two, so no lanes idle when K is not one. Each program writes -1/2 * its psi * lap sums, per (z, k)
# position, into its own partial row; one torch reduction folds the rows and z into ekin. No atomics, so the
# result is deterministic.
#
# Precision (hpcagent_bench/docs/numerical_validation.md): fp64 and fp32 compute and accumulate in themselves;
# fp16, bf16 and both fp8 formats load into fp32, compute and accumulate there (lap, the partial rows and the
# ekin reduction), and round to the storage format only when lap and ekin are written.
import torch
import triton
import triton.language as tl


def laplacian_configs():
    # Measured on the RTX 5060 Ti (fp32; N, K in (106, 14), (64, 8), (96, 64), (128, 33), (160, 16), (200, 5)),
    # the default budget keeps the first four: the best geometric-mean slowdown against each shape's best (1.04),
    # then the fastest at (106, 14) and (200, 5), at (160, 16), and at (96, 64).
    return [
        triton.Config({"BLOCK": 128, "X_CHUNK": 32}, num_warps=2),
        triton.Config({"BLOCK": 256, "X_CHUNK": 16}, num_warps=2),
        triton.Config({"BLOCK": 512, "X_CHUNK": 32}, num_warps=4),
        triton.Config({"BLOCK": 256, "X_CHUNK": 32}, num_warps=4),
        triton.Config({"BLOCK": 256, "X_CHUNK": 32}, num_warps=2),
        triton.Config({"BLOCK": 128, "X_CHUNK": 64}, num_warps=2),
    ]


# The partial buffer has a row per (x chunk, y); the config with the smallest X_CHUNK writes the most rows.
MIN_X_CHUNK = min(config.kwargs["X_CHUNK"] for config in laplacian_configs())


@triton.jit
def periodic(i, n):
    # Offsets stay within one period (|shift| <= 4 < N and 4 * K < N * K), so a single correction suffices.
    i = tl.where(i >= n, i - n, i)
    return tl.where(i < 0, i + n, i)


@triton.autotune(configs=laplacian_configs(), key=["N", "K"], cache_results=True)
@triton.jit
def laplacian_kernel(psi, lap, partial, coeff, N, K, ACC: tl.constexpr, BLOCK: tl.constexpr, X_CHUNK: tl.constexpr):
    """coeff = [3 * c0, w1, w2, w3, w4, inv_h2] and partial in ACC, the accumulation dtype.

    partial[xc * N + y, s] receives -1/2 * sum over this program's x planes of psi * lap at flat position s."""
    st = tl.program_id(0)
    y = tl.program_id(1)
    xc = tl.program_id(2)
    L = N * K  # stride of y
    plane = N.to(tl.int64) * L  # stride of x (N^3 * K may exceed 2^31)
    s = st * BLOCK + tl.arange(0, BLOCK)
    mask = s < L

    c0 = tl.load(coeff)
    w1 = tl.load(coeff + 1)
    w2 = tl.load(coeff + 2)
    w3 = tl.load(coeff + 3)
    w4 = tl.load(coeff + 4)
    scale = tl.load(coeff + 5)

    zm1 = periodic(s - K, L)
    zp1 = periodic(s + K, L)
    zm2 = periodic(s - 2 * K, L)
    zp2 = periodic(s + 2 * K, L)
    zm3 = periodic(s - 3 * K, L)
    zp3 = periodic(s + 3 * K, L)
    zm4 = periodic(s - 4 * K, L)
    zp4 = periodic(s + 4 * K, L)

    column = psi + y * L + s
    x0 = xc * X_CHUNK
    # Register queue: q0 = x - 4, ..., q4 = x (centre), ..., q8 = x + 4.
    q0 = tl.load(column + periodic(x0 - 4, N) * plane, mask=mask, other=0.0).to(ACC)
    q1 = tl.load(column + periodic(x0 - 3, N) * plane, mask=mask, other=0.0).to(ACC)
    q2 = tl.load(column + periodic(x0 - 2, N) * plane, mask=mask, other=0.0).to(ACC)
    q3 = tl.load(column + periodic(x0 - 1, N) * plane, mask=mask, other=0.0).to(ACC)
    q4 = tl.load(column + x0 * plane, mask=mask, other=0.0).to(ACC)
    q5 = tl.load(column + periodic(x0 + 1, N) * plane, mask=mask, other=0.0).to(ACC)
    q6 = tl.load(column + periodic(x0 + 2, N) * plane, mask=mask, other=0.0).to(ACC)
    q7 = tl.load(column + periodic(x0 + 3, N) * plane, mask=mask, other=0.0).to(ACC)
    energy = tl.zeros((BLOCK,), dtype=ACC)
    for x in range(x0, tl.minimum(x0 + X_CHUNK, N)):
        q8 = tl.load(column + periodic(x + 4, N) * plane, mask=mask, other=0.0).to(ACC)
        here = psi + x * plane
        # The reference's order of operations: axis 0, 1, 2; radius 1..4; acc + w * (psi[i - m] + psi[i + m]).
        # The results need not match the reference bit for bit: the compiler may contract a multiply and an add
        # into one FMA, which rounds once instead of twice. The repository allows that contraction (flags.py
        # pins -ffp-contract=fast for the compiled columns), and the tolerance band covers it.
        acc = c0 * q4
        acc = acc + w1 * (q3 + q5)
        acc = acc + w2 * (q2 + q6)
        acc = acc + w3 * (q1 + q7)
        acc = acc + w4 * (q0 + q8)
        yb = here + s
        acc = acc + w1 * (
            tl.load(yb + periodic(y - 1, N) * L, mask=mask, other=0.0).to(ACC)
            + tl.load(yb + periodic(y + 1, N) * L, mask=mask, other=0.0).to(ACC)
        )
        acc = acc + w2 * (
            tl.load(yb + periodic(y - 2, N) * L, mask=mask, other=0.0).to(ACC)
            + tl.load(yb + periodic(y + 2, N) * L, mask=mask, other=0.0).to(ACC)
        )
        acc = acc + w3 * (
            tl.load(yb + periodic(y - 3, N) * L, mask=mask, other=0.0).to(ACC)
            + tl.load(yb + periodic(y + 3, N) * L, mask=mask, other=0.0).to(ACC)
        )
        acc = acc + w4 * (
            tl.load(yb + periodic(y - 4, N) * L, mask=mask, other=0.0).to(ACC)
            + tl.load(yb + periodic(y + 4, N) * L, mask=mask, other=0.0).to(ACC)
        )
        zb = here + y * L
        acc = acc + w1 * (
            tl.load(zb + zm1, mask=mask, other=0.0).to(ACC) + tl.load(zb + zp1, mask=mask, other=0.0).to(ACC)
        )
        acc = acc + w2 * (
            tl.load(zb + zm2, mask=mask, other=0.0).to(ACC) + tl.load(zb + zp2, mask=mask, other=0.0).to(ACC)
        )
        acc = acc + w3 * (
            tl.load(zb + zm3, mask=mask, other=0.0).to(ACC) + tl.load(zb + zp3, mask=mask, other=0.0).to(ACC)
        )
        acc = acc + w4 * (
            tl.load(zb + zm4, mask=mask, other=0.0).to(ACC) + tl.load(zb + zp4, mask=mask, other=0.0).to(ACC)
        )
        out = scale * acc
        tl.store(lap + x * plane + y * L + s, out.to(lap.dtype.element_ty), mask=mask)
        energy += q4 * out
        q0, q1, q2, q3, q4, q5, q6, q7 = q1, q2, q3, q4, q5, q6, q7, q8
    row = (xc * N + y).to(tl.int64)
    # -1/2 is a power of two, so scaling the partial sums here rounds exactly as scaling ekin would.
    tl.store(partial + row * L + s, -0.5 * energy, mask=mask)


def accumulation_dtype(dtype):
    """The torch dtype ``dtype`` accumulates in: itself for fp64 and fp32, fp32 for fp16, bf16 and fp8."""
    return torch.float64 if dtype == torch.float64 else torch.float32


# The last coefficient tensor, keyed by (inv_h2, dtype, device): building it costs a host-to-device copy per call.
COEFFICIENTS: dict[tuple[float, torch.dtype, torch.device], torch.Tensor] = {}


def coefficients(inv_h2, dtype, device):
    """[3 * c0, w1, w2, w3, w4, inv_h2] with the reference's literals (_C0, _CW), stored in ``dtype``, the
    accumulation dtype."""
    key = (float(inv_h2), dtype, device)
    if key not in COEFFICIENTS:
        COEFFICIENTS.clear()
        COEFFICIENTS[key] = torch.tensor(
            [3.0 * (-205.0 / 72.0), 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0, float(inv_h2)],
            dtype=dtype,
            device=device,
        )
    return COEFFICIENTS[key]


def kernel(inv_h2, psi, lap, ekin):
    N = int(psi.shape[0])
    K = int(psi.shape[3])
    acc = accumulation_dtype(psi.dtype)
    coeff = coefficients(inv_h2, acc, psi.device)
    # Rows are overwritten, never accumulated, so autotune trials need no restore.
    partial = torch.empty((N * triton.cdiv(N, MIN_X_CHUNK), N * K), dtype=acc, device=psi.device)
    rows = {}

    def grid(meta):
        x_chunks = triton.cdiv(N, meta["X_CHUNK"])
        rows["used"] = N * x_chunks
        return (triton.cdiv(N * K, meta["BLOCK"]), N, x_chunks)

    laplacian_kernel[grid](psi, lap, partial, coeff, N, K, ACC=tl.float64 if acc == torch.float64 else tl.float32)
    # Each row holds (z, k) pairs with k innermost: viewing it as (rows * N, K) leaves one sum over all but k.
    energy = partial[: rows["used"]].view(-1, K)
    if ekin.dtype == acc:
        torch.sum(energy, dim=0, out=ekin)
    else:
        ekin.copy_(torch.sum(energy, dim=0))
