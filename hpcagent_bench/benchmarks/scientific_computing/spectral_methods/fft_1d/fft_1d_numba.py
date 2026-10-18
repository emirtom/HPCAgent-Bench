# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for fft_1d: a four-step FFT whose every step runs on all cores.

Numba cannot type ``np.fft``, and the NumpyToNumba emit's direct O(N^2) DFT cannot finish the judge's draw
(N ~ 1e8). One length-N transform through ``scipy.fft`` is pocketfft on ONE core: its ``workers`` split a batch
of transforms, never a single one. The four-step (Bailey) factorisation turns the one transform into batches:
with N = N1 * N2 and the input viewed as an (N1, N2) array,

1. N2 transforms of length N1 down the columns (one batched ``scipy.fft.fft`` call, ``workers`` = numba's thread
   budget);
2. the twiddle factors ``exp(-2 pi i k1 n2 / N)``, an elementwise multiply over the rows in parallel;
3. N1 transforms of length N2 along the rows (one batched call);
4. the transpose ``X[k1 + N1 k2] = C[k1, k2]``, in parallel blocks.

The inverse is the forward transform of the conjugate, conjugated and divided by N. N1 is the largest divisor of N
not above sqrt(N) (every draw is 7-smooth, so it is near sqrt(N)); a prime N is one transform along one row. The
result equals numpy's ``np.fft.fft`` / ``ifft`` up to the rounding of a different (equally O(log N)-accurate)
factorisation: each twiddle is evaluated from the exact integer ``k1 n2 mod N``.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import math

import numba as nb
import numpy as np
import scipy.fft

#: Columns one thread moves in a transpose block.
TRANSPOSE_BLOCK = 64


@nb.njit(cache=True)
def split(n: int) -> int:
    """The largest divisor of ``n`` not above ``sqrt(n)`` (1 when ``n`` is prime)."""
    best = 1
    d = 1
    while d * d <= n:
        if n % d == 0:
            best = d
        d += 1
    return best


@nb.njit(parallel=True, cache=True)
def twiddle(b: np.ndarray, n: int) -> None:
    """``b[k1, n2] *= exp(-2 pi i k1 n2 / n)``, each angle from the exact ``k1 n2 mod n``."""
    rows, cols = b.shape
    scale = -2.0 * math.pi / n
    for k1 in nb.prange(rows):
        for n2 in range(cols):
            angle = scale * ((k1 * n2) % n)
            b[k1, n2] *= complex(math.cos(angle), math.sin(angle))


@nb.njit(parallel=True, cache=True)
def transpose_flat(c: np.ndarray, out: np.ndarray, sign: float, divisor: float) -> None:
    """``out[k1 + N1 k2] = c[k1, k2]``, conjugated when ``sign < 0`` and divided by ``divisor``."""
    rows, cols = c.shape
    blocks = (cols + TRANSPOSE_BLOCK - 1) // TRANSPOSE_BLOCK
    for block in nb.prange(blocks):
        for k2 in range(block * TRANSPOSE_BLOCK, min((block + 1) * TRANSPOSE_BLOCK, cols)):
            for k1 in range(rows):
                value = c[k1, k2]
                if sign < 0.0:
                    value = value.conjugate()
                out[k1 + rows * k2] = value / divisor


@nb.njit(parallel=True, cache=True)
def conjugate_into(src: np.ndarray, dst: np.ndarray) -> None:
    for i in nb.prange(src.shape[0]):
        dst[i] = src[i].conjugate()


@nb.njit(cache=True)
def batched_fft(a: np.ndarray, axis: int, workers: int) -> np.ndarray:
    """One batched ``scipy.fft.fft`` along ``axis`` of a C-contiguous 2-D array, on ``workers`` threads."""
    with nb.objmode(out="complex128[:, ::1]"):
        out = np.ascontiguousarray(scipy.fft.fft(a, axis=axis, workers=workers))
    return out


@nb.njit(cache=True)
def four_step(x: np.ndarray, out: np.ndarray, inverse: bool) -> None:
    """``out = fft(x)`` (``inverse``: ``ifft(x)``) by the four-step factorisation. A prime ``n`` (``n1 = 1``)
    degenerates to one transform along the single row."""
    n = x.shape[0]
    workers = nb.get_num_threads()
    source = np.empty_like(x)
    if inverse:
        conjugate_into(x, source)
    else:
        source[:] = x
    n1 = split(n)
    b = batched_fft(source.reshape((n1, n // n1)), 0, workers)
    twiddle(b, n)
    c = batched_fft(b, 1, workers)
    transpose_flat(c, out, -1.0 if inverse else 1.0, float(n) if inverse else 1.0)


@nb.njit(cache=True)
def fft_1d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    four_step(x, y, False)
    four_step(y, z, True)
