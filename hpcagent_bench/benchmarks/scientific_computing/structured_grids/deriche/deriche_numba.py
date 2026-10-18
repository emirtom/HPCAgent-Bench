# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for deriche: each recursive filter pass with its independent lines in parallel.

The NumpyToNumba emit drops ``parallel=True`` for this kernel (its passes update arrays from shifted slices of
themselves) and the reference ran on one core. The horizontal passes recur along the second axis and are
independent per row, so a thread owns rows; the vertical passes recur along the first axis and are independent per
column, so a thread owns a block of columns and walks the rows. Every element is computed with the reference's
operations in the reference's order, so the result is the numpy reference's up to the last bit of the
coefficients' ``exp``.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np

#: Columns one thread carries down a vertical pass.
COLUMN_BLOCK = 64


@nb.njit(parallel=True, cache=True)
def kernel(alpha: float, imgIn: np.ndarray, imgOut: np.ndarray, H: int, W: int) -> None:
    k = (1.0 - np.exp(-alpha)) * (1.0 - np.exp(-alpha)) / (1.0 + 2.0 * alpha * np.exp(-alpha) - np.exp(2.0 * alpha))
    a1 = a5 = k
    a2 = a6 = k * np.exp(-alpha) * (alpha - 1.0)
    a3 = a7 = k * np.exp(-alpha) * (alpha + 1.0)
    a4 = a8 = -k * np.exp(-2.0 * alpha)
    b1 = 2.0 ** (-alpha)
    b2 = -np.exp(-2.0 * alpha)

    y1 = np.empty_like(imgIn)
    y2 = np.empty_like(imgIn)
    for i in nb.prange(W):
        y1[i, 0] = a1 * imgIn[i, 0]
        y1[i, 1] = a1 * imgIn[i, 1] + a2 * imgIn[i, 0] + b1 * y1[i, 0]
        for j in range(2, H):
            y1[i, j] = a1 * imgIn[i, j] + a2 * imgIn[i, j - 1] + b1 * y1[i, j - 1] + b2 * y1[i, j - 2]
        y2[i, H - 1] = 0.0
        y2[i, H - 2] = a3 * imgIn[i, H - 1]
        for j in range(H - 3, -1, -1):
            y2[i, j] = a3 * imgIn[i, j + 1] + a4 * imgIn[i, j + 2] + b1 * y2[i, j + 1] + b2 * y2[i, j + 2]
        for j in range(H):
            imgOut[i, j] = y1[i, j] + y2[i, j]

    blocks = (H + COLUMN_BLOCK - 1) // COLUMN_BLOCK
    for block in nb.prange(blocks):
        lo = block * COLUMN_BLOCK
        hi = min(lo + COLUMN_BLOCK, H)
        for j in range(lo, hi):
            y1[0, j] = a5 * imgOut[0, j]
            y1[1, j] = a5 * imgOut[1, j] + a6 * imgOut[0, j] + b1 * y1[0, j]
        for i in range(2, W):
            for j in range(lo, hi):
                y1[i, j] = a5 * imgOut[i, j] + a6 * imgOut[i - 1, j] + b1 * y1[i - 1, j] + b2 * y1[i - 2, j]
        for j in range(lo, hi):
            y2[W - 1, j] = 0.0
            y2[W - 2, j] = a7 * imgOut[W - 1, j]
        for i in range(W - 3, -1, -1):
            for j in range(lo, hi):
                y2[i, j] = a7 * imgOut[i + 1, j] + a8 * imgOut[i + 2, j] + b1 * y2[i + 1, j] + b2 * y2[i + 2, j]
        for i in range(W):
            for j in range(lo, hi):
                imgOut[i, j] = y1[i, j] + y2[i, j]
