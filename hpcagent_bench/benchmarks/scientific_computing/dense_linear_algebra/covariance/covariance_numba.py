# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for covariance: the centring by columns and rows in parallel, the Gramian in BLAS.

numba's parfor pass fails to lower the NumpyToNumba emit (``data -= mean`` broadcast in place), so numba produced no
reference and dropped out of the best-of race. Here the column means are a parallel loop over column blocks (each
column summed down its rows in order, as numpy's axis-0 sum adds row by row), the centring a parallel loop over rows,
and ``data.T @ data`` the BLAS product numpy calls. The result is the numpy reference's up to the summation order
inside the product.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np

#: Columns one thread sums down every row at a time.
COLUMN_BLOCK = 64


@nb.njit(parallel=True, cache=True)
def column_means(data: np.ndarray, N: int) -> np.ndarray:
    """``data.sum(axis=0) / N``, each column summed in row order."""
    M = data.shape[1]
    mean = np.zeros(M, dtype=data.dtype)
    blocks = (M + COLUMN_BLOCK - 1) // COLUMN_BLOCK
    for block in nb.prange(blocks):
        lo = block * COLUMN_BLOCK
        hi = min(lo + COLUMN_BLOCK, M)
        for i in range(N):
            for j in range(lo, hi):
                mean[j] += data[i, j]
        for j in range(lo, hi):
            mean[j] /= N
    return mean


@nb.njit(parallel=True, cache=True)
def kernel(M: int, float_n: float, data: np.ndarray, cov: np.ndarray, N: int) -> None:
    mean = column_means(data, N)
    for i in nb.prange(N):
        for j in range(M):
            data[i, j] -= mean[j]
    gram = data.T @ data
    scale = float_n - 1.0
    for i in nb.prange(M):
        for j in range(M):
            cov[i, j] = gram[i, j] / scale
