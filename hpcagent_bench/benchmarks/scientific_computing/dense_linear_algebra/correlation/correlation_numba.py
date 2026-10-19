# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for correlation: the column statistics and the scaling in parallel, the Gramian in BLAS.

numba's parfor pass fails to lower the NumpyToNumba emit (``data -= mean`` broadcast in place), so numba produced no
reference and dropped out of the best-of race. Here the column means and standard deviations are parallel loops over
column blocks (each column summed down its rows in order, as numpy's axis-0 sum adds row by row), the centring and
scaling a parallel loop over rows, and ``data.T @ data`` the BLAS product numpy calls. The result is the numpy
reference's up to the summation order inside the product.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np

#: Columns one thread sums down every row at a time.
COLUMN_BLOCK = 64


@nb.njit(parallel=True, cache=True)
def kernel(
    M: int,
    float_n: float,
    data: np.ndarray,
    corr: np.ndarray,
    N: int,
    stddev_eps: float = 0.1,
    stddev_replacement: float = 1.0,
) -> None:
    mean = np.zeros(M, dtype=data.dtype)
    stddev = np.zeros(M, dtype=data.dtype)
    blocks = (M + COLUMN_BLOCK - 1) // COLUMN_BLOCK
    for block in nb.prange(blocks):
        lo = block * COLUMN_BLOCK
        hi = min(lo + COLUMN_BLOCK, M)
        for i in range(N):
            for j in range(lo, hi):
                mean[j] += data[i, j]
        for j in range(lo, hi):
            mean[j] /= N
        for i in range(N):
            for j in range(lo, hi):
                centred = data[i, j] - mean[j]
                stddev[j] += centred * centred
        for j in range(lo, hi):
            stddev[j] = np.sqrt(stddev[j] / N)
            if stddev[j] <= stddev_eps:
                stddev[j] = stddev_replacement
    root_n = np.sqrt(float_n)
    for i in nb.prange(N):
        for j in range(M):
            data[i, j] -= mean[j]
            data[i, j] /= root_n * stddev[j]
    gram = data.T @ data
    for i in nb.prange(M):
        for j in range(M):
            corr[i, j] = gram[i, j]
        corr[i, i] = 1.0
