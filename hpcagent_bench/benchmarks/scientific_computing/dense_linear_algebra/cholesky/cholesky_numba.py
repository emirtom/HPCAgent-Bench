# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for cholesky: each column's trailing rows in parallel.

The NumpyToNumba emit keeps ``A[j + 1:, j] -= A[j + 1:, :j] @ A[j, :j]``, an in-place slice update that reads the
array it writes, so it drops ``parallel=True`` and the reference ran on one core. Column ``j`` reads only rows
of columns ``< j`` and the finished diagonal, and each row below the diagonal writes only its own entry: those
rows are the parallel loop, one contiguous dot each. Same column-oriented Crout order as the numpy reference, so
the factor is the reference's up to the summation order of each dot.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def kernel(A: np.ndarray, N: int) -> None:
    for j in range(N):
        A[j, j] = np.sqrt(A[j, j] - np.dot(A[j, :j], A[j, :j]))
        diagonal = A[j, j]
        for i in nb.prange(j + 1, N):
            acc = 0.0
            for p in range(j):
                acc += A[i, p] * A[j, p]
            A[i, j] = (A[i, j] - acc) / diagonal
