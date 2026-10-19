# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for lu: right-looking elimination with the trailing rows in parallel.

The NumpyToNumba emit keeps ``A[k + 1:, k + 1:] -= np.outer(...)``, an in-place slice update that reads the array
it writes, so it drops ``parallel=True`` and the reference ran on one core. Step ``k`` reads row ``k`` and writes
only the rows below it, each on its own: those rows are the parallel loop. Each row scales its multiplier and
applies the rank-1 update with numpy's operations in numpy's order, so the result is bitwise the numpy
reference's.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def kernel(A: np.ndarray, N: int) -> None:
    for k in range(N):
        pivot = A[k, k]
        for i in nb.prange(k + 1, N):
            multiplier = A[i, k] / pivot
            A[i, k] = multiplier
            for j in range(k + 1, N):
                A[i, j] -= multiplier * A[k, j]
