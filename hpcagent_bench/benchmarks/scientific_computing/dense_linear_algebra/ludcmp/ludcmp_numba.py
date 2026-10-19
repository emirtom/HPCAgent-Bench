# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for ludcmp: the parallel right-looking LU of ``lu``, then the two solves.

The NumpyToNumba emit keeps ``A[k + 1:, k + 1:] -= np.outer(...)``, an in-place slice update that reads the array
it writes, so it drops ``parallel=True`` and the reference ran on one core. The O(N^3) factorization runs the
rows below each pivot in parallel, with numpy's operations in numpy's order (bitwise the reference's factors).
The forward and back substitutions are sequential by definition and O(N^2); they keep the reference's dots.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def kernel(A: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray, N: int) -> None:
    for k in range(N):
        pivot = A[k, k]
        for i in nb.prange(k + 1, N):
            multiplier = A[i, k] / pivot
            A[i, k] = multiplier
            for j in range(k + 1, N):
                A[i, j] -= multiplier * A[k, j]
    for i in range(N):
        y[i] = b[i] - np.dot(A[i, :i], y[:i])
    for i in range(N - 1, -1, -1):
        x[i] = (y[i] - np.dot(A[i, i + 1 :], x[i + 1 :])) / A[i, i]
