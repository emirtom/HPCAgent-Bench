# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for gaussian: forward elimination with the rows below the pivot in parallel.

The NumpyToNumba emit keeps ``A[k + 1:, k:] -= mult[:, None] * A[k, k:]``, an in-place slice update that reads the
array it writes, so it drops ``parallel=True`` and the reference ran on one core. Step ``k`` only reads row ``k``
and writes the rows below it, each row on its own: those rows are the parallel loop. Each element is updated
with numpy's operations in numpy's order (``mult`` read before the row is touched), so the result is bitwise the
numpy reference's.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def gaussian(A: np.ndarray, b: np.ndarray, N: int) -> None:
    for k in range(N - 1):
        pivot = A[k, k]
        bk = b[k]
        for i in nb.prange(k + 1, N):
            mult = A[i, k] / pivot
            for j in range(k, N):
                A[i, j] -= mult * A[k, j]
            b[i] -= mult * bk
