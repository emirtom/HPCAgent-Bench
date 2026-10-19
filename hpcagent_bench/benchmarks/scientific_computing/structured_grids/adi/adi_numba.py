# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for adi: both Thomas sweeps with their independent lines in parallel.

The NumpyToNumba emit drops ``parallel=True`` for this kernel (its body updates arrays from shifted slices of
themselves) and the reference ran on one core. Each sweep is a recurrence along one axis and independent along
the other: the row sweep's recurrence runs down ``j`` for every column, so a thread owns a block of columns and
walks the rows; the column sweep's runs along ``j`` inside one row, so a thread owns rows. Every element is
computed with the reference's operations in the reference's order, so the result is bitwise the numpy
reference's.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np

#: Columns one thread carries down the row sweep's recurrence.
COLUMN_BLOCK = 64


@nb.njit(parallel=True, cache=True)
def kernel(TSTEPS: int, N: int, u: np.ndarray, b1: float = 2.0, b2: float = 1.0) -> None:
    v = np.empty((N, N), dtype=u.dtype)
    p = np.empty((N, N), dtype=u.dtype)
    q = np.empty((N, N), dtype=u.dtype)
    pt = np.empty((N, N), dtype=u.dtype)
    qt = np.empty((N, N), dtype=u.dtype)

    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    mul1 = b1 * DT / (DX * DX)
    mul2 = b2 * DT / (DY * DY)

    a = -mul1 / 2.0
    b = 1.0 + mul1
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d
    blocks = (N - 2 + COLUMN_BLOCK - 1) // COLUMN_BLOCK

    for _ in range(1, TSTEPS + 1):
        for block in nb.prange(blocks):
            lo = 1 + block * COLUMN_BLOCK
            hi = min(lo + COLUMN_BLOCK, N - 1)
            for i in range(lo, hi):
                v[0, i] = 1.0
                pt[0, i] = 0.0
                qt[0, i] = v[0, i]
            for j in range(1, N - 1):
                for i in range(lo, hi):
                    inv = 1.0 / (a * pt[j - 1, i] + b)
                    qt[j, i] = (-d * u[j, i - 1] + (1.0 + 2.0 * d) * u[j, i] - f * u[j, i + 1] - a * qt[j - 1, i]) * inv
                    pt[j, i] = -c * inv
            for i in range(lo, hi):
                v[N - 1, i] = 1.0
            for j in range(N - 2, 0, -1):
                for i in range(lo, hi):
                    v[j, i] = pt[j, i] * v[j + 1, i] + qt[j, i]

        for i in nb.prange(1, N - 1):
            u[i, 0] = 1.0
            p[i, 0] = 0.0
            q[i, 0] = u[i, 0]
            for j in range(1, N - 1):
                inv = 1.0 / (d * p[i, j - 1] + e)
                q[i, j] = (-a * v[i - 1, j] + (1.0 + 2.0 * a) * v[i, j] - c * v[i + 1, j] - d * q[i, j - 1]) * inv
                p[i, j] = -f * inv
            u[i, N - 1] = 1.0
            for j in range(N - 2, 0, -1):
                u[i, j] = p[i, j] * u[i, j + 1] + q[i, j]
