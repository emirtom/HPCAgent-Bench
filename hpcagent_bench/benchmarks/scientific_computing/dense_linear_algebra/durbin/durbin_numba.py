# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for durbin: the Levinson-Durbin recurrence with each step's vectors in parallel.

The NumpyToNumba emit keeps ``y[:k] += alpha * np.flip(y[:k])``, an in-place update that reads the array it writes
in reverse, so it drops ``parallel=True`` and the reference ran on one core. The steps are a true recurrence;
inside one step both O(k) pieces parallelize: the reversed dot as a reduction, and the update as pairs
``(i, k - 1 - i)`` that read each other's OLD values, one pair per iteration. Below ``PARALLEL_MIN`` elements a
step stays serial, where a parallel region would cost more than it saves. The update is bitwise the numpy
reference's; the dot differs only in its summation order.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np

#: The step length from which a step's dot and update run in parallel.
PARALLEL_MIN = 8192


@nb.njit(parallel=True, cache=True)
def parallel_step(r: np.ndarray, y: np.ndarray, k: int, beta: float) -> float:
    """One step at length ``k`` (its ``alpha``), the vector work in parallel."""
    acc = 0.0
    for i in nb.prange(k):
        acc += r[k - 1 - i] * y[i]
    alpha = -(r[k] + acc) / beta
    half = k // 2
    for i in nb.prange(half):
        low = y[i]
        high = y[k - 1 - i]
        y[i] = low + alpha * high
        y[k - 1 - i] = high + alpha * low
    if k % 2 == 1:
        y[half] += alpha * y[half]
    y[k] = alpha
    return alpha


@nb.njit(cache=True)
def serial_step(r: np.ndarray, y: np.ndarray, k: int, beta: float) -> float:
    """One step at length ``k`` (its ``alpha``), on one core."""
    acc = 0.0
    for i in range(k):
        acc += r[k - 1 - i] * y[i]
    alpha = -(r[k] + acc) / beta
    half = k // 2
    for i in range(half):
        low = y[i]
        high = y[k - 1 - i]
        y[i] = low + alpha * high
        y[k - 1 - i] = high + alpha * low
    if k % 2 == 1:
        y[half] += alpha * y[half]
    y[k] = alpha
    return alpha


@nb.njit(cache=True)
def kernel(r: np.ndarray, y: np.ndarray, N: int) -> None:
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]
    for k in range(1, N):
        beta *= 1.0 - alpha * alpha
        if k < PARALLEL_MIN:
            alpha = serial_step(r, y, k, beta)
        else:
            alpha = parallel_step(r, y, k, beta)
