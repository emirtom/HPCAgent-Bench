# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for conv_2d: the box convolution with the output rows in parallel.

The NumpyToNumba emit drops ``parallel=True`` for this kernel: the edge padding copies rows of ``padded`` into
other rows of ``padded`` (``padded[k, :] = padded[R, :]``), an in-place slice copy its race guard refuses, and
the only ``range`` loop it could hand to ``prange`` is that padding loop. Here the padding stays serial (it is
O(N R)) and the O(N^2 R^2) tap loop runs one output row per iteration. Each output sums its taps in the numpy
reference's order, so the result is bitwise the reference's.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def conv_2d(in_grid: np.ndarray, out_grid: np.ndarray, w_box: np.ndarray, N: int, R: int) -> None:
    padded = np.empty((N + 2 * R, N + 2 * R), dtype=in_grid.dtype)
    for i in nb.prange(N):
        for j in range(N):
            padded[R + i, R + j] = in_grid[i, j]
        for k in range(R):
            padded[R + i, k] = in_grid[i, 0]
            padded[R + i, R + N + k] = in_grid[i, N - 1]
    for k in range(R):
        for j in range(N + 2 * R):
            padded[k, j] = padded[R, j]
            padded[R + N + k, j] = padded[R + N - 1, j]
    K = 2 * R + 1
    for i in nb.prange(N):
        for j in range(N):
            acc = 0.0
            for di in range(K):
                for dj in range(K):
                    acc += w_box[di, dj] * padded[i + di, j + dj]
            out_grid[i, j] = acc
