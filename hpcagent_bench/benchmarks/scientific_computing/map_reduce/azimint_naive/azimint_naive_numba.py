# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for azimint_naive: the binning per point, the sums per thread.

The NumpyToNumba emit drops ``parallel=True`` because the body calls ``radius.max()``, which numba's parfor pass
answers differently from numpy on NaN. Here the maximum keeps numpy's NaN rule (the first NaN is the maximum),
every point is binned with the reference's exact arithmetic (the same edges and the same two one-step
corrections, element by element), and the scatter-add runs as one private (sum, count) pair per chunk of points,
added in chunk order. Counts are exact; a bin's sum differs from the reference's only in its summation order.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np

#: Points one private (sum, count) pair accumulates.
CHUNK = 65536


@nb.njit(parallel=True, cache=True)
def azimint_naive(data: np.ndarray, radius: np.ndarray, npt: int, res: np.ndarray) -> None:
    n = radius.shape[0]
    rmax = radius[0]
    for p in range(1, n):
        if np.isnan(rmax):
            break
        if radius[p] > rmax or np.isnan(radius[p]):
            rmax = radius[p]
    edges = np.empty(npt + 1, dtype=radius.dtype)
    for i in range(npt + 1):
        edges[i] = rmax * radius.dtype.type(i) / npt
    chunks = (n + CHUNK - 1) // CHUNK
    sums = np.zeros((chunks, npt), dtype=data.dtype)
    counts = np.zeros((chunks, npt), dtype=data.dtype)
    for chunk in nb.prange(chunks):
        for p in range(chunk * CHUNK, min((chunk + 1) * CHUNK, n)):
            r = radius[p]
            bin1 = min(max(np.int64(np.floor(r * npt / rmax)), -1), npt)
            safe = min(max(bin1, 0), npt)
            bin2 = bin1 + np.int64((bin1 + 1 <= npt) and (edges[min(safe + 1, npt)] <= r))
            bin3 = bin2 - np.int64(edges[min(max(bin2, 0), npt)] > r)
            bin4 = min(max(bin3, -1), npt)
            if bin4 >= 0 and bin4 < npt:
                sums[chunk, bin4] += data[p]
                counts[chunk, bin4] += 1.0
    total = np.zeros(npt, dtype=data.dtype)
    hits = np.zeros(npt, dtype=data.dtype)
    for chunk in range(chunks):
        for b in range(npt):
            total[b] += sums[chunk, b]
            hits[b] += counts[chunk, b]
    for b in range(npt):
        res[b] = total[b] / hits[b] if hits[b] != 0.0 else np.nan
