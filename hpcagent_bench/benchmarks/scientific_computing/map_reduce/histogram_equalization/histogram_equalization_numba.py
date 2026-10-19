# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for histogram_equalization: a per-thread histogram, then a parallel remap.

The NumpyToNumba emit drops ``parallel=True`` because the body calls ``.min()``, which numba's parfor pass answers
differently from numpy on NaN. Both O(H W) passes parallelize: the histogram as one private histogram per row
block, summed in block order (integer counts, so the order cannot change them), and the LUT gather as a row
loop. The O(nbins) cumulative sum and LUT stay serial and follow the numpy reference's arithmetic, so the output
is bitwise the reference's. Every pixel is below ``nbins`` (the manifest's contract), so a pixel's bin is its
value, as ``np.histogram(flat, nbins, range=(0, nbins))`` bins it.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np

#: Rows of ``img`` one private histogram counts.
ROW_BLOCK = 64


@nb.njit(parallel=True, cache=True)
def histogram_equalization(img: np.ndarray, out: np.ndarray, H: int, W: int, nbins: int) -> None:
    npix = H * W
    blocks = (H + ROW_BLOCK - 1) // ROW_BLOCK
    partial = np.zeros((blocks, nbins), dtype=np.int64)
    for block in nb.prange(blocks):
        for i in range(block * ROW_BLOCK, min((block + 1) * ROW_BLOCK, H)):
            for j in range(W):
                partial[block, img[i, j]] += 1
    hist = np.zeros(nbins, dtype=np.int64)
    for block in range(blocks):
        for v in range(nbins):
            hist[v] += partial[block, v]

    cdf = np.cumsum(hist)
    # Sentinel npix (an upper bound on any count) replaces empty leading bins so the minimum ignores them.
    cdf_min = npix
    for v in range(nbins):
        if cdf[v] > 0 and cdf[v] < cdf_min:
            cdf_min = cdf[v]
    lut = np.round((cdf - cdf_min) / max(npix - cdf_min, 1) * (nbins - 1))

    for i in nb.prange(H):
        for j in range(W):
            out[i, j] = lut[img[i, j]]
