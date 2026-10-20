# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for kmeans: Lloyd iterations with the assignment and the sums in parallel.

numba's parfor pass fails to lower the NumpyToNumba emit (the one-hot comparison broadcast), so numba produced no
reference and dropped out of the best-of race. Each iteration: the distance matrix with the reference's arithmetic
around the BLAS product ``X @ centroids.T``, each point's first minimum (numpy's argmin) in parallel, and the cluster
sums as one private (sum, count) pair per chunk of points, folded in chunk order. Labels are exact; a centroid
differs from the reference's only in its summation order.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np

#: Points one private (sum, count) pair accumulates.
CHUNK = 4096


@nb.njit(parallel=True, cache=True)
def kmeans(X: np.ndarray, centroids: np.ndarray, niter: int, nclusters: int) -> None:
    npoints, dim = X.shape
    K = nclusters
    x_sqnorm = np.empty(npoints, dtype=X.dtype)
    for i in nb.prange(npoints):
        acc = 0.0
        for d in range(dim):
            acc += X[i, d] * X[i, d]
        x_sqnorm[i] = acc
    labels = np.empty(npoints, dtype=np.int64)
    chunks = (npoints + CHUNK - 1) // CHUNK
    for iteration in range(niter):
        c_sqnorm = np.empty(K, dtype=X.dtype)
        for c in range(K):
            acc = 0.0
            for d in range(dim):
                acc += centroids[c, d] * centroids[c, d]
            c_sqnorm[c] = acc
        cross = X @ centroids.T
        for i in nb.prange(npoints):
            best = 0
            best_dist = x_sqnorm[i] - 2.0 * cross[i, 0] + c_sqnorm[0]
            for c in range(1, K):
                if np.isnan(best_dist):
                    break
                dist = x_sqnorm[i] - 2.0 * cross[i, c] + c_sqnorm[c]
                if dist < best_dist or np.isnan(dist):
                    best = c
                    best_dist = dist
            labels[i] = best
        sums = np.zeros((chunks, K, dim), dtype=X.dtype)
        counts = np.zeros((chunks, K), dtype=X.dtype)
        for chunk in nb.prange(chunks):
            for i in range(chunk * CHUNK, min((chunk + 1) * CHUNK, npoints)):
                c = labels[i]
                counts[chunk, c] += 1.0
                for d in range(dim):
                    sums[chunk, c, d] += X[i, d]
        for c in range(K):
            count = 0.0
            for chunk in range(chunks):
                count += counts[chunk, c]
            for d in range(dim):
                total = 0.0
                for chunk in range(chunks):
                    total += sums[chunk, c, d]
                centroids[c, d] = total / max(count, 1.0)
