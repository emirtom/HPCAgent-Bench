# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for chebyshev_filter_subspace: the Chebyshev recurrence with each H psi per x-plane in parallel.

numba cannot lower the NumpyToNumba emit (``np.tensordot`` along the middle axes), so numba produced no reference and
dropped out of the best-of race. The recurrence over degrees is sequential; each ``H psi`` applies the 8th-order periodic
stencil along every axis directly -- the circulant band ``_laplacian_1d`` builds, without the dense (N, N) product --
one x-plane per parallel iteration. The filtered block is the numpy reference's up to the summation order of the
stencil, which the dense product adds over zeros as well.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np

C0 = -205.0 / 72.0
CW = (8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0)


@nb.njit(parallel=True, cache=True)
def hpsi_minus(x: np.ndarray, vloc: np.ndarray, half_inv_h2: float, c: float, scale: float, out: np.ndarray) -> None:
    """``out = (H x - c x) * scale``, H = -half_inv_h2 * (periodic Laplacian along x, y, z) + vloc."""
    n0, n1, n2, nk = x.shape
    weights = np.array(CW)
    for i in nb.prange(n0):
        for j in range(n1):
            for k in range(n2):
                for q in range(nk):
                    t0 = C0 * x[i, j, k, q]
                    t1 = C0 * x[i, j, k, q]
                    t2 = C0 * x[i, j, k, q]
                    for m in range(1, 5):
                        w = weights[m - 1]
                        t0 += w * (x[(i + m) % n0, j, k, q] + x[(i - m) % n0, j, k, q])
                        t1 += w * (x[i, (j + m) % n1, k, q] + x[i, (j - m) % n1, k, q])
                        t2 += w * (x[i, j, (k + m) % n2, q] + x[i, j, (k - m) % n2, q])
                    h = -half_inv_h2 * (t0 + t1 + t2) + vloc[i, j, k] * x[i, j, k, q]
                    out[i, j, k, q] = (h - c * x[i, j, k, q]) * scale


@nb.njit(parallel=True, cache=True)
def chebyshev_filter_subspace(
    a: float,
    b: float,
    a0: float,
    half_inv_h2: float,
    m: float,
    vloc: np.ndarray,
    X: np.ndarray,
    out: np.ndarray,
    N: int,
) -> None:
    e = 0.5 * (b - a)
    c = 0.5 * (b + a)
    sigma = e / (a0 - c)
    sigma1 = sigma
    prev = X.copy()
    Y = np.empty_like(X)
    hpsi_minus(prev, vloc, half_inv_h2, c, sigma1 / e, Y)
    Ynew = np.empty_like(X)
    n0 = X.shape[0]
    for degree in range(2, int(m) + 1):
        sigma_new = 1.0 / (2.0 / sigma1 - sigma)
        hpsi_minus(Y, vloc, half_inv_h2, c, 2.0 * sigma_new / e, Ynew)
        damp = sigma * sigma_new
        for i in nb.prange(n0):
            Ynew[i] -= damp * prev[i]
        prev, Y, Ynew = Y, Ynew, prev
        sigma = sigma_new
    out[:] = Y
