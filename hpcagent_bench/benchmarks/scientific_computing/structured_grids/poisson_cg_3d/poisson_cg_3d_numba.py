# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for poisson_cg_3d: conjugate gradients with the stencil and the dots per x-plane in parallel.

numba cannot type the NumpyToNumba emit (``np.roll`` along an axis), so numba produced no reference and dropped out of
the best-of race. The iteration is a recurrence; inside it the periodic 7-point stencil runs one x-plane per parallel
iteration with the reference's operations in its order, and every dot product and mean is a per-plane partial sum
folded in plane order. The potential is the numpy reference's up to the summation order of those sums.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def neg_laplacian(x: np.ndarray, inv_h2: float, out: np.ndarray) -> None:
    """``inv_h2 * (6 x - the six periodic neighbours)``, subtracted in the reference's order."""
    nx, ny, nz = x.shape
    for i in nb.prange(nx):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = inv_h2 * (
                    6.0 * x[i, j, k]
                    - x[(i - 1) % nx, j, k]
                    - x[(i + 1) % nx, j, k]
                    - x[i, (j - 1) % ny, k]
                    - x[i, (j + 1) % ny, k]
                    - x[i, j, (k - 1) % nz]
                    - x[i, j, (k + 1) % nz]
                )


@nb.njit(parallel=True, cache=True)
def dot(a: np.ndarray, b: np.ndarray) -> float:
    """``sum(a * b)``: one partial per x-plane, folded in plane order."""
    nx, ny, nz = a.shape
    planes = np.zeros(nx, dtype=a.dtype)
    for i in nb.prange(nx):
        acc = 0.0
        for j in range(ny):
            for k in range(nz):
                acc += a[i, j, k] * b[i, j, k]
        planes[i] = acc
    total = 0.0
    for i in range(nx):
        total += planes[i]
    return total


@nb.njit(parallel=True, cache=True)
def mean(a: np.ndarray) -> float:
    """``a.mean()``: one partial per x-plane, folded in plane order."""
    nx, ny, nz = a.shape
    planes = np.zeros(nx, dtype=a.dtype)
    for i in nb.prange(nx):
        acc = 0.0
        for j in range(ny):
            for k in range(nz):
                acc += a[i, j, k]
        planes[i] = acc
    total = 0.0
    for i in range(nx):
        total += planes[i]
    return total / a.size


@nb.njit(parallel=True, cache=True)
def poisson_cg_3d(inv_h2: float, tol: float, niter: float, rho: np.ndarray, V: np.ndarray) -> None:
    nx, ny, nz = rho.shape
    b = 4.0 * np.pi * rho
    b_mean = mean(b)
    lap = np.empty_like(V)
    neg_laplacian(V, inv_h2, lap)
    r = np.empty_like(V)
    for i in nb.prange(nx):
        for j in range(ny):
            for k in range(nz):
                r[i, j, k] = (b[i, j, k] - b_mean) - lap[i, j, k]
    p = r.copy()
    Ap = np.empty_like(V)
    rs = dot(r, r)
    for iteration in range(int(niter)):
        neg_laplacian(p, inv_h2, Ap)
        alpha = rs / (dot(p, Ap) + 1.0e-30)
        for i in nb.prange(nx):
            for j in range(ny):
                for k in range(nz):
                    V[i, j, k] += alpha * p[i, j, k]
                    r[i, j, k] -= alpha * Ap[i, j, k]
        rs_new = dot(r, r)
        if np.sqrt(rs_new) < tol:
            break
        beta = rs_new / rs
        for i in nb.prange(nx):
            for j in range(ny):
                for k in range(nz):
                    p[i, j, k] = r[i, j, k] + beta * p[i, j, k]
        rs = rs_new
    v_mean = mean(V)
    for i in nb.prange(nx):
        for j in range(ny):
            for k in range(nz):
                V[i, j, k] -= v_mean
