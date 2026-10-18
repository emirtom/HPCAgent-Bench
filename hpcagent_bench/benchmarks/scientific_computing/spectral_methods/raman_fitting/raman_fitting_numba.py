# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for raman_fitting: the Levenberg-Marquardt fit with its grid work in parallel.

The NumpyToNumba emit drops ``parallel=True`` because the body calls ``np.min`` / ``np.max``, which numba's parfor
pass answers differently from numpy on NaN. The iteration is sequential; inside it the O(N K) work is the model
and the exact Jacobian over the grid, one grid point per parallel iteration. The O(N K^2) normal equations stay
the reference's matrix products. Every model value and Jacobian entry is computed with the reference's
operations in its order; the minima and maxima keep numpy's NaN rule. The fit is the numpy reference's up to the
summation order inside the BLAS products.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np

MAX_ITERATIONS = 200
INITIAL_DAMPING = 1.0e-3


@nb.njit(cache=True)
def nan_extremes(values: np.ndarray) -> tuple[float, float]:
    """``(np.min(values), np.max(values))``: NaN once a NaN is met."""
    lo = values[0]
    hi = values[0]
    for i in range(1, values.shape[0]):
        v = values[i]
        if np.isnan(v):
            return v, v
        lo = min(lo, v)
        hi = max(hi, v)
    return lo, hi


@nb.njit(cache=True)
def max_abs(values: np.ndarray) -> float:
    """``np.max(np.abs(values))``: NaN once a NaN is met."""
    best = abs(values[0])
    for i in range(1, values.shape[0]):
        v = abs(values[i])
        if np.isnan(v):
            return v
        best = max(best, v)
    return best


@nb.njit(parallel=True, cache=True)
def model_residual(grid: np.ndarray, y: np.ndarray, p: np.ndarray, npeaks: int, out: np.ndarray) -> None:
    """``lorentzian_model(grid, centres, widths, amplitudes, base) - y`` into ``out``, one grid point per iteration."""
    base = p[3 * npeaks]
    for n in nb.prange(grid.shape[0]):
        acc = 0.0
        for k in range(npeaks):
            delta = grid[n] - p[3 * k]
            width = p[3 * k + 1]
            denom = delta * delta + width * width
            acc += p[3 * k + 2] * width * width / denom
        out[n] = base + acc - y[n]


@nb.njit(parallel=True, cache=True)
def lorentzian_jacobian(grid: np.ndarray, p: np.ndarray, npeaks: int, jac: np.ndarray) -> None:
    """Exact d(model)/d(x0, gamma, amp) per peak, then the offset column, one grid point per iteration."""
    for n in nb.prange(grid.shape[0]):
        for k in range(npeaks):
            delta = grid[n] - p[3 * k]
            width = p[3 * k + 1]
            amplitude = p[3 * k + 2]
            gamma2 = width * width
            denom = delta * delta + gamma2
            denom2 = denom * denom
            jac[n, 3 * k] = amplitude * gamma2 * 2.0 * delta / denom2
            jac[n, 3 * k + 1] = amplitude * 2.0 * width * delta * delta / denom2
            jac[n, 3 * k + 2] = gamma2 / denom
        jac[n, 3 * npeaks] = 1.0


@nb.njit(cache=True)
def raman_fitting(x: np.ndarray, y: np.ndarray, params: np.ndarray, offset: np.ndarray) -> None:
    npeaks = params.shape[0]
    nparams = 3 * npeaks + 1
    lo, hi = nan_extremes(y)
    span = hi - lo
    p = np.empty(nparams, dtype=np.float64)
    for k in range(npeaks):
        # The two graphene bands, then evenly spaced fallbacks (the reference's initial centres).
        p[3 * k] = 1580.0 if k == 0 else 2670.0 if k == 1 else 1200.0 + 200.0 * k
        p[3 * k + 1] = 10.0
        p[3 * k + 2] = span
    p[3 * npeaks] = lo

    damping = INITIAL_DAMPING
    tol = np.sqrt(np.finfo(y.dtype).eps)
    n = x.shape[0]
    residual = np.empty(n, dtype=y.dtype)
    trial_residual = np.empty(n, dtype=y.dtype)
    jac = np.empty((n, nparams), dtype=x.dtype)
    model_residual(x, y, p, npeaks, residual)
    cost = float(residual @ residual)

    for _ in range(MAX_ITERATIONS):
        lorentzian_jacobian(x, p, npeaks, jac)
        normal = jac.T @ jac
        gradient = jac.T @ residual
        # Marquardt's own scaling: damp along the diagonal of J^T J.
        scale = np.diag(normal).copy()
        for i in range(nparams):
            if scale[i] <= 0.0:
                scale[i] = 1.0
        step = np.linalg.solve(normal + damping * np.diag(scale), -gradient)

        trial = p + step
        model_residual(x, y, trial, npeaks, trial_residual)
        trial_cost = float(trial_residual @ trial_residual)
        if trial_cost < cost:
            converged = (cost - trial_cost) <= tol * cost or max_abs(step) <= tol * (max_abs(trial) + tol)
            p[:] = trial
            residual[:] = trial_residual
            cost = trial_cost
            damping = max(damping * 0.1, 1.0e-14)
            if converged:
                break
        else:
            damping *= 10.0
            if damping > 1.0e14:
                break

    for j in range(npeaks):
        params[j, 0] = p[3 * j]
        params[j, 1] = p[3 * j + 1]
        params[j, 2] = p[3 * j + 2]
    offset[0] = p[3 * npeaks]
