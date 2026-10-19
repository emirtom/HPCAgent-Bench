# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for icon_w_solve: the tridiagonal solve of every column in parallel.

The NumpyToNumba emit drops ``parallel=True`` for this kernel (``w[jk, :]`` is rebuilt from ``w[jk - 1, :]`` in
place) and the reference ran on one core. Levels are sequential and columns independent, and the column axis is
innermost: a thread owns a block of columns and runs every step's forward and backward sweep and the relaxation
of ``z_w_expl`` on it, level by level. Every element is computed with the reference's operations in the
reference's order, so the result is bitwise the numpy reference's.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np

#: Columns one thread carries through the sweeps.
COLUMN_BLOCK = 256


@nb.njit(parallel=True, cache=True)
def icon_w_solve(
    z_alpha: np.ndarray,
    z_beta: np.ndarray,
    theta_v_ic: np.ndarray,
    ddqz_z_half: np.ndarray,
    vwind_impl_wgt: np.ndarray,
    z_w_expl: np.ndarray,
    z_exner_expl: np.ndarray,
    w_lb: np.ndarray,
    z_q: np.ndarray,
    w: np.ndarray,
    dtime: float,
    cpd: float,
    NLEV: int,
    NPROMA: int,
    nsteps: int,
) -> None:
    blocks = (NPROMA + COLUMN_BLOCK - 1) // COLUMN_BLOCK
    scale = dtime * cpd
    for block in nb.prange(blocks):
        lo = block * COLUMN_BLOCK
        hi = min(lo + COLUMN_BLOCK, NPROMA)
        for _step in range(nsteps):
            for c in range(lo, hi):
                z_q[0, c] = 0.0
                w[0, c] = 0.0
                w[NLEV, c] = w_lb[c]
            for jk in range(1, NLEV):
                for c in range(lo, hi):
                    z_gamma = scale * vwind_impl_wgt[c] * theta_v_ic[jk, c] / ddqz_z_half[jk, c]
                    z_a = -z_gamma * z_beta[jk - 1, c] * z_alpha[jk - 1, c]
                    z_c = -z_gamma * z_beta[jk, c] * z_alpha[jk + 1, c]
                    z_b = 1.0 + z_gamma * z_alpha[jk, c] * (z_beta[jk - 1, c] + z_beta[jk, c])
                    z_g = 1.0 / (z_b + z_a * z_q[jk - 1, c])
                    z_q[jk, c] = -z_c * z_g
                    explicit = z_w_expl[jk, c] - z_gamma * (z_exner_expl[jk - 1, c] - z_exner_expl[jk, c])
                    w[jk, c] = (explicit - z_a * w[jk - 1, c]) * z_g
            for jk in range(NLEV - 2, 0, -1):
                for c in range(lo, hi):
                    w[jk, c] = w[jk, c] + w[jk + 1, c] * z_q[jk, c]
            for jk in range(NLEV + 1):
                for c in range(lo, hi):
                    z_w_expl[jk, c] = 0.5 * (z_w_expl[jk, c] + w[jk, c])
