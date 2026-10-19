# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for vadv: the Thomas solve of every vertical column in parallel.

The NumpyToNumba emit drops ``parallel=True`` for this kernel (its sweeps update arrays from neighbouring levels
of themselves) and the reference ran on one core. The recurrence runs along ``k`` and every ``(i, j)`` column is
independent, so a thread owns columns and runs the whole forward and backward sweep of each, ``k`` innermost and
contiguous. Every element is computed with the reference's operations in the reference's order, so the result
is bitwise the numpy reference's.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def vadv(
    utens_stage: np.ndarray,
    u_stage: np.ndarray,
    wcon: np.ndarray,
    u_pos: np.ndarray,
    utens: np.ndarray,
    dtr_stage: float,
    K: int,
    bet_m: float = 0.5,
    bet_p: float = 0.5,
) -> None:
    I = utens_stage.shape[0]
    J = utens_stage.shape[1]
    ccol = np.empty_like(utens_stage)
    dcol = np.empty_like(utens_stage)
    for i in nb.prange(I):
        for j in range(J):
            gcv = 0.25 * (wcon[i + 1, j, 1] + wcon[i, j, 1])
            cs = gcv * bet_m
            ccol[i, j, 0] = gcv * bet_p
            bcol = dtr_stage - ccol[i, j, 0]
            correction_term = -cs * (u_stage[i, j, 1] - u_stage[i, j, 0])
            dcol[i, j, 0] = dtr_stage * u_pos[i, j, 0] + utens[i, j, 0] + utens_stage[i, j, 0] + correction_term
            divided = 1.0 / bcol
            ccol[i, j, 0] *= divided
            dcol[i, j, 0] *= divided

            for k in range(1, K - 1):
                gav = -gcv
                gcv = 0.25 * (wcon[i + 1, j, k + 1] + wcon[i, j, k + 1])
                as_ = gav * bet_m
                cs = gcv * bet_m
                acol = gav * bet_p
                ccol[i, j, k] = gcv * bet_p
                bcol = dtr_stage - acol - ccol[i, j, k]
                correction_term = -as_ * (u_stage[i, j, k - 1] - u_stage[i, j, k]) - cs * (
                    u_stage[i, j, k + 1] - u_stage[i, j, k]
                )
                dcol[i, j, k] = dtr_stage * u_pos[i, j, k] + utens[i, j, k] + utens_stage[i, j, k] + correction_term
                divided = 1.0 / (bcol - ccol[i, j, k - 1] * acol)
                ccol[i, j, k] *= divided
                dcol[i, j, k] = (dcol[i, j, k] - dcol[i, j, k - 1] * acol) * divided

            ktop = K - 1
            gav = -gcv
            as_ = gav * bet_m
            acol = gav * bet_p
            bcol = dtr_stage - acol
            correction_term = -as_ * (u_stage[i, j, ktop - 1] - u_stage[i, j, ktop])
            dcol[i, j, ktop] = (
                dtr_stage * u_pos[i, j, ktop] + utens[i, j, ktop] + utens_stage[i, j, ktop] + correction_term
            )
            divided = 1.0 / (bcol - ccol[i, j, ktop - 1] * acol)
            dcol[i, j, ktop] = (dcol[i, j, ktop] - dcol[i, j, ktop - 1] * acol) * divided

            data_col = dcol[i, j, K - 1]
            utens_stage[i, j, K - 1] = dtr_stage * (data_col - u_pos[i, j, K - 1])
            for k in range(K - 2, -1, -1):
                data_col *= -ccol[i, j, k]
                data_col += dcol[i, j, k]
                utens_stage[i, j, k] = dtr_stage * (data_col - u_pos[i, j, k])
