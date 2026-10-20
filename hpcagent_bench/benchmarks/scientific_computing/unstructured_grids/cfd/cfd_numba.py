# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for cfd: the face fluxes of every cell, one cell per iteration.

numba's parfor pass fails to lower the NumpyToNumba emit (the (ncells, 4, 3) flux broadcasts), so numba produced no
reference and dropped out of the best-of race. A cell's residual sums its four faces and writes only its own
entries: one cell per parallel iteration, every flux term computed with the reference's operations in its order and
the faces summed in face order, so the result is bitwise the numpy reference's.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def physical_flux(
    density: float, m0: float, m1: float, m2: float, energy: float, n0: float, n1: float, n2: float, gamma: float
) -> tuple[float, float, float, float, float]:
    """``(flux_density, flux_momentum[0..2], flux_energy)`` of one state across one face normal."""
    msq = m0 * m0 + m1 * m1 + m2 * m2
    pressure = (gamma - 1.0) * (energy - 0.5 * msq / density)
    mn = m0 * n0 + m1 * n1 + m2 * n2
    vn = mn / density
    return mn, vn * m0 + pressure * n0, vn * m1 + pressure * n1, vn * m2 + pressure * n2, (energy + pressure) * vn


@nb.njit(parallel=True, cache=True)
def cfd(
    density: np.ndarray,
    momentum: np.ndarray,
    energy: np.ndarray,
    neigh: np.ndarray,
    normals: np.ndarray,
    gamma: float,
    alpha: float,
    res_density: np.ndarray,
    res_momentum: np.ndarray,
    res_energy: np.ndarray,
) -> None:
    ncells = density.shape[0]
    faces = neigh.shape[1]
    half_alpha = 0.5 * alpha
    for i in nb.prange(ncells):
        sum_d = 0.0
        sum_m0 = 0.0
        sum_m1 = 0.0
        sum_m2 = 0.0
        sum_e = 0.0
        for f in range(faces):
            j = neigh[i, f]
            n0 = normals[i, f, 0]
            n1 = normals[i, f, 1]
            n2 = normals[i, f, 2]
            fd_i, fm0_i, fm1_i, fm2_i, fe_i = physical_flux(
                density[i], momentum[i, 0], momentum[i, 1], momentum[i, 2], energy[i], n0, n1, n2, gamma
            )
            fd_n, fm0_n, fm1_n, fm2_n, fe_n = physical_flux(
                density[j], momentum[j, 0], momentum[j, 1], momentum[j, 2], energy[j], n0, n1, n2, gamma
            )
            sum_d += 0.5 * (fd_i + fd_n) - half_alpha * (density[j] - density[i])
            sum_m0 += 0.5 * (fm0_i + fm0_n) - half_alpha * (momentum[j, 0] - momentum[i, 0])
            sum_m1 += 0.5 * (fm1_i + fm1_n) - half_alpha * (momentum[j, 1] - momentum[i, 1])
            sum_m2 += 0.5 * (fm2_i + fm2_n) - half_alpha * (momentum[j, 2] - momentum[i, 2])
            sum_e += 0.5 * (fe_i + fe_n) - half_alpha * (energy[j] - energy[i])
        res_density[i] += sum_d
        res_momentum[i, 0] += sum_m0
        res_momentum[i, 1] += sum_m1
        res_momentum[i, 2] += sum_m2
        res_energy[i] += sum_e
