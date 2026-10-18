# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for force_lj: the Lennard-Jones force on each atom in parallel.

numba's parfor pass fails to lower the NumpyToNumba emit (two (N, N, 3) broadcasts whose sizes it cannot match), so
numba produced no reference and dropped out of the best-of race. Each atom's force is the sum over every other atom
within the cutoff, independent of the other atoms': one atom per parallel iteration, the pair terms computed with the
reference's operations in its order and summed in atom order. The result is the numpy reference's up to the
summation order of that sum.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def force_lj(pos: np.ndarray, cutoff: float, force: np.ndarray, epsilon: float = 1.0, sigma: float = 1.0) -> None:
    n = pos.shape[0]
    cutoffsq = cutoff * cutoff
    sigma2 = sigma * sigma
    sigma6 = sigma2 * sigma2 * sigma2
    prefactor = 48.0 * epsilon * (sigma6 * sigma6)
    offset = 0.5 / sigma6
    for i in nb.prange(n):
        fx = 0.0
        fy = 0.0
        fz = 0.0
        for j in range(n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            rsq = dx * dx + dy * dy + dz * dz
            if rsq < cutoffsq and rsq > 0.0:
                r2inv = 1.0 / rsq
                r6inv = r2inv * r2inv * r2inv
                fpair = prefactor * r6inv * (r6inv - offset) * r2inv
                fx += fpair * dx
                fy += fpair * dy
                fz += fpair * dz
        force[i, 0] = fx
        force[i, 1] = fy
        force[i, 2] = fz
