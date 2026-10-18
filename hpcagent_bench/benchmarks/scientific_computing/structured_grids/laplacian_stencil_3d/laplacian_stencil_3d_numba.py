# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for laplacian_stencil_3d: the periodic 25-point stencil with each x-plane in parallel.

numba's parfor pass fails on the NumpyToNumba emit (the native call crashes on the rolled-array sums), so numba
produced no reference and dropped out of the best-of race. Each output point sums its periodic neighbours in the
reference's order (axis 0, 1, 2; offsets 1..4; ``roll(+m) + roll(-m)``), so ``lap`` is bitwise the reference's. The
kinetic energy is a per-plane partial sum over ``psi * lap``, folded in plane order, so it differs from numpy's
``einsum`` only in its summation order.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np

C0 = -205.0 / 72.0
CW = (8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0)


@nb.njit(parallel=True, cache=True)
def kernel(inv_h2: float, psi: np.ndarray, lap: np.ndarray, ekin: np.ndarray) -> None:
    nx, ny, nz, nk = psi.shape
    weights = np.array(CW)
    planes = np.zeros((nx, nk), dtype=psi.dtype)
    for x in nb.prange(nx):
        for y in range(ny):
            for z in range(nz):
                for q in range(nk):
                    acc = 3.0 * C0 * psi[x, y, z, q]
                    for m in range(1, 5):
                        acc = acc + weights[m - 1] * (psi[(x - m) % nx, y, z, q] + psi[(x + m) % nx, y, z, q])
                    for m in range(1, 5):
                        acc = acc + weights[m - 1] * (psi[x, (y - m) % ny, z, q] + psi[x, (y + m) % ny, z, q])
                    for m in range(1, 5):
                        acc = acc + weights[m - 1] * (psi[x, y, (z - m) % nz, q] + psi[x, y, (z + m) % nz, q])
                    value = inv_h2 * acc
                    lap[x, y, z, q] = value
                    planes[x, q] += psi[x, y, z, q] * value
    for q in range(nk):
        total = 0.0
        for x in range(nx):
            total += planes[x, q]
        ekin[q] = -0.5 * total
