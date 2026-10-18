# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written parallel numba reference for nbody: the pairwise forces and energies of every step, one particle per iteration.

numba's parfor pass fails to lower the NumpyToNumba emit (the (N, N) separation broadcasts), so numba produced no
reference and dropped out of the best-of race. The leapfrog steps are a recurrence; inside a step the O(N^2) work is
per particle: its acceleration is a sum over the others, and the potential energy a sum of the upper triangle, row by
row. Both run one particle per parallel iteration with the reference's operations in its order (the softened power,
the guarded inverse distance); a row's partial energy is folded serially in row order. The result is the numpy
reference's up to the summation order of those sums.

The missing autogen marker makes this file a hand override the NumpyToNumba regenerator leaves alone.
"""

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def accelerations(pos: np.ndarray, mass: np.ndarray, G: float, softening: float, acc: np.ndarray) -> None:
    """``acc[i] = G * sum_j (r_j - r_i) * m_j / (|r_j - r_i|^2 + softening^2)^1.5``, one particle per iteration."""
    n = pos.shape[0]
    soft2 = softening * softening
    for i in nb.prange(n):
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            inv_r31 = (dx * dx) + (dy * dy) + (dz * dz) + soft2
            inv_r32 = inv_r31 ** (-1.5) if inv_r31 > 0 else inv_r31
            ax += (dx * inv_r32) * mass[j, 0]
            ay += (dy * inv_r32) * mass[j, 0]
            az += (dz * inv_r32) * mass[j, 0]
        acc[i, 0] = G * ax
        acc[i, 1] = G * ay
        acc[i, 2] = G * az


@nb.njit(parallel=True, cache=True)
def energies(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, G: float) -> tuple[float, float]:
    """``(KE, PE)``: the kinetic energy, and the potential over every pair i < j, each row in parallel."""
    n = pos.shape[0]
    rows = np.zeros(n, dtype=pos.dtype)
    for i in nb.prange(n):
        acc = 0.0
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            inv_r1 = np.sqrt((dx * dx) + (dy * dy) + (dz * dz))
            inv_r2 = 1.0 / inv_r1 if inv_r1 > 0 else inv_r1
            acc += -(mass[i, 0] * mass[j, 0]) * inv_r2
        rows[i] = acc
    pe = 0.0
    ke = 0.0
    for i in range(n):
        pe += rows[i]
        for k in range(3):
            ke += mass[i, 0] * (vel[i, k] * vel[i, k])
    return 0.5 * ke, G * pe


@nb.njit(cache=True)
def nbody(
    mass: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    KE: np.ndarray,
    PE: np.ndarray,
    N: int,
    Nt: int,
    dt: float,
    G: float,
    softening: float,
) -> None:
    """The leapfrog integration; the energies of every step land in the ``KE`` / ``PE`` buffers the manifest declares
    (``returned: false``), which the numpy reference returns instead."""
    mean_mass = np.mean(mass)
    for k in range(3):
        momentum = 0.0
        for i in range(N):
            momentum += mass[i, 0] * vel[i, k]
        drift = momentum / N / mean_mass
        for i in range(N):
            vel[i, k] -= drift
    acc = np.empty((N, 3), dtype=pos.dtype)
    accelerations(pos, mass, G, softening, acc)
    KE[0], PE[0] = energies(pos, vel, mass, G)
    for i in range(Nt):
        vel += acc * dt / 2.0
        pos += vel * dt
        accelerations(pos, mass, G, softening, acc)
        vel += acc * dt / 2.0
        KE[i + 1], PE[i + 1] = energies(pos, vel, mass, G)
