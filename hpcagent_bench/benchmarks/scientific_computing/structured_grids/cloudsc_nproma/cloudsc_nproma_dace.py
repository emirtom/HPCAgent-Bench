# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written DaCe program of cloudsc_nproma (a hand override: no autogen marker).

The loops of ``cloudsc_nproma_numpy.py`` as the SCC GPU driver orders them: block map, ragged column map, level
loop, species loop of length 5. ``NCLV`` is a Python int, so the species loop unrolls.
"""

import dace as dc

nlev = dc.symbol("nlev", dtype=dc.int64, positive=True)
nproma = dc.symbol("nproma", dtype=dc.int64, positive=True)
ngptot = dc.symbol("ngptot", dtype=dc.int64, positive=True)
nblk = dc.symbol("nblk", dtype=dc.int64, positive=True)
ncldtop = dc.symbol("ncldtop", dtype=dc.int64, positive=True)

NCLV = 5
IRAIN = 2
ISNOW = 3
PTSPHY = 3600.0
RG = 9.80665
RD = 287.0597
RCOVPMIN = 0.1
RQPRE = 1.0e7
ZAMAX = 1.0 - 1.0e-6

f64 = dc.float64


@dc.program
def cloudsc_nproma(
    paph: f64[nblk, nlev + 1, nproma],
    pap: f64[nblk, nlev, nproma],
    pt: f64[nblk, nlev, nproma],
    za: f64[nblk, nlev, nproma],
    q: f64[nblk, 5, nlev, nproma],
    vfall: f64[5],
    pfplsx: f64[nblk, 5, nlev + 1, nproma],
    pcov: f64[nblk, nlev, nproma],
):
    for ibl in range(nblk):
        for jl in range(min(nproma, ngptot - ibl * nproma)):
            pcov[ibl, ncldtop - 1, jl] = 0.0
            for jm in range(NCLV):
                pfplsx[ibl, jm, ncldtop, jl] = 0.0
            for jk in range(ncldtop, nlev):
                dtgdp = PTSPHY * RG / (paph[ibl, jk + 1, jl] - paph[ibl, jk, jl])
                rho = pap[ibl, jk, jl] / (RD * pt[ibl, jk, jl])
                for jm in range(NCLV):
                    src = pfplsx[ibl, jm, jk, jl] * dtgdp
                    sink = dtgdp * vfall[jm] * rho
                    qnew = max((q[ibl, jm, jk, jl] + src) / (1.0 + sink), 0.0)
                    pfplsx[ibl, jm, jk + 1, jl] = sink * qnew / dtgdp
                qpre = pfplsx[ibl, IRAIN, jk + 1, jl] + pfplsx[ibl, ISNOW, jk + 1, jl]
                cov = 1.0 - (1.0 - pcov[ibl, jk - 1, jl]) * (1.0 - max(za[ibl, jk, jl], za[ibl, jk - 1, jl])) / (
                    1.0 - min(za[ibl, jk - 1, jl], ZAMAX)
                )
                pcov[ibl, jk, jl] = min(qpre * RQPRE, 1.0) * max(cov, RCOVPMIN)
