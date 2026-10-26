# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hand-written DaCe program of seissol_volume (a hand override: no autogen marker).

Output-stationary loop order: one entry ``upd[b, k, p]``, zeroed, accumulated over the three summed loops
``d, l, q`` in place, then applied to ``Q``. The zero init and the separate update keep it a loop nest rather
than one einsum library node.
"""

import dace as dc
from hpcagent_bench.frameworks.dace_framework import dc_float

NB = dc.symbol("NB", dtype=dc.int64, positive=True)
NBF = dc.symbol("NBF", dtype=dc.int64, positive=True)
NQ = dc.symbol("NQ", dtype=dc.int64, positive=True)
ND = dc.symbol("ND", dtype=dc.int64, positive=True)

__hpcagent_bench_program__ = "seissol_volume"


@dc.program
def seissol_volume(
    dt: dc_float,
    Q: dc_float[NB, NBF, NQ],
    upd: dc_float[NB, NBF, NQ],
    I: dc_float[NB, NBF, NQ],
    kDivM: dc_float[ND, NBF, NBF],
    star: dc_float[ND, NQ, NQ],
):
    for b in range(NB):
        for k in range(NBF):
            for p in range(NQ):
                upd[b, k, p] = 0.0
                for d in range(ND):
                    for l in range(NBF):
                        for q in range(NQ):
                            upd[b, k, p] = upd[b, k, p] + kDivM[d, k, l] * I[b, l, q] * star[d, q, p]
                Q[b, k, p] = Q[b, k, p] + dt * upd[b, k, p]
