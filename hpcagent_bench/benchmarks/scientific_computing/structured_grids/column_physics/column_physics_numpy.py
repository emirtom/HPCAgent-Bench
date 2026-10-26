# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""column_physics: per-column cover and tracer recurrences, then a level-carried diagnostic (numpy reference).

Synthetic. Each column ``jl`` runs a cloud-cover recurrence down the levels (a carried maximum in ``covc``) and
a tracer recurrence for ``NTRAC`` tracers (the level above times a fall factor, plus a source, clamped at
``cap``). A diagnostic pass then combines cover and the first tracer, level by level, carried over levels
(``out[jk]`` reads ``out[jk - 1]``; ``np.minimum`` keeps it from being a plain scan). ``tr`` is
(NLEV, NTRAC, KLON), the others (NLEV, KLON); ``KLON`` is contiguous. The column loop is outermost.
"""

import numpy as np


def column_physics(tr, src, fall, za, pcov, covc, out, cap, NLEV, KLON, NTRAC):
    # array shapes (numpy->dace): tr=(NLEV,NTRAC,KLON), src=(NLEV,KLON), fall=(NTRAC,), za=(NLEV,KLON),
    # pcov=(NLEV,KLON), covc=(KLON,), out=(NLEV,KLON), cap=()
    for jl in range(KLON):
        covc[jl] = 0.0
        for jk in range(1, NLEV):
            for jm in range(NTRAC):
                tr[jk, jm, jl] = min(tr[jk - 1, jm, jl] * fall[jm] + src[jk, jl], cap)
            cov = max(covc[jl], za[jk, jl])
            covc[jl] = cov
            pcov[jk, jl] = cov
    for jk in range(1, NLEV):
        out[jk, :] = np.minimum(out[jk - 1, :] * 0.5 + pcov[jk, :] * tr[jk, 0, :], cap)
