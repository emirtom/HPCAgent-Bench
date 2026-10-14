# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A reverse slice of a local whose extent is a compound shape token compiles and matches numpy.

``u = t[::-1]`` records u's extent as the expression ``(N + 0) // 1``. Reading ``u[::-1]`` seeds the
reverse start at ``extent - 1``, and that token used to be spliced into a ``Name`` verbatim, so the
Python ``//`` reached the C source (where it opens a comment) and the Fortran source (where it is
string concatenation). Both compiles failed.
"""

import numpy as np

from tests.translators import op_oracle

DOUBLE_REVERSE = (
    "import numpy as np\ndef f(x, out):\n    t = x * 2.0\n    u = t[::-1]\n    v = u[::-1]\n    out[:] = v - x\n"
)


def test_reverse_of_a_reversed_local_compiles_and_matches_numpy() -> None:
    no = op_oracle.native()
    x = np.random.default_rng(0).standard_normal(6)
    status = no.run_op(
        DOUBLE_REVERSE,
        "f",
        {"x": x},
        {"out": (6,)},
        {"N": 6},
        shapes={"x": "(N,)", "out": "(N,)"},
        backends=("c", "cpp", "fortran"),
    )
    no.assert_backends_ok(status, ("c", "cpp", "fortran"), "double reverse")
