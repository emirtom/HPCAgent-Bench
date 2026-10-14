# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fortran claims ``np.where`` as ``MERGE`` only when its operands conform, extent by extent.

``MERGE`` does not broadcast. A mask of shape ``(N, 1)`` against ``(N, M)`` branches has the same
RANK, which is all the claim used to check, so it emitted ``MERGE`` over non-conformable arrays and
the copy-out loop walked the mask's single column: every column of ``out`` but the first stayed 0.
"""

import numpy as np

from tests.translators import op_oracle

BROADCAST_MASK = "import numpy as np\ndef f(c, a, b, out):\n    out[:] = np.where(c > 0.5, a, b)\n"


def test_where_with_a_size_one_mask_axis_matches_numpy_in_fortran() -> None:
    no = op_oracle.native()
    rng = np.random.default_rng(0)
    status = no.run_op(
        BROADCAST_MASK,
        "f",
        {"c": rng.random((5, 1)), "a": rng.random((5, 3)), "b": rng.random((5, 3))},
        {"out": (5, 3)},
        {"N": 5, "M": 3},
        shapes={"c": "(N, 1)", "a": "(N, M)", "b": "(N, M)", "out": "(N, M)"},
        backends=("c", "fortran"),
    )
    no.assert_backends_ok(status, ("c", "fortran"), "where over a broadcast mask")


def test_where_over_conformable_operands_still_renders_merge() -> None:
    """The extent check must not cost the intrinsic its ordinary case."""
    from hpcagent_bench.translators.numpyto_common.lowering import lower
    from hpcagent_bench.translators.numpyto_fortran.emit import emit_fortran
    from hpcagent_bench.translators.numpyto_fortran.intrinsics import renders_natively

    shapes = {"c": "(N, M)", "a": "(N, M)", "b": "(N, M)", "out": "(N, M)"}
    kir = op_oracle.parse_source(BROADCAST_MASK, "f", ["c", "a", "b"], ["out"], shapes, {"N": 5, "M": 3})
    src = emit_fortran(lower(kir, native_call=renders_natively), fn_name="f")
    assert "MERGE(" in src.upper(), src
