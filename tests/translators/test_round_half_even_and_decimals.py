# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``np.round`` keeps numpy's meaning on the native backends: half to even, and ``decimals``.

C's ``round`` rounds a tie away from zero, so the scalar form of ``np.round(2.5)`` (and the builtin
``round``) came out 3 where numpy gives 2. The array form dropped every argument after the first, so
``np.round(x, 1)`` rounded to an integer.
"""

import numpy as np
import pytest

from tests.translators import op_oracle

NATIVE = ("c", "cpp", "fortran")
TIES = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.25, -7.75])
FRACTIONS = np.array([0.123, 1.25, -2.35, 10.05, 7.0, -0.45, 123.456, 0.5])


def run(body: str, x: np.ndarray) -> dict[str, str]:
    no = op_oracle.native()
    n = len(x)
    return no.run_op(
        "import numpy as np\ndef f(x, out):\n" + body,
        "f",
        {"x": x},
        {"out": (n,)},
        {"N": n},
        shapes={"x": "(N,)", "out": "(N,)"},
        backends=NATIVE,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "body",
    [
        "    for i in range(x.shape[0]):\n        out[i] = np.round(x[i])\n",
        "    for i in range(x.shape[0]):\n        out[i] = np.around(x[i])\n",
        "    for i in range(x.shape[0]):\n        out[i] = round(x[i])\n",
    ],
    ids=["np.round", "np.around", "builtin round"],
)
def test_a_scalar_tie_rounds_to_even(body: str) -> None:
    op_oracle.assert_backends_ok(run(body, TIES), NATIVE, body.strip())


@pytest.mark.parametrize(
    "body",
    [
        "    out[:] = np.round(x, 1)\n",
        "    out[:] = np.round(x, decimals=2)\n",
        "    out[:] = np.around(x, -1)\n",
        "    for i in range(x.shape[0]):\n        out[i] = np.round(x[i], 1)\n",
    ],
    ids=["array decimals=1", "array decimals= keyword", "array decimals=-1", "scalar decimals=1"],
)
def test_decimals_round_to_that_many_places(body: str) -> None:
    op_oracle.assert_backends_ok(run(body, FRACTIONS), NATIVE, body.strip())


def test_a_non_constant_decimals_is_refused() -> None:
    no = op_oracle.native()
    status = no.run_op(
        "import numpy as np\ndef f(x, d, out):\n    out[:] = np.round(x, d)\n",
        "f",
        {"x": FRACTIONS, "d": 1},
        {"out": (8,)},
        {"N": 8},
        shapes={"x": "(N,)", "out": "(N,)"},
        backends=NATIVE,
    )
    assert all(s.startswith("FAIL:emit:NotImplementedError") for s in status.values()), status
