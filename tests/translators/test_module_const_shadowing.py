# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A local rebinding of a module constant's name shadows it, whatever binding form it uses.

``inline_module_constants`` folds a module-level ``SCALE = 2.0`` into the kernel body unless the
kernel binds ``SCALE`` itself. Only a bare ``SCALE = ...`` was seen as a binding, so a tuple
unpack (``SCALE, other = 3.0, 1.0``) kept the local assign but folded the module value into every
read: numpy computes with 3.0, every native backend with 2.0, and nothing fails.
"""

import ast

import numpy as np
import pytest

from hpcagent_bench.translators.numpyto_common.frontend import inline_module_constants
from tests.translators.op_oracle import run_op

MODULE = "import numpy as np\nSCALE = 2.0\n"


@pytest.mark.parametrize(
    "binding",
    [
        " SCALE, other = 3.0, 1.0\n",
        " [SCALE, *rest] = [3.0, 1.0]\n",
        " SCALE: float = 3.0\n",
        " SCALE = 1.0\n SCALE += 2.0\n",
        " for SCALE in (3.0,):\n  pass\n",
        " if (SCALE := 3.0) > 0.0:\n  pass\n",
    ],
)
def test_every_binding_form_shadows_the_module_constant(binding: str) -> None:
    mod = ast.parse(MODULE + "def f(a, out):\n" + binding + " out[0] = a[0] * SCALE\n")
    fn = next(n for n in mod.body if isinstance(n, ast.FunctionDef))
    inline_module_constants(mod, fn, ["a", "out"])
    body = ast.unparse(fn)
    assert "a[0] * SCALE" in body, f"the module value was folded over the local binding:\n{body}"


def test_a_subscript_store_does_not_shadow() -> None:
    """``out[SCALE_I] = ...`` reads the constant; it binds nothing, so the fold still happens."""
    mod = ast.parse("import numpy as np\nSCALE_I = 1\ndef f(a, out):\n out[SCALE_I] = a[0]\n")
    fn = next(n for n in mod.body if isinstance(n, ast.FunctionDef))
    inline_module_constants(mod, fn, ["a", "out"])
    assert "out[1] = a[0]" in ast.unparse(fn)


def test_a_tuple_rebinding_computes_the_reference_numbers() -> None:
    src = (
        MODULE + "def f(a, out):\n"
        " SCALE, other = 3.0, 1.0\n"
        " for i in range(a.shape[0]):\n"
        "  out[i] = a[i] * SCALE + other\n"
    )
    status = run_op(
        src,
        "f",
        {"a": np.random.default_rng(3).standard_normal(8)},
        {"out": (8,)},
        {"N": 8},
        shapes={"a": "(N,)", "out": "(N,)"},
        backends=("c", "fortran"),
    )
    assert status["c"] == "ok", status
    assert status["fortran"] == "ok", status
