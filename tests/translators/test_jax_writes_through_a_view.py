# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A write through a numpy VIEW lands in the base array in the JAX translation.

``outg = out[:, g * k:(g + 1) * k]`` is a view: ``outg += ...`` writes into ``out``. jax arrays are
values, so the emitted ``outg = outg + ...`` rebound a local copy and ``out`` stayed at its bias --
eight grouped-conv kernels of the corpus came out wrong this way (conv2d_subtract_subtract_mish's
JAX output was the constant 0.038).
"""

import numpy as np
import pytest

from hpcagent_bench.translators.numpyto_jax import EmitError, emit_jax
from tests.translators import op_oracle

RNG = np.random.default_rng(0)

# (label, source, inputs, outputs, symbols, shapes)
CASES = [
    (
        "aug-assign of a whole column-group view",
        "import numpy as np\n"
        "def f(x, out):\n"
        "    for g in range(2):\n"
        "        outg = out[:, g * 3:(g + 1) * 3]\n"
        "        outg += x[:, g * 3:(g + 1) * 3] * (g + 1.0)\n",
        {"x": RNG.standard_normal((4, 6))},
        {"out": (4, 6)},
    ),
    (
        "strided store through a view's trailing axes",
        "import numpy as np\ndef f(x, out):\n    og = out[:, 1:3]\n    og[:, :, 0:4:2] += x\n",
        {"x": RNG.standard_normal((2, 2, 2))},
        {"out": (2, 4, 4)},
    ),
    (
        "element store through a slice view",
        "import numpy as np\ndef f(b, out):\n    out[:] = b\n    v = out[2:]\n    v[0] = 100.0\n",
        {"b": RNG.standard_normal(6)},
        {"out": (6,)},
    ),
    (
        "view of a view",
        "import numpy as np\ndef f(b, out):\n    out[:] = b\n    v = out[1:5]\n    w = v[1:3]\n    w *= 10.0\n",
        {"b": RNG.standard_normal(6)},
        {"out": (6,)},
    ),
]


@pytest.mark.parametrize(("label", "src", "inputs", "outputs"), CASES, ids=[c[0] for c in CASES])
def test_a_write_through_a_view_reaches_the_base(label: str, src: str, inputs: dict, outputs: dict) -> None:
    status = op_oracle.run_op(src, "f", inputs, outputs, {}, backends=("jax",))
    op_oracle.assert_backends_ok(status, ("jax",), label)


def test_a_view_read_only_is_left_alone() -> None:
    """Folding is for writes; a read-only slice binding keeps its one slicing op."""
    src = "import numpy as np\ndef f(x, out):\n    xs = x[1:]\n    out[1:] = xs * 2.0\n"
    assert "xs = x[1:]" in emit_jax(src, "f")


def test_a_written_view_with_a_rebound_index_is_refused() -> None:
    """``k`` moves after the view is taken, so the view's cells cannot be re-derived at the write."""
    src = (
        "import numpy as np\n"
        "def f(out):\n"
        "    for k in range(1, 2):\n"
        "        v = out[k:]\n"
        "        k = 2\n"
        "        v += 1.0\n"
    )
    with pytest.raises(EmitError):
        emit_jax(src, "f")
