# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A kept helper types its integer locals' bitwise literals exactly as the kernel body does.

``IAND`` / ``IOR`` / ``IEOR`` need both arguments of one kind, so a bare literal paired with a
typed local takes that local's kind suffix. The kernel body read the kind from the local's
declaration; the contained helper mapped every non-int64 integer local to ``int32``, so an
``integer(c_int16_t)`` local met a ``3_c_int32_t`` literal and gfortran rejected the ``IAND``.
"""

from hpcagent_bench.translators.numpyto_common.lowering import lower
from hpcagent_bench.translators.numpyto_fortran.emit import emit_fortran
from hpcagent_bench.translators.numpyto_fortran.intrinsics import renders_natively
from tests.translators.op_oracle import parse_source

# The early return keeps ``h`` a contained subroutine instead of inlining it.
SRC = (
    "import numpy as np\n"
    "def h(x):\n"
    " m = np.int16(x[0])\n"
    " r = m & 3\n"
    " if m < 0:\n"
    "  return 0.0\n"
    " return r * 1.0\n"
    "def f(x, out):\n"
    " out[0] = h(x)\n"
    " m = np.int16(x[1])\n"
    " out[0] += m & 3\n"
)


def emitted() -> str:
    kir = parse_source(SRC, "f", ["x"], ["out"], {"x": "(N,)", "out": "(1,)"}, {"N": 8})
    return emit_fortran(lower(kir, native_call=renders_natively), fn_name="f")


def test_helper_and_kernel_suffix_an_int16_bitwise_literal_alike() -> None:
    src = emitted()
    kernel, helper = src.split("contains", 1)
    assert "IAND(m, 3_c_int16_t)" in kernel, kernel
    assert "IAND(m, 3_c_int16_t)" in helper, helper
    assert "3_c_int32_t" not in src, src


def test_a_narrow_integer_scalar_parameter_is_declared_with_the_integers() -> None:
    """Every integer kind joins the integer-scalar group ahead of the arrays (``dtypes.is_integer``),
    not only int32/int64: an ``int16`` scalar landed after the arrays with the real scalars."""
    src = "import numpy as np\ndef f(a, k, out):\n for i in range(a.shape[0]):\n  out[i] = a[i] * k\n"
    kir = parse_source(src, "f", ["a", "k"], ["out"], {"a": "(N,)", "out": "(N,)"}, {"N": 8}, {"k": "int16"})
    text = emit_fortran(lower(kir, native_call=renders_natively), fn_name="f")
    assert text.index("integer(c_int16_t), value, intent(in) :: k") < text.index(":: a(N)"), text
