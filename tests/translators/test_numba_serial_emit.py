# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A serial numba emit carries no ``parallel=True``; the oracle asks for one only for the kernels it lists."""

from hpcagent_bench import numerical_oracle
from hpcagent_bench.translators.numpyto_numba.emit import emit_numba

SOURCE = "import numpy as np\ndef f(a, out):\n    out[:] = a * 2.0\n"


def test_the_default_emit_is_parallel() -> None:
    assert "parallel=True" in emit_numba(SOURCE)


def test_a_serial_emit_has_no_parfor_build() -> None:
    assert "parallel=True" not in emit_numba(SOURCE, serial=True)


def test_only_listed_kernels_are_checked_serial() -> None:
    assert numerical_oracle.NUMBA_SERIAL_EMIT == {"cloudsc", "cloudsc_monolith"}


if __name__ == "__main__":
    test_the_default_emit_is_parallel()
    test_a_serial_emit_has_no_parfor_build()
    test_only_listed_kernels_are_checked_serial()
