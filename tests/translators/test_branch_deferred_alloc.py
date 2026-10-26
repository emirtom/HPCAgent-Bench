# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A buffer first bound in two sibling branches is allocated in each, not only in the first one emitted.

The deferred-malloc table recorded sizes in emission order, so the ``else`` branch "reused" the
buffer the ``if`` branch allocated -- on a run taking only the ``else``, a NULL pointer
(warpx_esirkepov_deposition's ``cum_z`` in every non-3D geometry, a judge-side SIGSEGV).
"""

import pytest

from hpcagent_bench.translators.numpyto_c.emit import emit_c
from hpcagent_bench.translators.numpyto_common.lowering import lower
from tests.translators.native_tu import build_run_c, have_gcc
from tests.translators.op_oracle import parse_source

#: ``c``'s extent depends on a body-computed ``lo``, so its allocation is deferred into each branch.
#: N=8 takes only the ``else`` branch.
SOURCE = (
    "import numpy as np\n\n\n"
    "def k(a, out, N):\n"
    "    for i in range(N):\n"
    "        lo = i % 2\n"
    "        if N > 100:\n"
    "            c = np.cumsum(a[lo:N])\n"
    "            out[i] = c[N - lo - 1]\n"
    "        else:\n"
    "            c = np.cumsum(a[lo:N] * 2.0)\n"
    "            out[i] = c[N - lo - 1]\n"
)

DRIVER = (
    "int main(void) {\n"
    "    enum { N = 8 };\n"
    "    double a[N], out[N];\n"
    "    for (int i = 0; i < N; ++i) a[i] = 1.0;\n"
    "    k(a, out, N);\n"
    "    for (int i = 0; i < N; ++i) if (out[i] != 2.0 * (N - i % 2)) return 1;\n"
    "    return 0;\n"
    "}\n"
)


@pytest.mark.skipif(not have_gcc(), reason="needs gcc")
def test_the_else_branch_allocates_its_own_buffer() -> None:
    kir = lower(parse_source(SOURCE, "k", ["a"], ["out"], {"a": "(N,)", "out": "(N,)"}, {"N": 8}))
    run = build_run_c(emit_c(kir, fn_name="k"), DRIVER, sanitize=True)
    assert run.returncode == 0, run.stderr[-2000:]


if __name__ == "__main__":
    test_the_else_branch_allocates_its_own_buffer()
