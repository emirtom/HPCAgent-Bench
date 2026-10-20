# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A store whose right-hand side reads the same array at other cells keeps numpy's meaning.

numpy evaluates the whole right-hand side before it stores. The fused C/Fortran loop nest stores one
element per iteration, so a read of a cell an earlier iteration already stored saw the new value:
``a[1:] = a[:-1] + 1`` became a running sum, ``a[:] = a[::-1]`` a palindrome, ``x[:] = x[perm]`` a
partial permutation -- the baseline every speedup is divided by, silently wrong. numba's in-place
``a[1:] += a[:-1]`` had the same running-sum defect even in a serial build.
"""

import numpy as np
import pytest

from hpcagent_bench.translators.numpyto_common.lowering import lower
from hpcagent_bench.translators.numpyto_common.lowering.overlap import OVERLAP_TEMP_PREFIX
from hpcagent_bench.translators.numpyto_numba.emit import emit_numba
from tests.translators import op_oracle

RNG = np.random.default_rng(0)
VEC = RNG.standard_normal(6)
MAT = RNG.standard_normal((6, 6))
PERM = np.array([3, 1, 4, 0, 5, 2])

HEAD = "import numpy as np\ndef f(x, perm, out):\n"

# (label, body, input, output shape symbol)
CASES = [
    ("shift right", "    out[:] = x\n    out[1:] = out[:-1] + 1.0\n", VEC, "(N,)"),
    ("augmented shift", "    out[:] = x\n    out[1:] += out[:-1]\n", VEC, "(N,)"),
    ("reverse", "    out[:] = x\n    out[:] = out[::-1]\n", VEC, "(N,)"),
    ("gather", "    out[:] = x\n    out[:] = out[perm]\n", VEC, "(N,)"),
    ("local gather rebind", "    u = x * 2.0\n    u = u[perm]\n    out[:] = u\n", VEC, "(N,)"),
    ("local reverse rebind", "    t = x * 2.0\n    t = t[::-1]\n    out[:] = t\n", VEC, "(N,)"),
    ("earlier row, later column", "    out[:, :] = x\n    out[1:, :-1] = out[:-1, 1:]\n", MAT, "(N, N)"),
    ("row shift", "    out[:, :] = x\n    out[1:, :] = out[:-1, :] * 2.0\n", MAT, "(N, N)"),
    ("column reverse", "    out[:, :] = x\n    out[:, :] = out[:, ::-1]\n", MAT, "(N, N)"),
    ("fixed row inside the written range", "    out[:, :] = x\n    out[1:, :] = out[3, :] * 2.0\n", MAT, "(N, N)"),
    ("transpose through a row", "    out[:, :] = x\n    out[2, :] = out[:, 2] + 1.0\n", MAT, "(N, N)"),
]


@pytest.mark.parametrize(("label", "body", "x", "shape"), CASES, ids=[c[0] for c in CASES])
def test_overlapping_self_read_matches_numpy(label: str, body: str, x: np.ndarray, shape: str) -> None:
    no = op_oracle.native()
    backends = ("c", "cpp", "fortran", "numba")
    status = no.run_op(
        HEAD + body,
        "f",
        {"x": x, "perm": PERM},
        {"out": x.shape},
        {"N": 6},
        shapes={"x": shape, "perm": "(N,)", "out": shape},
        backends=backends,
    )
    no.assert_backends_ok(status, backends, label)


@pytest.mark.parametrize(
    ("label", "body", "shape"),
    [
        ("read ahead", "    out[:] = x\n    out[:-1] = out[1:] + 1.0\n", "(N,)"),
        ("same cell", "    out[:] = x\n    out[1:] = out[1:] * 2.0\n", "(N,)"),
        ("later row, earlier column", "    out[:, :] = x\n    out[:-1, 1:] = out[1:, :-1]\n", "(N, N)"),
        ("fixed row below the written range", "    out[:, :] = x\n    out[2:, :] = out[1, :] * 2.0\n", "(N, N)"),
        ("other fixed row", "    out[:, :] = x\n    out[2, 1:] = out[0, :-1]\n", "(N, N)"),
    ],
)
def test_a_read_no_earlier_iteration_stored_stays_one_fused_nest(label: str, body: str, shape: str) -> None:
    """Staging costs a temp and a second pass; a read of an unchanged cell must not pay it
    (seidel_2d reads ahead of its write)."""
    kir = op_oracle.parse_source(
        HEAD + body, "f", ["x", "perm"], ["out"], {"x": shape, "perm": "(N,)", "out": shape}, {"N": 6}
    )
    lowered = lower(kir)
    assert not any(n.startswith(OVERLAP_TEMP_PREFIX) for n in lowered.zeros_locals), (label, lowered.zeros_locals)


def test_numba_spells_out_an_overlapping_augmented_slice_store() -> None:
    """The overlapping store is spelled out and stores a copy, which numba's parfor pass cannot race."""
    src = HEAD + "    out[1:] += out[:-1]\n"
    assert "out[1:] = materialized(out[1:] + out[:-1])" in emit_numba(src)


def test_numba_keeps_an_elementwise_augmented_store_in_place() -> None:
    src = HEAD + "    out[1:] += x[:-1]\n"
    assert "out[1:] += x[:-1]" in emit_numba(src)
