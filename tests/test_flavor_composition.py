# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A framework entry composes from mixins: :func:`hpcagent_bench.registry.fields_of` reads inherited attributes, a
subclass overrides its bases, and a DaCe flavor's names follow from its arch, stage and tuner alone."""

import itertools

import pytest

from hpcagent_bench import models
from hpcagent_bench.columns import DACE_STAGES, DACE_TUNERS, FRAMEWORKS
from hpcagent_bench.registry import fields_of
from hpcagent_bench.vocabulary import OPTIMIZERS


class Base:
    __slots__ = ()

    shared = "base"
    overridden = "base"


class Mixin:
    __slots__ = ()

    overridden = "mixin"
    extra = 1


class Leaf(Mixin, Base):
    __slots__ = ()

    def method(self) -> None:
        """Not a field."""


def test_fields_of_merges_bases_and_the_leftmost_base_wins() -> None:
    assert fields_of(Leaf) == {"shared": "base", "overridden": "mixin", "extra": 1}


DACE_FLAVORS = list(itertools.product(("cpu", "gpu"), DACE_STAGES, (None, *DACE_TUNERS)))


@pytest.mark.parametrize(("arch", "stage", "tuner"), DACE_FLAVORS)
def test_every_dace_stage_and_tuner_is_a_flavor_on_both_devices(arch: str, stage: str, tuner: str | None) -> None:
    flavor = stage if tuner is None else f"{stage}_{tuner}"
    meta = FRAMEWORKS.entries[f"dace_{arch}_{flavor}"]
    derived = (meta.get("column"), meta.get("flavor"), meta.get("pipelines"), meta.get("stage"), meta.get("tuner"))
    assert derived == (f"dace_{arch}", flavor, (f"{stage}_{arch}",), stage, tuner)
    assert meta["display"].startswith(f"DaCe ({arch.upper()}, ")
    tuned = tuner is not None
    assert meta["adapter"].startswith("hpcagent_bench.harness.optimizers:") == tuned
    assert meta["sweep_deterministic"] is not tuned


def test_dace_columns_name_no_stage() -> None:
    for arch in ("cpu", "gpu"):
        meta = FRAMEWORKS.entries[f"dace_{arch}"]
        assert "stage" not in meta
        assert "pipelines" not in meta
        assert meta["display"] == f"DaCe ({arch.upper()})"


def test_dace_optimizers_are_the_flavors_of_their_stage() -> None:
    for stage, optimizer in (("parallel", "dace"), ("canon", "cpf")):
        flavors = {key for key, meta in FRAMEWORKS.entries.items() if meta.get("stage") == stage}
        assert flavors == set(models.dace_flavors(stage))
        assert all(OPTIMIZERS.canonical(key) == optimizer for key in flavors)


def test_tvm_names_the_gpu_column() -> None:
    assert FRAMEWORKS.canonical("tvm") == "tvm_gpu"
