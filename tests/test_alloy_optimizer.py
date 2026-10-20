# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Alloy (the optional ``alloy`` extra) end to end on one kernel: every region gets several verified
variants, and both CPU profile flavors submit a library the judge grades correct. The ``alloy`` CI job runs it."""

import pathlib
from collections.abc import Callable
from http.server import ThreadingHTTPServer

import pytest

from hpcagent_bench import config
from hpcagent_bench.categories import Baseline, InputMode, Language, Oracle, SourceMode
from hpcagent_bench.harness import tools
from hpcagent_bench.harness.judge_wire import GradeAnswer
from hpcagent_bench.harness.optimizers import optimizer_registry
from hpcagent_bench.harness.service import ServiceConfig
from hpcagent_bench.harness.task import Task
from hpcagent_bench.precision import Precision
from hpcagent_bench.spec import BenchSpec
from hpcagent_bench.config import SubmitFeedback

KERNEL = "tsvc_2_vpv"


def test_every_region_gets_several_verified_variants(tmp_path: pathlib.Path) -> None:
    # Deferred: both import the optional alloy extra.
    from alloy import configuration_matrix

    from hpcagent_bench.harness import alloy_adapter as bench

    spec = BenchSpec.load(KERNEL)
    sdfg = bench.pipeline_sdfg(spec, "parallel_cpu", Precision.FP64)

    sut = bench.generated(sdfg, spec, Precision.FP64, configuration_matrix.load(), tmp_path)

    assert sut.regions
    assert sut.verification is not None
    for rep in sut.regions:
        variants = bench.verified(rep, sut.verification.verdicts[rep.region.symbol])
        assert len(variants) >= 2, f"{rep.region.symbol}: {[b.name for b in variants]}"


@pytest.mark.parametrize("flavor", ["dace_cpu_parallel_alloy_profile", "dace_cpu_canon_alloy_profile"])
def test_the_profile_flavor_submits_a_library_the_judge_grades_correct(
    flavor: str,
    make_judge: Callable[..., tuple[ThreadingHTTPServer, str]],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HPCAGENT_BENCH_SHARED_DIR", str(tmp_path))
    url = make_judge(ServiceConfig(baseline=Baseline.C, oracle=Oracle.AUTO, input_mode=InputMode.ANY, repeat=3))[1]
    monkeypatch.setenv("JUDGE_URL", url)  # the optimizer scores its variants on this judge

    sub = optimizer_registry()[flavor](workdir=tmp_path).solve(Task(KERNEL, SourceMode.ANY, Language.C))

    assert sub.library is not None
    with config.overridden("service.submit_feedback", SubmitFeedback.FULL):  # need the measured grade, not the verdict
        r = tools.JudgeClient(url).submit(sub, KERNEL)
    assert isinstance(r, GradeAnswer), r
    assert r.build_ok is True, r.detail
    assert r.correct is True, r.detail
    assert r.speedup > 0.0
