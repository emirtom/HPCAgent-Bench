# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Alloy adapter's inner tier without a judge: the stub ranking keeps each region's fastest candidate over the
best prefix so far, survives a failing stub, stops when the budget does, and reads a stub's time from either profile."""

import dataclasses

import pytest

from hpcagent_bench.harness.alloy_adapter import ranked, stub_time
from hpcagent_bench.harness.profile_wire import DeviceTraceReport, LinuxperfReport, ProfileAnswer


@dataclasses.dataclass(frozen=True, slots=True)
class Fake:
    name: str


def always() -> bool:
    return True


def test_each_region_keeps_its_fastest_stub_over_the_best_prefix() -> None:
    start = {"r0": Fake("r0.ref"), "r1": Fake("r1.ref")}
    candidates = {"r0": [Fake("r0.fast"), Fake("r0.slow")], "r1": [Fake("r1.fast")]}
    cost = {"r0.ref": 5.0, "r0.fast": 1.0, "r0.slow": 9.0, "r1.ref": 4.0, "r1.fast": 2.0}
    prefixes: list[str] = []

    def time_stub(selection: dict[str, Fake], symbol: str) -> float:
        if symbol == "r1":
            prefixes.append(selection["r0"].name)
        return cost[selection[symbol].name]

    sut = ranked(["r0", "r1"], start, candidates, time_stub, always)

    assert {s: b.name for s, b in sut.items()} == {"r0": "r0.fast", "r1": "r1.fast"}
    assert set(prefixes) == {"r0.fast"}, "the later region's stubs run over the earlier region's winner"


def test_a_failing_stub_drops_out_and_the_start_build_stays_when_all_fail() -> None:
    start = {"r0": Fake("r0.ref")}

    def time_stub(selection: dict[str, Fake], symbol: str) -> float:
        raise RuntimeError("stub did not build")

    sut = ranked(["r0"], start, {"r0": [Fake("r0.a")]}, time_stub, always)

    assert sut["r0"].name == "r0.ref"


def test_the_ranking_stops_when_the_budget_does() -> None:
    start = {"r0": Fake("r0.ref"), "r1": Fake("r1.ref")}
    timed: list[str] = []

    def time_stub(selection: dict[str, Fake], symbol: str) -> float:
        timed.append(selection[symbol].name)
        return 1.0

    sut = ranked(["r0", "r1"], start, {"r1": [Fake("r1.fast")]}, time_stub, lambda: len(timed) < 1)

    assert timed == ["r0.ref"]
    assert sut == start


def test_a_cpu_stub_time_is_the_elapsed_time_of_the_graded_thread_count() -> None:
    configs = [{"threads": 1, "elapsed_ns": 900}, {"threads": 8, "elapsed_ns": 200}]
    answer = ProfileAnswer.model_construct(root=LinuxperfReport.model_construct(configs=configs))

    assert stub_time(answer) == 200.0


def test_a_gpu_stub_time_is_the_device_time_per_repetition() -> None:
    answer = ProfileAnswer.model_construct(root=DeviceTraceReport.model_construct(device_ns_per_rep=1234.5))

    assert stub_time(answer) == 1234.5


def test_a_profile_without_a_time_is_refused() -> None:
    answer = ProfileAnswer.model_construct(root=LinuxperfReport.model_construct(configs=[]))

    with pytest.raises(ValueError, match="no stub time"):
        stub_time(answer)
