# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The optimize time a framework column reports (``canon.optimize_s``): one number per kernel, cached work included.

A column's optimize time is the wall time HPCAgent-Bench measures around the framework's emit, optimize and compile
steps (:meth:`hpcagent_bench.frameworks.test.Test.run`), plus the time every cached artifact those steps loaded took
to produce in the first place. A cache stores that production time beside the artifact and :func:`credit` adds it
when it serves a hit, so a column that read its canonical SDFG from the CPF prerender, or its emitted sources and
parsed SDFG from the kernel's ``.cache/``, reports what the work costs, not what the cache saved.

    with optimize_clock.measuring() as clock:
        ...                      # emit, optimize, compile; caches call optimize_clock.credit(seconds)
    clock.seconds                # wall time + credited production time
"""

import contextlib
import dataclasses
import time
from collections.abc import Iterator

__all__ = ["Clock", "credit", "measuring"]


@dataclasses.dataclass(slots=True, eq=False)  # identity: RUNNING removes this measurement, not an equal one
class Clock:
    """One measurement: the credited production seconds while it runs, and the total once it ends."""

    credited: float = 0.0
    seconds: float = 0.0


#: The measurements running in this process, innermost last; a credit goes to every one of them.
RUNNING: list[Clock] = []


def credit(seconds: float) -> None:
    """Add a cached artifact's production ``seconds`` to every running measurement (none: nothing to add to)."""
    for clock in RUNNING:
        clock.credited += max(float(seconds), 0.0)


@contextlib.contextmanager
def measuring() -> Iterator[Clock]:
    """Measure the enclosed block: on exit ``seconds`` is its wall time plus the credited production time."""
    clock = Clock()
    RUNNING.append(clock)
    started = time.perf_counter()
    try:
        yield clock
    finally:
        RUNNING.remove(clock)
        clock.seconds = time.perf_counter() - started + clock.credited
