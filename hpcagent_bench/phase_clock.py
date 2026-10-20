# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""How long a framework column takes to turn a kernel into a binary, split into the three phases the canon table
records (``transform_time``, ``codegen_time``, ``compile_time``), cached work included.

* ``transform``: everything that is not one of the other two -- parsing, simplify, canonicalize, auto-optimize,
  a polyhedral transform (``polycc``);
* ``codegen``: emitting source (the numpy-to-C/Fortran translators, DaCe's code generator, a CPF form render);
* ``compile``: the backend compiler building the emitted source.

The span is measured by :func:`measuring` around a column's emit, optimize and compile steps
(:meth:`hpcagent_bench.frameworks.test.Test.run`); a step inside it marks its phase with :func:`phase`, and the rest of
the span is ``transform``. A cache that serves an artifact credits the seconds producing it took to the phase that
produced it (:func:`credit`), so a column that read its canonical SDFG or its emitted sources from a cache reports what
the work costs, not what the cache saved.

The running measurements and marks are per context (:mod:`contextvars`), so each thread -- a judge thread rendering
a form on demand next to another -- times and credits only its own spans.

    with phase_clock.measuring() as clock:
        with phase_clock.phase(Phase.COMPILE):
            ...                                    # the backend compiler
        phase_clock.credit(12.5, Phase.CODEGEN)  # a cache hit on emitted sources
    clock.times()                                  # {"transform_time": ..., "codegen_time": ..., "compile_time": ...}
"""

import contextlib
import contextvars
import dataclasses
import enum
import time
from collections.abc import Iterator

from hpcagent_agent.coded import Coded

__all__ = ["MARKS", "PHASES", "RUNNING", "Clock", "Mark", "Phase", "credit", "measuring", "phase", "shift"]


class Phase(Coded):
    """A build phase; its spelling names the canon column ``<phase>_time``."""

    TRANSFORM = enum.auto()
    CODEGEN = enum.auto()
    COMPILE = enum.auto()


#: The phases, in pipeline order.
PHASES: tuple[Phase, ...] = tuple(Phase)


@dataclasses.dataclass(slots=True)
class Clock:
    """One measurement: the seconds each phase took, filled in as the span runs and finished when it ends."""

    seconds: dict[Phase, float] = dataclasses.field(default_factory=lambda: dict.fromkeys(PHASES, 0.0))
    #: Wall seconds spent inside marked phases (credits excluded): the span's rest is :attr:`Phase.TRANSFORM`.
    marked_wall: float = 0.0

    def times(self) -> dict[str, float]:
        """The canon columns: ``<phase>_time`` for every phase."""
        return {f"{name.text}_time": self.seconds[name] for name in PHASES}


@dataclasses.dataclass(slots=True)
class Mark:
    """A running :func:`phase`: its name, start, and the wall seconds phases nested in it took."""

    name: Phase
    started: float
    nested: float = 0.0


#: The measurements running in this context, innermost last; time and credit go to every one of them.
RUNNING: contextvars.ContextVar[tuple[Clock, ...]] = contextvars.ContextVar("phase_clock_running", default=())
#: The marked phases running in this context, innermost last.
MARKS: contextvars.ContextVar[tuple[Mark, ...]] = contextvars.ContextVar("phase_clock_marks", default=())


def credit(seconds: float, name: Phase) -> None:
    """Add a cached artifact's production ``seconds`` to ``name`` in every measurement running in this context."""
    for clock in RUNNING.get():
        clock.seconds[name] += max(float(seconds), 0.0)


def shift(seconds: float, source: Phase, target: Phase) -> None:
    """Move ``seconds`` of a marked phase to ``target`` in every measurement running in this context: a step that
    reports its own split after the fact (DaCe's ``SDFG.compile(stats=...)`` generating code before it compiles)."""
    for clock in RUNNING.get():
        moved = min(max(float(seconds), 0.0), clock.seconds[source])
        clock.seconds[source] -= moved
        clock.seconds[target] += moved


@contextlib.contextmanager
def phase(name: Phase) -> Iterator[None]:
    """Mark the enclosed block as ``name``. A phase marked inside another takes its own seconds from the outer one."""
    mark = Mark(name, time.perf_counter())
    token = MARKS.set((*MARKS.get(), mark))
    try:
        yield
    finally:
        MARKS.reset(token)
        elapsed = time.perf_counter() - mark.started
        own = elapsed - mark.nested
        for clock in RUNNING.get():
            clock.seconds[name] += own
            clock.marked_wall += own
        outer = MARKS.get()
        if outer:
            outer[-1].nested += elapsed


@contextlib.contextmanager
def measuring() -> Iterator[Clock]:
    """Measure the enclosed block. On exit, :attr:`Phase.TRANSFORM` also holds the span's wall time no marked
    phase took."""
    clock = Clock()
    token = RUNNING.set((*RUNNING.get(), clock))
    started = time.perf_counter()
    try:
        yield clock
    finally:
        RUNNING.reset(token)
        clock.seconds[Phase.TRANSFORM] += max(time.perf_counter() - started - clock.marked_wall, 0.0)
