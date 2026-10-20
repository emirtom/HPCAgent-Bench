# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The judge's tool routes: ``GET /<name>/...`` answered by a ``@route`` handler in this package, one module per
tool, so adding a judge-side tool never edits :mod:`hpcagent_bench.harness.service`.

    @route("canonical_parallel_form")
    def serve(judge: Judge, parts: list[str], qs: dict[str, list[str]]) -> None: ...

``parts`` is the path split on ``/`` (``parts[0]`` is the route), ``qs`` the parsed query. The judge router
(``hpcagent_bench/cluster/judge_service.py``) relays every GET it does not answer itself, so a new route needs
no router code; the agent-side half is a ``@tool`` in ``agent/hpcagent_agent/tools/``.
"""

import contextlib
import importlib
import pkgutil
from collections.abc import Callable, Mapping
from typing import Protocol

from hpcagent_bench.api import RunConfig
from hpcagent_bench.harness.judge_scheduler import DeviceSlot
from hpcagent_bench.harness.judge_wire import JudgeAnswer

__all__ = ["RESERVED", "ROUTES", "Handler", "Judge", "first_param", "load", "route"]


class Judge(Protocol):
    """What a route handler may use of the request handler serving it (``service.JudgeHandler``)."""

    cfg: RunConfig

    def misrouted(self, requested: object) -> bool: ...

    def reply(self, code: int, payload: JudgeAnswer) -> None: ...

    def device_slot(self) -> contextlib.AbstractContextManager[DeviceSlot | None]: ...


#: A route's handler: the judge, the path parts and the query; it answers through ``judge.reply``.
Handler = Callable[[Judge, list[str], dict[str, list[str]]], None]

#: The GET routes ``service.JudgeHandler.serve_get`` answers itself; a tool route may not take one.
RESERVED = frozenset({"health", "build", "baseline"})

#: Every tool route, by its first path segment.
ROUTES: dict[str, Handler] = {}


def first_param(qs: Mapping[str, list[str]], name: str) -> str | None:
    """The first value of query parameter ``name``, or None."""
    values = qs.get(name)
    return values[0] if values else None


def route(name: str) -> Callable[[Handler], Handler]:
    """Register the decorated handler for ``GET /<name>/...``."""

    def register(handler: Handler) -> Handler:
        if name in RESERVED:
            raise ValueError(f"route {name!r} of {handler.__module__} is one the judge answers itself")
        existing = ROUTES.get(name)
        if existing is not None and existing.__module__ != handler.__module__:  # a reload re-registers its own
            raise ValueError(f"route {name!r} of {handler.__module__} is already registered by {existing.__module__}")
        ROUTES[name] = handler
        return handler

    return register


def load() -> dict[str, Handler]:
    """Import every route module of this package (each registers itself) and return :data:`ROUTES`."""
    for module in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{module.name}")
    return ROUTES
