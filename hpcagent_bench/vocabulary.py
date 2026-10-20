# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The vocabulary of a figure and a results database: LLMs, standalone optimizers, agent harnesses, languages,
devices, skill packets and framework columns -- one :class:`~hpcagent_bench.registry.Kind` each, and the entry
types they hold.

The entries are added in :mod:`hpcagent_bench.models`, :mod:`hpcagent_bench.columns` and
:mod:`hpcagent_bench.skill_packets` (and by plugins, :mod:`hpcagent_bench.plugins`). ``docs/extending/registry.md``
is the author's guide.

An entry's slot is its registration position (:class:`~hpcagent_bench.registry.Kind`); :mod:`hpcagent_bench.stats.palette`
derives its colour and marker from it.
"""

import dataclasses
import enum
import re

from hpcagent_agent.coded import Coded

from hpcagent_bench.precision import Precision
from hpcagent_bench.registry import Kind, RegistryError

__all__ = [
    "ADAPTER_PATTERN",
    "DEVICES",
    "FRAMEWORKS",
    "HARNESSES",
    "KINDS",
    "LANGUAGES",
    "MODELS",
    "OPTIMIZERS",
    "OPTIMIZER_ADAPTER_MODULE",
    "PACKETS",
    "TUNERS",
    "Arch",
    "Column",
    "DaceStage",
    "ModelEntry",
    "PacketDef",
    "PacketDevice",
    "Transform",
    "TunerEntry",
    "check_vocabulary",
    "is_optimizer_adapter",
]


class Arch(Coded):
    """The device a framework column runs on, so the device anti-cheats a grade meets."""

    CPU = enum.auto()
    GPU = enum.auto()


class DaceStage(Coded):
    """A DaCe flavor's pipeline stage."""

    CANON = enum.auto()
    PARALLEL = enum.auto()
    AUTOOPT = enum.auto()


class Transform(Coded):
    """The source-to-source tool whose output a pluto-family column compiles."""

    PLUTO = enum.auto()
    PPCG = enum.auto()
    POLYGEIST = enum.auto()


class PacketDevice(Coded):
    """Whose tools a packet's pages teach."""

    CPU = enum.auto()
    AMD = enum.auto()
    NVIDIA = enum.auto()


@dataclasses.dataclass(frozen=True, slots=True)
class ModelEntry:
    """A model's display name and the checkpoint it is expected to serve.

    The checkpoint is recorded so an experiment that swaps one cannot silently keep the old name on an axis;
    ``tests/test_display_names.py`` checks it against what the setups really ran."""

    name: str
    serves: str


@dataclasses.dataclass(frozen=True, slots=True)
class PacketDef:
    """One packet's definition, before ``${VAR}`` placeholders are filled or ``lang``/``*`` are expanded into
    concrete skill pages -- see :mod:`hpcagent_bench.packets`, the resolver that reads this.

    A registered key's definition is immutable once a results database has recorded it (the ``packets`` table in
    :mod:`hpcagent_bench.harness.recording`): a changed meaning goes under a NEW key."""

    #: The display name.
    name: str
    #: Skill page directories to stage; ``lang`` and ``*`` expand at resolve time.
    skills: tuple[str, ...] = ()
    #: Other registered packet keys this one composes.
    packets: tuple[str, ...] = ()
    #: ``KEY=value`` switches, in order; a value may hold ``${VAR}``.
    env: tuple[tuple[str, str], ...] = ()
    #: The method this packet runs: a method directory a plugin ships (its ``method.md`` fills ``{{HINTS}}``);
    #: resolving it exports ``AGENT_METHOD`` (:data:`hpcagent_bench.packets.METHOD_ENV`). At most one per packet.
    method: str = ""
    #: MCP tools this packet CARRIES, served only in its setups; its ``skills`` are then that tool's manual, which
    #: ``*`` does not expand to (:func:`hpcagent_bench.packets.tool_pages`).
    tools: tuple[str, ...] = ()
    #: Whose tools the pages teach; None for a device-neutral packet.
    device: PacketDevice | None = None
    #: Why a recorded key takes no new submissions; "" while it still does.
    frozen: str = ""
    #: The shape this packet wears instead of the next free one from the pool ("": the pool's).
    marker: str = ""
    #: A figure's short spelling, for a column naming delivery and packet at once ("C-Skills").
    short: str = ""


#: ``package.module:Class``.
ADAPTER_PATTERN = re.compile(r"^[a-z_][a-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
#: The module whose :class:`~hpcagent_bench.harness.agent.Agent` classes an autotuning flavor's ``adapter`` names;
#: every other adapter is a :class:`~hpcagent_bench.frameworks.framework.Framework` subclass.
OPTIMIZER_ADAPTER_MODULE = "hpcagent_bench.harness.optimizers"


def is_optimizer_adapter(adapter: str) -> bool:
    """Whether ``adapter`` names an autotuning :class:`~hpcagent_bench.harness.agent.Agent`, not a Framework."""
    return adapter.partition(":")[0] == OPTIMIZER_ADAPTER_MODULE


@dataclasses.dataclass(frozen=True, slots=True)
class Column:
    """One framework column (or flavor of one): what runs it, where, and how a figure names it
    (:mod:`hpcagent_bench.columns` has the naming scheme)."""

    #: The name a figure prints: the compiler or library, not the flavor.
    display: str
    #: ``package.module:Class`` of the Framework (or autotuning Agent), imported on first use.
    adapter: str
    #: The backend this column is a flavor of; columns sharing a base share an adapter.
    base: str
    #: The long name a table prints.
    full_name: str
    #: Selects the kernel impl file, ``<module>_<postfix>.py``.
    postfix: str
    arch: Arch
    #: Whether a deterministic (unjudged, no-agent) sweep may select it.
    sweep_deterministic: bool
    #: The precisions it can execute; ``run-framework`` does not enforce it.
    precisions: frozenset[Precision]
    #: The pipelines the adapter compiles and scores for this column (a DaCe flavor's SDFG pipeline).
    pipelines: tuple[str, ...] = ()
    #: The framework's own pipeline stage this flavor stops at (:class:`DaceStage` for DaCe).
    stage: Coded | None = None
    #: The :data:`TUNERS` key of the autotuner run after the stage; the column's ``adapter`` is that tuner's.
    tuner: str | None = None
    #: With ``flavor``: the DB column this name groups under; the name must be ``<column>_<flavor>``.
    column: str | None = None
    flavor: str | None = None
    #: What the native/pluto column compiles.
    language: str | None = None
    #: The language of the translator output the sources start from.
    emit_language: str | None = None
    #: The ``compilers.yaml`` block the build forces.
    compiler: str | None = None
    #: The :mod:`hpcagent_bench.flags` preset appended to the baseline.
    flags: str | None = None
    #: The ``flags.<probe>()`` that must read OK before the column builds.
    autopar_gate: str | None = None
    transform: Transform | None = None

    def __post_init__(self) -> None:
        if not ADAPTER_PATTERN.match(self.adapter):
            raise RegistryError(
                f"frameworks {self.display!r}: adapter {self.adapter!r} must read 'package.module:Class'"
            )
        if not self.precisions:
            raise RegistryError(f"frameworks {self.display!r}: precisions is empty, so the column can execute nothing")
        if (self.column is None) != (self.flavor is None):
            raise RegistryError(f"frameworks {self.display!r}: a flavor declares both column and flavor, or neither")

    @property
    def optimizer(self) -> bool:
        """Whether an autotuning Agent runs this column, not a Framework."""
        return is_optimizer_adapter(self.adapter)


@dataclasses.dataclass(frozen=True, slots=True)
class TunerEntry:
    """An autotuner a column can run after its framework's stage: how a figure names it and the
    :class:`~hpcagent_bench.harness.agent.Agent` that runs it (``package.module:Class``)."""

    display: str
    full_name: str
    adapter: str

    def __post_init__(self) -> None:
        if not is_optimizer_adapter(self.adapter):
            raise RegistryError(
                f"tuners {self.display!r}: adapter {self.adapter!r} must name a class of {OPTIMIZER_ADAPTER_MODULE}"
            )


MODELS: Kind[ModelEntry] = Kind("models")
#: Standalone optimizers by display name: a compiler or pipeline that stands where an LLM stands on a figure.
OPTIMIZERS: Kind[str] = Kind("optimizers")
#: Agent harnesses (the loop driving the model) by display name.
HARNESSES: Kind[str] = Kind("harnesses")
#: The language a setup asked for, by display name.
LANGUAGES: Kind[str] = Kind("languages")
#: Autotuners a framework column runs after its stage (Alloy, ...), by key.
TUNERS: Kind[TunerEntry] = Kind("tuners")
#: Where a kernel was timed, by display name.
DEVICES: Kind[str] = Kind("devices")
PACKETS: Kind[PacketDef] = Kind("packets")
FRAMEWORKS: Kind[Column] = Kind("frameworks")

#: Every vocabulary kind by its plural name, the spelling a figure and the display-name lookups use.
KINDS: dict[str, Kind[ModelEntry] | Kind[str] | Kind[PacketDef] | Kind[Column]] = {
    kind.name: kind for kind in (MODELS, OPTIMIZERS, HARNESSES, LANGUAGES, DEVICES, PACKETS, FRAMEWORKS)
}


def check_vocabulary() -> None:
    """The rules that span entries, run once the whole vocabulary is registered: every entry but the no-packet
    control has a slot, columns sharing a base share one Framework adapter, and a composed packet exists."""
    for kind in KINDS.values():
        slotless = sorted(key for key, order in kind.orders.items() if order is None and (kind is not PACKETS or key))
        if slotless:
            raise RegistryError(f"{kind.name}: {slotless} have no slot, so no colour or marker")
    for key, column in FRAMEWORKS.entries.items():
        if column.tuner is not None and (column.tuner not in TUNERS or TUNERS[column.tuner].adapter != column.adapter):
            raise RegistryError(f"frameworks {key!r}: tuner {column.tuner!r} is not registered with its adapter")
    for base in {column.base for column in FRAMEWORKS.entries.values()}:
        adapters = {
            column.adapter for column in FRAMEWORKS.entries.values() if column.base == base and not column.optimizer
        }
        if len(adapters) != 1:
            raise RegistryError(
                f"frameworks: base {base!r} names Framework adapters {sorted(adapters)}; it needs exactly one"
            )
    if PACKETS.orders.get("", 0) is not None or "" not in PACKETS.entries:
        raise RegistryError("packets: the no-packet control (key '') must be registered with slotted=False")
    for key, definition in PACKETS.entries.items():
        for part in definition.packets:
            if PACKETS.canonical(part) not in PACKETS.entries:
                raise RegistryError(f"packets {key!r}: composes {part!r}, which is not registered")
