# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""The envelope's ``requests`` block: how a submission asks the judge to change what reaches its kernel.

``libraries`` (catalog names), ``precision`` (per-array element type), ``layout`` (per-array op
sequence, MPI ``distribute`` included), ``grid`` (the processor grid, extents over the rank count ``P``),
``location`` (per-array residency of a distributed run), ``sparse`` (per-array sparse format) and the
group ops ``zip`` / ``unzip``. See docs/requests_api.md.

The layout ops, the grid expressions, their JSON, inverses and lowering are relayout's
(:mod:`relayout.ops`): this module parses the block around them and lowers it, at a rank count, to the
distribution plan the scatter descriptor reads.
"""

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from relayout.ops import (
    GROUP_OPS,
    RANKS_SYMBOL,
    AxisSplit,
    GroupOp,
    LayoutOp,
    Scheme,
    expr_json,
    grid_extents,
    groups_from_json,
    groups_json,
    lower_layout,
    op_json,
    ops_for_axes,
    parse_grid,
    parse_op,
)

from hpcagent_bench import config
from hpcagent_bench.dtypes import canonical, is_float_dtype
from hpcagent_bench.precision import Precision, numpy_dtype, precision_from_datatype
from hpcagent_bench.support.bindings.contract import graded_datatype
from hpcagent_bench.support.helpers.sparse.abi import parse_sparse_config

if TYPE_CHECKING:
    from hpcagent_bench.spec import BenchSpec
    from hpcagent_bench.support.bindings.contract import Binding

__all__ = [
    "Location",
    "Requests",
    "grid_request",
    "mpi_residency",
    "precision_problem_dtypes",
    "precision_refusal",
]


class Location(enum.Enum):
    """Where a distributed array's tiles live during the call; the harness scatters on the host."""

    HOST = "host"
    DEVICE = "device"


def mpi_residency() -> Location:
    """The run-wide residency of a distributed array no ``requests.location`` names (``mpi.residency``)."""
    return Location(config.get_str("mpi.residency", Location.HOST.value))


def grid_request(grid: Sequence[int]) -> tuple[str, ...]:
    """The ``requests.grid`` that is ``grid`` at its rank count and follows ``P`` like a scaling run does:
    ``[P]`` in 1-D, ``root(P, d)`` per edge of an equal-edge ``d``-D grid, else the literal extents."""
    if len(grid) == 1:
        return (RANKS_SYMBOL,)
    if len(set(grid)) == 1:
        return (f"root({RANKS_SYMBOL}, {len(grid)})",) * len(grid)
    return tuple(str(int(extent)) for extent in grid)


@dataclass(frozen=True, slots=True)
class Requests:
    """One submission's ``requests`` block, structurally validated; checks against the kernel (array
    names, dtypes, the binding's ranks) are the judge's."""

    libraries: tuple[str, ...] = ()
    precision: Mapping[str, Precision] = field(default_factory=dict)
    #: The processor grid's extents as expressions over the rank count ``P`` (:func:`relayout.ops.grid_extents`).
    grid: tuple[str, ...] | None = None
    layout: Mapping[str, tuple[LayoutOp, ...]] = field(default_factory=dict)
    location: Mapping[str, Location] = field(default_factory=dict)
    #: The group ops (``zip`` / ``unzip``), unzips first.
    groups: tuple[GroupOp, ...] = ()
    sparse: Mapping[str, Any] | None = None

    @property
    def distributed(self) -> bool:
        """Whether this asks for an MPI run: it names a processor grid."""
        return self.grid is not None

    @classmethod
    def from_obj(cls, raw: object) -> "Requests":
        """Parse the ``requests`` JSON object (None = no requests); a malformed block raises ValueError."""
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"requests must be an object; got {type(raw).__name__}")
        known = ["libraries", "precision", "grid", "layout", "location", "sparse", *GROUP_OPS.keys()]
        unknown = set(raw) - set(known)
        if unknown:
            raise ValueError(f"requests has unknown field(s) {sorted(unknown)}; fields are {', '.join(known)}")
        libraries = raw.get("libraries") or []
        if not isinstance(libraries, list) or not all(isinstance(name, str) for name in libraries):
            raise ValueError("requests.libraries must be a list of catalog names")
        grid = None if raw.get("grid") is None else parse_grid(raw["grid"], "requests.grid")
        location = parse_location(raw.get("location"))
        if location and grid is None:
            raise ValueError("requests.location places the tiles of a distributed run; it needs requests.grid")
        groups = [
            group
            for key in ("unzip", "zip")
            if key in raw
            for group in groups_from_json(key, raw[key], f"requests.{key}")
        ]
        return cls(
            libraries=tuple(libraries),
            precision=parse_precision(raw.get("precision")),
            grid=grid,
            layout=parse_layout(raw.get("layout")),
            location=location,
            groups=tuple(groups),
            sparse=parse_sparse(raw.get("sparse")),
        )

    def to_json(self) -> dict[str, Any]:
        """The block as JSON, empty fields left out (``{}`` = no requests)."""
        out: dict[str, Any] = {}
        if self.libraries:
            out["libraries"] = list(self.libraries)
        if self.precision:
            out["precision"] = {name: p.value for name, p in sorted(self.precision.items())}
        if self.grid is not None:
            out["grid"] = [expr_json(extent) for extent in self.grid]
        if self.layout:
            out["layout"] = {name: [op_json(op) for op in ops] for name, ops in sorted(self.layout.items())}
        if self.location:
            out["location"] = {name: loc.value for name, loc in sorted(self.location.items())}
        out.update(groups_json(self.groups))
        if self.sparse is not None:
            out["sparse"] = dict(self.sparse)
        return out

    def grid_at(self, ranks: int) -> tuple[int, ...] | None:
        """The grid's extents at ``ranks`` ranks; None when there is no grid or it does not span ``ranks``."""
        return None if self.grid is None else grid_extents(self.grid, ranks, "requests.grid")

    def plan(self, ndims: Mapping[str, int], ranks: int) -> dict[str, Any]:
        """The distribution plan the descriptor scatters with at ``ranks`` ranks (``{"grid", "arrays": {name:
        {"axes" | "replicated", "location"}}}``). Every op sequence is lowered here, so a sequence that cannot
        be is refused before anything is built."""
        if self.grid is None:
            raise ValueError("requests.layout distributes nothing without requests.grid")
        dims = self.grid_at(ranks)
        if dims is None:
            raise ValueError(f"requests.grid {list(self.grid)} does not span {ranks} rank(s)")
        if self.groups:
            raise ValueError(f"requests.{self.groups[0].op} is not implemented yet")
        lowered: dict[str, list[AxisSplit | None]] = {}
        for name, ops in sorted(self.layout.items()):
            if name not in ndims:
                raise ValueError(f"requests.layout names unknown array {name!r}; arrays are {sorted(ndims)}")
            lowered[name] = lower_layout(ops, ndims[name], dims, f"requests.layout[{name!r}]")
        cyclic = any(axis is not None and axis.scheme is not Scheme.BLOCK for axes in lowered.values() for axis in axes)
        if cyclic and len(set(dims)) > 1:
            raise ValueError(
                f"a cyclic / block-cyclic layout needs an equal-edge grid so the cyclic wrap is symmetric; "
                f"requests.grid {list(self.grid)} is {list(dims)} at {ranks} rank(s)"
            )
        arrays: dict[str, dict[str, Any]] = {name: axes_json(axes) for name, axes in lowered.items()}
        for name, location in self.location.items():
            if name not in ndims:
                raise ValueError(f"requests.location names unknown array {name!r}; arrays are {sorted(ndims)}")
            arrays.setdefault(name, {"replicated": True})["location"] = location.value
        return {"grid": list(dims), "arrays": arrays}

    @classmethod
    def from_plan(cls, plan: Mapping[str, Any], libraries: Sequence[str] = ()) -> "Requests":
        """The requests whose :meth:`plan` at the plan's rank count is ``plan``, with the grid following ``P``
        (:func:`grid_request`): how a kernel's default distribution (and a grade recorded before the
        ``requests`` column) is shown as op sequences."""
        layout: dict[str, tuple[LayoutOp, ...]] = {}
        location: dict[str, Location] = {}
        for name, entry in plan["arrays"].items():
            if "location" in entry:
                location[name] = Location(entry["location"])
            if entry.get("replicated"):
                layout[name] = ()
                continue
            axes = [
                None
                if axis.get("grid_dim") is None
                else AxisSplit(
                    int(axis["grid_dim"]),
                    Scheme(axis.get("scheme", Scheme.BLOCK.value)),
                    int(axis.get("block_size", 1)),
                )
                for axis in entry["axes"]
            ]
            layout[name] = ops_for_axes(axes)
        grid = grid_request([int(extent) for extent in plan["grid"]])
        return cls(libraries=tuple(libraries), grid=grid, layout=layout, location=location)


def axes_json(axes: Sequence[AxisSplit | None]) -> dict[str, Any]:
    """One array's plan entry: ``{"replicated": true}`` when no dimension is split, else its ``axes``."""
    if all(axis is None for axis in axes):
        return {"replicated": True}
    return {
        "axes": [
            {"grid_dim": None}
            if axis is None
            else {"grid_dim": axis.grid_dim, "scheme": axis.scheme.value, "block_size": axis.block_size}
            for axis in axes
        ]
    }


def parse_precision(raw: object) -> dict[str, Precision]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("requests.precision must be an object {array: precision}")
    out: dict[str, Precision] = {}
    for name, value in raw.items():
        if not isinstance(value, str):
            raise ValueError(f"requests.precision[{name!r}] must be a precision name; got {value!r}")
        out[str(name)] = Precision.from_str(value)
    return out


def parse_layout(raw: object) -> dict[str, tuple[LayoutOp, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("requests.layout must be an object {array: [op, ...]}")
    out: dict[str, tuple[LayoutOp, ...]] = {}
    for name, ops in raw.items():
        if not isinstance(ops, list):
            raise ValueError(f"requests.layout[{name!r}] must be a list of ops")
        out[str(name)] = tuple(parse_op(op, f"requests.layout[{name!r}][{k}]") for k, op in enumerate(ops))
    return out


def parse_location(raw: object) -> dict[str, Location]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("requests.location must be an object {array: 'host' | 'device'}")
    out: dict[str, Location] = {}
    for name, value in raw.items():
        try:
            out[str(name)] = Location(value)
        except ValueError:
            raise ValueError(f"requests.location[{name!r}] must be 'host' or 'device'; got {value!r}") from None
    return out


def parse_sparse(raw: object) -> dict[str, Any] | None:
    if raw is None:
        return None
    parse_sparse_config(raw)  # the shape only, with its own refusals; the kernel is consulted by the judge
    if not isinstance(raw, dict):  # unreachable past parse_sparse_config; narrows the type
        raise ValueError("requests.sparse must be an object {array: format}")
    return {str(name): label for name, label in raw.items()}


def precision_refusal(precision: Mapping[str, Precision], problem: Mapping[str, str]) -> str | None:
    """Why ``precision`` cannot be honoured, or None. ``problem`` maps every array a precision may be
    requested for (floating, dense, not an index array) to its dtype at the problem precision."""
    for name, wanted in sorted(precision.items()):
        if name not in problem:
            return (
                f"requests.precision names {name!r}, which is not a floating dense array of this kernel; "
                f"precision may be requested for {sorted(problem)}"
            )
        if np.dtype(numpy_dtype(wanted)) == np.dtype(problem[name]):
            return f"requests.precision[{name!r}] = {wanted.value!r} is already the array's problem precision; drop it"
    return None


def precision_problem_dtypes(spec: "BenchSpec", binding: "Binding", datatype: str) -> dict[str, str]:
    """Every array a precision may be requested for -- floating, dense, not an index array -- with its
    dtype at the problem precision: a manifest ``init.dtypes`` override, else the graded datatype's."""
    problem = np.dtype(numpy_dtype(precision_from_datatype(graded_datatype(spec, datatype)))).name
    sparse = {member for group in binding.packed for member in group.members}
    out: dict[str, str] = {}
    for arg in binding.pointers:
        if arg.is_index or arg.name in sparse or not is_float_dtype(arg.dtype):
            continue
        declared = spec.init.dtypes.get(arg.name) if spec.init is not None else None
        out[arg.name] = canonical(declared) if declared is not None else problem
    return out
