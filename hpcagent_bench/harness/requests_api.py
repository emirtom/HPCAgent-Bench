# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""The envelope's ``requests`` block: how a submission asks the judge to change what reaches its kernel.

``libraries`` (catalog names), ``precision`` (per-array element type), ``layout`` (per-array op
sequence, MPI ``distribute`` included), ``grid`` (the processor grid the layout distributes over),
``location`` (per-array residency of a distributed run), ``sparse`` (per-array sparse format) and the
group ops ``zip`` / ``unzip``. See docs/requests_api.md.

The layout ops, their JSON, inverses and lowering are relayout's (:mod:`relayout.ops`): this module
parses the block around them and lowers it to the distribution plan the scatter descriptor reads.
"""

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from hpcagent_bench.dtypes import canonical, is_float_dtype
from hpcagent_bench.precision import Precision, numpy_dtype, precision_from_datatype
from hpcagent_bench.support.bindings.contract import graded_datatype
from hpcagent_bench.support.helpers.sparse.abi import parse_sparse_config
from relayout.ops import (
    GROUP_OPS,
    SCHEMES,
    GroupOp,
    LayoutOp,
    Scheme,
    Split,
    from_json,
    lower_layout,
    op_json,
    ops_for_axes,
    parse_op,
    to_json,
)

if TYPE_CHECKING:
    from hpcagent_bench.spec import BenchSpec
    from hpcagent_bench.support.bindings.contract import Binding

__all__ = [
    "SCHEMES",
    "Location",
    "Requests",
    "precision_problem_dtypes",
    "precision_refusal",
]


class Location(enum.Enum):
    """Where a distributed array's tiles live during the call; the harness scatters on the host."""

    HOST = "host"
    DEVICE = "device"


@dataclass(frozen=True, slots=True)
class Requests:
    """One submission's ``requests`` block, structurally validated; checks against the kernel (array
    names, dtypes, the binding's ranks) are the judge's."""

    libraries: tuple[str, ...] = ()
    precision: Mapping[str, Precision] = field(default_factory=dict)
    grid: tuple[int, ...] | None = None
    layout: Mapping[str, tuple[LayoutOp, ...]] = field(default_factory=dict)
    location: Mapping[str, Location] = field(default_factory=dict)
    #: The group ops (``zip`` / ``unzip``), unzips first, each list in request order.
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
        return cls(
            libraries=tuple(libraries),
            precision=parse_precision(raw.get("precision")),
            grid=parse_grid(raw.get("grid")),
            layout=parse_layout(raw.get("layout")),
            location=parse_location(raw.get("location")),
            groups=tuple(
                group
                for key in sorted(GROUP_OPS.keys(), key="unzip".__ne__)
                for group in parse_group(key, raw.get(key))
            ),
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
            out["grid"] = list(self.grid)
        if self.layout:
            out["layout"] = {name: [op_json(op) for op in ops] for name, ops in sorted(self.layout.items())}
        if self.location:
            out["location"] = {name: loc.value for name, loc in sorted(self.location.items())}
        for key in GROUP_OPS.keys():
            if listed := [to_json(group) for group in self.groups if group.op == key]:
                out[key] = listed
        if self.sparse is not None:
            out["sparse"] = dict(self.sparse)
        return out

    def plan(self, ndims: Mapping[str, int], grid: Sequence[int] | None = None) -> dict[str, Any]:
        """The distribution plan the descriptor scatters with (``{"grid", "arrays": {name: {"axes" |
        "replicated", "location"}}}``), over ``grid`` (default: the requested one). Every op sequence is
        lowered here, so a sequence that cannot be is refused before anything is built."""
        chosen = self.grid if grid is None else grid
        if chosen is None:
            raise ValueError("requests.layout distributes nothing without requests.grid")
        dims = tuple(chosen)
        if self.groups:
            raise ValueError(
                f"requests.{self.groups[0].op} is not implemented yet (see docs/requests_api.md section 4.5)"
            )
        arrays: dict[str, dict[str, Any]] = {}
        for name, ops in sorted(self.layout.items()):
            if name not in ndims:
                raise ValueError(f"requests.layout names unknown array {name!r}; arrays are {sorted(ndims)}")
            lowered = lower_layout(ops, ndims[name], dims, f"requests.layout[{name!r}]")
            if all(axis is None for axis in lowered):
                entry: dict[str, Any] = {"replicated": True}
            else:
                entry = {
                    "axes": [
                        {"grid_dim": None}
                        if axis is None
                        else {"grid_dim": axis.grid_dim, "scheme": axis.scheme.value, "block_size": axis.block_size}
                        for axis in lowered
                    ]
                }
            arrays[name] = entry
        cyclic = any(
            axis.get("scheme") in ("cyclic", "block_cyclic")
            for entry in arrays.values()
            for axis in entry.get("axes", [])
        )
        if cyclic and len(set(dims)) > 1:
            raise ValueError(
                f"a cyclic / block-cyclic layout needs an equal-edge hypercube grid ([P], [P, P], ...) so the cyclic "
                f"wrap is symmetric; got requests.grid {list(dims)}"
            )
        for name, location in self.location.items():
            if name not in ndims:
                raise ValueError(f"requests.location names unknown array {name!r}; arrays are {sorted(ndims)}")
            arrays.setdefault(name, {"replicated": True})["location"] = location.value
        return {"grid": list(dims), "arrays": arrays}

    @classmethod
    def from_plan(cls, plan: Mapping[str, Any], libraries: Sequence[str] = ()) -> "Requests":
        """The requests whose :meth:`plan` is ``plan``: how a kernel's default distribution (and a grade
        recorded before the ``requests`` column) is shown as op sequences."""
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
                else Split(int(axis["grid_dim"]), Scheme(axis.get("scheme", "block")), int(axis.get("block_size", 1)))
                for axis in entry["axes"]
            ]
            layout[name] = ops_for_axes(axes)
        return cls(
            libraries=tuple(libraries), grid=tuple(int(p) for p in plan["grid"]), layout=layout, location=location
        )

    def with_grid(self, grid: Sequence[int]) -> "Requests":
        """These requests over another grid; factor expressions re-evaluate there."""
        return replace(self, grid=tuple(grid))


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


def parse_grid(raw: object) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(p, int) and not isinstance(p, bool) and p >= 1 for p in raw)
    ):
        raise ValueError("requests.grid must be a non-empty list of positive ints")
    return tuple(raw)


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


def parse_group(key: str, raw: object) -> list[GroupOp]:
    """The ``requests.<key>`` list of one group op."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"requests.{key} must be a list of {{name, fields}} objects")
    groups: list[GroupOp] = []
    for k, entry in enumerate(raw):
        where = f"requests.{key}[{k}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be an object; got {entry!r}")
        group = from_json(GROUP_OPS.entries[key], entry, "", where)
        group.validate(where)
        groups.append(group)
    return groups


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
