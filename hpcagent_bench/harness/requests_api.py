# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""The envelope's ``requests``: how a submission asks the judge to change what reaches its kernel.

A request is relayout's (:mod:`relayout.ops`, its JSON :mod:`relayout.requests`): ``startup`` loads
catalog libraries and sets the processor grid, and ``operations`` run top to bottom over the kernel's
arrays (``cast``, ``format``, ``zip`` / ``unzip``, the digit cuts, ``distribute``, ``replicate``,
``place``). This module reads it against a kernel: which arrays exist, which precisions and sparse
formats apply, and the distribution plan the scatter descriptor reads at a rank count. See
docs/requests_api.md.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import relayout.ops
from relayout.ops import (
    RANKS_SYMBOL,
    ArrayName,
    ArraySpec,
    AxisSplit,
    Cast,
    DigitOperation,
    Format,
    FormatLabel,
    Location,
    Operation,
    Place,
    Replicate,
    Scheme,
    Startup,
    State,
    Unzip,
    Zip,
    lower,
    operations_for_axes,
)
from relayout.convert import Group, Member
from relayout.layout import IndexMap
from relayout.requests import parse_requests, requests_json

from hpcagent_bench import config
from hpcagent_bench.dtypes import canonical, is_float_dtype
from hpcagent_bench.harness.precision_casts import Regroup, extent_names
from hpcagent_bench.precision import Precision, numpy_dtype, precision_from_datatype
from hpcagent_bench.support.bindings.contract import graded_datatype
from hpcagent_bench.support.helpers.sparse.abi import array_layout

if TYPE_CHECKING:
    from hpcagent_bench.spec import BenchSpec
    from hpcagent_bench.support.bindings.contract import Binding

__all__ = [
    "Location",
    "Requests",
    "grid_request",
    "kernel_arrays",
    "mpi_residency",
    "precision_problem_dtypes",
    "precision_refusal",
]


def mpi_residency() -> Location:
    """The run-wide residency of a distributed array no ``place`` names (``mpi.residency``)."""
    return Location(config.get_str("mpi.residency", Location.HOST.value))


def grid_request(grid: Sequence[int]) -> tuple[str, ...]:
    """The ``startup.grid`` that is ``grid`` at its rank count and follows ``P`` like a scaling run does:
    ``[P]`` in 1-D, ``root(P, d)`` per edge of an equal-edge ``d``-D grid, else the literal extents."""
    if len(grid) == 1:
        return (RANKS_SYMBOL,)
    if len(set(grid)) == 1:
        return (f"root({RANKS_SYMBOL}, {len(grid)})",) * len(grid)
    return tuple(str(int(extent)) for extent in grid)


def kernel_arrays(binding: "Binding", ndims: Mapping[str, int]) -> list[ArraySpec]:
    """The arrays a request's operations start from: every shaped dense pointer of ``ndims`` (a complex
    one is the native struct ``(re, im)``: laid out as one element, unzipped into its parts) and every
    sparse logical array."""
    sparse = {member for group in binding.packed for member in group.members}
    dtypes = {arg.name: arg.dtype for arg in binding.pointers}
    complex_names = {name for name, dtype in dtypes.items() if np.dtype(canonical(dtype)).kind == "c"}
    dense = [
        ArraySpec(name, ndim, members=("re", "im"), native=True) if name in complex_names else ArraySpec(name, ndim)
        for name, ndim in ndims.items()
        if name in dtypes and name not in sparse
    ]
    return dense + [ArraySpec(group.logical, 2, sparse=True) for group in binding.packed]


@dataclass(frozen=True, slots=True)
class Requests(relayout.ops.Requests):
    """One submission's request, structurally validated; its checks against a kernel (array names,
    dtypes, the binding's ranks) are the judge's."""

    @classmethod
    def from_obj(cls, raw: object) -> "Requests":
        """Parse the ``requests`` JSON object (None = no requests); a malformed one raises ValueError."""
        parsed = parse_requests(raw)
        for op in parsed.operations:
            if isinstance(op, Format):
                array_layout(op.array, op.to)
        return cls(parsed.startup, parsed.operations)

    @classmethod
    def of(
        cls,
        *,
        libraries: Sequence[str] = (),
        precision: Mapping[str, Precision] | None = None,
        sparse: Mapping[str, str] | None = None,
    ) -> "Requests":
        """The single-node request loading ``libraries``, casting by ``precision`` and storing each sparse
        array in its ``sparse`` format."""
        ops: list[Operation] = []
        ops += [Cast(ArrayName(name), relayout.ops.Precision(p.value)) for name, p in (precision or {}).items()]
        ops += [Format(ArrayName(name), FormatLabel(label)) for name, label in (sparse or {}).items()]
        return cls(Startup(tuple(libraries)), tuple(ops))

    def to_json(self) -> dict[str, Any]:
        """The request as JSON, empty parts left out (``{}`` = no requests)."""
        return requests_json(self)

    @property
    def libraries(self) -> tuple[str, ...]:
        """The catalog libraries ``startup.load`` names."""
        return self.startup.load

    @property
    def grid(self) -> tuple[str, ...] | None:
        """``startup.grid``: the grid's extents over the rank count ``P``."""
        return self.startup.grid

    @property
    def precision(self) -> dict[str, Precision]:
        """Each array a ``cast`` names, with the precision its last cast leaves."""
        return {name: Precision(p.value) for name, p in self.precisions().items()}

    @property
    def location(self) -> dict[str, Location]:
        """Each array a ``place`` names, with the memory its last one leaves."""
        return self.locations()

    @property
    def sparse(self) -> dict[str, str] | None:
        """Each sparse array a ``format`` names, with its format label; None when none does."""
        return self.formats() or None

    def layouts(self, arrays: Sequence[ArraySpec]) -> dict[str, IndexMap]:
        """Each dense array of ``arrays`` (the kernel's, :func:`kernel_arrays`) the digit operations lay out on a
        single node, as its layout over the logical extents ``d0``, ``d1``, ... (:func:`precision_casts.extent_names`);
        an identity is left out, and a request with no digit operation runs nothing."""
        if self.distributed or not any(isinstance(op, DigitOperation) for op in self.operations):
            return {}
        state = self.run(arrays)
        laid = {
            spec.name: state.layout(spec.name, extent_names(spec.ndim))
            for spec in arrays
            if not spec.sparse and spec.name in state.arrays and state.arrays[spec.name].touched
        }
        return {name: layout for name, layout in laid.items() if not layout.is_identity()}

    def regroups(self, arrays: Sequence[ArraySpec], dtypes: Mapping[str, str]) -> tuple[Regroup, ...]:
        """Each single-node ``zip`` / ``unzip`` of the kernel's arrays (``arrays``, :func:`kernel_arrays`, with
        their problem ``dtypes``) as the relayout group between its parts and its whole, each part at the type
        its casts left and each laid out as the later digit operations leave it; a request without either
        runs nothing. A zip or unzip of an array another one made is refused (ValueError)."""
        if self.distributed or not any(isinstance(op, Zip | Unzip) for op in self.operations):
            return ()
        state = self.run(arrays)
        ndims = {spec.name: spec.ndim for spec in arrays if not spec.sparse}
        cast_to: dict[str, str] = {}
        out = []
        for op in self.operations:
            if isinstance(op, Cast):
                cast_to[op.array] = np.dtype(numpy_dtype(Precision(op.to.value))).name
            elif isinstance(op, Zip):
                out.append(zip_group(op, state, ndims, dtypes, cast_to))
            elif isinstance(op, Unzip):
                out.append(unzip_group(op, state, ndims, dtypes, cast_to))
        return tuple(out)

    def lowered(self, ndims: Mapping[str, int], ranks: int) -> State:
        """The operations run over the arrays of ``ndims`` on the grid at ``ranks`` ranks."""
        return self.run([ArraySpec(name, ndim) for name, ndim in ndims.items()], ranks)

    def plan(self, ndims: Mapping[str, int], ranks: int) -> dict[str, Any]:
        """The distribution plan the descriptor scatters with at ``ranks`` ranks (``{"grid", "arrays": {name:
        {"axes" | "replicated", "location"}}}``); an array no operation lays out is left out (replicated)."""
        state = self.lowered(ndims, ranks)
        splits = lower(state)
        cyclic = any(
            axis is not None and axis.scheme is not Scheme.BLOCK for axes in splits.values() for axis in axes or ()
        )
        grid = state.grid or ()
        if cyclic and len(set(grid)) > 1:
            raise ValueError(
                f"a cyclic / block-cyclic layout needs an equal-edge grid so the cyclic wrap is symmetric; "
                f"startup.grid {list(self.grid or ())} is {list(grid)} at {ranks} rank(s)"
            )
        arrays: dict[str, dict[str, Any]] = {name: axes_json(axes) for name, axes in splits.items()}
        for name, array in state.arrays.items():
            if array.location is not None:
                arrays.setdefault(name, {"replicated": True})["location"] = array.location.value
        return {"grid": list(grid), "arrays": arrays}

    def unplaced(self, ndims: Mapping[str, int], ranks: int) -> list[str]:
        """The arrays of ``ndims`` the operations neither distribute nor replicate."""
        state = self.lowered(ndims, ranks)
        return sorted(
            name for name, array in state.arrays.items() if not (array.digits.distributed or array.replicated)
        )

    @classmethod
    def from_plan(
        cls, plan: Mapping[str, Any], libraries: Sequence[str] = (), arrays: Sequence[str] = ()
    ) -> "Requests":
        """The request whose :meth:`plan` at the plan's rank count is ``plan``, with the grid following ``P``
        (:func:`grid_request`) and every array of ``arrays`` the plan leaves out replicated: how a kernel's
        default distribution (and a grade recorded before the ``requests`` column) is shown."""
        ops: list[Operation] = []
        for name, entry in plan["arrays"].items():
            if entry.get("replicated"):
                ops.append(Replicate(ArrayName(name)))
            else:
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
                ops += operations_for_axes(name, axes)
            if "location" in entry:
                ops.append(Place(ArrayName(name), Location(entry["location"])))
        ops += [Replicate(ArrayName(name)) for name in arrays if name not in plan["arrays"]]
        grid = grid_request([int(extent) for extent in plan["grid"]])
        return cls(Startup(tuple(libraries), grid), tuple(ops))


def own_arrays(names: Sequence[str], ndims: Mapping[str, int], action: str) -> None:
    unknown = [name for name in names if name not in ndims]
    if unknown:
        raise ValueError(f"{action} of {unknown}: on a single node only the kernel's own dense arrays regroup")


def zip_group(
    op: Zip, state: State, ndims: Mapping[str, int], dtypes: Mapping[str, str], cast_to: Mapping[str, str]
) -> Regroup:
    """A zip's parts (the problem's arrays) and its whole (the kernel's): one element type stacks along ``dim``,
    several make the C struct ``hb_zip_<into>`` with one member per part, named after it."""
    own_arrays(op.arrays, ndims, "zip")
    names = extent_names(ndims[op.arrays[0]])
    types = [canonical(cast_to.get(part, dtypes[part])) for part in op.arrays]
    if len(set(types)) == 1:
        dim = len(names) if op.dim is None else int(op.dim)
        shape = [*names[:dim], str(len(op.arrays)), *names[dim:]]
        struct, whole_dtype = "", np.dtype(types[0])
    else:
        dim, shape, struct = None, names, f"hb_zip_{op.into}"
        whole_dtype = np.dtype([(part, np.dtype(t)) for part, t in zip(op.arrays, types, strict=True)], align=True)
    members = tuple(
        Member(part, IndexMap.identity(names), canonical(dtypes[part]), t, part)
        for part, t in zip(op.arrays, types, strict=True)
    )
    group = Group(op.into, state.layout(op.into, shape), members, dim, struct)
    return Regroup(group, True, whole_dtype, tuple(canonical(dtypes[part]) for part in op.arrays))


def unzip_group(
    op: Unzip, state: State, ndims: Mapping[str, int], dtypes: Mapping[str, str], cast_to: Mapping[str, str]
) -> Regroup:
    """An unzip's whole (the problem's array) and its parts (the kernel's): along ``dim`` one part per index,
    or a complex array into its real and imaginary parts."""
    own_arrays([op.array], ndims, "unzip")
    whole = np.dtype(canonical(dtypes[op.array]))
    target = np.dtype(canonical(cast_to.get(op.array, dtypes[op.array])))
    if op.dim is not None:
        names = extent_names(ndims[op.array] - 1)
        dim = int(op.dim)
        shape = [*names[:dim], str(len(op.into)), *names[dim:]]
        members = tuple(Member(part, state.layout(part, names), target.name, whole.name) for part in op.into)
        group = Group(op.array, IndexMap.identity(shape), members, dim)
    elif whole.kind == "c":
        names = extent_names(ndims[op.array])
        part, element = np.dtype(f"float{target.itemsize * 4}").name, np.dtype(f"float{whole.itemsize * 4}").name
        members = tuple(
            Member(name, state.layout(name, names), part, element, field)
            for name, field in zip(op.into, ("re", "im"), strict=True)
        )
        struct = "relayout_c128" if whole.itemsize == 16 else "relayout_c64"
        group = Group(op.array, IndexMap.identity(names), members, None, struct)
    else:
        raise ValueError(f"unzip of {op.array!r}: a real array splits along a dim; name it")
    return Regroup(group, False, whole, tuple(member.logical for member in group.parts))


def axes_json(axes: Sequence[AxisSplit | None] | None) -> dict[str, Any]:
    """One array's plan entry: ``{"replicated": true}`` when no dimension is split, else its ``axes``."""
    if axes is None or all(axis is None for axis in axes):
        return {"replicated": True}
    return {
        "axes": [
            {"grid_dim": None}
            if axis is None
            else {"grid_dim": axis.grid_dim, "scheme": axis.scheme.value, "block_size": axis.block_size}
            for axis in axes
        ]
    }


def precision_refusal(precision: Mapping[str, Precision], problem: Mapping[str, str]) -> str | None:
    """Why ``precision`` cannot be honoured, or None. ``problem`` maps every array a precision may be
    requested for (floating, dense, not an index array) to its dtype at the problem precision."""
    for name, wanted in sorted(precision.items()):
        if name not in problem:
            return (
                f"a cast names {name!r}, which is not a floating dense array of this kernel; "
                f"precision may be requested for {sorted(problem)}"
            )
        if np.dtype(numpy_dtype(wanted)) == np.dtype(problem[name]):
            return f"cast of {name!r} to {wanted.value!r}: that is already the array's problem precision; drop it"
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
