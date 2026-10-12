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
    Format,
    FormatLabel,
    Location,
    Operation,
    Place,
    Replicate,
    Scheme,
    Startup,
    State,
    lower,
    operations_for_axes,
)
from relayout.requests import parse_requests, requests_json

from hpcagent_bench import config
from hpcagent_bench.dtypes import canonical, is_float_dtype
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
    one is the struct ``(re, im)``) and every sparse logical array."""
    sparse = {member for group in binding.packed for member in group.members}
    dtypes = {arg.name: arg.dtype for arg in binding.pointers}
    dense = [
        ArraySpec(name, ndim, members=("re", "im") if np.dtype(canonical(dtypes[name])).kind == "c" else ())
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
