# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""The envelope's ``requests`` block: how a submission asks the judge to change what reaches its kernel.

``libraries`` (catalog names), ``precision`` (per-array element type), ``layout`` (per-array op
sequence over the array's digit tuple, MPI ``distribute`` included), ``grid`` (the processor grid
the layout distributes over), ``location`` (per-array residency of a distributed run) and ``sparse``
(per-array sparse format). See docs/requests_api.md.

A layout is the mixed-radix digit algebra of dace extended ``dace/libraries/layout/algebra.py``: an
array starts as one digit per dimension, ``block`` splits a dimension's finest digit, ``permute``
reorders the digits and ``distribute`` hands a digit to a grid dimension. The judge lowers a sequence
to the :class:`~hpcagent_bench.harness.mpi_descriptor.ArrayDist` it scatters with; a sequence the
lowering does not cover yet (pad, shuffle, zip, unzip, a non-canonical local order) is refused as a
request fault, never approximated.
"""

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from hpcagent_bench.dtypes import canonical, is_float_dtype
from hpcagent_bench.fuzz import eval_int
from hpcagent_bench.precision import Precision, numpy_dtype, precision_from_datatype
from hpcagent_bench.support.bindings.contract import graded_datatype
from hpcagent_bench.support.helpers.sparse.abi import parse_sparse_config

if TYPE_CHECKING:
    from hpcagent_bench.spec import BenchSpec
    from hpcagent_bench.support.bindings.contract import Binding

__all__ = [
    "GRID_SYMBOL",
    "SCHEMES",
    "Block",
    "Digit",
    "Distribute",
    "LayoutOp",
    "Location",
    "Pad",
    "Permute",
    "Requests",
    "Shuffle",
    "Split",
    "Unblock",
    "Unzip",
    "Zip",
    "canonical_permutation",
    "grid_names",
    "inverse_sequence",
    "lower_layout",
    "ops_for_axes",
    "parse_op",
    "precision_problem_dtypes",
    "precision_refusal",
]

#: The per-axis split schemes a layout lowers to; ``mpi_descriptor.AXIS_SCHEMES`` is this tuple.
SCHEMES = ("block", "block_cyclic", "cyclic")
#: Grid dimension ``g``'s extent in a factor/amount expression: ``P0``, ``P1``, ...
GRID_SYMBOL = "P{}"


class Location(enum.Enum):
    """Where a distributed array's tiles live during the call; the harness scatters on the host."""

    HOST = "host"
    DEVICE = "device"


def grid_names(grid: Sequence[int]) -> dict[str, int]:
    """``{"P0": grid[0], ...}``: the names a factor expression reads."""
    return {GRID_SYMBOL.format(g): int(p) for g, p in enumerate(grid)}


def expr_text(value: object, what: str) -> str:
    """An int or expression string as expression text; anything else is a request fault."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{what} must be an int or an expression over {GRID_SYMBOL.format('<g>')}; got {value!r}")
    return str(value)


def expr_json(text: str) -> int | str:
    """``text`` back as JSON: a plain integer literal as an int."""
    return int(text) if text.lstrip("-").isdigit() else text


def non_negative_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{what} must be a non-negative int; got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Block:
    """Split dimension ``dim``'s finest digit ``(d, s, e)`` into outer ``(d, s*b, ceil(e/b))`` (in
    place) and inner ``(d, s, b)`` (appended)."""

    dim: int
    factor: str

    def inverse(self) -> "Unblock":
        return Unblock(self.dim, self.factor)


@dataclass(frozen=True, slots=True)
class Unblock:
    """Merge dimension ``dim``'s inner digit of extent ``factor`` back into its outer partner."""

    dim: int
    factor: str

    def inverse(self) -> Block:
        return Block(self.dim, self.factor)


@dataclass(frozen=True, slots=True)
class Permute:
    """Reorder the digit tuple: ``new[k] = old[perm[k]]``."""

    perm: tuple[int, ...]

    def inverse(self) -> "Permute":
        inverse = [0] * len(self.perm)
        for position, source in enumerate(self.perm):
            inverse[source] = position
        return Permute(tuple(inverse))


@dataclass(frozen=True, slots=True)
class Pad:
    """Grow dimension ``dim``'s coarsest digit by ``amount`` dead cells (a negative amount unpads)."""

    dim: int
    amount: str

    def inverse(self) -> "Pad":
        return Pad(self.dim, f"-({self.amount})")


@dataclass(frozen=True, slots=True)
class Shuffle:
    """Renumber index ``i[dim]`` by the bijection ``f`` (``f_inv`` its inverse), both over ``i``."""

    dim: int
    f: str
    f_inv: str

    def inverse(self) -> "Shuffle":
        return Shuffle(self.dim, self.f_inv, self.f)


@dataclass(frozen=True, slots=True)
class Unzip:
    """Split a complex array into its ``re`` and ``im`` fields; inverse: zip them back."""

    def inverse(self) -> "Unzip":
        raise ValueError("unzip's inverse is the zip of its re/im fields, a group op (requests.zip)")


@dataclass(frozen=True, slots=True)
class Distribute:
    """Hand digit ``digit`` (a position in the current tuple) to grid dimension ``grid_dim``. A digit of
    extent ``P_g`` becomes the rank coordinate and leaves the local tuple; a whole undivided dimension
    is split into ``P_g`` load-balanced contiguous parts and stays, at its local extent."""

    digit: int
    grid_dim: int


type LayoutOp = Block | Unblock | Permute | Pad | Shuffle | Unzip | Distribute


@dataclass(frozen=True, slots=True)
class Zip:
    """Fuse ``fields`` (arrays of one logical shape) into one array of C structs named ``name``."""

    name: str
    fields: tuple[str, ...]


def inverse_sequence(ops: Sequence[LayoutOp]) -> tuple[LayoutOp, ...]:
    """The sequence that undoes ``ops``: reversed, each op inverted. ``distribute`` is undone by the
    gather, not by an op, so a distributed sequence has no op-level inverse."""
    inverted: list[LayoutOp] = []
    for op in reversed(ops):
        if isinstance(op, Distribute):
            raise ValueError("a distribute is undone by the gather; invert the sequence before it")
        inverted.append(op.inverse())
    return tuple(inverted)


def parse_op(raw: object, where: str) -> LayoutOp:
    """One layout op from its JSON object; a malformed op is a request fault naming ``where``."""
    if not isinstance(raw, dict) or not isinstance(raw.get("op"), str):
        raise ValueError(f"{where} must be an object with an 'op' name; got {raw!r}")
    kind = raw["op"]
    if kind in ("block", "unblock"):
        dim = non_negative_int(raw.get("dim"), f"{where}.dim")
        factor = expr_text(raw.get("factor"), f"{where}.factor")
        return Block(dim, factor) if kind == "block" else Unblock(dim, factor)
    if kind == "permute":
        perm = raw.get("perm")
        if not isinstance(perm, list) or sorted(perm) != list(range(len(perm))):
            raise ValueError(f"{where}.perm must be a permutation of 0..n-1; got {perm!r}")
        return Permute(tuple(int(p) for p in perm))
    if kind == "pad":
        return Pad(non_negative_int(raw.get("dim"), f"{where}.dim"), expr_text(raw.get("amount"), f"{where}.amount"))
    if kind == "shuffle":
        f, f_inv = raw.get("f"), raw.get("f_inv")
        if not isinstance(f, str) or not isinstance(f_inv, str):
            raise ValueError(f"{where} needs both 'f' and its inverse 'f_inv' as expressions over i")
        return Shuffle(non_negative_int(raw.get("dim"), f"{where}.dim"), f, f_inv)
    if kind == "unzip":
        return Unzip()
    if kind == "distribute":
        return Distribute(
            non_negative_int(raw.get("digit"), f"{where}.digit"),
            non_negative_int(raw.get("grid_dim"), f"{where}.grid_dim"),
        )
    raise ValueError(
        f"{where}.op {kind!r} is not a layout op; ops are block, unblock, permute, pad, shuffle, unzip, distribute"
    )


def op_json(op: LayoutOp) -> dict[str, Any]:
    """``op`` as the JSON object :func:`parse_op` reads."""
    if isinstance(op, Block):
        return {"op": "block", "dim": op.dim, "factor": expr_json(op.factor)}
    if isinstance(op, Unblock):
        return {"op": "unblock", "dim": op.dim, "factor": expr_json(op.factor)}
    if isinstance(op, Permute):
        return {"op": "permute", "perm": list(op.perm)}
    if isinstance(op, Pad):
        return {"op": "pad", "dim": op.dim, "amount": expr_json(op.amount)}
    if isinstance(op, Shuffle):
        return {"op": "shuffle", "dim": op.dim, "f": op.f, "f_inv": op.f_inv}
    if isinstance(op, Unzip):
        return {"op": "unzip"}
    return {"op": "distribute", "digit": op.digit, "grid_dim": op.grid_dim}


# ---- digit simulation and lowering ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Digit:
    """One mixed-radix digit: ``(i[dim] // stride) % extent``; ``extent`` None = the dimension's whole
    (symbolic) remaining extent."""

    dim: int
    stride: int
    extent: int | None


def canonical_permutation(digits: Sequence[Digit]) -> tuple[int, ...]:
    """The permutation that orders ``digits`` row-major: by dimension, coarsest stride first."""
    return tuple(sorted(range(len(digits)), key=lambda k: (digits[k].dim, -digits[k].stride)))


class Split(NamedTuple):
    """How one dimension is split over the grid: one of :data:`SCHEMES` along ``grid_dim``."""

    grid_dim: int
    scheme: str
    block_size: int


def finest(digits: Sequence[Digit], dim: int, where: str) -> int:
    positions = [k for k, digit in enumerate(digits) if digit.dim == dim]
    if not positions:
        raise ValueError(f"{where}: dimension {dim} has no digit left to split")
    return min(positions, key=lambda k: digits[k].stride)


def block_digits(digits: list[Digit], dim: int, factor: int, where: str) -> None:
    """Apply ``block(dim, factor)`` to ``digits`` in place."""
    if factor < 1:
        raise ValueError(f"{where}: block factor must be >= 1; got {factor}")
    position = finest(digits, dim, where)
    old = digits[position]
    outer = None if old.extent is None else math.ceil(old.extent / factor)
    digits[position] = Digit(dim, old.stride * factor, outer)
    digits.append(Digit(dim, old.stride, factor))


def unblock_digits(digits: list[Digit], dim: int, factor: int, where: str) -> None:
    """Apply ``unblock(dim, factor)`` to ``digits`` in place: the finest inner digit of extent ``factor``
    merges with the digit of stride ``stride * factor``."""
    inners = sorted(
        (k for k, d in enumerate(digits) if d.dim == dim and d.extent == factor), key=lambda k: digits[k].stride
    )
    for inner in inners:
        partner = next(
            (k for k, d in enumerate(digits) if d.dim == dim and d.stride == digits[inner].stride * factor), None
        )
        if partner is None:
            continue
        outer = digits[partner]
        digits[partner] = Digit(dim, digits[inner].stride, None if outer.extent is None else outer.extent * factor)
        del digits[inner]
        return
    raise ValueError(f"{where}: no blocked digit pair of dimension {dim} with factor {factor} to unblock")


def lower_layout(ops: Sequence[LayoutOp], ndim: int, grid: Sequence[int], where: str) -> list[Split | None]:
    """Lower one array's op sequence over ``grid`` to its per-dimension split (None = not split), or raise the request
    fault that says why it cannot be (yet).

    Lowerable: any blocks, unblocks and permutes, then distributes of either a whole undivided
    dimension (``block``, load-balanced) or a digit of extent ``P_g`` sitting between a coarse digit of
    stride ``s*P_g`` and, when ``s > 1``, a fine digit ``(d, 1, s)`` (``block_cyclic`` of width ``s``;
    ``cyclic`` at ``s == 1``). The rank-local digits must end in row-major order."""
    names = grid_names(grid)
    digits = [Digit(d, 1, None) for d in range(ndim)]
    distributed: dict[int, tuple[Digit, int]] = {}  # dim -> (its distributed digit, grid dim)
    for index, op in enumerate(ops):
        here = f"{where}[{index}]"
        if isinstance(op, Distribute):
            distribute_digit(digits, op, grid, distributed, here)
        else:
            apply_shape_op(digits, op, names, here)
    axes: list[Split | None] = [None] * ndim
    for dim, (digit, grid_dim) in distributed.items():
        axes[dim] = split_scheme(digits, digit, grid[grid_dim], grid_dim, where)
    order = canonical_permutation(digits)
    if list(order) != list(range(len(digits))):
        raise ValueError(
            f"{where}: the rank-local digits are not row-major, which is a local relayout and not implemented "
            f'yet; append {{"op": "permute", "perm": {list(order)}}}'
        )
    return axes


def apply_shape_op(digits: list[Digit], op: LayoutOp, names: Mapping[str, int], where: str) -> None:
    """Apply one non-distributing op to ``digits`` in place; an op the lowering does not cover yet is a
    request fault."""
    ndim = 1 + max((d.dim for d in digits), default=-1)
    if isinstance(op, Block):
        block_digits(digits, check_dim(op.dim, ndim, where), eval_int(op.factor, names), where)
    elif isinstance(op, Unblock):
        unblock_digits(digits, check_dim(op.dim, ndim, where), eval_int(op.factor, names), where)
    elif isinstance(op, Permute):
        if len(op.perm) != len(digits):
            raise ValueError(f"{where}: permute of {len(op.perm)} positions over a {len(digits)}-digit tuple")
        digits[:] = [digits[k] for k in op.perm]
    else:
        raise ValueError(
            f"{where}: layout op {op_json(op)['op']!r} is not implemented yet (only block, unblock, permute and "
            f"distribute lower today; see docs/requests_api.md section 4.5)"
        )


def check_dim(dim: int, ndim: int, where: str) -> int:
    if dim >= ndim:
        raise ValueError(f"{where}: dimension {dim} does not exist (the array has {ndim})")
    return dim


def distribute_digit(
    digits: list[Digit], op: Distribute, grid: Sequence[int], distributed: dict[int, tuple[Digit, int]], where: str
) -> None:
    """Apply one ``distribute`` to ``digits`` in place, recording it under its dimension."""
    if op.grid_dim >= len(grid):
        raise ValueError(f"{where}: grid_dim {op.grid_dim} does not exist (the grid is {list(grid)})")
    if op.digit >= len(digits):
        raise ValueError(f"{where}: digit {op.digit} does not exist (the tuple has {len(digits)})")
    if any(g == op.grid_dim for _, g in distributed.values()):
        raise ValueError(f"{where}: grid_dim {op.grid_dim} is already distributed; each grid dimension drives one axis")
    digit = digits[op.digit]
    if digit.dim in distributed:
        raise ValueError(f"{where}: dimension {digit.dim} is already distributed")
    distributed[digit.dim] = (digit, op.grid_dim)
    if digit.extent is not None:
        del digits[op.digit]  # the rank coordinate; split_scheme checks the extent is P_g


def split_scheme(digits: Sequence[Digit], digit: Digit, parts: int, grid_dim: int, where: str) -> Split:
    """The split of the distributed ``digit`` against the local ``digits`` left."""
    own = sorted((d for d in digits if d.dim == digit.dim), key=lambda d: -d.stride)
    if digit.extent is None:
        if digit.stride != 1 or len(own) != 1:
            raise ValueError(
                f"{where}: distributing the coarse digit of a blocked dimension {digit.dim} is not implemented yet; "
                f"distribute the whole dimension (block) or a digit of extent P{grid_dim} (cyclic / block-cyclic)"
            )
        return Split(grid_dim, "block", 1)
    if digit.extent != parts:
        raise ValueError(
            f"{where}: a distributed digit of a blocked dimension must have extent P{grid_dim} = {parts}; "
            f"dimension {digit.dim}'s has {digit.extent}"
        )
    width = digit.stride
    wanted = [Digit(digit.dim, width * parts, None)] + ([Digit(digit.dim, 1, width)] if width > 1 else [])
    if own != wanted:
        raise ValueError(
            f"{where}: dimension {digit.dim} must be blocked as block(d, b*P{grid_dim}) then block(d, b) for a "
            f"block-cyclic distribution of width b (or block(d, P{grid_dim}) for cyclic); its local digits are {own}"
        )
    return Split(grid_dim, "cyclic" if width == 1 else "block_cyclic", width)


def normal_split(grid_dim: int, scheme: str, block_size: int) -> Split:
    """The split :func:`lower_layout` gives back for ``scheme``: block and cyclic carry width 1, and a
    block-cyclic of width 1 is cyclic."""
    if scheme == "block_cyclic" and block_size == 1:
        scheme = "cyclic"
    return Split(grid_dim, scheme, 1 if scheme in ("block", "cyclic") else block_size)


def ops_for_axes(axes: Sequence[Split | None]) -> tuple[LayoutOp, ...]:
    """The op sequence whose lowering is ``axes`` (None = not split): the inverse of :func:`lower_layout`, used to show a kernel's default layout.

    Digit positions do not depend on the grid's extents, only on stride order, so they are found by
    simulating on a representative grid of 2s; the result is lowered back as a self-check."""
    ops: list[LayoutOp] = []
    for dim, axis in enumerate(axes):
        if axis is not None and axis[1] not in SCHEMES:
            raise ValueError(f"unknown axis scheme {axis[1]!r}; split schemes are {list(SCHEMES)}")
        if axis is None or axis[1] == "block":
            continue
        grid_dim, scheme, width = axis
        symbol = GRID_SYMBOL.format(grid_dim)
        if scheme == "cyclic" or width == 1:
            ops.append(Block(dim, symbol))
        else:
            ops += [Block(dim, f"{width}*{symbol}"), Block(dim, str(width))]
    sample = [2] * (1 + max((axis[0] for axis in axes if axis is not None), default=0))
    names = grid_names(sample)
    digits = [Digit(d, 1, None) for d in range(len(axes))]
    for op in ops:
        apply_shape_op(digits, op, names, "default layout")
    order = canonical_permutation(digits)
    if list(order) != list(range(len(digits))):
        ops.append(Permute(order))
        digits = [digits[k] for k in order]
    targets: list[tuple[int, int]] = []
    for dim, axis in enumerate(axes):
        if axis is None:
            continue
        width = 1 if axis[1] in ("block", "cyclic") else axis[2]
        extent = None if axis[1] == "block" else sample[axis[0]]
        targets += [(k, axis[0]) for k, d in enumerate(digits) if (d.dim, d.stride, d.extent) == (dim, width, extent)]
    # Descending positions, so a removed digit never shifts one distributed after it.
    ops += [Distribute(position, grid_dim) for position, grid_dim in sorted(targets, reverse=True)]
    lowered = lower_layout(ops, len(axes), sample, "default layout")
    expected = [None if a is None else normal_split(*a) for a in axes]
    if lowered != expected:
        raise AssertionError(f"ops_for_axes({list(axes)}) lowers to {lowered}")
    return tuple(ops)


# ---- the requests block ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Requests:
    """One submission's ``requests`` block, structurally validated; checks against the kernel (array
    names, dtypes, the binding's ranks) are the judge's."""

    libraries: tuple[str, ...] = ()
    precision: Mapping[str, Precision] = field(default_factory=dict)
    grid: tuple[int, ...] | None = None
    layout: Mapping[str, tuple[LayoutOp, ...]] = field(default_factory=dict)
    location: Mapping[str, Location] = field(default_factory=dict)
    zip: tuple[Zip, ...] = ()
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
        unknown = set(raw) - {"libraries", "precision", "grid", "layout", "location", "zip", "sparse"}
        if unknown:
            raise ValueError(
                f"requests has unknown field(s) {sorted(unknown)}; fields are libraries, precision, grid, layout, "
                f"location, zip, sparse"
            )
        libraries = raw.get("libraries") or []
        if not isinstance(libraries, list) or not all(isinstance(name, str) for name in libraries):
            raise ValueError("requests.libraries must be a list of catalog names")
        return cls(
            libraries=tuple(libraries),
            precision=parse_precision(raw.get("precision")),
            grid=parse_grid(raw.get("grid")),
            layout=parse_layout(raw.get("layout")),
            location=parse_location(raw.get("location")),
            zip=parse_zip(raw.get("zip")),
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
        if self.zip:
            out["zip"] = [{"name": z.name, "fields": list(z.fields)} for z in self.zip]
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
        if self.zip:
            raise ValueError("requests.zip is not implemented yet (see docs/requests_api.md section 4.5)")
        arrays: dict[str, dict[str, Any]] = {}
        for name, ops in sorted(self.layout.items()):
            if name not in ndims:
                raise ValueError(f"requests.layout names unknown array {name!r}; arrays are {sorted(ndims)}")
            lowered = lower_layout(ops, ndims[name], dims, f"requests.layout[{name!r}]")
            if all(axis is None for axis in lowered):
                entry: dict[str, Any] = {"replicated": True}
            else:
                entry = {"axes": [{"grid_dim": None} if axis is None else axis._asdict() for axis in lowered]}
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
                else Split(int(axis["grid_dim"]), str(axis.get("scheme", "block")), int(axis.get("block_size", 1)))
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


def parse_zip(raw: object) -> tuple[Zip, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("requests.zip must be a list of {name, fields}")
    groups: list[Zip] = []
    for k, group in enumerate(raw):
        fields = group.get("fields") if isinstance(group, dict) else None
        name = group.get("name") if isinstance(group, dict) else None
        if not isinstance(name, str) or not isinstance(fields, list) or len(fields) < 2:
            raise ValueError(f"requests.zip[{k}] must be {{'name': str, 'fields': [two or more arrays]}}")
        groups.append(Zip(name, tuple(str(f) for f in fields)))
    return tuple(groups)


def parse_sparse(raw: object) -> dict[str, Any] | None:
    if raw is None:
        return None
    parse_sparse_config(raw)  # the shape only; the kernel is consulted by the judge
    if not isinstance(raw, dict):
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
