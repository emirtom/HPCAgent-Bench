# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""The envelope's ``requests`` block: how a submission asks the judge to change what reaches its kernel.

``libraries`` (catalog names), ``precision`` (per-array element type), ``layout`` (per-array op
sequence over the array's digit tuple, MPI ``distribute`` included), ``grid`` (the processor grid
the layout distributes over), ``location`` (per-array residency of a distributed run), ``sparse``
(per-array sparse format) and the group ops ``zip`` / ``unzip``. See docs/requests_api.md.

A layout is the mixed-radix digit algebra of dace extended ``dace/libraries/layout/algebra.py``: an
array starts as one digit per dimension, ``block`` splits a dimension's finest digit, ``permute``
reorders the digits and ``distribute`` hands a digit to a grid dimension. Dimensions are numpy's, over
the C-contiguous buffer the harness stages; a Fortran kernel sees it with reversed extents, a codegen
step after the ops. The judge lowers a sequence to the
:class:`~hpcagent_bench.harness.mpi_descriptor.ArrayDist` it scatters with; a sequence the lowering does
not cover yet (pad, shuffle, a non-canonical local order) is refused as a request fault, never
approximated.

Every op is one class registered under its JSON name: :func:`layout_op` for the per-array ops (each
with its inverse and its effect on the digit tuple), :func:`group_op` for ``zip`` / ``unzip``, which
turn several arrays into one or one into several and so live outside any array's sequence. A field's
type (:data:`FIELD_CODECS`) is how it is read from and written to JSON, so adding an op is one class.
"""

import ast
import dataclasses
import enum
import math
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, NewType, NoReturn

import numpy as np

from hpcagent_bench.dtypes import canonical, is_float_dtype
from hpcagent_bench.fuzz import eval_int
from hpcagent_bench.precision import Precision, numpy_dtype, precision_from_datatype
from hpcagent_bench.registry import Kind, RegistryError
from hpcagent_bench.support.bindings.contract import graded_datatype
from hpcagent_bench.support.helpers.sparse.abi import parse_sparse_config

if TYPE_CHECKING:
    from hpcagent_bench.spec import BenchSpec
    from hpcagent_bench.support.bindings.contract import Binding

__all__ = [
    "GRID_SYMBOL",
    "GROUP_OPS",
    "LAYOUT_OPS",
    "Block",
    "Digit",
    "DigitTuple",
    "Distribute",
    "GroupOp",
    "LayoutOp",
    "LinearShuffle",
    "Location",
    "Pad",
    "Permute",
    "SCHEMES",
    "Requests",
    "Scheme",
    "Shuffle",
    "Split",
    "Unblock",
    "Unpad",
    "Unzip",
    "Zip",
    "canonical_permutation",
    "gf2_inverse",
    "grid_names",
    "inverse_sequence",
    "lower_layout",
    "ops_for_axes",
    "parse_op",
    "precision_problem_dtypes",
    "precision_refusal",
]

#: Grid dimension ``g``'s extent in a factor/amount expression: ``P0``, ``P1``, ...
GRID_SYMBOL = "P{}"


class Scheme(enum.Enum):
    """The per-axis split a layout lowers to; the value is the descriptor's ``scheme`` spelling."""

    BLOCK = "block"
    BLOCK_CYCLIC = "block_cyclic"
    CYCLIC = "cyclic"


#: The per-axis split scheme names; ``mpi_descriptor.AXIS_SCHEMES`` is this tuple.
SCHEMES = tuple(scheme.value for scheme in Scheme)


class Location(enum.Enum):
    """Where a distributed array's tiles live during the call; the harness scatters on the host."""

    HOST = "host"
    DEVICE = "device"


def grid_names(grid: Sequence[int]) -> dict[str, int]:
    """``{"P0": grid[0], ...}``: the names a factor expression reads."""
    return {GRID_SYMBOL.format(g): int(p) for g, p in enumerate(grid)}


# ---- op fields: one JSON codec per field type -----------------------------------------------------

#: An array dimension.
Dim = NewType("Dim", int)
#: A position in the current digit tuple.
Position = NewType("Position", int)
#: A processor-grid dimension.
GridDim = NewType("GridDim", int)
#: An integer expression over the size symbols and ``P0, P1, ...`` (a factor, an amount).
Expr = NewType("Expr", str)
#: A permutation of ``0..n-1``.
Perm = NewType("Perm", tuple[int, ...])
#: Distinct array dimensions.
Dims = NewType("Dims", tuple[int, ...])
#: One index expression per shuffled dimension, over ``i0, i1, ...`` (the shuffled dimensions' indices
#: in ``dims`` order), the size symbols and ``P0, P1, ...``.
IndexExprs = NewType("IndexExprs", tuple[str, ...])
#: The bit width of each shuffled dimension (its extent is ``2**bits``).
Bits = NewType("Bits", tuple[int, ...])
#: A square 0/1 matrix over GF(2): row ``r`` is output bit ``r`` as the XOR of the input bits it marks.
BitMatrix = NewType("BitMatrix", tuple[tuple[int, ...], ...])
#: An array name.
ArrayName = NewType("ArrayName", str)
#: Two or more array names.
Fields = NewType("Fields", tuple[str, ...])

#: The operators an index expression may use: integer arithmetic, floor and mod, and the bitwise ones
#: XOR swizzles need. Evaluation belongs to the relayouter; the judge only checks the form.
INDEX_BINOPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.FloorDiv,
    ast.Mod,
    ast.BitXor,
    ast.BitAnd,
    ast.BitOr,
    ast.LShift,
    ast.RShift,
)
INDEX_CALLS = ("min", "max")


def non_negative_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative int; got {value!r}")
    return value


def int_list(value: object, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty list of non-negative ints; got {value!r}")
    return tuple(non_negative_int(v, f"{where}[{k}]") for k, v in enumerate(value))


def check_index_expr(text: str, where: str) -> str:
    """``text`` when it is an integer expression of the allowed form, else a request fault."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ValueError(f"{where}: {text!r} is not an expression") from None
    for node in ast.walk(tree.body):
        if isinstance(node, (ast.BinOp, ast.UnaryOp)):
            node = node.op
        allowed = (
            isinstance(node, (ast.Name, ast.Load, ast.USub, *INDEX_BINOPS))
            or (isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool))
            or (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in INDEX_CALLS)
        )
        if not allowed:
            raise ValueError(
                f"{where}: {text!r} uses {type(node).__name__}; an index expression takes ints, names, "
                f"+ - * // % ^ & | << >> and min/max"
            )
    return text


def parse_expr(value: object, where: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{where} must be an int or an expression over {GRID_SYMBOL.format('<g>')}; got {value!r}")
    return check_index_expr(str(value), where)


def parse_perm(value: object, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or sorted(value) != list(range(len(value))):
        raise ValueError(f"{where} must be a permutation of 0..n-1; got {value!r}")
    return tuple(int(p) for p in value)


def parse_dims(value: object, where: str) -> tuple[int, ...]:
    dims = int_list(value, where)
    if len(set(dims)) != len(dims):
        raise ValueError(f"{where} must name distinct dimensions; got {list(dims)}")
    return dims


def parse_index_exprs(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{where} must be a list of index expressions over i0, i1, ...; got {value!r}")
    return tuple(check_index_expr(v, f"{where}[{k}]") for k, v in enumerate(value))


def parse_bit_matrix(value: object, where: str) -> tuple[tuple[int, ...], ...]:
    rows = value if isinstance(value, list) else None
    if not rows or not all(isinstance(r, list) and len(r) == len(rows) and set(r) <= {0, 1} for r in rows):
        raise ValueError(f"{where} must be a square 0/1 matrix (a list of rows); got {value!r}")
    return tuple(tuple(int(b) for b in r) for r in rows)


def parse_name(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be an array name; got {value!r}")
    return value


def parse_fields(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < 2 or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{where} must be a list of two or more arrays; got {value!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"{where} names an array twice: {value!r}")
    return tuple(value)


def expr_json(text: str) -> int | str:
    """``text`` back as JSON: a plain integer literal as an int."""
    return int(text) if text.lstrip("-").isdigit() else text


def plain_json(value: object) -> object:
    """A tuple field as a JSON list (nested for a matrix)."""
    return [plain_json(v) for v in value] if isinstance(value, tuple) else value


class FieldCodec(NamedTuple):
    """How one field type is read from JSON (a request fault naming ``where`` when malformed) and written."""

    parse: Callable[[object, str], Any]
    dump: Callable[[Any], object] = plain_json


#: Field type -> codec. A registered op may only declare fields of these types (checked at import).
FIELD_CODECS: dict[object, FieldCodec] = {
    Dim: FieldCodec(non_negative_int),
    Position: FieldCodec(non_negative_int),
    GridDim: FieldCodec(non_negative_int),
    Expr: FieldCodec(parse_expr, expr_json),
    Perm: FieldCodec(parse_perm),
    Dims: FieldCodec(parse_dims),
    IndexExprs: FieldCodec(parse_index_exprs),
    Bits: FieldCodec(int_list),
    BitMatrix: FieldCodec(parse_bit_matrix),
    ArrayName: FieldCodec(parse_name),
    Fields: FieldCodec(parse_fields),
}


# ---- the two registries -----------------------------------------------------------------------------


class LayoutOp:
    """One op of an array's layout sequence. A subclass is a frozen slotted dataclass registered with
    :func:`layout_op`; it names its JSON ``op``, inverts itself and applies itself to a :class:`DigitTuple`.
    An op without an ``apply`` of its own is accepted and refused at lowering as not implemented yet."""

    __slots__ = ()
    op: ClassVar[str]

    def validate(self, where: str) -> None:
        """Checks across fields, past each field's own codec (a request fault naming ``where``)."""

    def inverse(self) -> "LayoutOp":
        raise NotImplementedError

    def apply(self, digits: "DigitTuple") -> None:
        raise ValueError(
            f"{digits.where}: layout op {self.op!r} is not implemented yet (only {', '.join(lowered_ops())} "
            f"lower today; see docs/requests_api.md section 4.5)"
        )


class GroupOp:
    """One op that turns several arrays into one or one into several (``zip``, ``unzip``). A subclass is a
    frozen slotted dataclass registered with :func:`group_op`; it has no inverse: each direction is its
    own request."""

    __slots__ = ()
    op: ClassVar[str]

    def validate(self, where: str) -> None:
        """Checks across fields, past each field's own codec (a request fault naming ``where``)."""


def refuse_class(key: str, attrs: dict[str, Any]) -> NoReturn:
    raise RegistryError(f"{key!r}: register an op with its decorator ({sorted(attrs)})")


#: JSON ``op`` name -> the per-array layout op class.
LAYOUT_OPS: Kind[type[LayoutOp]] = Kind("layout op", {}, refuse_class)
#: Top-level request key -> the group op class.
GROUP_OPS: Kind[type[GroupOp]] = Kind("group op", {}, refuse_class)


def op_fields(cls: type) -> dict[str, FieldCodec]:
    """``cls``'s dataclass fields with their codecs; a field of an unregistered type is refused."""
    hints = typing.get_type_hints(cls)
    codecs: dict[str, FieldCodec] = {}
    for spec in dataclasses.fields(cls):
        codec = FIELD_CODECS.get(hints[spec.name])
        if codec is None:
            raise RegistryError(f"{cls.__name__}.{spec.name}: field type {hints[spec.name]!r} has no JSON codec")
        codecs[spec.name] = codec
    return codecs


def layout_op[C: type[LayoutOp]](cls: C) -> C:
    """Register a layout op class under its ``op`` name; it must define its own ``inverse``."""
    op_fields(cls)
    if "inverse" not in vars(cls):
        raise RegistryError(f"layout op {cls.op!r} must define its inverse")
    LAYOUT_OPS.add(cls.op, cls, order=None)
    return cls


def group_op[C: type[GroupOp]](cls: C) -> C:
    """Register a group op class under its ``op`` name, which is also its top-level request key."""
    op_fields(cls)
    GROUP_OPS.add(cls.op, cls, order=None)
    return cls


def lowered_ops() -> list[str]:
    """The layout ops the lowering applies today (the others are refused as not implemented yet)."""
    return [key for key in LAYOUT_OPS.keys() if "apply" in vars(LAYOUT_OPS.entries[key])]


def from_json[T](cls: type[T], raw: Mapping[str, object], skip: str, where: str) -> T:
    """An op of ``cls`` from its JSON object (``skip`` = the key naming the op, not a field)."""
    codecs = op_fields(cls)
    unknown = sorted(set(raw) - set(codecs) - {skip})
    if unknown:
        raise ValueError(f"{where} has unknown field(s) {unknown}; its fields are {sorted(codecs)}")
    return cls(**{name: codec.parse(raw.get(name), f"{where}.{name}") for name, codec in codecs.items()})


def to_json(op: LayoutOp | GroupOp) -> dict[str, object]:
    """``op``'s fields as JSON (without the key naming the op)."""
    values = dataclasses.asdict(op)  # type: ignore[call-overload]  # every registered op is a dataclass
    return {name: codec.dump(values[name]) for name, codec in op_fields(type(op)).items()}


def parse_op(raw: object, where: str) -> LayoutOp:
    """One layout op from its JSON object; a malformed op is a request fault naming ``where``."""
    if not isinstance(raw, dict) or not isinstance(raw.get("op"), str):
        raise ValueError(f"{where} must be an object with an 'op' name; got {raw!r}")
    cls = LAYOUT_OPS.entries.get(raw["op"])
    if cls is None:
        raise ValueError(f"{where}.op {raw['op']!r} is not a layout op; ops are {', '.join(LAYOUT_OPS.keys())}")
    op = from_json(cls, raw, "op", where)
    op.validate(where)
    return op


def op_json(op: LayoutOp) -> dict[str, object]:
    """``op`` as the JSON object :func:`parse_op` reads."""
    return {"op": op.op, **to_json(op)}


def inverse_sequence(ops: Sequence[LayoutOp]) -> tuple[LayoutOp, ...]:
    """The sequence that undoes ``ops``: reversed, each op inverted."""
    return tuple(op.inverse() for op in reversed(ops))


# ---- the digit tuple ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Digit:
    """One mixed-radix digit: ``(i[dim] // stride) % extent``; ``extent`` None = the dimension's whole
    (symbolic) remaining extent."""

    dim: int
    stride: int
    extent: int | None


class Split(NamedTuple):
    """How one dimension is split over the grid: ``scheme`` along ``grid_dim``."""

    grid_dim: int
    scheme: Scheme
    block_size: int


def canonical_permutation(digits: Sequence[Digit]) -> tuple[int, ...]:
    """The permutation that orders ``digits`` row-major: by dimension, coarsest stride first."""
    return tuple(sorted(range(len(digits)), key=lambda k: (digits[k].dim, -digits[k].stride)))


class DigitTuple:
    """One array's digit tuple over a grid while its ops apply: the ops' only state."""

    __slots__ = ("digits", "distributed", "grid", "names", "ndim", "where")

    def __init__(self, ndim: int, grid: Sequence[int], where: str) -> None:
        self.ndim = ndim
        self.grid = tuple(grid)
        self.names = grid_names(grid)
        self.digits = [Digit(d, 1, None) for d in range(ndim)]
        #: dim -> (its distributed digit, grid dim)
        self.distributed: dict[int, tuple[Digit, int]] = {}
        #: The op being applied, for refusals.
        self.where = where

    def eval(self, expr: str) -> int:
        return eval_int(expr, self.names)

    def dim(self, dim: int) -> int:
        if dim >= self.ndim:
            raise ValueError(f"{self.where}: dimension {dim} does not exist (the array has {self.ndim})")
        return dim

    def finest(self, dim: int) -> int:
        positions = [k for k, digit in enumerate(self.digits) if digit.dim == dim]
        if not positions:
            raise ValueError(f"{self.where}: dimension {dim} has no digit left to split")
        return min(positions, key=lambda k: self.digits[k].stride)

    def block(self, dim: int, factor: int) -> None:
        """Split ``dim``'s finest digit ``(d, s, e)`` into outer ``(d, s*b, ceil(e/b))`` and inner ``(d, s, b)``."""
        if factor < 1:
            raise ValueError(f"{self.where}: block factor must be >= 1; got {factor}")
        position = self.finest(self.dim(dim))
        old = self.digits[position]
        outer = None if old.extent is None else math.ceil(old.extent / factor)
        self.digits[position] = Digit(dim, old.stride * factor, outer)
        self.digits.append(Digit(dim, old.stride, factor))

    def unblock(self, dim: int, factor: int) -> None:
        """Merge ``dim``'s finest inner digit of extent ``factor`` with its outer partner of stride ``s * factor``."""
        self.dim(dim)
        inners = sorted(
            (k for k, d in enumerate(self.digits) if d.dim == dim and d.extent == factor),
            key=lambda k: self.digits[k].stride,
        )
        for inner in inners:
            stride = self.digits[inner].stride
            partner = next((k for k, d in enumerate(self.digits) if d.dim == dim and d.stride == stride * factor), None)
            if partner is None:
                continue
            outer = self.digits[partner]
            self.digits[partner] = Digit(dim, stride, None if outer.extent is None else outer.extent * factor)
            del self.digits[inner]
            return
        raise ValueError(f"{self.where}: no blocked digit pair of dimension {dim} with factor {factor} to unblock")

    def permute(self, perm: Sequence[int]) -> None:
        if len(perm) != len(self.digits):
            raise ValueError(f"{self.where}: permute of {len(perm)} positions over a {len(self.digits)}-digit tuple")
        self.digits[:] = [self.digits[k] for k in perm]

    def distribute(self, position: int, grid_dim: int) -> None:
        """Hand the digit at ``position`` to ``grid_dim``: a digit of extent ``P_g`` leaves the tuple as the
        rank coordinate; a whole undivided dimension stays, split into load-balanced bands."""
        if grid_dim >= len(self.grid):
            raise ValueError(f"{self.where}: grid_dim {grid_dim} does not exist (the grid is {list(self.grid)})")
        if position >= len(self.digits):
            raise ValueError(f"{self.where}: digit {position} does not exist (the tuple has {len(self.digits)})")
        if any(g == grid_dim for _, g in self.distributed.values()):
            raise ValueError(
                f"{self.where}: grid_dim {grid_dim} is already distributed; each grid dimension drives one axis"
            )
        digit = self.digits[position]
        if digit.dim in self.distributed:
            raise ValueError(f"{self.where}: dimension {digit.dim} is already distributed")
        self.distributed[digit.dim] = (digit, grid_dim)
        if digit.extent is not None:
            del self.digits[position]  # the rank coordinate; split() checks the extent is P_g

    def split(self, digit: Digit, grid_dim: int, where: str) -> Split:
        """The split of the distributed ``digit`` against the local digits left."""
        parts = self.grid[grid_dim]
        own = sorted((d for d in self.digits if d.dim == digit.dim), key=lambda d: -d.stride)
        if digit.extent is None:
            if digit.stride != 1 or len(own) != 1:
                raise ValueError(
                    f"{where}: distributing the coarse digit of a blocked dimension {digit.dim} is not implemented "
                    f"yet; distribute the whole dimension (block) or a digit of extent P{grid_dim} (cyclic / "
                    f"block-cyclic)"
                )
            return Split(grid_dim, Scheme.BLOCK, 1)
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
                f"block-cyclic distribution of width b (or block(d, P{grid_dim}) for cyclic); its local digits are "
                f"{own}"
            )
        return Split(grid_dim, Scheme.CYCLIC if width == 1 else Scheme.BLOCK_CYCLIC, width)

    def lower(self, where: str) -> list[Split | None]:
        """Per dimension its split (None = not split); the rank-local digits must end row-major."""
        axes: list[Split | None] = [None] * self.ndim
        for dim, (digit, grid_dim) in self.distributed.items():
            axes[dim] = self.split(digit, grid_dim, where)
        order = canonical_permutation(self.digits)
        if list(order) != list(range(len(self.digits))):
            raise ValueError(
                f"{where}: the rank-local digits are not row-major, which is a local relayout and not implemented "
                f'yet; append {{"op": "permute", "perm": {list(order)}}}'
            )
        return axes


# ---- the layout ops ----------------------------------------------------------------------------------


@layout_op
@dataclass(frozen=True, slots=True)
class Block(LayoutOp):
    """Split dimension ``dim``'s finest digit ``(d, s, e)`` into outer ``(d, s*b, ceil(e/b))`` (in place)
    and inner ``(d, s, b)`` (appended)."""

    op: ClassVar[str] = "block"
    dim: Dim
    factor: Expr

    def inverse(self) -> "Unblock":
        return Unblock(self.dim, self.factor)

    def apply(self, digits: DigitTuple) -> None:
        digits.block(self.dim, digits.eval(self.factor))


@layout_op
@dataclass(frozen=True, slots=True)
class Unblock(LayoutOp):
    """Merge dimension ``dim``'s inner digit of extent ``factor`` back into its outer partner."""

    op: ClassVar[str] = "unblock"
    dim: Dim
    factor: Expr

    def inverse(self) -> Block:
        return Block(self.dim, self.factor)

    def apply(self, digits: DigitTuple) -> None:
        digits.unblock(self.dim, digits.eval(self.factor))


@layout_op
@dataclass(frozen=True, slots=True)
class Permute(LayoutOp):
    """Reorder the digit tuple: ``new[k] = old[perm[k]]``."""

    op: ClassVar[str] = "permute"
    perm: Perm

    def inverse(self) -> "Permute":
        inverse = [0] * len(self.perm)
        for position, source in enumerate(self.perm):
            inverse[source] = position
        return Permute(Perm(tuple(inverse)))

    def apply(self, digits: DigitTuple) -> None:
        digits.permute(self.perm)


@layout_op
@dataclass(frozen=True, slots=True)
class Pad(LayoutOp):
    """Grow dimension ``dim``'s coarsest digit by ``amount`` dead cells."""

    op: ClassVar[str] = "pad"
    dim: Dim
    amount: Expr

    def inverse(self) -> "Unpad":
        return Unpad(self.dim, self.amount)


@layout_op
@dataclass(frozen=True, slots=True)
class Unpad(LayoutOp):
    """Drop ``amount`` dead cells from dimension ``dim``'s coarsest digit."""

    op: ClassVar[str] = "unpad"
    dim: Dim
    amount: Expr

    def inverse(self) -> Pad:
        return Pad(self.dim, self.amount)


@layout_op
@dataclass(frozen=True, slots=True)
class Shuffle(LayoutOp):
    """Renumber the indices of ``dims`` by the bijection ``f`` (one expression per dimension, over ``i0,
    i1, ...``); the agent supplies its inverse ``f_inv``, checked exhaustively before the build."""

    op: ClassVar[str] = "shuffle"
    dims: Dims
    f: IndexExprs
    f_inv: IndexExprs

    def validate(self, where: str) -> None:
        if not len(self.f) == len(self.f_inv) == len(self.dims):
            raise ValueError(f"{where}: f and f_inv need one expression per shuffled dimension ({len(self.dims)})")

    def inverse(self) -> "Shuffle":
        return Shuffle(self.dims, self.f_inv, self.f)


@layout_op
@dataclass(frozen=True, slots=True)
class LinearShuffle(LayoutOp):
    """Renumber the indices of ``dims`` (extents ``2**bits``) by an invertible linear map over GF(2) on
    their bits (Linear Layouts, arXiv:2505.23819): the input bit vector is each dimension's bits, least
    significant first, in ``dims`` order. XOR swizzles are this form; the inverse is the inverse matrix."""

    op: ClassVar[str] = "linear_shuffle"
    dims: Dims
    bits: Bits
    matrix: BitMatrix

    def validate(self, where: str) -> None:
        if len(self.bits) != len(self.dims):
            raise ValueError(f"{where}: bits needs one width per shuffled dimension ({len(self.dims)})")
        if len(self.matrix) != sum(self.bits):
            raise ValueError(f"{where}: the matrix must be {sum(self.bits)} x {sum(self.bits)} (sum of bits)")
        if gf2_inverse(self.matrix) is None:
            raise ValueError(f"{where}: the matrix is singular over GF(2), so the shuffle is not a bijection")

    def inverse(self) -> "LinearShuffle":
        inverse = gf2_inverse(self.matrix)
        if inverse is None:
            raise ValueError("a singular GF(2) matrix has no inverse")
        return LinearShuffle(self.dims, self.bits, BitMatrix(inverse))


@layout_op
@dataclass(frozen=True, slots=True)
class Distribute(LayoutOp):
    """Hand digit ``digit`` (a position in the current tuple) to grid dimension ``grid_dim``. A digit of
    extent ``P_g`` becomes the rank coordinate and leaves the local tuple; a whole undivided dimension
    is split into ``P_g`` load-balanced contiguous parts and stays, at its local extent."""

    op: ClassVar[str] = "distribute"
    digit: Position
    grid_dim: GridDim

    def inverse(self) -> NoReturn:
        raise ValueError("a distribute is undone by the gather; invert the sequence before it")

    def apply(self, digits: DigitTuple) -> None:
        digits.distribute(self.digit, self.grid_dim)


def gf2_inverse(matrix: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...] | None:
    """The inverse of a square 0/1 matrix over GF(2) by Gauss-Jordan elimination, or None when singular."""
    n = len(matrix)
    rows = [[*row, *(int(k == r) for k in range(n))] for r, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if rows[r][col]), None)
        if pivot is None:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(n):
            if r != col and rows[r][col]:
                rows[r] = [a ^ b for a, b in zip(rows[r], rows[col], strict=True)]
    return tuple(tuple(row[n:]) for row in rows)


# ---- the group ops -----------------------------------------------------------------------------------


@group_op
@dataclass(frozen=True, slots=True)
class Zip(GroupOp):
    """Fuse ``fields`` (arrays of one logical shape) into one array ``name`` of C structs, one member per
    field in order, each at its own (requested) precision."""

    op: ClassVar[str] = "zip"
    name: ArrayName
    fields: Fields


@group_op
@dataclass(frozen=True, slots=True)
class Unzip(GroupOp):
    """Split the struct array ``name`` into one array per member, named ``fields`` in member order. A
    complex array is the struct ``(re, im)``."""

    op: ClassVar[str] = "unzip"
    name: ArrayName
    fields: Fields


# ---- lowering ----------------------------------------------------------------------------------------


def lower_layout(ops: Sequence[LayoutOp], ndim: int, grid: Sequence[int], where: str) -> list[Split | None]:
    """Lower one array's op sequence over ``grid`` to its per-dimension split (None = not split), or raise
    the request fault that says why it cannot be (yet).

    Lowerable: any blocks, unblocks and permutes, then distributes of either a whole undivided
    dimension (``block``, load-balanced) or a digit of extent ``P_g`` sitting between a coarse digit of
    stride ``s*P_g`` and, when ``s > 1``, a fine digit ``(d, 1, s)`` (``block_cyclic`` of width ``s``;
    ``cyclic`` at ``s == 1``). The rank-local digits must end in row-major order."""
    digits = DigitTuple(ndim, grid, where)
    for index, op in enumerate(ops):
        digits.where = f"{where}[{index}]"
        op.apply(digits)
    return digits.lower(where)


def normal_split(grid_dim: int, scheme: Scheme, block_size: int) -> Split:
    """The split :func:`lower_layout` gives back for ``scheme``: block and cyclic carry width 1, and a
    block-cyclic of width 1 is cyclic."""
    if scheme is Scheme.BLOCK_CYCLIC and block_size == 1:
        scheme = Scheme.CYCLIC
    return Split(grid_dim, scheme, block_size if scheme is Scheme.BLOCK_CYCLIC else 1)


def ops_for_axes(axes: Sequence[Split | None]) -> tuple[LayoutOp, ...]:
    """The op sequence whose lowering is ``axes`` (None = not split): the inverse of :func:`lower_layout`,
    used to show a kernel's default layout.

    Digit positions do not depend on the grid's extents, only on stride order, so they are found by
    simulating on a representative grid of 2s; the result is lowered back as a self-check."""
    normal = [None if axis is None else normal_split(*axis) for axis in axes]
    ops: list[LayoutOp] = []
    for dim, axis in enumerate(normal):
        if axis is None or axis.scheme is Scheme.BLOCK:
            continue
        symbol = GRID_SYMBOL.format(axis.grid_dim)
        if axis.scheme is Scheme.CYCLIC:
            ops.append(Block(Dim(dim), Expr(symbol)))
        else:
            ops += [Block(Dim(dim), Expr(f"{axis.block_size}*{symbol}")), Block(Dim(dim), Expr(str(axis.block_size)))]
    sample = [2] * (1 + max((axis.grid_dim for axis in normal if axis is not None), default=0))
    digits = DigitTuple(len(axes), sample, "default layout")
    for op in ops:
        op.apply(digits)
    order = canonical_permutation(digits.digits)
    if list(order) != list(range(len(digits.digits))):
        ops.append(Permute(Perm(order)))
        digits.permute(order)
    targets: list[tuple[int, int]] = []
    for dim, axis in enumerate(normal):
        if axis is None:
            continue
        extent = None if axis.scheme is Scheme.BLOCK else sample[axis.grid_dim]
        wanted = Digit(dim, axis.block_size, extent)
        targets += [(k, axis.grid_dim) for k, d in enumerate(digits.digits) if d == wanted]
    # Descending positions, so a removed digit never shifts one distributed after it.
    ops += [Distribute(Position(position), GridDim(grid_dim)) for position, grid_dim in sorted(targets, reverse=True)]
    lowered = lower_layout(ops, len(axes), sample, "default layout")
    if lowered != normal:
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
