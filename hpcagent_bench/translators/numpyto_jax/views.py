"""Writes through a numpy view, re-addressed to the array the view looks into.

``outg = out[:, a:b]`` is a VIEW in numpy: ``outg += x`` and ``outg[i] = y`` change ``out``. jax arrays
are values, so the functional translation of those writes rebinds the local ``outg`` and ``out`` never
sees them. :func:`fold_written_views` substitutes every use of such a view by the subscript it stands
for -- ``out[:, a:b] += x``, ``out[:, a:b][i]`` composed to ``out[i, a:b]`` -- before translation, so
the write is an ordinary ``.at[...]`` update of the base array. A view that is only READ is left
alone: its copy reads the same values, and folding it would only repeat the slicing at every use.
"""

import ast
import copy

from hpcagent_bench.translators.numpyto_jax.errors import EmitError

__all__ = ["compose_index", "fold_written_views"]

STMT_BLOCKS = ("body", "orelse", "finalbody")


def index_entries(index: ast.expr) -> list[ast.expr]:
    return list(index.elts) if isinstance(index, ast.Tuple) else [index]


def index_node(entries: list[ast.expr]) -> ast.expr:
    return entries[0] if len(entries) == 1 else ast.Tuple(elts=entries, ctx=ast.Load())


def is_full(entry: ast.expr) -> bool:
    return isinstance(entry, ast.Slice) and entry.lower is None and entry.upper is None and entry.step is None


def is_negative_literal(node: ast.expr | None) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return True
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value < 0


def is_unit_step(step: ast.expr | None) -> bool:
    return step is None or (isinstance(step, ast.Constant) and step.value == 1)


def plus(a: ast.expr | None, b: ast.expr) -> ast.expr:
    return copy.deepcopy(b) if a is None else ast.BinOp(left=copy.deepcopy(a), op=ast.Add(), right=copy.deepcopy(b))


def compose_index(view: list[ast.expr], outer: list[ast.expr], slice_names: set[str]) -> list[ast.expr]:
    """The base-array index that ``view_array[outer]`` names, where ``view_array = base[view]``.

    Each slice axis of the view takes the next outer entry; an integer entry of the view has removed
    its axis and takes none. A full ``:`` on either side passes the other side through; two
    non-trivial entries compose only for a unit-step view slice and a non-negative outer entry (a
    negative one counts from the VIEW's end, which needs its length). Anything else raises
    :class:`EmitError` -- an emit that cannot place the write must not drop it."""
    out: list[ast.expr] = []
    rest = list(outer)
    for d in view:
        if not isinstance(d, ast.Slice):
            out.append(copy.deepcopy(d))
            continue
        if not rest:
            out.append(copy.deepcopy(d))
            continue
        e = rest.pop(0)
        if is_full(d):
            out.append(copy.deepcopy(e))
        elif is_full(e):
            out.append(copy.deepcopy(d))
        elif not is_unit_step(d.step) or (isinstance(e, ast.Name) and e.id in slice_names):
            raise EmitError(f"cannot compose a write through view entry {ast.unparse(d)!r} with {ast.unparse(e)!r}")
        elif isinstance(e, ast.Slice):
            if not is_unit_step(e.step) or is_negative_literal(e.lower) or is_negative_literal(e.upper):
                raise EmitError(f"cannot compose a write through view entry {ast.unparse(d)!r} with {ast.unparse(e)!r}")
            lower = plus(d.lower, e.lower) if e.lower is not None else copy.deepcopy(d.lower)
            upper = plus(d.lower, e.upper) if e.upper is not None else copy.deepcopy(d.upper)
            out.append(ast.Slice(lower=lower, upper=upper, step=None))
        else:
            if is_negative_literal(e):
                raise EmitError(f"a negative index {ast.unparse(e)!r} into view entry {ast.unparse(d)!r}")
            out.append(plus(d.lower, e))
    return out + [copy.deepcopy(e) for e in rest]


class ViewDef:
    """One ``name = base[index]`` binding of a basic-indexed view."""

    __slots__ = ("base", "entries", "name")

    def __init__(self, name: str, base: str, entries: list[ast.expr]) -> None:
        self.name = name
        self.base = base
        self.entries = entries


def view_def(stmt: ast.stmt) -> ViewDef | None:
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.Subscript)
        and isinstance(stmt.value.value, ast.Name)
        and any(isinstance(e, ast.Slice) for e in index_entries(stmt.value.slice))
    ):
        return None
    return ViewDef(stmt.targets[0].id, stmt.value.value.id, index_entries(stmt.value.slice))


def is_basic(entries: list[ast.expr], loop_vars: set[str]) -> bool:
    """Every entry is a slice or an integer -- what makes numpy return a view, not a copy. A name
    counts as an integer when it is a ``range`` loop variable; any other name could be an index
    array, whose gather is a copy."""

    def integer(node: ast.expr) -> bool:
        if isinstance(node, ast.Constant):
            return type(node.value) is int
        if isinstance(node, ast.Name):
            return node.id in loop_vars
        if isinstance(node, ast.UnaryOp):
            return integer(node.operand)
        if isinstance(node, ast.BinOp):
            return integer(node.left) and integer(node.right)
        return False

    return all(isinstance(e, ast.Slice) or integer(e) for e in entries)


def written_through(fn: ast.FunctionDef, views: dict[str, ViewDef]) -> set[str]:
    """View names something writes through -- directly, or through a view of the view."""
    written: set[str] = set()
    for node in ast.walk(fn):
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AugAssign) else []
        )
        for t in targets:
            base = t
            while isinstance(base, ast.Subscript):
                base = base.value
            if isinstance(base, ast.Name) and (isinstance(t, ast.Subscript) or isinstance(node, ast.AugAssign)):
                written.add(base.id)
    grown = written & set(views)
    frontier = set(grown)
    while frontier:
        under = views[frontier.pop()].base
        if under in views and under not in grown:
            grown.add(under)
            frontier.add(under)
    return grown


def rebinds(stmt: ast.stmt, name: str) -> bool:
    """True when ``stmt`` binds ``name`` as a whole. An augmented assignment to an array name writes
    in place, so it does not count."""
    in_place = {id(n.target) for n in ast.walk(stmt) if isinstance(n, ast.AugAssign)}
    return any(
        isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == name and id(n) not in in_place
        for n in ast.walk(stmt)
    )


def scalar_rebinds(stmt: ast.stmt, names: set[str]) -> bool:
    """True when ``stmt`` changes the value of any of ``names`` -- an index the view was taken at."""
    return any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in names for n in ast.walk(stmt))


class Substitute(ast.NodeTransformer):
    """Every use of the view ``v`` re-addressed to ``base[...]``."""

    __slots__ = ("slice_names", "view")

    def __init__(self, view: ViewDef, slice_names: set[str]) -> None:
        self.view = view
        self.slice_names = slice_names

    def whole(self, ctx: ast.expr_context) -> ast.Subscript:
        return ast.Subscript(
            value=ast.Name(id=self.view.base, ctx=ast.Load()),
            slice=index_node(copy.deepcopy(self.view.entries)),
            ctx=ctx,
        )

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        if isinstance(node.value, ast.Name) and node.value.id == self.view.name:
            node.slice = self.visit(node.slice)
            entries = compose_index(self.view.entries, index_entries(node.slice), self.slice_names)
            return ast.Subscript(
                value=ast.Name(id=self.view.base, ctx=ast.Load()), slice=index_node(entries), ctx=node.ctx
            )
        self.generic_visit(node)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        node.value = self.visit(node.value)
        if isinstance(node.target, ast.Name) and node.target.id == self.view.name:
            node.target = self.whole(ast.Store())
        else:
            node.target = self.visit(node.target)
        return node

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id == self.view.name and isinstance(node.ctx, ast.Load):
            return self.whole(ast.Load())
        return node


def fold_written_views(fn: ast.FunctionDef) -> None:
    """Re-address every write through a basic-indexed view in ``fn`` to its base array, in place.

    A view's uses are substituted from its binding to the next statement of the same block that
    rebinds it, and the binding is dropped. Raises :class:`EmitError` when a written view cannot be
    folded faithfully: its index is not provably basic, an index it was taken at changes while it
    is live, or a use of it survives outside the block it was bound in."""
    loop_vars = {
        n.target.id
        for n in ast.walk(fn)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and isinstance(n.iter, ast.Call)
        and isinstance(n.iter.func, ast.Name)
        and n.iter.func.id == "range"
    }
    slice_names = {
        n.targets[0].id
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "slice"
    }
    views: dict[str, ViewDef] = {}
    plain: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.stmt):
            vd = view_def(node)
            if vd is not None:
                views.setdefault(vd.name, vd)
            elif isinstance(node, ast.Assign):
                plain |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    written = written_through(fn, views)
    if not written:
        return
    for name in written:
        for node in ast.walk(fn):
            vd = view_def(node) if isinstance(node, ast.stmt) else None
            if vd is not None and vd.name == name and not is_basic(vd.entries, loop_vars):
                raise EmitError(f"write through {name!r}, bound to a subscript that may be a copy")
    fold_block(fn.body, written, slice_names)
    for name in written - plain:
        if any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(fn)):
            raise EmitError(f"a use of the written view {name!r} is outside the block that binds it")
    ast.fix_missing_locations(fn)


def fold_block(block: list[ast.stmt], written: set[str], slice_names: set[str]) -> None:
    i = 0
    while i < len(block):
        vd = view_def(block[i])
        if vd is None or vd.name not in written:
            for field, inner in ast.iter_fields(block[i]):
                if field in STMT_BLOCKS and isinstance(inner, list):
                    fold_block(inner, written, slice_names)
            if isinstance(stmt := block[i], ast.Try | ast.TryStar):
                for handler in stmt.handlers:
                    fold_block(handler.body, written, slice_names)
            i += 1
            continue
        deps = {n.id for e in vd.entries for n in ast.walk(e) if isinstance(n, ast.Name)}
        end = next((k for k in range(i + 1, len(block)) if rebinds(block[k], vd.name)), len(block))
        live = block[i + 1 : end]
        if any(rebinds(stmt, vd.base) for stmt in live):
            raise EmitError(f"the base of the written view {vd.name!r} is rebound while the view is live")
        moved = next((k for k, stmt in enumerate(live) if scalar_rebinds(stmt, deps)), None)
        if moved is not None and any(uses(stmt, vd.name) for stmt in live[moved:]):
            raise EmitError(f"an index of the written view {vd.name!r} changes while it is live")
        sub = Substitute(vd, slice_names)
        block[i + 1 : end] = [sub.visit(stmt) for stmt in live]
        if end < len(block):
            rebind = block[end]
            if not (isinstance(rebind, ast.Assign) and all(isinstance(t, ast.Name) for t in rebind.targets)):
                raise EmitError(f"the written view {vd.name!r} is rebound by a {type(rebind).__name__}")
            rebind.value = sub.visit(rebind.value)
        del block[i]


def uses(stmt: ast.stmt, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(stmt))
