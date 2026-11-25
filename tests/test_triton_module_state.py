# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""No scientific Triton baseline keeps state between calls, read with ``ast``.

Our GPU runs use the child clock (the student cluster refuses the parent clock), which drops v1.0.0's protection
against a kernel that saves work across timed calls. So every ``scientific_computing/**/*_triton.py`` must do all of
a call's work inside the call. This test enforces it statically, with NO exceptions:

- Module level holds only imports, functions, the docstring and immutable constants (numbers, strings, tuples,
  dtype attributes, ``tl.constexpr(...)``, ``min``/``max``/... of those). No dict, list or set, no tensor, no lambda,
  no class, no call whose result could be mutable.
- Functions use no ``global``/``nonlocal``, no decorator other than ``triton.jit``/``triton.autotune``/
  ``triton.heuristics`` (so no ``functools.cache``/``lru_cache``), no mutable default argument, no ``setattr``/
  ``globals()``/``vars()``/``sys.modules``, and never assign to, delete from or call a mutating method on a
  module-level name (a function attribute, an imported module's attribute such as a torch flag, ...).

Triton's own autotune and compile caches are not ours and are allowed. ``probe.py`` in the pipeline adds a runtime
check that the module's globals are unchanged after repeated calls.
"""

import ast
import pathlib

from hpcagent_bench import paths

SCOPE = paths.BENCHMARKS / "scientific_computing"
TRITON_DECORATORS = ("triton.jit", "triton.autotune", "triton.heuristics")
#: Calls whose result is an immutable scalar or constexpr (their arguments are not stored).
SAFE_CALLS = frozenset(
    {
        "min",
        "max",
        "abs",
        "int",
        "float",
        "round",
        "len",
        "sum",
        "bool",
        "tl.constexpr",
        "triton.cdiv",
        "math.sqrt",
        "math.log2",
        "math.ceil",
        "math.floor",
    }
)
MUTATING_METHODS = frozenset(
    {
        "append",
        "extend",
        "insert",
        "pop",
        "popitem",
        "clear",
        "update",
        "setdefault",
        "add",
        "discard",
        "remove",
        "copy_",
        "fill_",
        "zero_",
        "__setitem__",
        "__setattr__",
    }
)
REFLECTION = frozenset({"setattr", "delattr", "globals", "vars"})


def name_of(node: ast.expr) -> str:
    """``a.b.c`` for a Name/Attribute chain, else ''."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else ""


def root_of(node: ast.expr) -> str:
    """The Name at the bottom of an Attribute/Subscript chain (``x`` for ``x.a[1].b``), else ''."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def immutable_value(node: ast.expr) -> bool:
    """Whether a module-level value is an immutable constant expression."""
    if isinstance(node, (ast.Constant, ast.Name, ast.Attribute)):
        return True
    if isinstance(node, ast.Tuple):
        return all(immutable_value(e) for e in node.elts)
    if isinstance(node, ast.BinOp):
        return immutable_value(node.left) and immutable_value(node.right)
    if isinstance(node, ast.UnaryOp):
        return immutable_value(node.operand)
    if isinstance(node, ast.BoolOp):
        return all(immutable_value(v) for v in node.values)
    if isinstance(node, ast.Compare):
        return immutable_value(node.left) and all(immutable_value(c) for c in node.comparators)
    if isinstance(node, ast.IfExp):
        return all(immutable_value(x) for x in (node.test, node.body, node.orelse))
    if isinstance(node, ast.Call):
        return name_of(node.func) in SAFE_CALLS
    return False


def module_level(tree: ast.Module) -> tuple[list[str], set[str]]:
    """Violations at module level, and the names the module binds there."""
    found: list[str] = []
    names: set[str] = set()
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            pass  # the docstring
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not all(isinstance(t, ast.Name) for t in targets):
                found.append(f"line {node.lineno}: module-level assignment to something other than a plain name")
            names.update(t.id for t in targets if isinstance(t, ast.Name))
            if node.value is None or not immutable_value(node.value):
                found.append(
                    f"line {node.lineno}: module-level value is not an immutable constant: {ast.unparse(node)[:80]}"
                )
        else:
            found.append(
                f"line {node.lineno}: module-level {type(node).__name__} (only imports, functions, the "
                "docstring and constants are allowed)"
            )
    return found, names


def local_names(fn: ast.FunctionDef) -> set[str]:
    """Parameters and names the function itself binds."""
    a = fn.args
    params = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    params |= {p.arg for p in (a.vararg, a.kwarg) if p}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            params.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.Lambda)) and node is not fn:
            if isinstance(node, ast.FunctionDef):
                params.add(node.name)
            params |= {p.arg for p in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
        elif isinstance(node, ast.comprehension):
            params |= {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
    return params


def function_level(tree: ast.Module, module_names: set[str]) -> list[str]:
    found: list[str] = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for d in fn.decorator_list:
            dname = name_of(d.func if isinstance(d, ast.Call) else d)
            if dname not in TRITON_DECORATORS:
                found.append(f"line {fn.lineno}: decorator @{ast.unparse(d)[:50]} on {fn.name}")
        defaults = [*fn.args.defaults, *(d for d in fn.args.kw_defaults if d is not None)]
        found += [f"line {d.lineno}: mutable default argument in {fn.name}" for d in defaults if not immutable_value(d)]
        shared = module_names - local_names(fn)
        for node in ast.walk(fn):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                found.append(f"line {node.lineno}: {type(node).__name__.lower()} {', '.join(node.names)}")
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            elif isinstance(node, ast.Delete):
                targets = node.targets
            for t in targets:
                if isinstance(t, (ast.Attribute, ast.Subscript)) and root_of(t) in shared:
                    found.append(f"line {node.lineno}: writes module-level {root_of(t)!r}: {ast.unparse(t)[:60]}")
            if isinstance(node, ast.Call):
                called = name_of(node.func)
                if called in REFLECTION or called.startswith("sys.modules"):
                    found.append(f"line {node.lineno}: {called}()")
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in MUTATING_METHODS
                    and root_of(node.func.value) in shared
                ):
                    found.append(
                        f"line {node.lineno}: mutates module-level {root_of(node.func.value)!r}: "
                        f"{ast.unparse(node)[:60]}"
                    )
    return found


def violations(source: str) -> list[str]:
    tree = ast.parse(source)
    found, names = module_level(tree)
    return found + function_level(tree, names)


def triton_files() -> list[pathlib.Path]:
    return sorted(SCOPE.rglob("*_triton.py"))


def test_no_scientific_triton_baseline_keeps_state_between_calls() -> None:
    report = [
        f"{path.relative_to(paths.ROOT)}: {problem}"
        for path in triton_files()
        for problem in violations(path.read_text(encoding="utf-8"))
    ]
    assert not report, "state kept between calls (see the module docstring):\n" + "\n".join(report)


def test_the_scope_is_not_empty() -> None:
    assert len(triton_files()) >= 14


def test_each_kind_of_state_is_caught() -> None:
    for source, expected in (
        ("CACHE = {}\n", "not an immutable constant"),
        ("C: dict[int, int] = {}\n", "not an immutable constant"),
        ("BUF = []\n", "not an immutable constant"),
        ("import torch\nT = torch.zeros(3)\n", "not an immutable constant"),
        ("def f():\n    return []\nX = f()\n", "not an immutable constant"),
        ("F = lambda x: x\n", "not an immutable constant"),
        ("class K:\n    pass\n", "module-level ClassDef"),
        ("import torch\ntorch.backends.cuda.matmul.allow_tf32 = True\n", "other than a plain name"),
        ("for i in range(3):\n    pass\n", "module-level For"),
        ("import functools\n@functools.cache\ndef f(n):\n    return n\n", "decorator @functools.cache"),
        ("from functools import lru_cache\n@lru_cache(None)\ndef f(n):\n    return n\n", "decorator @lru_cache"),
        ("X = 0\ndef f():\n    global X\n    X = 1\n", "global X"),
        ("def f(a, cache={}):\n    return cache\n", "mutable default argument"),
        ("def f(a, cache=None):\n    return a\ndef g():\n    f.memo = 1\n", "writes module-level 'f'"),
        ("import torch\ndef g():\n    torch.backends.cudnn.benchmark = True\n", "writes module-level 'torch'"),
        ("X = (1, 2)\ndef g():\n    del X[0]\n", "writes module-level 'X'"),
        ("def g():\n    setattr(g, 'k', 1)\n", "setattr()"),
        ("import sys\ndef g():\n    sys.modules['m'].x = 1\n", "writes module-level 'sys'"),
        ("def g():\n    globals()['X'] = 1\n", "globals()"),
    ):
        found = violations(source)
        assert any(expected in f for f in found), (source, found)


def test_constants_and_triton_decorators_pass() -> None:
    ok = (
        '"""Doc."""\n'
        "import torch\nimport triton\nimport triton.language as tl\n"
        "STEPS = 4\nHALO = 2 * STEPS\nAX = tl.constexpr(0.98)\nDT = torch.float32\nPAIR = (1, -2.5, 'x')\n"
        "MIN_CHUNK = min(c for c in (8, 16))\n"
        "@triton.autotune(configs=[triton.Config({'B': 64})], key=['n'])\n@triton.jit\n"
        "def k(x, n, B: tl.constexpr):\n    pass\n"
        "def kernel(x, out=None, steps=STEPS):\n    cache = {}\n    cache[1] = x\n    buf = []\n    buf.append(x)\n"
        "    out = torch.empty_like(x) if out is None else out\n    out.zero_()\n    return out\n"
    )
    assert violations(ok) == []
