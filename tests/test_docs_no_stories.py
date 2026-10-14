# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Docstrings, docs pages and config.yaml comments state the current contract: no history, no job ids.

Scanned text: every module, class and function docstring under ``hpcagent_bench/`` and ``agent/``, every
``docs/**/*.md`` page, and every ``#`` comment of ``hpcagent_bench/config.yaml``. A line matching a
:data:`STORY_PATTERNS` entry fails unless an :data:`ALLOWED` entry names its file, pattern and a substring of
the line.
"""

import ast
import functools
import pathlib
import re
from collections.abc import Iterator
from typing import NamedTuple

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
#: Roots whose ``.py`` docstrings are scanned, relative to :data:`REPO`.
DOCSTRING_ROOTS = ("hpcagent_bench", "agent")
#: Root whose ``.md`` pages are scanned whole, relative to :data:`REPO`.
DOCS_ROOT = "docs"
#: File whose ``#`` comments are scanned, relative to :data:`REPO`.
CONFIG_YAML = "hpcagent_bench/config.yaml"

#: Pattern name -> case-insensitive regex of wording that narrates history or cites a run as evidence.
STORY_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(regex, re.IGNORECASE)
    for name, regex in {
        "used to": r"\bused to\b",
        "previously": r"\bpreviously\b",
        "we found": r"\bwe found\b",
        "since PR": r"\bsince PR\b",
        "historically": r"\bhistorically\b",
        "job id": r"\bjobs? 6\d{5}\b",
        "node name": r"\bnid\d+\b",
    }.items()
}
#: Union of :data:`STORY_PATTERNS`; a line that does not match it is not tested per pattern.
ANY_STORY = re.compile("|".join(f"(?:{p.pattern})" for p in STORY_PATTERNS.values()), re.IGNORECASE)


class Match(NamedTuple):
    """One line of scanned text that matches one :data:`STORY_PATTERNS` entry."""

    path: str  # repo-relative, "/"-separated
    line: int  # 1-based line in the file
    pattern: str  # key of STORY_PATTERNS
    text: str  # the matching line, stripped


class Allowed(NamedTuple):
    """A match that is not a story: ``path`` and ``pattern`` equal, ``fragment`` a substring of the line."""

    path: str
    pattern: str
    fragment: str
    reason: str


#: Matches that are passive "is used to <verb>" or algorithmic "previously computed", not history.
ALLOWED: tuple[Allowed, ...] = (
    Allowed("hpcagent_bench/translators/numpyto_pythran/emit.py", "used to", "(used to build", "passive voice"),
    Allowed(
        "hpcagent_bench/translators/numpyto_common/lowering/signature.py", "used to", "Used to keep", "passive voice"
    ),
    Allowed("hpcagent_bench/translators/numpyto_common/lib_nodes/helpers.py", "used to", "-- used to", "passive voice"),
    Allowed(
        "hpcagent_bench/translators/numpyto_common/frontend/shape_arith.py",
        "used to",
        "used to harvest",
        "passive voice",
    ),
    Allowed("hpcagent_bench/translators/numpyto_common/frontend/shapes.py", "used to", "Used to size", "passive voice"),
    Allowed(
        "hpcagent_bench/translators/numpyto_common/lowering/whole_array.py", "used to", "Used to type", "passive voice"
    ),
    Allowed(
        "hpcagent_bench/translators/numpyto_common/lowering/whole_array.py",
        "used to",
        "used to recognise",
        "passive voice",
    ),
    Allowed(
        "hpcagent_bench/translators/numpyto_common/lowering/masks.py",
        "previously",
        "previously-computed",
        "algorithm order",
    ),
    Allowed("hpcagent_bench/translators/numpyto_c/emit.py", "used to", "used to emit", "passive voice"),
    Allowed("hpcagent_bench/harness/mpi_descriptor.py", "used to", "used to scatter", "passive voice"),
    Allowed(
        "hpcagent_bench/benchmarks/scientific_computing/structured_grids/cp2k_grid_integrate/cp2k_grid_integrate.py",
        "used to",
        "override used to construct",
        "passive voice",
    ),
    Allowed(
        "hpcagent_bench/benchmarks/scientific_computing/structured_grids/hotspot_rodinia/hotspot_rodinia_numpy.py",
        "previously",
        "reuses the previously",
        "algorithm order",
    ),
    Allowed("hpcagent_bench/cluster/make_problems.py", "used to", "used to tell", "passive voice"),
    Allowed("hpcagent_bench/cpf_bridge.py", "used to", "USED to appear", "passive voice"),
    Allowed("agent/hpcagent_agent/tools/http_json.py", "used to", "never used to pick", "passive voice"),
)


#: Statement fields that hold nested statements (``handlers`` and ``cases`` hold nodes with a ``body``).
STATEMENT_FIELDS = ("body", "orelse", "finalbody", "handlers", "cases")


def documented_nodes(statements: list[ast.AST]) -> Iterator[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every class and function definition in ``statements`` and their nested statements; expressions are
    not visited, since a docstring is never inside one."""
    for statement in statements:
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            yield statement
        for field in STATEMENT_FIELDS:
            nested = getattr(statement, field, None)
            if isinstance(nested, list):
                yield from documented_nodes(nested)


def docstring_lines(path: pathlib.Path) -> Iterator[tuple[int, str]]:
    """``(line, text)`` of every line of every module, class and function docstring in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in (tree, *documented_nodes(tree.body)):
        body = node.body
        if not body or not isinstance(body[0], ast.Expr):
            continue
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for offset, text in enumerate(value.value.splitlines()):
                yield value.lineno + offset, text


def comment_lines(path: pathlib.Path) -> Iterator[tuple[int, str]]:
    """``(line, text)`` of the ``#`` comment part of every line of the YAML file ``path``."""
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "#" in text:
            yield number, text[text.index("#") :]


def all_lines(path: pathlib.Path) -> Iterator[tuple[int, str]]:
    """``(line, text)`` of every line of ``path``."""
    yield from enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)


def scanned(repo: pathlib.Path) -> Iterator[tuple[str, int, str]]:
    """``(repo-relative path, line, text)`` of every scanned line under ``repo``."""
    sources: list[tuple[pathlib.Path, Iterator[tuple[int, str]]]] = []
    for root in DOCSTRING_ROOTS:
        sources += [(path, docstring_lines(path)) for path in sorted((repo / root).rglob("*.py"))]
    sources += [(path, all_lines(path)) for path in sorted((repo / DOCS_ROOT).rglob("*.md"))]
    config_yaml = repo / CONFIG_YAML
    sources.append((config_yaml, comment_lines(config_yaml)))
    for path, lines in sources:
        relative = path.relative_to(repo).as_posix()
        for number, text in lines:
            yield relative, number, text


@functools.cache
def story_matches(repo: pathlib.Path) -> tuple[Match, ...]:
    """Every scanned line under ``repo`` matching a :data:`STORY_PATTERNS` entry, once per pattern."""
    return tuple(
        Match(path, number, name, text.strip())
        for path, number, text in scanned(repo)
        if ANY_STORY.search(text)
        for name, pattern in STORY_PATTERNS.items()
        if pattern.search(text)
    )


def allowed_by(match: Match) -> Allowed | None:
    """The :data:`ALLOWED` entry covering ``match``, or None."""
    for entry in ALLOWED:
        if entry.path == match.path and entry.pattern == match.pattern and entry.fragment in match.text:
            return entry
    return None


@pytest.mark.smoke
def test_no_story_wording_outside_the_allowlist() -> None:
    """A docstring, docs page or config.yaml comment states no history, run anecdote, job id or node name."""
    offending = [match for match in story_matches(REPO) if allowed_by(match) is None]
    assert not offending, "story wording; rewrite as the current contract or add an ALLOWED entry:\n" + "\n".join(
        f"  {m.path}:{m.line} [{m.pattern}] {m.text}" for m in offending
    )


def test_every_allowlist_entry_still_matches() -> None:
    """An ALLOWED entry whose line was rewritten or deleted is removed, so the allowlist stays exact."""
    used = {allowed_by(match) for match in story_matches(REPO)}
    stale = [entry for entry in ALLOWED if entry not in used]
    assert not stale, f"ALLOWED entries matching nothing: {stale}"


@pytest.mark.parametrize(
    ("text", "pattern"),
    [
        ("The judge used to time c-autopar.", "used to"),
        ("Previously the cap was 10 GB.", "previously"),
        ("We found that RCCL hangs.", "we found"),
        ("Fixed since PR #42.", "since PR"),
        ("Historically the default was host.", "historically"),
        ("It failed in job 671243.", "job id"),
        ("Jobs 675850 and 675956 died.", "job id"),
        ("The server ran on nid002968.", "node name"),
    ],
)
def test_each_story_pattern_fires(text: str, pattern: str) -> None:
    assert STORY_PATTERNS[pattern].search(text)


@pytest.mark.parametrize(
    "text",
    ["The refused to bind case is a build error.", "job <jobid> on <node>", "A 600 s timeout.", "jobs run in order"],
)
def test_contract_wording_does_not_fire(text: str) -> None:
    assert not any(pattern.search(text) for pattern in STORY_PATTERNS.values())


def test_scan_reads_docstrings_docs_and_config_comments(tmp_path: pathlib.Path) -> None:
    """A story in a docstring, a docs page and a config.yaml comment is found; code and YAML values are not."""
    package = tmp_path / "hpcagent_bench"
    package.mkdir()
    (tmp_path / "agent").mkdir()
    (package / "mod.py").write_text('"""Module."""\n\n\ndef f():\n    """It previously raised."""\n    used_to = 1\n')
    (package / "config.yaml").write_text("key: previously  # job 671243\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "page.md").write_text("# Page\n\nRan on nid001234.\n")
    found = {(m.path, m.line, m.pattern) for m in story_matches(tmp_path)}
    assert found == {
        ("hpcagent_bench/mod.py", 5, "previously"),
        ("hpcagent_bench/config.yaml", 1, "job id"),
        ("docs/page.md", 3, "node name"),
    }
