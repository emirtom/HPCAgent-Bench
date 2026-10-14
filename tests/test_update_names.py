# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A database written before the framework naming scheme moves to it once, and only once: the bare DaCe column
ran the parallel pipeline, while the old ``parallel`` flavor was the removed loop2map recipe, so a second pass
that renamed again would relabel real parallel rows as loop2map."""

import contextlib
import pathlib
import sqlite3

import pytest

from hpcagent_bench.support import update_names
from hpcagent_bench.support.collect import canon_db

OLD_PAIRS = [
    ("dace_cpu", None),
    ("dace_cpu", "parallel"),
    ("dace_cpu", "canonicalize"),
    ("dace_gpu", "canonicalize"),
    ("dace_cpu", "autoopt"),
    ("tvm", None),
    ("numba", None),
]


def old_framework_shard(path: pathlib.Path) -> None:
    with contextlib.closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE results (framework TEXT NOT NULL, flavor TEXT)")
        conn.executemany("INSERT INTO results VALUES (?, ?)", OLD_PAIRS)


def canon_row(column: str) -> dict[str, str]:
    return {"run": "r", "column": column, "kernel": "k", "preset": "S", "datatype": "float64", "validated": "True"}


def old_canon_db(path: pathlib.Path) -> None:
    """A canon DB as the sweep wrote it before the scheme: the real table, no names stamp."""
    with contextlib.closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(f"CREATE TABLE canon ({', '.join(f'{n} {t}' for n, t in canon_db.SCHEMA)})")
        conn.executemany(
            "INSERT INTO canon (run, column, kernel, preset, datatype, validated) "
            "VALUES (:run, :column, :kernel, :preset, :datatype, :validated)",
            [canon_row(update_names.flat(*pair)) for pair in OLD_PAIRS],
        )


def rows(path: pathlib.Path, query: str) -> list[tuple[str | None, ...]]:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return [tuple(row) for row in conn.execute(query)]


EXPECTED_PAIRS = [
    ("dace_cpu", "parallel"),
    ("dace_cpu", "loop2map"),
    ("dace_cpu", "canon"),
    ("dace_gpu", "canon"),
    ("dace_cpu", "autoopt"),
    ("tvm_gpu", None),
    ("numba", None),
]


def test_a_framework_shard_moves_to_the_current_names_once(tmp_path: pathlib.Path) -> None:
    shard = tmp_path / "shard.db"
    old_framework_shard(shard)

    assert update_names.update(shard) == 5
    assert update_names.update(shard) == 0

    assert rows(shard, "SELECT framework, flavor FROM results ORDER BY rowid") == EXPECTED_PAIRS


def test_a_canon_db_moves_its_flat_column_names(tmp_path: pathlib.Path) -> None:
    canon = tmp_path / "canon.db"
    old_canon_db(canon)

    update_names.update(canon)

    assert rows(canon, "SELECT column FROM canon ORDER BY rowid") == [
        (update_names.flat(*pair),) for pair in EXPECTED_PAIRS
    ]


def test_a_retired_alloy_optimizer_becomes_its_flavor(tmp_path: pathlib.Path) -> None:
    results = tmp_path / "results.db"
    with contextlib.closing(sqlite3.connect(results)) as conn, conn:
        conn.execute("CREATE TABLE setups (harness TEXT NOT NULL)")
        conn.executemany("INSERT INTO setups VALUES (?)", [("alloy-par",), ("alloy-canon",), ("claude",)])

    update_names.update(results)

    assert rows(results, "SELECT harness FROM setups ORDER BY rowid") == [
        ("dace_cpu_parallel_alloy_profile",),
        ("dace_cpu_canon_alloy_profile",),
        ("claude",),
    ]


def test_a_writer_refuses_a_database_under_the_old_names(tmp_path: pathlib.Path) -> None:
    canon = tmp_path / "canon.db"
    old_canon_db(canon)

    with pytest.raises(update_names.StaleNamesError, match="update_names"):
        canon_db.record(canon, [canon_row("dace_cpu_canon")])

    update_names.update(canon)
    assert canon_db.record(canon, [canon_row("dace_cpu_canon")]) == 1


def test_a_new_database_is_written_under_the_current_names(tmp_path: pathlib.Path) -> None:
    canon = tmp_path / "canon.db"

    canon_db.record(canon, [canon_row("dace_cpu_parallel")])

    assert rows(canon, "PRAGMA application_id") == [(update_names.NAMES_VERSION,)]
    assert update_names.update(canon) == 0
    assert rows(canon, "SELECT column FROM canon") == [("dace_cpu_parallel",)]


def test_reading_a_canon_db_writes_nothing(tmp_path: pathlib.Path) -> None:
    """read() went through the writer's connect(): it created an absent file and failed on a read-only copy."""
    absent = tmp_path / "typo.db"
    assert canon_db.read(absent) == []
    assert not absent.exists()

    canon = tmp_path / "canon.db"
    canon_db.record(canon, [canon_row("dace_cpu_parallel")])
    canon.chmod(0o444)
    tmp_path.chmod(0o555)  # no journal can be created beside it either
    try:
        assert [row["column"] for row in canon_db.read(canon)] == ["dace_cpu_parallel"]
    finally:
        tmp_path.chmod(0o755)
        canon.chmod(0o644)


def test_a_reader_refuses_a_database_under_the_old_names_without_stamping_it(tmp_path: pathlib.Path) -> None:
    canon = tmp_path / "canon.db"
    old_canon_db(canon)

    with pytest.raises(update_names.StaleNamesError, match="update_names"):
        canon_db.read(canon)
    assert rows(canon, "PRAGMA application_id") == [(0,)]
