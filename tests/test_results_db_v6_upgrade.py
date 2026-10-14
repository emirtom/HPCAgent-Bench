# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A version-6 results file (per-request columns ``requested_libraries``, ``distribution``,
``layout_request``, ``layout_prep_ns``) opens, upgrades in place and merges into version 7 with every
request folded into ``requests`` and its sparse conversion into ``convert_in_ns``."""

import json
import pathlib
import sqlite3
import tempfile

from hpcagent_bench.harness import results_db

PLAN = {
    "grid": [2, 2],
    "arrays": {
        "A": {
            "axes": [
                {"grid_dim": 0, "scheme": "block", "block_size": 1},
                {"grid_dim": 1, "scheme": "block", "block_size": 1},
            ]
        },
        "x": {"replicated": True},
    },
}

FOLDED = {
    "startup": {"grid": ["root(P, 2)", "root(P, 2)"], "load": ["openblas"]},
    "operations": [
        {"action": "distribute", "array": "A", "dim": 0, "grid_dim": 0, "scheme": "block"},
        {"action": "distribute", "array": "A", "dim": 1, "grid_dim": 1, "scheme": "block"},
        {"action": "replicate", "array": "x"},
        {"action": "format", "array": "A", "to": "csr"},
    ],
}


def write_v6(path: pathlib.Path) -> None:
    """A version-6 file: the current schema with v6's per-request ``grades`` columns in place of v7's,
    holding a grade with every request column set and one whose recorded distribution is no plan (a
    refused call's)."""
    conn = sqlite3.connect(path)
    conn.executescript(results_db.SCHEMA_PATH.read_text(encoding="utf-8"))
    for name, ddl in (
        ("requested_libraries", "TEXT NOT NULL DEFAULT ''"),
        ("distribution", "TEXT NOT NULL DEFAULT ''"),
        ("layout_request", "TEXT NOT NULL DEFAULT ''"),
        ("layout_prep_ns", "INTEGER NOT NULL DEFAULT 0"),
    ):
        conn.execute(f"ALTER TABLE grades ADD COLUMN {name} {ddl}")
    for name in ("requests", "convert_in_ns", "convert_out_ns"):
        conn.execute(f"ALTER TABLE grades DROP COLUMN {name}")
    conn.execute("PRAGMA user_version = 6")
    conn.execute("INSERT INTO setups (setup, language, device, harness) VALUES ('s', 'c', 'cpu', 'claude')")
    conn.execute("INSERT INTO episodes (id, setup, label, rep, slot) VALUES (1, 's', 'L', 1, 1)")
    conn.execute(
        "INSERT INTO grades (id, episode_id, kernel, ts_ms, kind, requested_libraries, distribution, layout_request,"
        " layout_prep_ns) VALUES (1, 1, 'gemm', 1, 'score', ?, ?, ?, 77)",
        (json.dumps(["openblas"]), json.dumps(PLAN), json.dumps({"A": "csr"})),
    )
    conn.execute(
        "INSERT INTO grades (id, episode_id, kernel, ts_ms, kind, distribution) VALUES (2, 1, 'gemm', 2, 'score', ?)",
        (json.dumps({"bogus": 1}),),
    )
    conn.commit()
    conn.close()


def folded_grades(conn: sqlite3.Connection) -> dict[int, tuple[object, int]]:
    rows = conn.execute("SELECT id, requests, convert_in_ns FROM grades ORDER BY id").fetchall()
    return {int(row[0]): (json.loads(row[1]) if row[1] else "", int(row[2])) for row in rows}


def test_opening_a_v6_file_upgrades_it_in_place(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "v6.db"
    write_v6(path)
    conn = results_db.open_db(path)
    try:
        assert results_db.schema_version(conn) == results_db.SCHEMA_VERSION
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(grades)")}
        assert not columns & set(results_db.LEGACY_GRADE_COLUMNS)
        assert folded_grades(conn) == {1: (FOLDED, 77), 2: ("", 0)}
        assert len(conn.execute("SELECT * FROM grades_flat").fetchall()) == 2
    finally:
        conn.close()


def test_merging_a_v6_file_folds_its_requests(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "v6.db"
    write_v6(source)
    dest = results_db.open_db(tmp_path / "merged.db")
    try:
        assert results_db.merge_one(dest, source)["grades"] == 2
        assert folded_grades(dest) == {1: (FOLDED, 77), 2: ("", 0)}
    finally:
        dest.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as first:
        test_opening_a_v6_file_upgrades_it_in_place(pathlib.Path(first))
    with tempfile.TemporaryDirectory() as second:
        test_merging_a_v6_file_folds_its_requests(pathlib.Path(second))
