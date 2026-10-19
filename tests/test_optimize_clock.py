# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The optimize time a column reports counts cached work: a cache hit credits what producing the artifact took.

A canon row's ``optimize_s`` is one number per kernel whether the run produced its canonical SDFG, emitted sources
and parsed SDFG itself or read them from a cache (:mod:`hpcagent_bench.optimize_clock`); a results file written
before the column existed (user_version 13) is upgraded in place when opened for writing, and read as it is.
"""

import contextlib
import pathlib
import sqlite3

from hpcagent_bench import optimize_clock
from hpcagent_bench.harness import results_db
from hpcagent_bench.support.collect import canon_db


def test_a_measurement_adds_the_credited_production_time_to_its_wall_time() -> None:
    with optimize_clock.measuring() as clock:
        optimize_clock.credit(3.0)
    assert clock.credited == 3.0
    assert 3.0 <= clock.seconds < 3.5


def test_a_credit_reaches_every_running_measurement_and_none_after() -> None:
    with optimize_clock.measuring() as outer:
        with optimize_clock.measuring() as inner:
            optimize_clock.credit(1.0)
        optimize_clock.credit(2.0)
    optimize_clock.credit(5.0)
    assert inner.credited == 1.0
    assert outer.credited == 3.0


def test_a_negative_credit_counts_as_nothing() -> None:
    with optimize_clock.measuring() as clock:
        optimize_clock.credit(-1.0)
    assert clock.credited == 0.0


def version_13_canon_file(path: pathlib.Path) -> None:
    """A results file as user_version 13 wrote it: the current schema without ``canon.optimize_s``."""
    with contextlib.closing(results_db.open_db(path)) as conn:
        conn.execute("ALTER TABLE canon DROP COLUMN optimize_s")
        conn.execute("PRAGMA user_version = 13")
        conn.commit()


def test_a_version_13_file_is_upgraded_in_place_when_opened_for_writing(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "canon.db"
    version_13_canon_file(path)
    row = {"run": "r", "column": "numba", "kernel": "gemm", "preset": "S", "datatype": "", "status": "ok"}
    canon_db.record(path, [{**row, "optimize_s": 1.5}])
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert results_db.schema_version(conn) == results_db.SCHEMA_VERSION
    assert canon_db.read(path)[0]["optimize_s"] == 1.5


def test_a_version_13_file_is_read_and_merged_as_it_is(tmp_path: pathlib.Path) -> None:
    old = tmp_path / "old.db"
    version_13_canon_file(old)
    with contextlib.closing(sqlite3.connect(old)) as conn:
        conn.execute(
            'INSERT INTO canon (run, "column", kernel, preset, datatype, status) VALUES (?, ?, ?, ?, ?, ?)',
            ("r", "numba", "gemm", "S", "", "ok"),
        )
        conn.commit()
    assert "optimize_s" not in canon_db.read(old)[0]
    merged = tmp_path / "merged.db"
    assert results_db.merge(merged, [old])["canon"] == 1
    assert canon_db.read(merged)[0]["optimize_s"] is None


if __name__ == "__main__":
    import tempfile

    test_a_measurement_adds_the_credited_production_time_to_its_wall_time()
    test_a_credit_reaches_every_running_measurement_and_none_after()
    test_a_negative_credit_counts_as_nothing()
    test_a_version_13_file_is_upgraded_in_place_when_opened_for_writing(pathlib.Path(tempfile.mkdtemp()))
    test_a_version_13_file_is_read_and_merged_as_it_is(pathlib.Path(tempfile.mkdtemp()))
