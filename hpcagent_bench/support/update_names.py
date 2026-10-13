# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Move results databases to the current framework names (the naming scheme of :mod:`hpcagent_bench.columns`).

    python -m hpcagent_bench.support.update_names DB [DB ...]

A framework results shard (``results``: ``framework`` + ``flavor``) and the canon DB (``canon``: the flat ``column``)
carry the names scheme they were written under as ``PRAGMA user_version``; :func:`guard` stamps a new one with
:data:`NAMES_VERSION` and refuses to write into an older one, so a database never mixes two schemes. Agent results
(``setups.harness``) only rename the retired Alloy optimizers. Running the script twice changes nothing the second
time. Standard library only, like :mod:`hpcagent_bench.support.collect.canon_db`.
"""

import argparse
import contextlib
import pathlib
import sqlite3
import sys

__all__ = ["FLAVOR_RENAMES", "HARNESS_RENAMES", "NAMES_VERSION", "StaleNamesError", "guard", "main", "update"]

#: The names scheme current writers use; 0 (SQLite's default) is every database written before it.
NAMES_VERSION = 1

#: ``(framework, flavor) -> (framework, flavor)``, applied in order. The removed loop2map recipe was the ``parallel``
#: flavor, so its rows move out of the way first and keep a name of their own; the bare DaCe column then becomes the
#: ``parallel`` flavor it always ran.
FLAVOR_RENAMES: tuple[tuple[tuple[str, str | None], tuple[str, str | None]], ...] = (
    (("dace_cpu", "parallel"), ("dace_cpu", "loop2map")),
    (("dace_gpu", "parallel"), ("dace_gpu", "loop2map")),
    (("dace_cpu", None), ("dace_cpu", "parallel")),
    (("dace_gpu", None), ("dace_gpu", "parallel")),
    (("dace_cpu", "canonicalize"), ("dace_cpu", "canon")),
    (("dace_gpu", "canonicalize"), ("dace_gpu", "canon")),
    (("tvm", None), ("tvm_gpu", None)),
)

#: The retired Alloy optimizers -> the flavors that replaced them.
HARNESS_RENAMES: tuple[tuple[str, str], ...] = (
    ("alloy-par", "dace_cpu_parallel_alloy_profile"),
    ("alloy-canon", "dace_cpu_canon_alloy_profile"),
)


class StaleNamesError(RuntimeError):
    """A database written under an older names scheme."""


def flat(framework: str, flavor: str | None) -> str:
    """The CLI name of a ``(framework, flavor)`` pair, the canon DB's ``column``."""
    return framework if flavor is None else f"{framework}_{flavor}"


def has_table(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone() is not None
    )


def names_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def guard(path: str | pathlib.Path, table: str) -> None:
    """Stamp ``path`` with :data:`NAMES_VERSION` when ``table`` is not there yet; raise :class:`StaleNamesError`
    when ``table`` holds rows written under an older names scheme."""
    with contextlib.closing(sqlite3.connect(path, timeout=60.0)) as conn, conn:
        if names_version(conn) >= NAMES_VERSION:
            return
        if has_table(conn, table):
            raise StaleNamesError(
                f"{path} holds rows under the old framework names; run `python -m hpcagent_bench.support.update_names "
                f"{path}` or write a new database"
            )
        conn.execute(f"PRAGMA user_version = {NAMES_VERSION}")


def update(path: pathlib.Path) -> int:
    """Rename every row of ``path`` to the current names; returns how many rows changed."""
    changed = 0
    with contextlib.closing(sqlite3.connect(path, timeout=60.0)) as conn, conn:
        if has_table(conn, "setups"):
            for old, new in HARNESS_RENAMES:
                changed += conn.execute("UPDATE setups SET harness = ? WHERE harness = ?", (new, old)).rowcount
        if names_version(conn) >= NAMES_VERSION:
            return changed
        for (old_fw, old_flavor), (new_fw, new_flavor) in FLAVOR_RENAMES:
            if has_table(conn, "results"):
                changed += conn.execute(
                    "UPDATE results SET framework = ?, flavor = ? WHERE framework = ? AND flavor IS ?",
                    (new_fw, new_flavor, old_fw, old_flavor),
                ).rowcount
            if has_table(conn, "canon"):
                changed += conn.execute(
                    "UPDATE canon SET column = ? WHERE column = ?", (flat(new_fw, new_flavor), flat(old_fw, old_flavor))
                ).rowcount
        if has_table(conn, "results") or has_table(conn, "canon"):
            conn.execute(f"PRAGMA user_version = {NAMES_VERSION}")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hpcagent_bench.support.update_names", description=__doc__)
    parser.add_argument("databases", nargs="+", type=pathlib.Path, help="results, shard or canon databases")
    args = parser.parse_args(argv)
    for path in args.databases:
        if not path.is_file():
            print(f"{path}: no such file", file=sys.stderr)
            return 1
        print(f"{path}: {update(path)} row(s) renamed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
