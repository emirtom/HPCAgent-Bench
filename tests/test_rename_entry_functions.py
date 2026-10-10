# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""rename_entry_functions: a recorded delivery defining the old entry has the name renamed (not
comments, not attributes) and its grade repointed; an alias is left as recorded; a dry run writes
nothing."""

import contextlib
import pathlib
import tempfile

from hpcagent_bench.cluster import rename_entry_functions
from hpcagent_bench.harness import results_db
from tests import results_seed

DELIVERY = (
    "# kernel: the entry\ndef kernel(N, seq, table):\n    table[:] = 0\n\n\n"
    'if __name__ == "__main__":\n    kernel(4, None, np.kernel)  # kernel\n'
)
RENAMED = (
    "# kernel: the entry\ndef nussinov(N, seq, table):\n    table[:] = 0\n\n\n"
    'if __name__ == "__main__":\n    nussinov(4, None, np.kernel)  # kernel\n'
)
ALIAS = "from reference import kernel as impl\n\nkernel = impl\n"


def seeded(db: pathlib.Path) -> None:
    """Three nussinov deliveries: one to rename, one aliasing the reference, one C comment mention."""
    python = results_db.Setup("p", "python", "cpu")
    results_seed.grade(db, "p.n0.p0.w0", "nussinov", "score", 1, setup=python, source=DELIVERY)
    results_seed.grade(db, "p.n0.p0.w0", "nussinov", "score", 2, setup=python, source=ALIAS)
    results_seed.grade(db, "c.n0.p0.w0", "nussinov", "score", 3, source="/* kernel */ void nussinov_fp64(void) {}\n")


def stored_texts(db: pathlib.Path) -> list[str]:
    with results_db.reading(db) as conn:
        return [str(r[0]) for r in conn.execute("SELECT text FROM sources ORDER BY text")]


def test_the_top_level_entry_def_is_renamed_and_the_rest_left() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = pathlib.Path(tmp) / "results.db"
        seeded(db)
        with contextlib.closing(results_db.open_db(db)) as conn, conn:
            counts = rename_entry_functions.migrate(conn, dry_run=False)
        assert (counts.sources, counts.rewritten, counts.left) == (3, 1, 1)
        texts = stored_texts(db)
        assert RENAMED in texts
        assert DELIVERY not in texts, "the old text stays behind although no grade references it"
        assert ALIAS in texts


def test_a_dry_run_counts_and_writes_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = pathlib.Path(tmp) / "results.db"
        seeded(db)
        before = stored_texts(db)
        assert rename_entry_functions.main([str(db), "--dry-run"]) == 0
        assert stored_texts(db) == before


if __name__ == "__main__":
    test_the_top_level_entry_def_is_renamed_and_the_rest_left()
    test_a_dry_run_counts_and_writes_nothing()
