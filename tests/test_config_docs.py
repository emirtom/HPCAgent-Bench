# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""docs/configuration.md embeds the key table generated from hpcagent_bench/config.py."""

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_the_configuration_key_table_matches_the_model() -> None:
    """A key added, removed or redescribed in the model without rerunning the generator leaves the page stale."""
    checked = subprocess.run(
        [sys.executable, "helpers/scripts/gen_config_docs.py", "--check"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
