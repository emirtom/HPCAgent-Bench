# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``pin_env_kv``: set one ``KEY=VALUE`` line of an env file, replacing every earlier line of that key."""

import pathlib
import subprocess

from hpcagent_bench import paths

SCRIPT = paths.ROOT / "hpcagent_bench" / "cluster" / "pin_env_kv.sh"


def pin(env: pathlib.Path, kv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", '. "$1" && pin_env_kv "$2" "$3"', "_", str(SCRIPT), str(env), kv],
        capture_output=True,
        text=True,
        check=False,
    )


def test_an_existing_key_is_replaced_and_a_missing_one_appended(tmp_path: pathlib.Path) -> None:
    env = tmp_path / ".env"
    env.write_text("A=1\nB=2\nA=3\n")
    assert pin(env, "A=9").returncode == 0
    assert pin(env, "C=4").returncode == 0
    assert env.read_text().splitlines() == ["B=2", "A=9", "C=4"]


def test_a_key_is_literal_text_not_a_pattern(tmp_path: pathlib.Path) -> None:
    """`A.B` must not drop `AxB`, and a longer key sharing the prefix is another key."""
    env = tmp_path / ".env"
    env.write_text("AxB=1\nA.B=2\nA.BC=3\n")
    assert pin(env, "A.B=9").returncode == 0
    assert env.read_text().splitlines() == ["AxB=1", "A.BC=3", "A.B=9"]


def test_a_missing_env_file_is_refused(tmp_path: pathlib.Path) -> None:
    done = pin(tmp_path / "absent.env", "A=1")
    assert done.returncode == 2
    assert "no such env file" in done.stderr
