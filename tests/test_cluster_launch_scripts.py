# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Launch-environment lines of the cluster job scripts, lifted from the files and run through bash.

The scripts cannot run end to end in a unit test (they need Slurm, containers and images), so each test
takes the EXACT text it checks out of the script (never retyped) and runs only that.
"""

import pathlib
import shutil
import subprocess

import pytest

CLUSTER = pathlib.Path(__file__).resolve().parents[1] / "hpcagent_bench" / "cluster"


def lines_between(path: pathlib.Path, first: str, last: str) -> str:
    """The lines of ``path`` from the one starting with ``first`` through the next one starting with ``last``."""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(first))
    stop = next(i for i in range(start, len(lines)) if lines[i].startswith(last))
    return "\n".join(lines[start : stop + 1])


def bash(script: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script, "script", *args],
        env={"PATH": "/usr/bin:/bin", **(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )


def quote(text: str) -> str:
    """``text`` as one single-quoted bash word."""
    return "'" + text.replace("'", "'\\''") + "'"


@pytest.mark.skipif(shutil.which("taskset") is None, reason="needs taskset to own one CPU")
def test_a_role_step_derives_its_thread_limit_from_the_cores_it_owns() -> None:
    """srun --export=ALL hands a role step the batch shell's whole-node OMP_THREAD_LIMIT; a judge owning 24
    cores must clamp at 24, not at the node's 192, or a 4 * omp_get_num_procs() team is not clamped."""
    block = lines_between(CLUSTER / "run_cluster.sh", "ulimit -s ", "export OMP_THREAD_LIMIT=")
    script = f'{block}\necho "${{OMP_THREAD_LIMIT}}"'
    role = bash(f"exec taskset -c 0 bash -c {quote(script)} script --judge-node", env={"OMP_THREAD_LIMIT": "192"})
    assert role.returncode == 0, role.stderr
    assert role.stdout.strip() == "1"
    batch = bash(f"exec taskset -c 0 bash -c {quote(script)} script", env={"OMP_THREAD_LIMIT": "192"})
    assert batch.stdout.strip() == "192", "the batch shell keeps a limit its launcher set"


def test_serve_only_reads_the_agents_readiness_key_first() -> None:
    """The agent driver waits AGENT_READY_TIMEOUT_SECONDS first; Kimi's layer sets it to 7200 beside a
    VLLM_READY_TIMEOUT_SECONDS of 3600, so serve-only must not give up after 3600 s."""
    line = next(x for x in (CLUSTER / "serve-only.sbatch").read_text().splitlines() if x.startswith("READY_TIMEOUT="))
    done = bash(
        f'{line}\necho "${{READY_TIMEOUT}}"',
        env={"VLLM_READY_TIMEOUT_SECONDS": "3600", "AGENT_READY_TIMEOUT_SECONDS": "7200"},
    )
    assert done.stdout.strip() == "7200"
    assert (
        bash(f'{line}\necho "${{READY_TIMEOUT}}"', env={"VLLM_READY_TIMEOUT_SECONDS": "3600"}).stdout.strip() == "3600"
    )


def test_grade_under_cleanup_survives_a_relay_that_already_died(tmp_path: pathlib.Path) -> None:
    """A gang relay that crashed mid-job fails its kill at EXIT; the trap must still remove the job's files
    and leave the job's own exit status."""
    cleanup = lines_between(CLUSTER / "grade-under.sbatch", "cleanup() {", "}")
    leftover = tmp_path / "judge-1.edf.toml"
    leftover.write_text("x")
    script = "\n".join(
        [
            "set -euo pipefail",
            'true & dead=$!; wait "${dead}"',
            'relay_pids=("${dead}")',
            f"job_files=({quote(str(leftover))})",
            cleanup,
            "trap cleanup EXIT",
            "exit 0",
        ]
    )
    done = bash(script)
    assert done.returncode == 0, done.stderr
    assert not leftover.exists()
