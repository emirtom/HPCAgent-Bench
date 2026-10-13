#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# The tests of the submitting checkout on one compute node inside the judge image (CONTRIBUTING.md): the registered
# judge EDF with its package mount pointed at this checkout (edf_with_checkout), so the image's launch venv installs
# this checkout's locked dependencies and imports its code, then helpers/scripts/run_tests.sh with the pytest args.
#   JUDGE_EDF=~/.edf/hpcagent-bench-judge-mi300-latest.toml sbatch -p <partition> -A <account> \
#       helpers/scripts/run_tests.sbatch [pytest args...]
#SBATCH --job-name=run-tests
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=%x-%j.out
set -euo pipefail
ulimit -c 0

repo=${HPCAGENT_BENCH_REPO:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}
. "${repo}/hpcagent_bench/cluster/env.sh"
. "${repo}/hpcagent_bench/cluster/container_runtime.sh"
mkdir -p "${HPCAGENT_BENCH_SCRATCH}"
edf="${HPCAGENT_BENCH_SCRATCH}/run-tests-${SLURM_JOB_ID}.edf.toml"
edf_with_checkout "${JUDGE_EDF:?JUDGE_EDF names the registered judge EDF}" "${repo}" "${edf}" || exit 2
trap 'rm -f "${edf}"' EXIT
# The image's interpreter, not the submitting shell's; no editor hooks, so a test's sdfg.view() opens nothing; idle
# OpenMP workers sleep instead of spinning on the cores the pytest workers share.
srun --mem=0 --environment="${edf}" env -u VSCODE_IPC_HOOK -u VSCODE_IPC_HOOK_CLI -u VSCODE_GIT_IPC_HANDLE \
    -u TERM_PROGRAM -u HPCAGENT_BENCH_HOST_PYTHON OMP_WAIT_POLICY=passive \
    bash -c 'HPCAGENT_BENCH_HOST_PYTHON="${HPCAGENT_BENCH_IMAGE_PYTHON:?no image python}" exec "$0" "$@"' \
    "${repo}/helpers/scripts/run_tests.sh" "$@"
