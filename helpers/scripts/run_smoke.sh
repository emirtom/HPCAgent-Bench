#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# Run the release smoke suite (every test marked `smoke`, tests/test_smoke.py) through run_tests.sh, on a CPU host
# without a GPU or SLURM:
#
#   helpers/scripts/run_smoke.sh [pytest args...]
#
# SMOKE_BUDGET_S (default 300): wall-clock seconds for the whole session (pytest-timeout --session-timeout); the
#   session fails once it is spent.
# SMOKE_CPUS (default 8): the run is pinned to the first SMOKE_CPUS CPUs of this process's affinity (taskset), so a
#   shared login node keeps its other CPUs; 0 runs unpinned.
set -Eeuo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_BUDGET_S="${SMOKE_BUDGET_S:-300}"
SMOKE_CPUS="${SMOKE_CPUS:-8}"

pin=()
if [[ "${SMOKE_CPUS}" -gt 0 ]]; then
    cpus=$(python3 -c "import os, sys; print(','.join(map(str, sorted(os.sched_getaffinity(0))[:int(sys.argv[1])])))" \
        "${SMOKE_CPUS}")
    pin=(taskset -c "${cpus}")
fi

exec "${pin[@]}" "${REPO}/helpers/scripts/run_tests.sh" -p no:cacheprovider -rfEs --durations=10 -m smoke \
    --session-timeout="${SMOKE_BUDGET_S}" "$@" "${REPO}/tests"
