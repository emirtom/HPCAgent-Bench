#!/bin/bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# The scientific_computing kernels that still have no *_triton.py: the pool Triton batches are drawn from
# (73 at v1.0.0). A kernel drops out once its Triton file is committed. Run from the repository root.
#
#   todo.sh            "<dwarf> <kernel>" per line, sorted by dwarf
#   todo.sh --count    the number of kernels left, per dwarf and in total
set -euo pipefail

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
cd "$(git rev-parse --show-toplevel)"
SC=hpcagent_bench/benchmarks/scientific_computing
list() {
  # A kernel is a folder holding <kernel>.yaml (its manifest); dwarf folders hold none.
  find "$SC" -mindepth 3 -maxdepth 3 -name "*.yaml" | while read -r manifest; do
    dir=$(dirname "$manifest")
    [ "$(basename "$manifest" .yaml)" = "$(basename "$dir")" ] || continue
    compgen -G "$dir/*_triton.py" >/dev/null && continue
    echo "$(basename "$(dirname "$dir")") $(basename "$dir")"
  done | sort
}
if [ "${1:-}" = --count ]; then
  list | awk '{n[$1]++; t++} END {for (d in n) printf "%-28s %d\n", d, n[d]; printf "%-28s %d\n", "total", t}' | sort
else
  list
fi
