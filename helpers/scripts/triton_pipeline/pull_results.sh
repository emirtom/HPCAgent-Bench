#!/bin/bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copy every measurement off the clusters' scratch file systems, which are not backed up (Euler's scratch also
# deletes old files), into a local folder (run LOCALLY, at the end of each batch and after each Euler run):
#   student cluster: /scratch/users/emuete/pipeline/{ledger,ledger-archive}/ -> <dest>/student-cluster/
#   Euler:           $SCRATCH/sc-baseline*/ canon.db, binding.txt and job summaries -> <dest>/euler/
#
#   pull_results.sh [<dest>]      (default /home/emreute/Code/triton-reports/results)
#
# Files are only added or updated, never deleted locally, so a file that vanished from scratch keeps its last copy.
set -euo pipefail

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
DEST=${1:-/home/emreute/Code/triton-reports/results}
mkdir -p "$DEST/student-cluster" "$DEST/euler"
rsync -a student-cluster:/scratch/users/emuete/pipeline/ledger student-cluster:/scratch/users/emuete/pipeline/ledger-archive \
  "$DEST/student-cluster/"
# Euler is optional (no access, or no run yet): report it and keep the student-cluster copy.
if euler_scratch=$(ssh -o BatchMode=yes -o ConnectTimeout=20 euler 'echo $SCRATCH' 2>/dev/null); then
  rsync -a --prune-empty-dirs --include='*/' --include='canon.db' --include='binding.txt' --include='*.csv' \
    --include='*.pdf' --exclude='*' "euler:$euler_scratch/sc-baseline*" "$DEST/euler/" 2>/dev/null \
    || echo "pull_results.sh: no Euler results under $euler_scratch/sc-baseline* yet"
else
  echo "pull_results.sh: Euler not reachable; only the student-cluster ledger was copied" >&2
fi
echo "PULLED into $DEST:"
find "$DEST" -type f -printf '  %P (%s bytes)\n' | sort
