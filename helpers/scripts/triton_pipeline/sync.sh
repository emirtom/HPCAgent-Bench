#!/bin/bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Put a local commit into a per-batch cluster checkout (run LOCALLY, from any directory).
#
#   sync.sh <local-repo-or-worktree> <batch-name> [<sha>]      (sha defaults to HEAD)
#
# Creates /scratch/users/emuete/batches/<batch-name> on first use (clone from the bundle + uv sync --extra nvgpu),
# then fetches the bundle and checks out the SHA detached, leaving a clean tree. Never pushes anywhere else.
# Prints "SYNCED <checkout> <sha>" on success.
set -euo pipefail

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
REPO=$1 BATCH=$2
SHA=$(git -C "$REPO" rev-parse "${3:-HEAD}")
REMOTE_DIR=/scratch/users/emuete/batches/$BATCH
BUNDLE_LOCAL=$(mktemp -d)/$BATCH.bundle
# A bundle needs a ref: a private one per batch (refs/pipeline/<batch>), never a branch.
git -C "$REPO" update-ref "refs/pipeline/$BATCH" "$SHA"
git -C "$REPO" bundle create -q "$BUNDLE_LOCAL" "refs/pipeline/$BATCH"
ssh student-cluster "mkdir -p /scratch/users/emuete/batches/bundles"
scp -q "$BUNDLE_LOCAL" "student-cluster:/scratch/users/emuete/batches/bundles/$BATCH.bundle"
rm -rf "$(dirname "$BUNDLE_LOCAL")"
ssh student-cluster bash -s -- "$REMOTE_DIR" "$SHA" "$BATCH" <<'EOF'
set -euo pipefail
dir=$1 sha=$2 batch=$3
bundle=/scratch/users/emuete/batches/bundles/$batch.bundle
export PATH=/scratch/users/emuete/bin:$PATH UV_CACHE_DIR=/scratch/users/emuete/.cache/uv
fresh=0
if [ ! -d "$dir/.git" ]; then
  git init -q "$dir"
  fresh=1
fi
git -C "$dir" fetch -q "$bundle" "+refs/pipeline/$batch:refs/pipeline/$batch"
git -C "$dir" checkout -q --detach "$sha"
if [ "$fresh" = 1 ] || [ ! -x "$dir/.venv/bin/python" ]; then
  (cd "$dir" && uv sync -q --extra nvgpu)
fi
test -z "$(git -C "$dir" status --porcelain)" || { echo "checkout $dir is not clean"; exit 4; }
echo "SYNCED $dir $(git -C "$dir" rev-parse HEAD)"
EOF
