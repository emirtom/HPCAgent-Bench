#!/bin/bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# The ONE command an agent uses for GPU work (run LOCALLY). Allocates one RTX 5060 Ti on the student cluster
# and runs gpu_run.sh inside it against the batch checkout that sync.sh prepared.
#
#   gpu.sh [-t MINUTES] <batch-name> <sha> <kernel> <step> [<step> ...]
#
# -t defaults to 5 (short checks: S, preset:M:float32, probe:M, ncu:M). Use -t 30 for the `default` step.
# The QOS allows 3 submitted jobs per user; when srun is refused for that limit (or the GPUs are busy), this
# waits and retries for up to 30 minutes instead of failing.
set -u

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
MIN=5
if [ "${1:-}" = -t ]; then MIN=$2; shift 2; fi
BATCH=$1 SHA=$2 KERNEL=$3
shift 3
TOOLS=/scratch/users/emuete/pipeline/tools
CHECKOUT=/scratch/users/emuete/batches/$BATCH
STEP_TIMEOUT=$(( MIN * 60 - 60 ))
for attempt in $(seq 1 60); do
  out=$(ssh -o BatchMode=yes student-cluster \
    "srun --gpus 5060ti:1 -A dphpc -t $MIN bash --login -c 'STEP_TIMEOUT=$STEP_TIMEOUT bash $TOOLS/gpu_run.sh $CHECKOUT $SHA $KERNEL $*'" 2>&1)
  if grep -q -e "QOSMaxSubmitJobPerUserLimit" -e "Job violates accounting/QOS policy" <<<"$out"; then
    echo "gpu.sh: submit limit reached (3 jobs per user); retry $attempt in 30 s" >&2
    sleep 30
    continue
  fi
  printf '%s\n' "$out" | grep -v -e "^srun: GPU count" -e "^srun: CPU count" -e "^srun: Memory"
  exit 0
done
echo "gpu.sh: gave up after 60 attempts (submit limit)" >&2
exit 5
