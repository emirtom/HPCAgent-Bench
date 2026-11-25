#!/bin/bash -l
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Euler, CPU: the SC track's numba baseline for our Triton kernels, examples/sc_triton/cpu_baseline.sbatch adapted
# to a guest account on the public share (48 cores, 128 GiB in total, no -A account). Euler's CPU nodes have 64- or
# 96-core sockets, so a full socket does not fit: one rank takes 48 of the 64 cores of one AMD EPYC 7742 socket
# (the CPU model of the sample). Report it as "48 of 64 cores of one EPYC 7742 socket".
# Output: $RESULTS/canon.db (default $SCRATCH/sc-baseline), run label cpu-<COLUMN>, plus binding.txt.
#   cd ~/HPCAgent-Bench && sbatch helpers/scripts/triton_pipeline/euler_cpu_baseline.sbatch            # L, our kernels
#   PRESET=S sbatch --time=01:00:00 helpers/scripts/triton_pipeline/euler_cpu_baseline.sbatch          # quick check
#SBATCH --job-name=sc-cpu-baseline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2500M
#SBATCH --constraint=EPYC_7742
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=sc-cpu-baseline-%j.out
set -euo pipefail

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
cd "${SLURM_SUBMIT_DIR:-.}"
# Load one at a time: the stack has to be loaded before its compiler, and the compiler before openblas.
module load stack/2024-06
module load gcc/12.2.0
module load openblas/0.3.24
source .venv/bin/activate
export HPCAGENT_BENCH_RESULTS_DIR=${RESULTS:-$SCRATCH/sc-baseline}
# The GPU rows come from the student cluster, whose ptrace_scope=2 forces the child clock (child-clock-v1);
# canon rows of different timing protocols are never paired, so the baseline is timed the same way.
export HPCAGENT_BENCH_MEASUREMENT_PARENT_CLOCK=false
unshare --user true 2>/dev/null || export HPCAGENT_BENCH_GRADING_SEAL=false
column=${COLUMN:-numba}
out=$HPCAGENT_BENCH_RESULTS_DIR/cpu-$column
mkdir -p "$out"
kernels=${KERNELS:-reduce_2d,ls3df_fragment_patch_density,ls3df_lda_xc_potential,rk4_ensemble,rk45_ensemble,conv_2d,harris_corner,laplacian_stencil_3d}
# What the report needs next to the plot: machine, binding (all 48 cores must sit on one socket), toolchain, commit.
{
  echo "host: $(hostname)  job: ${SLURM_JOB_ID:-}  commit: $(git rev-parse HEAD)  preset: ${PRESET:-L}  column: $column"
  lscpu | grep -E "Model name|Socket\(s\)|Core\(s\) per socket|NUMA node[0-9]+ CPU"
  echo "bound cpus: $(taskset -cp $$ | sed 's/.*: //')"
  for cpu in $(taskset -cp $$ | sed 's/.*: //' | tr ',' ' '); do
    for c in $(seq "${cpu%-*}" "${cpu#*-}"); do cat "/sys/devices/system/cpu/cpu$c/topology/physical_package_id"; done
  done | sort | uniq -c | awk '{print "socket " $2 ": " $1 " cpus"}'
  echo "gcc: $(gcc --version | head -1)  openblas: $(pkg-config --modversion openblas)  python: $(python --version)"
  echo "seal: ${HPCAGENT_BENCH_GRADING_SEAL:-on}  parent_clock: $HPCAGENT_BENCH_MEASUREMENT_PARENT_CLOCK"
} | tee "$out/binding.txt"
args=(--column "$column" --out-root "$out" --preset "${PRESET:-L}" --datatype fp32-if-allowed --kernels "$kernels")
srun --ntasks=1 hpcagent-bench job baseline "${args[@]}" --phase begin
srun --ntasks=1 --cpu-bind=cores hpcagent-bench job baseline "${args[@]}" --phase run
srun --ntasks=1 hpcagent-bench job baseline "${args[@]}" --phase finish
hpcagent-bench run-framework --summarize "$HPCAGENT_BENCH_RESULTS_DIR/canon.db" --canon-run "cpu-$column"
