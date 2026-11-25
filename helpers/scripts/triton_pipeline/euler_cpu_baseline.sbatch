#!/bin/bash -l
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Euler, CPU: the SC track's numba baseline for our Triton kernels, examples/sc_triton/cpu_baseline.sbatch adapted
# to a guest account on the public share (48 cores, 128 GiB in total, no -A account). Euler's CPU nodes have 64- or
# 96-core sockets, so a full socket does not fit: each kernel runs on 48 cores of an AMD EPYC 7742 node (the CPU
# model of the sample; the supervisor accepted 48 cores). Euler shares nodes, so the 48 may straddle both sockets:
# binding.txt records where they landed, and the report states it.
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
# What the report needs next to the plot: machine, binding (which sockets the 48 cores landed on: Euler shares
# nodes, so they may straddle both), toolchain, commit.
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
# Not `job baseline`: it forks one worker per NUMA domain the task touches (jobs.task_slots, no setting), and on a
# shared Euler node 48 cores span 4-6 uneven domains, so kernels were timed on 2-16 threads each. The supervisor
# accepted 48 cores, so every kernel gets all 48: one run-framework per kernel into the same canon.db and run label
# cpu-<column> that examples/sc_triton/gpu_triton.sbatch reads, with the sample's 20 timed reps (1 warm-up from
# config). The OpenMP launch environment job baseline would set comes from flags.openmp_launch_env (thread limit =
# cores owned); it must be in place before numpy loads, hence before the step starts.
eval "$(python -c 'from hpcagent_bench import flags
print("\n".join(f"export {k}={v}" for k, v in flags.openmp_launch_env().items()))')"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# judge.cpu_slots_per_node defaults to one slot per NUMA domain the task touches, and run-framework then times the
# kernel in that multi-slot mode: on 48 cores over 3 domains, conv_2d at S took 23.9 ms (at 48 or 16 threads) and
# 0.53 ms with one slot (diagnostic job 22381535). One slot = the kernel gets the whole 48-core allocation.
export HPCAGENT_BENCH_JUDGE_CPU_SLOTS_PER_NODE=1
ulimit -s "$(ulimit -H -s)"
echo "omp: OMP_THREAD_LIMIT=$OMP_THREAD_LIMIT OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_STACKSIZE=$OMP_STACKSIZE" \
  " cpu_slots_per_node=$HPCAGENT_BENCH_JUDGE_CPU_SLOTS_PER_NODE" \
  | tee -a "$out/binding.txt"
db=$HPCAGENT_BENCH_RESULTS_DIR/canon.db
for kernel in ${kernels//,/ }; do
  # -e: a rerun times only the kernels without a validated row, as the sample's scripts do.
  srun --ntasks=1 --cpu-bind=cores hpcagent-bench run-framework -f "$column" -b "$kernel" -p "${PRESET:-L}" \
    -d fp32-if-allowed -r 20 -e --canon-db "$db" --canon-run "cpu-$column" || echo "baseline: $kernel FAILED"
done
hpcagent-bench run-framework --summarize "$db" --canon-run "cpu-$column"
