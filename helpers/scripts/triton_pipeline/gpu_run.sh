#!/bin/bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# One compute-node allocation for the Triton baseline pipeline (run INSIDE srun on the student cluster).
#
#   gpu_run.sh <checkout> <expected-sha> <kernel> <step> [<step> ...]
#
# Steps (each prints one "RESULT {...}" JSON line that roofline.py / the ledger picks up):
# Precision is fp32-if-allowed everywhere (the sample_sc_triton.md protocol): float32 where the kernel's manifest
# allows fp32, else float64. The ledger records the resolved dtype.
#   S                    harness S preset, -r 3                  (validation + median)
#   default              harness default fuzzed preset: 20 XL-anchored draws, each validated against
#                        the CPU reference on the allocation's 3 cores. It does NOT fit a 5-minute allocation for
#                        most kernels: run it once per final SHA in a long allocation (-t 30, STEP_TIMEOUT=1500).
#   preset:<P>[:<dtype>] any harness preset (dtype defaults to fp32-if-allowed), e.g. preset:M, preset:S:bf16
#   probe:<P>            probe.py timing at preset P (after JIT and autotune), CUDA events, median of 20
#   ncu:<P>              probe.py under Nsight Compute, one call: executed FLOPs, DRAM and L2 bytes, kernel time
#                        (the hardware roofline; roofline.py report places each version on it)
#   ncu-full:<P>         probe.py under ncu --set full (first launch of each distinct kernel, at most 8): saves the
#                        report to $NCU_REPORTS/<kernel>/<sha9>-<P>.ncu-repz (open it in the Nsight Compute GUI) and
#                        prints ncu_digest.py's tuning digest: occupancy limiter, stall reasons, ncu's rules with
#                        estimated speedups, and the Triton source lines with the most stalls
#
# The revision check runs first and nothing is measured unless HEAD == expected SHA, the tree is clean, the
# kernel's _triton.py exists and the GPU is an RTX 5060 Ti. The grading sandbox is disabled for harness steps
# because the node's AppArmor policy (kernel.apparmor_restrict_unprivileged_userns=1) blocks user namespaces;
# the user approved HPCAGENT_BENCH_GRADING_SEAL=false for compute-node harness runs only. The node also sets
# kernel.yama.ptrace_scope=2, so v1.0.0's parent clock (process_vm_readv into the grading child) is refused and an
# fp64 reference fails with "the trusted clock cannot run here"; HPCAGENT_BENCH_MEASUREMENT_PARENT_CLOCK=false makes
# the child time itself (child-clock-v1), which is what every fp32 run (a host precision cast) already uses.
set -u
CHECKOUT=$1 EXPECT=$2 KERNEL=$3
shift 3
HERE=$(cd "$(dirname "$0")" && pwd -P)
ulimit -c 0
ulimit -s "$(ulimit -H -s)"
export OMP_STACKSIZE=512M OMP_THREAD_LIMIT=$(nproc) HPCAGENT_BENCH_MEASUREMENT_PARENT_CLOCK=false
export PATH=/scratch/users/emuete/bin:$PATH UV_CACHE_DIR=/scratch/users/emuete/.cache/uv
cd "$CHECKOUT" || { echo "REVISION CHECK FAILED: no checkout $CHECKOUT"; exit 2; }

# By the kernel's folder, not its name: a renamed kernel keeps its old module_name (ls3df_fragment_patch_density
# holds fragment_patch_density_triton.py).
KDIR=$(find hpcagent_bench/benchmarks -type d -name "$KERNEL" | head -1)
KFILE=$( [ -n "$KDIR" ] && find "$KDIR" -maxdepth 1 -name "*_triton.py" | head -1)
HEAD_SHA=$(git rev-parse HEAD)
STATUS=$(git status --porcelain)
GPU=$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1)
echo "== revision check"
echo "host: $(hostname)  gpu: $GPU  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "path: $(pwd -P)"
echo "HEAD: $HEAD_SHA (expected $EXPECT)"
echo "status: [$STATUS]"
echo "kernel file: ${KFILE:-MISSING}"
echo "cpu: $(nproc) cores usable, $(lscpu | sed -n 's/^Model name: *//p' | head -1)"
if [ "$HEAD_SHA" != "$EXPECT" ] || [ -n "$STATUS" ] || [ -z "$KFILE" ] || [[ "$GPU" != *"5060 Ti"* ]]; then
  echo "REVISION CHECK FAILED; not measuring"
  exit 2
fi
echo "REVISION CHECK OK"
PY=.venv/bin/python
$PY -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__)"

harness() {  # harness <framework> <preset> <dtype> [extra]
  local f=$1 p=$2 d=$3; shift 3
  local pa=(-p "$p"); [ "$p" = fuzzed ] && pa=()
  local log start=$SECONDS
  # STEP_TIMEOUT (s) caps one harness step so a slow step cannot eat the rest of the allocation.
  log=$(HPCAGENT_BENCH_GRADING_SEAL=false timeout "${STEP_TIMEOUT:-240}" uv run --extra nvgpu hpcagent-bench \
    run-framework -f "$f" -b "$KERNEL" "${pa[@]}" -d "$d" "$@" 2>&1 | grep -v "^numpyto_c")
  [ ${PIPESTATUS[0]:-0} = 124 ] || [ $((SECONDS - start)) -ge "${STEP_TIMEOUT:-240}" ] \
    && log="$log"$'\n'"ERROR: step TIMEOUT after ${STEP_TIMEOUT:-240}s"
  echo "step wall time: $((SECONDS - start))s"
  echo "$log" | grep -e "validation" -e "median" -e "FAIL" -e "SUCCESS" -e "rror" | cut -c1-400 | tail -12
  $PY "$HERE/parse_harness.py" "$KERNEL" "$EXPECT" "$p" "$d" "$f" <<<"$log"
}
# Nsight Compute 2026.3 comes from the cluster's CUDA module (not on the compute node's default PATH).
NCU=$(command -v ncu || { module load cuda/13.2.2 >/dev/null 2>&1; command -v ncu; } || echo /cluster/data/cuda/x86_64/13.2.2/bin/ncu)
NCU_METRICS=gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sector_hit_rate.pct,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active
# Hardware roofline: executed FP instructions (an FMA is 2 FLOPs), tensor-core FLOPs (sm__ops_path_* already counts
# FLOPs) and L2 bytes. peaks.py --ncu checked that these reproduce known FLOP and byte counts on this GPU.
NCU_METRICS=$NCU_METRICS,lts__t_bytes.sum
for op in ffma fadd fmul dfma dadd dmul hfma hadd hmul; do NCU_METRICS=$NCU_METRICS,sm__sass_thread_inst_executed_op_${op}_pred_on.sum; done
for src in fp16_dst_fp32 bf16_dst_fp32 fp16_dst_fp16 tf32_dst_fp32; do NCU_METRICS=$NCU_METRICS,sm__ops_path_tensor_src_${src}.sum; done

DT=fp32-if-allowed
NCU_REPORTS=/scratch/users/emuete/pipeline/ncu-reports
LEDGER=/scratch/users/emuete/pipeline/ledger/$KERNEL.jsonl
mkdir -p "$(dirname "$LEDGER")"
run_steps() {
for step in "$@"; do
  echo "== step $step"
  case "$step" in
    S) harness triton S "$DT" -r 3 ;;
    default) harness triton fuzzed "$DT" ;;
    preset:*) IFS=: read -r _ p d <<<"$step"; harness triton "$p" "${d:-$DT}" ;;
    probe:*) $PY "$HERE/probe.py" "$KERNEL" "${step#probe:}" --dtype "$DT" --sha "$EXPECT" ;;
    ncu-full:*)
      P=${step#ncu-full:}
      REP=$NCU_REPORTS/$KERNEL/${EXPECT:0:9}-$P
      mkdir -p "$(dirname "$REP")"
      echo "ncu: $NCU; --set full --import-source yes --clock-control none --cache-control all --kernel-id :::1 -c 8"
      "$NCU" --profile-from-start off --set full --import-source yes --clock-control none --cache-control all \
        --kernel-id :::1 -c 8 -f -o "$REP" $PY "$HERE/probe.py" "$KERNEL" "$P" --dtype "$DT" --sha "$EXPECT" --ncu 2>&1 \
        | grep -E "^==(ERROR|WARNING)==|Profiling \"|MODULE STATE ERROR" | grep -v "nvidia-smi" | tee "$REP.log"
      if grep -q "MODULE STATE ERROR" "$REP.log"; then
        echo "NCU ERROR: kernel keeps module state between calls; no ncu-full row"
      elif [ -f "$REP.ncu-repz" ]; then
        $PY "$HERE/ncu_digest.py" "$REP.ncu-repz" --ncu "$NCU"
        echo "RESULT {\"kind\": \"ncu-full\", \"kernel\": \"$KERNEL\", \"sha\": \"$EXPECT\", \"preset\": \"$P\", \"report\": \"$REP.ncu-repz\"}"
      else
        echo "NCU ERROR: no report written"
      fi ;;
    ncu:*)
      # GPU's own clocks, caches flushed before each kernel: one probe call after JIT + autotune.
      echo "ncu: $NCU ($("$NCU" --version | tail -1)); settings: --clock-control none --cache-control all"
      "$NCU" --profile-from-start off --clock-control none --cache-control all --csv --print-units base --metrics "$NCU_METRICS" \
        $PY "$HERE/probe.py" "$KERNEL" "${step#ncu:}" --dtype "$DT" --sha "$EXPECT" --ncu 2>&1 | grep -v "^==PROF==" \
        | $PY "$HERE/roofline.py" ncu-csv "$KERNEL" "$EXPECT" "${step#ncu:}" ;;
    *) echo "unknown step $step"; return 3 ;;
  esac
done
}
# Every RESULT line is appended to the kernel's ledger (one JSON object per line) for roofline.py report.
# Each row is written as it arrives, so a step killed by the Slurm time limit still leaves earlier rows. The write
# runs in the pipeline itself, not in a tee >(...) process substitution: the script didn't wait for that, so srun
# could end the step before the last (large, many-launch ncu) row landed.
run_steps "$@" 2>&1 | while IFS= read -r line || [ -n "$line" ]; do
  printf '%s\n' "$line"
  [[ $line == "RESULT "* ]] && printf '%s\n' "${line#RESULT }" >>"$LEDGER"
done
