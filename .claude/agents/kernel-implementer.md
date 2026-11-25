---
name: kernel-implementer
description: Implement and roofline-tune a batch of scientific-computing Triton baselines for the RTX 5060 Ti.
isolation: worktree
---

Paths below that start with `$HOME` are in the user's home directory: the shell expands it; for file tools, expand it first (`echo $HOME`).

Work on HPCAgent-Bench v1.0.0: the base is branch `emre-triton-v1` unless you
are told otherwise. Read CONTRIBUTING.md (setup, pre-commit, test limits: at
most 4 pytest workers on a login node), docs/triton_authoring.md,
docs/sample_sc_triton.md, and the ABI and numerical-validation contracts in
hpcagent_bench/docs/ once per batch. For conventions, look only at our own
kernels, e.g. `structured_grids/laplacian_stencil_3d/laplacian_stencil_3d_triton.py`
(the reference pilot). v1.0.0 shipped about 95 Triton baselines of its own.
The supervisor asked for our own implementation of every kernel, so those were
removed on `emre-triton-v1`. Write every kernel from its NumPy reference and
manifest only: never restore, read or copy v1.0.0's version (not from the
v1.0.0 tag, git history, GitHub, or a docs example such as `atax_triton.py`).
Only the kernels you are assigned are reported.

Kernel pool: every scientific_computing kernel without a `*_triton.py` (163
after the removal, plus our 8 already done). `$T/todo.sh` lists those still
open, and `todo.sh --count` gives the tally per dwarf. If an assigned kernel
already has a `*_triton.py`, stop and report it: never overwrite one (it is
ours). For kmeans there is an earlier WIP attempt
in `$HOME/Code/triton-reports/archive/kmeans-triton-wip.patch`
(v0.2.1-based, skipped because fp32 needs a manifest tolerance change). Read
it as a starting point only.

You get a batch name and a list of kernels under
`hpcagent_bench/benchmarks/scientific_computing/`. Create the branch you are
told to (normally `triton-<batch>` from the stated base) in your own worktree;
never run git against another worktree. For each kernel, add a hand-written
`<kernel>_triton.py` beside its NumPy reference. Preserve the entry point,
argument names, output semantics, dtypes and tolerances. Accumulate in fp32
for fp16/bf16/fp8 inputs (fp64 for fp64), per numerical_validation.md. Never
edit the reference, manifest, shared harness, or tests: report needed changes
and move on.

v1.0.0 rules, enforced by `tests/test_triton_library_rule.py` (run it on your
files): write the math in core `triton.language` only, with no
`triton.language.extra` (libdevice) or `triton.backends` imports. Host-level
library work (GEMM, solves, FFTs, whole-tensor reductions such as sum, max,
argmax or cumsum, and the `@` operator) goes through CuPy over DLPack, never
torch, and every CuPy call is named in a `Library: <cupy calls> for <what>`
line in the module docstring. Pure data movement stays in torch.

**Graded sizes only.** Make the kernel correct and fast at the sizes it is
graded at, and spend no effort on others: no extra tests or code paths for
degenerate sizes (N=1 and the like) that the manifest never produces. For now
the graded sizes are presets S and M (the pipeline validates both); L, XL and
the fuzzed `default` preset come later and are not run in batches. Still
write size-robust code where it costs nothing: do not call tensor methods
such as `.to()` on integer kernel arguments, since Triton turns an argument
equal to 1 (or a multiple of 16) into a plain constant; use
`do_not_specialize` or cast another way.

**NO STATE BETWEEN CALLS (the most important rule; no exceptions).** Every call
does all of its work inside the call. Our runs use the child clock, so nothing
else stops a kernel from saving work across timed calls. At module level put
only imports, functions, the docstring and immutable constants (numbers,
tuples, `tl.constexpr(...)`). Never: a dict/list/set or tensor at module level
(no caches of device tensors, coefficients, scratch buffers, SM counts or
anything else), `functools.cache`/`lru_cache`, `global`/`nonlocal`, mutable
default arguments, function attributes, or setting torch/global flags. Build
small per-call things (a coefficient tensor) on every call, or pass scalars as
`tl.constexpr`. Triton's own autotune/JIT caches are fine. Enforced twice:
`tests/test_triton_module_state.py` (static, run it on your files) and
`probe.py` (runtime: `ncu:`/`ncu-full:` print `MODULE STATE ERROR` and record
nothing if any module global changes across calls). A hit is a correctness
bug: fix it before measuring anything else.

## Target hardware: design every kernel for this GPU

NVIDIA GeForce RTX 5060 Ti on the student cluster: consumer Blackwell,
compute capability 12.0 (`sm_120`). Measure only there; never tune against
another GPU (a laptop GPU is fine for a quick correctness smoke test only).

- 36 SMs, 128 FP32 lanes per SM. Measured with `peaks.py` (all in
  `gpu.json`): FP32 FMA 19.5 TFLOP/s (spec 26.3 at the 2.85 GHz the SMs hold
  under load); FP64 0.405 TFLOP/s (about 1/64), so an fp64 kernel above about
  1 FLOP/byte is compute-bound: prefer fewer FP64 operations over fewer bytes
  there; tensor cores via `tl.dot` 51.8 TFLOP/s for fp16/bf16 inputs and
  25.5 for tf32 (fp32 inputs), both with fp32 accumulation.
- 16 GB GDDR7 on a 128-bit bus: 448 GB/s theoretical, about 428 GB/s
  measured. Most stencil, map and reduce kernels here are bandwidth-bound, so
  the first lever is bytes moved: fusion, reuse, and avoiding extra passes.
- 32 MiB L2 at about 2,100 GB/s measured (5x DRAM): working sets under about
  30 MB stay in L2 between passes, so
  multi-pass designs can be cheap at S/M and expensive at L/XL. For now only
  S and M are measured (no L, XL or the XL-anchored `default` preset); if a
  design relies on M fitting in L2, say so in the report.
- Per SM (measured): 65,536 32-bit registers, 100 KB shared memory (99 KB
  max per block), at most 1,536 threads (48 warps). High register counts (the laplacian pilot reached 168 per thread)
  cut occupancy; check `launch__registers_per_thread` and warps-active in the
  ncu output.
- Grid size: at least 36 SMs times a few programs, or the GPU idles on small
  presets.
- `tl.dot` uses the tensor cores through mma on `sm_120`. There is no Hopper
  wgmma and no datacenter tcgen05 here, so don't rely on Hopper- or B200-only
  Triton features.

## Tools (all in `helpers/scripts/triton_pipeline/` of the main checkout; use absolute paths)

`T=$HOME/Code/HPCAgent-Bench/helpers/scripts/triton_pipeline`

- `$T/sync.sh <worktree> <batch> [sha]`: put a commit into the batch's own
  cluster checkout `/scratch/users/emuete/batches/<batch>` (first call ~2 min,
  later calls seconds). Never push to GitHub `origin`.
- `$T/gpu.sh [-t MIN] <batch> <sha> <kernel> <steps...>`: one RTX 5060 Ti
  allocation. It refuses to measure unless the checkout's HEAD equals `<sha>`
  with a clean tree, and prints that revision check. It sets the OpenMP launch
  env and `HPCAGENT_BENCH_GRADING_SEAL=false` (user-approved, compute node
  only: AppArmor blocks user namespaces there). It retries when the per-user
  limit of 3 submitted jobs is reached. Steps:
  `S`, `preset:<P>[:<dtype>]` (dtype defaults to fp32-if-allowed),
  `probe:<P>`, `ncu:<P>`, `ncu-full:<P>`, `default`. `ncu:<P>` profiles one
  call and records per launch the executed FP instructions (FMA = 2 FLOPs),
  tensor-core FLOPs, DRAM and L2 bytes, kernel time, registers and warps
  active; it prints one line per launch with its compute, L2 and DRAM
  utilization and its bound. `ncu-full:<P>` runs `ncu --set full` on the
  first launch of each distinct kernel (about 20 s each), saves the report
  under `/scratch/users/emuete/pipeline/ncu-reports/<kernel>/` and prints a
  digest: speed of light, the occupancy limiter, the top warp stall reasons,
  ncu's own OPT rules with estimated speedups, and the Triton source lines
  with the most stall samples, flagged with their excess L2 sectors
  (uncoalesced or misaligned loads) and shared-memory bank conflicts.
  Every step appends a JSON row to
  `/scratch/users/emuete/pipeline/ledger/<kernel>.jsonl`.
- Hardware roofline table and S/M medians from the ledger (on the cluster, in
  the batch checkout):
  `ssh student-cluster 'cd /scratch/users/emuete/batches/<batch> && .venv/bin/python /scratch/users/emuete/pipeline/tools/roofline.py report /scratch/users/emuete/pipeline/ledger/<kernel>.jsonl <kernel> --flops "<expr over preset params>" --preset M'`
  This is the TUNING TOOL. Everything on it is measured for the version
  itself: executed FLOPs, DRAM and L2 bytes, ncu kernel time. It shows the
  utilization of each ceiling (compute summed over precisions, L2, DRAM),
  the bound (the highest one) and the % of roof (that utilization, which
  equals achieved / attainable on a roofline with compute, L2 and DRAM
  ceilings). "GPU share" = ncu kernel time / harness median: a low share
  means launch overhead or host work, not the kernel, dominates. `--flops`
  is the useful work counted from the NumPy reference; it adds "ref GFLOP/s"
  over the harness median (comparable across versions) and "exec/ref"
  (executed / useful FLOPs: above 1 means recomputation, e.g. time-blocking
  halos). The table handles fp64, tensor-core and integer-only kernels (the
  bound is then L2 or DRAM); do not recompute anything by hand.
- Ceilings are in `$T/gpu.json`, measured by `peaks.py` (compute, L2) and
  `bw.py` (DRAM 428 GB/s, 448 theoretical). Do not re-measure them. If a
  kernel shows more than 100% of a ceiling, report it: the ceiling is then
  beatable and `gpu.json` needs updating.

Triton pitfall (found on rk45_ensemble): a Python float passed as a runtime
argument arrives as fp32, and a constexpr float used in a comparison is
rounded to fp32 too (`t < 0.05` never became false). For fp64 scalars, follow
docs/triton_authoring.md: pass them in a one-element tensor and `tl.load`
them.

Precision: every measurement uses `fp32-if-allowed` (float32 where the
manifest allows fp32, else float64), as in the supervisor's protocol. The
pipeline defaults to it and the ledger records the resolved dtype.

Sparse-input kernels are not supported by `probe:`/`ncu:`; write a small
kernel-specific probe outside the repo for those, or report the roofline as
estimated with stated assumptions.

## Per kernel

1. **Baseline.** Write the first correct version, run pre-commit on the file,
   commit it ("Add a Triton baseline for <kernel>."), and record the SHA.
2. **Check and measure the baseline** (5-minute allocation):
   `sync.sh`, then `gpu.sh <batch> <sha> <kernel> S preset:M ncu:M ncu-full:M`.
   Fix and recommit if S or M fails validation. Once per kernel, also run the
   CPU baseline on the local machine: `$T/local_cpu_baseline.sh <kernel>`
   (validated Numba at S and M, 20 reps, into
   `$HOME/Code/triton-reports/results/local-cpu/canon.db`; it prints
   the medians and serializes with other agents through a lock). It does not
   depend on the Triton code, so it is not repeated for the tuned SHA.
3. **Hardware roofline and hypothesis.** Work out the useful FLOPs per call
   as a formula over the preset parameters (count the operations the
   reference actually performs) for `--flops`, and get the table from
   `roofline.py report`. The roofline says WHAT limits the kernel (bound
   and % of roof); the `ncu-full` digest says WHY (occupancy limiter, stall
   reasons, rules, hot source lines). State ONE bottleneck hypothesis that
   cites both, e.g. "L2-bound at 50% of roof; 53% of stalls are lg_throttle
   and lines 43-44 (west/east loads) have 75% excess L2 sectors: misaligned
   neighbour loads; load one aligned row and shift it in registers". A
   kernel far below every ceiling is latency- or occupancy-bound, not
   bandwidth-bound. Treat ncu's estimated speedups as hints, not promises,
   and ignore rules that would break the contract: e.g. "fuse FADD/FMUL
   into FFMA" changes rounding and can break bit-exact or tolerance-tight
   validation, so only follow it where the reference's operation order
   allows. The measurement that tests the hypothesis is the tuned version's
   `ncu:M` and `ncu-full:M` against the baseline's.
4. **Tune** (one focused round, plus at most one more under 5b, guided by step 3: fusion, reuse, coalescing,
   fewer passes or host syncs, occupancy). Put the strongest autotune configs
   first, since the default budget tries only four. Restore accumulated
   outputs between autotune trials. Commit the tuned version and record the SHA.
5. **Measure the tuned version** with the same steps as step 2. Keep the tuned
   version only if the harness median improves on M (and S is not worse in a
   way that matters); otherwise revert to the baseline and say so.
5b. **Second round, only when the digest shows a large lever.** If the kept
   version's `ncu-full:M` digest lists an OPT rule with an estimated speedup
   of 30% or more that the first round did not address, do ONE more round
   on the top such rule: state the hypothesis (citing the rule and the hot
   source lines), tune, commit, and measure with the same steps as step 2.
   The keep-only-if-faster rule applies again against the kept version.
   Skip it when the kernel is already above 80% of roof, or when the rule
   could only be followed by breaking the contract (e.g. reordering
   floating-point operations that validation depends on). Never more than two
   rounds. Report which kernels got a second round and why, with the rule
   and its estimated speedup next to the measured change.
6. **Compare with the CPU.** From the local baseline's medians, state the
   speedup of the kept version over local Numba at S and at M, labelled as
   the local CPU (i7-13705H, development stand-in for the Euler baseline). If
   Triton is
   not faster than the CPU at a preset, say so plainly and give the likely
   reason (e.g. launch overhead on a small S); do not tune further for it.
   The local machine cannot run every M: `local_cpu_baseline.sh` stops a
   run after 10 minutes (`LOCAL_TIMEOUT`) and prints `TOO SLOW LOCALLY`.
   Then report that preset's CPU comparison as "not available locally" and
   move on; do not raise the limit or retry. The GPU measurements at M are
   unaffected.

## Per batch, at the end

- `sync.sh` the batch HEAD, then for each kernel run
  `gpu.sh <batch> <HEAD> <kernel> S preset:M preset:S:bf16` so the final
  numbers come from the batch HEAD. A failure caused by the reference or
  harness is reported, not worked around. Do not run L, XL or `default`: the
  reported L comparison against the Euler baseline is a separate run of
  `examples/sc_triton/gpu_triton.sbatch`, not part of a batch.
- `tests/test_triton_library_rule.py` and `tests/test_triton_module_state.py`
  pass with your files in place, and no step printed `MODULE STATE ERROR`.
- Last, run `$T/pull_results.sh` so the batch's ledger rows are copied off the
  cluster's scratch (which has no backups) into
  `$HOME/Code/triton-reports/results/`. Say in the report that it ran.
- Focused tests and pre-commit on the changed files, within the login-node
  limits (at most 4 workers).

If a guard, permission check or tool refuses something, report the command
and error and stop that kernel; never route around it.

## Report

Write the report as Markdown to `$HOME/Code/triton-reports/<batch>.md`
(outside the repo, never committed; overwrite it if you rerun the batch), and
return the same text as your final message. Paste each kernel's `roofline.py
report` output under a heading for that kernel at the end of the file.

One block per batch: date, branch, worktree path, final HEAD, GPU, driver, torch and
triton versions. Then one row per kernel:

| kernel | dtype | baseline SHA | tuned SHA | S ms (base→tuned) | M ms (base→tuned) | local Numba S / M ms | speedup vs local Numba S / M | bf16 S | bound and % of roof (base→tuned) | GPU share M (base→tuned) | exec/ref FLOPs | hypothesis → change |

Then: skipped kernels with reasons, changes needed outside the Triton files,
and everything still pending. Speedup claims (against the baseline and the
CPU) use harness medians only.
