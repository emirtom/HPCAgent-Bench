# Authoring Triton kernels

The `triton` column is registered (`hpcagent_bench/columns.py`), so a kernel gets a Triton baseline from one
hand-written `<kernel>_triton.py` beside its `<kernel>_numpy.py`; no registry, adapter or test slot changes.
Triton has no autogeneration target (`EMITTERS` in `hpcagent_bench/autogen.py` has no `triton` entry), so the
file is always written by hand. Start from `atax_triton.py`
(`scientific_computing/dense_linear_algebra/atax/`) or `compute_triton.py` (`scientific_computing/map_reduce/compute/`).
A new framework, not a new kernel for this one, is [extending/optimizer.md](extending/optimizer.md#b-add-a-framework-column).

## The entry point

- **Name.** The function the manifest's `func_name` names, the same one `<kernel>_numpy.py` defines
  (`load_impl` in `hpcagent_bench/frameworks/framework.py`).
- **Arguments.** Bound by name from the manifest's `args:` (`Framework.call_args`): use the NumPy reference's
  parameter names. A parameter the manifest does not name needs a default, else the call falls back to the
  positional order of `input_args`.
- **Values.** Every dense array arrives as a `torch` tensor on the GPU (`TritonFramework.copy_func`); a sparse
  matrix stays a scipy matrix; a scalar is a NumPy scalar, so convert it (`int(a)`, `float(a)`) before it reaches a
  `@triton.jit` kernel.
- **Outputs.** Write each output in place into its tensor (`compute(array_1, array_2, a, b, c, out)`), or return
  all of them in the manifest's output order and leave the output parameter out (`atax_triton.kernel(A, x)`
  returns `out`); `resolve_outputs` in `hpcagent_bench/frameworks/utilities.py` binds either.
- **Timing.** The harness synchronizes the CUDA stream after the call (`TritonFramework.post_call`) and times with
  CUDA events; do not synchronize inside the entry.

## The search

Put the tuning space in `@triton.autotune(configs=..., key=[...])`. The harness keeps only the first
`OptimizeBudget.configs` entries of every config list (`HPCAGENT_BENCH_OPTIMIZE_BUDGET`, `hpcagent_bench/optimize.py`;
the default `small` scale keeps 4, `full` keeps all), so order the list best first. An output the kernel
accumulates into (`tl.atomic_add`, `+=`) needs `restore_value=[<that output>]`, or each autotune trial adds onto
the last one's result (`atax_triton.py`).

Shared helpers are in `hpcagent_bench/support/helpers/triton_utilities.py`: `powers_of_2` for config ranges,
`use_grid` and `derive_launch_arguments` to fix a kernel's grid and derived sizes once, the `get_*d_tile_offsets`
tile-index helpers and `matmul`.

## Precision

The column declares no fp64 (`precisions` in `columns.py`), but `run-framework` does not enforce a column's
`precisions` and its default datatype is float64: pass `-d float32` (or `fp16`, `bf16`) yourself.

## Check and time it

```sh
uv sync --extra nvgpu      # torch and triton; --extra amdgpu on ROCm
uv run hpcagent-bench run-framework -f triton -b <kernel> -p S -d float32 -r 3   # validates against NumPy
uv run hpcagent-bench run-framework -f triton -b <kernel> -d float32              # the default fuzzed preset
```

A `validation: SUCCESS` line at preset `S` is the wiring check; report times from the default preset.
