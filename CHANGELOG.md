# Changelog

## 0.2.1 (2026-10)

Same pins as 0.2.0 (dace extended `3f5d9a884`, Alloy `aeca753`, relayout `2718b2f`). Fixes the CI failures the
0.2.0 release left and makes every scientific_computing numba reference parallel.

### Changed
- **Parallel numba references.** The scientific_computing oracle is best-of(numba, c), and 30 kernels' numba half
  ran on one core (the emit drops `parallel=True` for a `max`/`min` call, an in-place self-dependent slice update or
  an FFT) or could not be lowered at all. Each now ships a hand-written parallel override, including a four-step
  `fft_1d` and a column-wise `cloudsc_monolith` that compiles in seconds instead of over 30 minutes and leads its
  oracle at XL. `tests/test_numba_reference_parallel.py` fails any new kernel whose numba reference is serial.
- **No numba on the machine_learning track.** Its oracle is torch only; the ML numba kernels are deleted and the
  correctness sweep compares against the eager torch model.
- **Low-precision bands follow the precision** (fp32 1e-2, fp16 2e-2) against the fp64 reference; `cloudsc` is fp64
  only, like `cloudsc_monolith`, because its physics thresholds branch differently at fp32.

### Fixed
- The correctness sweep hands the compiled reference every size symbol and the returned-output buffers
  (`no-reference` failures), widens complex64 inputs to complex128 for the C reference, and launches the
  per-kernel reference tests with the OpenMP environment grading requires.
- Single-precision C/C++/Fortran compare against a float constant expression in double, as the numpy reference
  does (`za > 1.0 - 1e-8` had folded to `za > 1.0f`).
- A CPF form gets the workspace and libraries it requests in the ABI check; DaCe integer helpers (`ipow`, ...) in a
  workspace size are translated to Python.
- CI sizes OpenMP from the physical cores of the runner.

## 0.2.0 (2026-10)

Pinned with dace extended `3f5d9a884`, Alloy `aeca753` and relayout `2718b2f`. Not backwards compatible with 0.1:
the results schema, wire formats and environment names changed without aliases or migrations.

### New
- **Canonical Parallel Form (CPF) track.** Four tracks (`cpf-cpu-src`, `cpf-gpu-src`, `cpf-tool-cpu`,
  `cpf-tool-gpu`) with two hint variants: `proofs` (dependence witnesses and the assumptions a parallel loop relies
  on) or `kinds` (`par | seq | unknown` per loop). Forms are pre-rendered, cached and independently verified; GPU
  forms come from the new DaCe GPU code generator, bf16 forms are native, and a form requests the workspace its
  transients are carved from.
- **Requests API.** One `requests` block asks for per-array precision, libraries and an MPI layout. Low-precision
  outputs are upcast and graded against the fp64 reference, layouts are converted back, conversions are timed
  separately and never scored, and every grade records what was asked for and granted.
- **External skill packets.** A study's own skill pages run from folders outside the repository
  (`HPCAGENT_BENCH_SKILLS_EXTERNAL`), without committing them; they carry pages only and never reuse a shipped name.
- **Running elsewhere.** The seams a port to another site or a cloud changes (site values, job shape, container
  runtime, role launcher, user namespaces) are documented in `docs/launch.md`.

### Changed
- **DaCe canonicalization** is correct, deterministic (`PYTHONHASHSEED=0`) and fast on CPU and GPU, including
  cloudsc, cloudsc_monolith and velocity_tendencies. HB passes sign facts explicitly, reduction fusion removes
  per-thread stack arrays, and strided host/device copies are lowered to one transfer. A CI perf gate holds
  canonicalization within 2x of the compiled baseline.
- **Results database.** One clean schema (`hb-results-v1.0`) with a `canon` table and a `grade_requests` table:
  every row is identifiable, merging shards refuses conflicting rows instead of silently keeping one, setups record
  their tag version, and episodes that never reached the judge are still recorded with their token spend.
- **Pydantic everywhere** for wire, JSON, YAML, config, every judge HTTP answer and the results DB; the agent
  containers stay standard-library only, with regression tests checking their hand-written JSON against the models.
- **Grading.** One correctness region per output and lower peak memory (cloudsc_sedimentation XL fits 4 ranks per
  node); one cut rule (`max(1 s, 2x the reference)`); stopped cells recorded as upper bounds; 512M OpenMP thread
  stacks; a statistical test registry; bit-identical NumPy/CuPy inputs from a counter-based RNG.
- **Corpus.** ~720 kernels, with the relayout kernels ported and a `public:` manifest block (scales,
  configurations, sign facts). Numba references compile without fastmath (hand-written where numba cannot type
  them); ICON neighbour tables are mesh-local; tolerance and input-generation fixes.
- **Frameworks.** DaCe on the new GPU code generator with pooled device transients, Alloy autotuning flavors,
  Polygeist columns, `ppcg_hip` for override-backed and rank >= 2 kernels.
- **Launch and serving.** One Slurm launcher (`job submit`) and one agent budget (120M tokens, 20 h) on every track;
  big-model serving sized for 20 live agents; checkouts outside the registered container mounts run their prepare
  steps.

### Removed
- Migrations and 0.1 compatibility shims, `/bench`, `/oracle`, `/web-search`, SQLModel and the `bench_info` dict.
- Duplicated container stages (shared scripts), duplicated helpers, dead code and stale files.

## 0.1.0 (2026-09)

First PyPI release (`uv pip install hpcagent-bench`).

- Kernel corpus: NumPy references with manifests across three tracks (`loop_level_reasoning`,
  `scientific_computing`, `machine_learning`), plus the optional `distributed` MPI residency.
- Judge: `hpcagent-bench serve` grades submissions over HTTP; `hpcagent_bench.score` /
  `submit` grade in-process.
- NumPy -> C / C++ / Fortran / CuPy / Numba / Pythran / JAX translators (`numpyto*` commands).
- Agent harnesses, prompts, skills and tool fragments; framework baselines (numba, dace, tvm,
  triton, pluto, ...) behind the `cpu` / `nvidia` / `amd` extras.
- Experiment runtime in `hpcagent_bench/cluster/` (`experiments/` holds configuration only); helper jobs as
  `hpcagent-bench job grade-under|prepare|baseline`, their tasks splitting the work by
  `SLURM_PROCID` / `SLURM_NTASKS` (samples in `docs/jobs/`); logs, core dumps and native-mode submissions under
  `$HPCAGENT_BENCH_SCRATCH` (default `<repo>/.scratch`).
