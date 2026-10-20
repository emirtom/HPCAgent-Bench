# Changelog

## 0.2.1 (2026-10)

Pins: dace extended `bda29a3d9` (overrides Alloy's), Alloy `aeca753`, relayout `2718b2f`. No backwards
compatibility with 0.2.0 except the results database, which 0.2.1 reads and extends in place.

### dace extended `0fd6219ce` -> `bda29a3d9`
Bug fixes and optimizations only. More sibling loops parallelize (per-loop private scalars); CloudSC's
canonicalize-GPU, vectorized-GPU and parallelize pipelines match the reference on the device (231 -> 19 kernels);
the inliner emits no identity copies; canonicalize is 6-12% faster.

### Extending the benchmark
- **Plugins.** A folder with `plugin.toml` brings agent tools, judge routes, skill pages, packets and methods
  without editing the repository: from `plugins/`, `HPCAGENT_BENCH_USER_PLUGIN_DIRS` or the
  `hpcagent_bench.plugins` entry point (`docs/extending/plugins.md`). AutoKernel is the shipped example.
- **Tools.** The tool server runs on the `mcp` SDK; a tool is one `@tool` function with a pydantic input model
  (schemas are generated, never written by hand), served per setup. A packet gates tools by name; `AGENT_METHOD`
  replaces `AGENT_PACKET`. Judge routes take `POST` with per-episode state (`EpisodeStore`).
- **Registries.** Every named kind (models, frameworks, packets, harnesses, anti-cheat gates, ...) is one typed
  `registry.Kind`; closed categories are `enum.auto()` enums spelled only at the DB/wire edge.
- **Harnesses** `anthropic` and `openai` (native SDKs) and `optimas` again; each harness has its own venv.

### Changed
- `/profile` may measure at a smaller size than the run's preset (fast profile).
- Numba references are `njit(parallel=True)` with no `prange` in generated code; the machine_learning track's
  oracle is torch only.
- Narrow formats (fp16/bf16) are graded against the fp64 reference; `cloudsc` is fp64 only.
- Canon rows record build time by phase (`transform_time`, `codegen_time`, `compile_time`; schema 14), and `setups.plugins` the digest of every plugin the judge loaded (schema 15); older files gain both when opened for writing.
- `any`-mode tasks accept one C or Fortran source built with the agent's own flags.

### Fixed
- Each agent proves its episode with a token, so no agent can spend a sibling's submission.
- Claude Code stream stalls are relaunched like timeouts; output-cap and malformed-tool-call ends are recorded as such.
- The anthropic harness never runs a tool call cut off at `max_tokens`; the shell tool kills a timed-out command's
  whole process group.
- Numba translator (PR #113): outer-product peeling, `None` placeholders, axis scans; four more kernels run.
- Correctness sweep, CPF workspace/library requests, the phase clock, image registry push and RTX 30/40 builds.
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
