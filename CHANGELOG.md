# Changelog

## 0.2.1 (2026-10)

Pinned with dace extended `0fd6219ce` (the bench's pin now overrides the one Alloy names), Alloy `aeca753` and
relayout `2718b2f`. Fixes the CI failures the 0.2.0 release left and makes every scientific_computing numba reference
parallel.

### Changed
- **Parallel numba references.** The scientific_computing oracle is best-of(numba, c), and 30 kernels' numba half
  ran on one core (the emit drops `parallel=True` for a `max`/`min` call, an in-place self-dependent slice update or
  an FFT) or could not be lowered at all. Each now ships a hand-written parallel override, including a four-step
  `fft_1d` and a column-wise `cloudsc_monolith` that compiles in seconds instead of over 30 minutes and leads its
  oracle at XL. `tests/test_numba_reference_parallel.py` fails any new kernel whose numba reference is serial.
- **No numba on the machine_learning track.** Its oracle is torch only; the ML numba kernels are deleted and the
  correctness sweep compares against the eager torch model.
- **Narrow formats against the fp64 reference.** Every path of a grade hands the numba and C references fp16/bf16
  inputs promoted to fp64 (and complex64 as complex128 to C). `cloudsc` is fp64 only, like
  `cloudsc_monolith`, because its physics thresholds branch differently at fp32.

### New
- **Build time per framework, by phase.** Canon rows carry three optional columns, `transform_time` (parse,
  simplify, canonicalize, auto-optimize, polycc), `codegen_time` (emitting source) and `compile_time` (the backend
  compiler), in seconds; their sum is the total, and NULL means not measured. Results schema 14 is 13 plus these
  columns: a 13 file gains them when opened for writing and is read and merged as it is. A cached or prerendered
  artifact (emitted sources, parsed SDFG, canonical SDFG, CPF form) credits the time producing it took to its phase,
  so a cache hit reports what a cold run costs. DaCe builds split the two from `SDFG.compile(stats=)`;
  `hpcagent-bench canon-times` reports p50/p90/max per column and phase.
- **Single-source submissions in `any` mode.** An `any`-mode task accepts one C or Fortran source built with the
  agent's own compile flags (fast-math included); link flags stay restricted to `-l`/`-L`. Multi-file builds go
  through the existing `library` delivery.
- **Tools on the official MCP SDK.** The agent's tool server (`agent/hpcagent_agent/tools/mcp_server.py`) runs on the
  `mcp` SDK's low-level server over stdio, and a tool is one file: `@tool` on its
  `run()` with a pydantic input model (`agent/hpcagent_agent/tools/registry.py`), plus `@route` on a judge handler in
  `hpcagent_bench/harness/routes/` when the judge serves it. The agent package now depends on `mcp` and `pydantic`.
- **Packets gate tools by name.** A packet's `tools` reach the agent as `HPCAGENT_BENCH_PACKET_TOOLS`; a non-core tool
  is served only where it is listed. `search` is no longer switched on by `AGENT_SEARCH_TOOL` but by the new
  `web-search` packet, and the per-tool env switches are gone.
- **A skill page declares its own packet.** A `packet:` block in a page's frontmatter registers it (the CPF, tracer
  and caveman packets moved there; their definitions are unchanged), and staging drops the block from the agent's copy.
- **New harnesses** `anthropic` (Messages API) and `openai` (Agents SDK, any OpenAI-compatible endpoint), both MCP
  clients of the one tool server, and **Optimas** again, on the Agents SDK with no tool shim. claude-code 2.1.303,
  openhands 1.55.1. mini-SWE and OpenHands now really send the effort rung (litellm dropped it).
- **Markers by kind.** The first `harness_shapes` free shapes of the pool are the harnesses', the rest the packets':
  registering either kind never reshapes the other.
- **Flavor composition** (from `registry-flavor-composition`): framework entries compose from mixins, adding the
  `dace_{cpu,gpu}_autoopt_alloy_{profile,heuristic}` flavors and the `tvm` alias.

### Fixed
- **Episode identity is proven.** Each agent gets a token for its own episode (HMAC of the episode id under a job
  secret the seal hides), the tools send it, and the router refuses a request naming an episode its token does not
  prove: an agent can no longer spend a sibling's single submission.
- Claude Code's "Response stalled mid-stream" and "Connection closed mid-response" are transport faults like the
  client timeout (rc 127): the agent is relaunched within its deadline instead of ending as an agent fault (21 v0.2
  episodes). A reply over the output cap (rc 128, `output_cap`) and a tool call the engine refuses as malformed
  JSON (rc 129, `model_error`) are recorded as what they are, not as faults, and are not relaunched.
- A Python submission without the task's entry function is refused before grading, naming the function and what the
  module defines (it failed on import, with the reason only in the judge log).
- The correctness sweep hands the compiled reference every size symbol and the returned-output buffers
  (`no-reference` failures), never shrinks a torch-oracle kernel, and launches the per-kernel reference tests
  with the OpenMP environment grading requires.
- The raman_fitting numba reference compiles at fp32; fft_1d's no longer asks numba to cache an objmode function.
- A CPF form gets the workspace and libraries it requests in the ABI check; DaCe integer helpers (`ipow`, ...) in a
  workspace size are translated to Python.
- CI sizes OpenMP from the physical cores of the runner.
- **Members are static.** Production code reads a model, dataclass, module or AST node by its declared field
  (`vars(obj)[name]`, `isinstance`), never `getattr` / `hasattr` by name; only C-extension objects without a
  `__dict__` (a cffi `Lib`, a tree-sitter node) are read by name. `tests/test_static_members.py` holds the line.
  The tuple desugar's per-node marker attribute became a per-run registry.
- The Anthropic harness answers a tool call cut off at `max_tokens` as an error instead of running it with
  truncated input; Optimas reads its reward and usage from the Agents SDK's actual result shapes.
- The phase clock is per context, so a judge thread rendering a CPF form does not bill another thread's span.
- `registry.sh push` drops the session bus sbatch inherits and retries a load once (the job's first image came
  back without an ID); `mirror-repos.sh` mirrors only the repositories of the image it builds and exits 2 on an
  unknown one.
- The x86_64 NVIDIA image builds native sm_86 / sm_89 (RTX 30 / 40) next to sm_75 to sm_120.

### CI
- 13 jobs plus coverage, each ~35-40 min (was 25): short legs ride along on longer jobs, and the nvc++ toolchain
  goes to whichever unit shard holds `test_parallelism_dispatch.py` (it had drifted to a shard without it).

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
