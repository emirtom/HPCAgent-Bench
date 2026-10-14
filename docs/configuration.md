# Configuration: site, storage and job shape

Nothing in this repository names a cluster's filesystems, Slurm account, partition or node shape. Those
values resolve in one order, for every script:

1. a command-line flag (`hpcagent-bench job submit --cpus-per-task 72`, `NICE=10 submit.sh`);
2. an environment variable (`export SBATCH_PARTITION=gpu`); a job script's own `#SBATCH` line beats every one but
   Slurm's `SBATCH_*` ([below](#job-shape-per-system));
3. the site layer, `experiments/layers/site.env`, which holds this cluster's values;
4. a generic default (for a job's node shape: the named system's entry, [below](#job-shape-per-system)).

The site layer keeps the `VAR="${VAR:-value}"` form, so a value already exported beats the file
and sourcing it twice changes nothing.

## Set up on your system

```bash
cd hpcagent-bench
cp experiments/layers/site-example.env experiments/layers/site.env   # gitignored
$EDITOR experiments/layers/site.env                                  # fill in your cluster's values
export SCRATCH=/path/to/your/scratch                                 # most HPC sites already set it
. hpcagent_bench/cluster/env.sh                                      # loads the layer, caches, interpreter, OpenMP env
ulimit -s "$(ulimit -H -s)"                                          # once per shell: the main thread stack, to its hard limit
echo "$FAST_SCRATCH $SBATCH_PARTITION $SBATCH_ACCOUNT $HF_HOME"
```

A minimal `site.env` for a cluster with a flash tier and a GPU partition named `gpu`:

```bash
FAST_SCRATCH="${FAST_SCRATCH:-/flash/${USER}}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"
SALLOC_PARTITION="${SALLOC_PARTITION:-${SBATCH_PARTITION}}"
```

To keep the layer outside the checkout, point `HPCAGENT_BENCH_SITE_ENV` at it. With no layer every variable
takes its generic default, which runs wherever `SCRATCH` is set. `experiments/layers/site-cscs.env` is the layer
of the CSCS Alps MI300A partition, the reference setup of this repository's experiments.

| Where | What it resolves |
|---|---|
| `experiments/layers/site-<name>.env`, loaded by `helpers/scripts/site_env.sh` | this cluster's values: account, partition, fast storage, node exclusions, the host interpreter |
| `helpers/scripts/cache_env.sh` | every cache and work directory, from `SCRATCH` and `FAST_SCRATCH` |
| `helpers/scripts/host_python.sh` | the host-side interpreter (`HPCAGENT_BENCH_HOST_PYTHON`), checked to be Python >= 3.10 |
| each image's EDF | the interpreter of every step inside it (`HPCAGENT_BENCH_IMAGE_PYTHON`) and `PYTHONHASHSEED=0` |
| `hpcagent_bench/cluster/env.sh` | the checkout, `PYTHONHASHSEED=0`, the OpenMP launch environment; sources the two scripts above |
| `pyproject.toml` (`[tool.uv.sources] dace`) | the dace commit a release installs, bakes and runs ([below](#dace)) |
| `hpcagent_bench/paths.py` | the Python side of the same roots |

Experiment knobs (models, agents, judges, budgets) are not site values: they live in `experiments/layers/common.env`
and the model layers ([launch.md](launch.md), [`experiments/LAUNCH.md`](../experiments/LAUNCH.md)).

## Job shape per system

Tasks per node, cores per task and GPUs per node or per task differ between machines, so every job script (the
helper jobs in `hpcagent_bench/cluster/`, [jobs.md](jobs.md); the container jobs in `containers/`) is started with
`hpcagent-bench job submit`, which passes each field the script does not pin as an `sbatch` option:

```bash
hpcagent-bench job submit --system daint.alps hpcagent_bench/cluster/grade-under.sbatch worklist.jsonl out
hpcagent-bench job submit --ntasks-per-node 2 --cpus-per-task 32 --gpus-per-task 1 hpcagent_bench/cluster/baseline.sbatch ...
hpcagent-bench job submit --dry-run hpcagent_bench/cluster/prepare.sbatch --tag llr40 --language c   # print the sbatch line only
```

Each field is resolved in this order, the first that sets it winning (`hpcagent_bench/cluster/systems.py`):

1. the flag;
2. Slurm's own variable (`SBATCH_PARTITION`, `SBATCH_ACCOUNT`; partition and account only);
3. the script's own `#SBATCH` line (long or short form);
4. the `HPCAGENT_BENCH_*` variable, or the site layer's value of it;
5. the system's entry in `hpcagent_bench/cluster/systems.yaml`.

| Field | Flag | Variable | `systems.yaml` key |
|---|---|---|---|
| partition | `--partition` | `SBATCH_PARTITION` | `partition` |
| account | `--account` | `SBATCH_ACCOUNT` | -- |
| nodes | `--nodes` | `HPCAGENT_BENCH_JOB_NODES` | -- |
| time limit | `--time` | `HPCAGENT_BENCH_JOB_TIME` | -- |
| nice | `--nice` | `HPCAGENT_BENCH_NICE` (default 100) | -- |
| tasks | `--ntasks` | `HPCAGENT_BENCH_JOB_NTASKS` | -- |
| tasks per node | `--ntasks-per-node` | `HPCAGENT_BENCH_JOB_NTASKS_PER_NODE` | `ntasks_per_node` |
| cores per task | `--cpus-per-task` | `HPCAGENT_BENCH_JOB_CPUS_PER_TASK` | `cpus_per_task` |
| GPUs per node | `--gpus-per-node` | `HPCAGENT_BENCH_JOB_GPUS_PER_NODE` | `gpus_per_node` |
| GPUs per task | `--gpus-per-task` | `HPCAGENT_BENCH_JOB_GPUS_PER_TASK` | -- |
| hardware (`submit.sh` only, not an sbatch option) | `--hardware` | `HPCAGENT_BENCH_HARDWARE` | `hardware` |
| longest partition time (`submit.sh` only) | -- (`job options --max-time-hours`) | `HPCAGENT_BENCH_MAX_TIME_HOURS` | `max_time_hours` |

The two GPU fields are one choice, and so are the two task fields: each pair comes whole from the highest source
that sets either. The shipped systems are `beverin` (MI300A, partition `mi300`, 4 tasks x 24 cores, 4 GPUs),
`beverin-mi200` (partition `mi200`, 4 tasks x 16 cores, 8 GPUs) and `daint.alps` (GH200, partition `normal`,
4 tasks x 72 cores, 4 GPUs). A script that pins its shape keeps it (`build_and_verify.sbatch` stays one task of
96 cores); one that pins nothing gets the system's. The system is `--system`, else
`HPCAGENT_BENCH_SYSTEM`, else the entry whose `cluster` is `SLURM_CLUSTER_NAME`, else none: a cluster with no entry
runs from flags and the environment alone. Without `--system`, a partition picks the entry of the same cluster that
serves it (`--partition mi200` on Beverin is `beverin-mi200`: 8 GPUs, 16 cores per task). A new machine is a file of
the same shape named by `HPCAGENT_BENCH_SYSTEMS_FILE` (its entries add to or replace the shipped ones). An image build or verify job runs on its role's `images.env` partition, as if
its header named it, unless `--system` names a machine ([containers/README.md](../containers/README.md#amd-beverin)).

The experiment submitter (`hpcagent_bench/cluster/submit.sh`) resolves the same way, through the same code
(`hpcagent-bench job options`): `--partition`, `--account`, `--gpus-per-node`, `--system` and `--hardware` over their
variables over the system's entry. Its node count is the sum of its roles and its time limit comes from the tag
(`--time` overrides it), so neither is resolved. An experiment cannot run without an account and a GPU count: a
missing one is an error naming its flag and its variable (`--account` / `SBATCH_ACCOUNT`, `--gpus-per-node` /
`HPCAGENT_BENCH_JOB_GPUS_PER_NODE`). The partition may stay unset, which is the cluster's default partition. The
hardware (`--hardware`, `HPCAGENT_BENCH_HARDWARE`, the entry's `hardware`) is not an `sbatch` option: it names
the GPU generation whose images and serving layers the experiment uses ([below](#hardware-is-not-a-site-value)).

## Variables

### Site layer

| Variable | Default | Controls |
|---|---|---|
| `HPCAGENT_BENCH_SITE_ENV` | `experiments/layers/site.env` when it exists | which layer `helpers/scripts/site_env.sh` loads; a named file that does not exist is an error |
| `SBATCH_PARTITION`, `SALLOC_PARTITION` | unset (the cluster's default) | the partition of every `sbatch` / `salloc`; Slurm reads it and it overrides a script's `#SBATCH --partition`, and `--partition=` on the command line overrides it (the submitters resolve it with their `--partition` flag, over this variable, over the system's entry) |
| `SBATCH_ACCOUNT` | empty | the account every `sbatch` bills; `hpcagent_bench/cluster/submit.sh` refuses to submit without one, and `root`. `SLURM_ACCOUNT` and `SALLOC_ACCOUNT` follow it. No script passes `-A` |
| `HPCAGENT_BENCH_LOGIN_HOST`, `HPCAGENT_BENCH_SSH_JUMP` | empty: a placeholder | the login host and ssh jump chain in the laptop tunnel commands `containers/inference/serve-private.sbatch` prints |
| `HPCAGENT_BENCH_NICE` | `100` | the `--nice` every submitter passes (`NICE=<n>` for one submission); Slurm has no environment variable for it, so a bare `sbatch` runs at nice 0 |

### Storage

| Variable | Default | Controls |
|---|---|---|
| `SCRATCH` | set by the site | bulk storage: runs, result DBs, logs, JIT caches, the venv |
| `HPCAGENT_BENCH_SCRATCH` | `<checkout>/.scratch` (git-ignored) | submitter Slurm output (`logs/`), the judge core dump of a crash-diagnosis setup (`core/`), native-mode submissions (`native_runs/`) |
| `FAST_SCRATCH` | `SCRATCH`, else `<checkout>/.cache` | model weights and large read-mostly caches (`HF_HOME`, the judge's disk result store) |
| `HPCAGENT_BENCH_CACHE` | `$FAST_SCRATCH/.hpcagentbench-cache` | root of the read-mostly caches; `HF_HOME` is `$HPCAGENT_BENCH_CACHE/hf` |
| `JIT_CACHE_ROOT` | `$SCRATCH/.hpcagentbench-cache`, else `<checkout>/.cache/jit` | compile/JIT caches (aiter, triton, inductor, vLLM), keyed by image below it |
| `HPCAGENT_BENCH_CPF_PRERENDER_DIR`, `HPCAGENT_BENCH_CPF_CACHE` | under `$JIT_CACHE_ROOT` | Canonical Parallel Form views and their content-addressed cache (the judge renders a kernel on its first request, `python -m hpcagent_bench.cpf_prerender` warms it) |
| `HPCAGENT_BENCH_TOOLS_DIR` | `$JIT_CACHE_ROOT/tools` | build tools not in the images (e.g. `ppcg`) |
| `HPCAGENT_BENCH_RUNS_ROOT`, `HPCAGENT_BENCH_RESULTS_DIR` | under `$JIT_CACHE_ROOT` | per-job work directories of deterministic-framework jobs, and the persistent results they merge into |
| `HPCAGENT_BENCH_DATA_ROOTS` | derived: top-level filesystems of `SCRATCH` and `FAST_SCRATCH` | what container EDFs bind-mount |

### Checkout, Python and containers

| Variable | Default | Controls |
|---|---|---|
| `HPCAGENT_BENCH_REPO` | the checkout `cluster/env.sh` lives in | the tree scripts and jobs run from |
| `HPCAGENT_BENCH_HOST_PYTHON` | `python3` on PATH | the host-side interpreter, with the package installed (`uv sync`) |
| `HPCAGENT_BENCH_IMAGE_PYTHON` | the image's EDF | the interpreter of every step inside a container |
| `EDF_PATH` | `$HOME/.edf` | where container EDFs are looked up |
| `CONTAINER_RUNTIME` | `ce` | how `cluster/services.sbatch` starts containers, through one seam (`cluster/container_runtime.sh`, [runtime.md](runtime.md)): `ce`, `apptainer`, `podman` or `docker` |
| `HPCAGENT_BENCH_HARDWARE`, `HPCAGENT_BENCH_MAX_TIME_HOURS` | the system's `hardware`, `max_time_hours`; else unset | the hardware, and the longest time limit of the partition, which clamps a scaled wall clock (`TIME_SCALE`, minus `STAGING_HOURS`); unset, nothing is clamped |
| `HPCAGENT_BENCH_HOST` | `SLURMD_NODENAME`, else the host name | the node name recorded with each result |
| `OMP_STACKSIZE`, `OMP_THREAD_LIMIT` | `512M`, the cores the shell owns (`nproc`); set by `cluster/env.sh` and `run_cluster.sh` | the OpenMP launch environment every grading process needs; the main thread's stack is the shell step `ulimit -s "$(ulimit -H -s)"` (its hard limit) |

Every command runs `<python> -m hpcagent_bench...` (or `-m hpcagent_agent...` in an agent step) with one of the two
interpreters, never a PATH lookup. Nothing sets `PYTHONPATH` or edits `sys.path` (the `import-path` rule of `helpers/scripts/checks/check_repo_rules.py`): both
packages are installed, editable in a checkout (`uv sync`) and through the image hooks in a container.

### Container image builds

Defaults are in `containers/images/images.env` and `build_common.sh`; `IMAGE_REQUIREMENTS.md` says what the knobs do.

| Variable | Default | Controls |
|---|---|---|
| `CE_IMAGES` | `$SCRATCH/ce-images` | squashfs images, their sidecars and build logs |
| `CE_TMPFS` | `/dev/shm/$USER` | per-user tmpfs for podman stores and image mounts (Lustre cannot hold them) |
| `CE_BUILD_CACHE` | `1` | keep the node's podman layer store and mount the spack and pip caches; `0` builds cold |
| `CE_PULL` | `1` | pull a registry image whose build-inputs label matches instead of building; `only`, `0` |
| `REGISTRY_REPO`, `PULL_REPO` | `docker.io/spcleth/hpcagent-bench`, `PUSH_REPO` | the published image repository, and the one pull-first reads |
| `SPACK_BUILDCACHE`, `UV_BUILD_CACHE`, `BASE_CACHE`, `GIT_MIRRORS` | under `$SCRATCH` | the binary, wheel, base-image and git caches the builds mount |

### dace

DaCe comes from the spcl/dace `extended` branch, pinned by the `dace` entry of `[tool.uv.sources]` in
`pyproject.toml` (the one place it is written; the published metadata says plain `dace`). `uv sync` installs the pin
(point `[tool.uv.sources]` at a checkout of your own for development); no image carries it: every job installs the
pin of its own checkout at start (`containers/images/lib/launch_venv.sh`), so moving the pin (only to an extended commit
whose CI is green) needs no rebuild. The installed dace records its commit in its PEP 610 `direct_url.json`, which the judge prints into the job log;
canon columns stamp `dace <sha>` into `record.build` and `canon.db`'s `build` column.

## Hardware is not a site value

`mi300` and `mi200` in image and EDF names (`hpcagent-bench-agent-mi300-latest`), in `--hardware mi200` and in
`experiments/layers/hardware-mi200.env` name a GPU generation, not a site's Slurm partition: they select the image
built for that architecture and its serving layers. The partition that hardware sits in (`--partition`) and the
GPUs per node (`--gpus-per-node`) are job-shape values, so a cluster whose MI250X partition is called `gpu` runs
`submit.sh --hardware mi200 --partition gpu --gpus-per-node 8`, or names the three in a `systems.yaml` entry.

The Container Engine names its images by EDF, and the EDF name carries the hardware. `layers/common.env` names the
EDFs of its base hardware (`HPCAGENT_BENCH_BASE_HARDWARE=mi300`); `submit.sh` pins `HPCAGENT_BENCH_HARDWARE` and, for
any other hardware, renames every `*_CE_ENV` to the `-<hardware>-` EDF and pins `layers/hardware-<hardware>.env`
(and `hardware-<hardware>-<model>.env` for a model served on our nodes). A recorded experiment on another hardware
must name it, so its rows never pool with the base hardware's. Under the Container Engine no hardware is an error;
the other runtimes name an image per role (`INFERENCE_IMAGE`, `BENCH_IMAGE`) and need none.

## The guard

The `site-values` rule of `helpers/scripts/checks/check_repo_rules.py` (a pre-commit hook) scans every tracked file for storage mounts, home directories, user names,
site emails, Slurm accounts, `#SBATCH` partition/account/node directives, node and login host names, literal
partitions, one experiment's run directories and the site image registry, in live code (comments and docstrings may
name a site to explain it). A file that legitimately carries such a value is allowlisted in the check with one
reason: the CSCS site layer, the system entries, the hardware layer, this page, and the MI300A serving
recipe's partition check.

## Benchmark settings

Seeds, timeouts, measurement rigor, grading switches and every other benchmark setting are fields of one typed
model, `hpcagent_bench/config.py` (pydantic). The model is the only place a default is written; code reads a
value as `config.current().<section>.<key>`. A key resolves in this order, first match wins:

1. a runtime override (`config.set_override`, `config.overridden`), process-global;
2. the key's environment variable in the active `config.scoped_environment` (a fused judge's per-request setup
   scope);
3. the key's environment variable in the process environment;
4. the model default.

The environment variable of key `a.b_c` is `HPCAGENT_BENCH_A_B_C`. Its text is parsed by the key's type:

| Type | Accepted text |
|---|---|
| nullable (`X \| null`) | `""`, `null` or `none` (any case) for null, else as `X` |
| `str` | the text unchanged |
| `bool` | `true`/`false`, `1`/`0`, `yes`/`no`, `on`/`off`, `t`/`f`, `y`/`n` (any case) |
| `int`, `float`, `one of ...` | the value; surrounding whitespace ignored |
| `list[...]` | a JSON array, or a comma-separated list |
| `map[...]` | a JSON object; the map is replaced whole |

A value that does not parse, a constraint violation (`int >= 1` given `0`) and an `HPCAGENT_BENCH_<SECTION>_*`
name that is no key are errors when the configuration is next read; names that other code owns under a section
prefix (`HPCAGENT_BENCH_JUDGE_URLS`, `HPCAGENT_BENCH_MPI_GANG_*`, ...) are listed in `config.NON_CONFIG_ENV`, and
`HPCAGENT_BENCH_PROMPT_SECTIONS_<KEY>` sets one entry of `prompt.sections`. `config.set_override` raises on an
unknown key or a value of the wrong type.

<!-- BEGIN generated by helpers/scripts/gen_config_docs.py -->

| Key | Environment variable | Type | Default | Meaning |
| --- | --- | --- | --- | --- |
| `seeds.input_dist` | `HPCAGENT_BENCH_SEEDS_INPUT_DIST` | `int >= 0` | `0` | Base seed of input array values (frameworks/benchmark.py). |
| `seeds.fuzz` | `HPCAGENT_BENCH_SEEDS_FUZZ` | `int >= 0` | `42` | Base seed of public fuzz draws of sizes and distributions (fuzz.base_seed). |
| `seeds.kernel_sample` | `HPCAGENT_BENCH_SEEDS_KERNEL_SAMPLE` | `int >= 0` | `0` | Seed of tags.sample() when the CLI passes no seed. |
| `seeds.secret_shape` | `HPCAGENT_BENCH_SEEDS_SECRET_SHAPE` | `int >= 0 \| null` | `null` | Judge only. Seed of the timed-shape draw under perf.mode=secret_3shapes; null = a fresh os.urandom draw per call (fuzz.secret_shape_seed). |
| `fuzz.iterations` | `HPCAGENT_BENCH_FUZZ_ITERATIONS` | `int > 0` | `20` | Draws per config of framework sweeps and the default --k of the Harbor grader. |
| `fuzz.anchor` | `HPCAGENT_BENCH_FUZZ_ANCHOR` | `one of S, M, L, XL` | `"XL"` | Rung the default fuzz range is drawn around. spec.resolve_preset overrides it for every preset token it resolves (bare fuzzed = XL), so this value applies only in a process that resolves none. |
| `fuzz.correctness_iterations` | `HPCAGENT_BENCH_FUZZ_CORRECTNESS_ITERATIONS` | `int > 0` | `8` | Correctness draws per config on the agent grading path, counting draw 0 (the declared maximum); edge shapes are additional. |
| `fuzz.size_cap` | `HPCAGENT_BENCH_FUZZ_SIZE_CAP` | `int >= 0` | `0` | Upper bound on every drawn size dimension; 0 = none. |
| `fuzz.xl_lo_mult` | `HPCAGENT_BENCH_FUZZ_XL_LO_MULT` | `float > 0` | `0.5` | Correctness sizes are drawn in [xl_lo_mult, xl_hi_mult] x anchor rung; timed sizes in [(xl_lo_mult + xl_hi_mult) / 2, xl_hi_mult] x anchor rung. |
| `fuzz.xl_hi_mult` | `HPCAGENT_BENCH_FUZZ_XL_HI_MULT` | `float > 0` | `1.0` | Upper bound of every draw as a multiple of the anchor rung; <= 1.0 keeps draws at or below it. |
| `fuzz.correctness_size_cap` | `HPCAGENT_BENCH_FUZZ_CORRECTNESS_SIZE_CAP` | `int >= 0` | `1024` | Per-dimension upper bound on correctness draws only; 0 = none. Timed draws are unaffected. |
| `fuzz.hidden_correctness_presets` | `HPCAGENT_BENCH_FUZZ_HIDDEN_CORRECTNESS_PRESETS` | `list[str]` | `["XL", "M", "M", "L", "S"]` | Preset of held-out case i, positional against hidden_tests.VARIANTS; a preset the kernel does not declare, or an empty list, falls back to the timed preset. |
| `fuzz.size_distribution` | `HPCAGENT_BENCH_FUZZ_SIZE_DISTRIBUTION` | `one of log_uniform, uniform` | `"log_uniform"` | Distribution of a drawn size within its [lo, hi] range. |
| `fuzz.data_distribution` | `HPCAGENT_BENCH_FUZZ_DATA_DISTRIBUTION` | `str` | `"uniform"` | Input value distribution of a fuzz draw whose manifest names none (a name registered in support/distributions). |
| `inputs.noise` | `HPCAGENT_BENCH_INPUTS_NOISE` | `bool` | `false` | true: every float input array is multiplied by 1 + noise_eps * u, u uniform in [-1, 1), after initialization; integer, index, sparse and structural arrays are unchanged. |
| `inputs.noise_eps` | `HPCAGENT_BENCH_INPUTS_NOISE_EPS` | `float >= 0` | `0.0` | Relative noise amplitude; 0 = the per-format default (float64 1e-6, float32 1e-5, float16 4e-3, bfloat16 3e-2). |
| `service.oracle` | `HPCAGENT_BENCH_SERVICE_ORACLE` | `one of auto, compiled, numba, c, torch` | `"auto"` | Correctness reference (api.Oracle). auto = compiled on loop_level_reasoning and scientific_computing, torch on machine_learning. |
| `service.input_mode` | `HPCAGENT_BENCH_SERVICE_INPUT_MODE` | `one of py-binding, source, library, any` | `"source"` | What POST /submit accepts (api.InputMode). |
| `service.submit_feedback` | `HPCAGENT_BENCH_SERVICE_SUBMIT_FEEDBACK` | `one of verdict, full` | `"verdict"` | /submit response body. verdict = correct yes/no + request_id (+ build log on a build failure); full = the whole grade, only for the loopback upstream of cluster/judge_service.py. |
| `service.preset` | `HPCAGENT_BENCH_SERVICE_PRESET` | `str` | `"XL+fuzz"` | Preset token the judge grades at (spec.parse_preset). |
| `service.datatype` | `HPCAGENT_BENCH_SERVICE_DATATYPE` | `str` | `"float64"` | Storage datatype a grade uses when the request names none. |
| `service.canonical_parallel_form_dir` | `HPCAGENT_BENCH_SERVICE_CANONICAL_PARALLEL_FORM_DIR` | `str` | `""` | Directory of the canonical-parallel-form view the /cpf route serves; empty = the route answers unavailable and the prompt omits the tool. |
| `service.warm_problems` | `HPCAGENT_BENCH_SERVICE_WARM_PROBLEMS` | `str` | `""` | Problems file whose cells the judge compiles at start; empty = no warm-up. |
| `service.warm_language` | `HPCAGENT_BENCH_SERVICE_WARM_LANGUAGE` | `str` | `"c"` | Language of the warm-up builds. |
| `service.warm_shards` | `HPCAGENT_BENCH_SERVICE_WARM_SHARDS` | `int > 0` | `1` | Judges sharing the warm-up; judge rank k warms cells[k::warm_shards]. |
| `measurement.baseline` | `HPCAGENT_BENCH_MEASUREMENT_BASELINE` | `one of auto, numba, c, c-autopar, cpp-autopar, fortran-autopar, torch-autotune, torch-autotune-cpu, torch-autotune-gpu` | `"auto"` | Speedup denominator kind (grading.BASELINE_OPTIONS); auto = measurement.denominator.<track>. A reference that cannot be emitted, built or run is a scored judge fault (timing.measurement_baseline). |
| `measurement.denominator.loop_level_reasoning` | `HPCAGENT_BENCH_MEASUREMENT_DENOMINATOR_LOOP_LEVEL_REASONING` | `one of numba, c, c-autopar, vendored, best-of(numba,c), best-of(numba,c,c-autopar), torch-autotune` | `"best-of(numba,c)"` | Denominator of loop_level_reasoning. |
| `measurement.denominator.scientific_computing` | `HPCAGENT_BENCH_MEASUREMENT_DENOMINATOR_SCIENTIFIC_COMPUTING` | `one of numba, c, c-autopar, vendored, best-of(numba,c), best-of(numba,c,c-autopar), torch-autotune` | `"best-of(numba,c)"` | Denominator of scientific_computing. |
| `measurement.denominator.machine_learning` | `HPCAGENT_BENCH_MEASUREMENT_DENOMINATOR_MACHINE_LEARNING` | `one of numba, c, c-autopar, vendored, best-of(numba,c), best-of(numba,c,c-autopar), torch-autotune` | `"torch-autotune"` | Denominator of machine_learning. |
| `measurement.baseline_race` | `HPCAGENT_BENCH_MEASUREMENT_BASELINE_RACE` | `one of leader-first, complete` | `"leader-first"` | How best-of(numba,c) is timed. leader-first: the expected winner first, the other cut by the early stop; complete: both timed in full (grading.py). |
| `measurement.early_stop_factor` | `HPCAGENT_BENCH_MEASUREMENT_EARLY_STOP_FACTOR` | `float > 0` | `2.0` | Per-rep budget of a candidate timed after the leader = max(early_stop_floor_s, early_stop_factor x the leader's slowest rep); a rep past it cuts the candidate. |
| `measurement.early_stop_floor_s` | `HPCAGENT_BENCH_MEASUREMENT_EARLY_STOP_FLOOR_S` | `float >= 0` | `1.0` | s. Lower bound of the early-stop budget. |
| `measurement.repeat` | `HPCAGENT_BENCH_MEASUREMENT_REPEAT` | `int > 0` | `20` | Timed reps per measurement of /profile, /baseline and a promotion's first grade; /submit, /score and grade-under use their protocol's runs. |
| `measurement.local_repeat` | `HPCAGENT_BENCH_MEASUREMENT_LOCAL_REPEAT` | `int > 0` | `5` | Timed reps of /score on a distributed (MPI / ML-scaling) task, reduced with median_of_k. |
| `measurement.warmup` | `HPCAGENT_BENCH_MEASUREMENT_WARMUP` | `int >= 0` | `1` | Untimed reps before the timed reps of a submission and of every baseline; 0 = none. |
| `measurement.vary_inputs` | `HPCAGENT_BENCH_MEASUREMENT_VARY_INPUTS` | `bool` | `true` | true: each timed rep runs on a fresh draw from the cell's input pool (harness/rep_variation.py). |
| `measurement.pin_threads` | `HPCAGENT_BENCH_MEASUREMENT_PIN_THREADS` | `bool` | `true` | Pin the timing process to one thread per physical core (timing.pin_threads). |
| `measurement.timing_lock` | `HPCAGENT_BENCH_MEASUREMENT_TIMING_LOCK` | `str` | `""` | Path the Harbor verifier holds flock() on around its timed section (harbor.timing_lock); empty = no lock. The judge service does not read it. |
| `measurement.timing_backend` | `HPCAGENT_BENCH_MEASUREMENT_TIMING_BACKEND` | `one of min_of_k, median_of_k, mannwhitney_delta` | `"mannwhitney_delta"` | Reduction of a cell's runs to one speedup: min_of_k (ratio of minima), median_of_k (ratio of medians), mannwhitney_delta (ratio of medians, credited only when a Mann-Whitney U test confirms its sign, else 1.0). |
| `measurement.credited_protocol` | `HPCAGENT_BENCH_MEASUREMENT_CREDITED_PROTOCOL` | `str` | `"mw4x5"` | Grade protocol a credited number is stated under and /submit grades under (a name registered in protocols.py). |
| `measurement.quiescence.residual_ns` | `HPCAGENT_BENCH_MEASUREMENT_QUIESCENCE_RESIDUAL_NS` | `float >= 0` | `11340.0` | ns. Floor of the allowed post-clock resynchronize residual; 0 with residual_factor 0 = gate off. |
| `measurement.quiescence.residual_factor` | `HPCAGENT_BENCH_MEASUREMENT_QUIESCENCE_RESIDUAL_FACTOR` | `float >= 0` | `0.319` | Allowed residual above residual_ns as a fraction of the credited sample. |
| `measurement.quiescence.divergence_factor` | `HPCAGENT_BENCH_MEASUREMENT_QUIESCENCE_DIVERGENCE_FACTOR` | `float >= 0` | `3.47` | Maximum host-bracket / event-time ratio of one rep; 0 = gate off. |
| `measurement.quiescence.divergence_slack_ns` | `HPCAGENT_BENCH_MEASUREMENT_QUIESCENCE_DIVERGENCE_SLACK_NS` | `float >= 0` | `77164.0` | ns. Host-minus-event gap added to divergence_factor x event time. |
| `measurement.mannwhitney.p` | `HPCAGENT_BENCH_MEASUREMENT_MANNWHITNEY_P` | `float > 0, <= 1 \| null` | `null` | Significance level of the Mann-Whitney timing test; null = the credited protocol's alpha. |
| `measurement.mannwhitney.repeats` | `HPCAGENT_BENCH_MEASUREMENT_MANNWHITNEY_REPEATS` | `int > 0 \| null` | `null` | Timed runs a side the test needs; null = the credited protocol's repeat. |
| `statistics.paired_test` | `HPCAGENT_BENCH_STATISTICS_PAIRED_TEST` | `str` | `"sign-flip"` | Paired test on per-kernel log ratios of two setups. |
| `statistics.proportion_test` | `HPCAGENT_BENCH_STATISTICS_PROPORTION_TEST` | `str` | `"fisher"` | Test on two setups' solved/total runs. |
| `statistics.paired_proportion_test` | `HPCAGENT_BENCH_STATISTICS_PAIRED_PROPORTION_TEST` | `str` | `"mcnemar"` | Test on the kernels exactly one of two setups solved. |
| `statistics.correction` | `HPCAGENT_BENCH_STATISTICS_CORRECTION` | `str` | `"benjamini-hochberg"` | Multiple-comparison correction over every reported family. |
| `statistics.two_sample_test` | `HPCAGENT_BENCH_STATISTICS_TWO_SAMPLE_TEST` | `str` | `"mannwhitney_delta"` | Two-sided timing test on two setups' scored runs of one kernel (report only). |
| `perf.mode` | `HPCAGENT_BENCH_PERF_MODE` | `one of all_configs_3shapes, secret_3shapes` | `"all_configs_3shapes"` | all_configs_3shapes: fixed public timed shapes; secret_3shapes: shapes drawn from seeds.secret_shape. Both time every config. |
| `perf.n_large_shapes` | `HPCAGENT_BENCH_PERF_N_LARGE_SHAPES` | `int > 0 \| null` | `null` | Timed inputs per grade, set by a grading scope (grade_under.final_settings); null = the credited protocol's input count. |
| `preset.seed` | `HPCAGENT_BENCH_PRESET_SEED` | `int \| null` | `null` | Seed of a fuzzed:<seed> token (fuzz.base_seed); null = seeds.fuzz. |
| `repo.speedup_min` | `HPCAGENT_BENCH_REPO_SPEEDUP_MIN` | `float > 0` | `1.2` | Minimum speedup over the sequential-C baseline for a pull request to be accepted. |
| `prompt.template` | `HPCAGENT_BENCH_PROMPT_TEMPLATE` | `str` | `"task.j2"` | Top-level template rendered. |
| `prompt.template_dir` | `HPCAGENT_BENCH_PROMPT_TEMPLATE_DIR` | `str \| null` | `null` | Template directory searched before the built-in prompts/ directory; null = none. |
| `prompt.template_dirs` | `HPCAGENT_BENCH_PROMPT_TEMPLATE_DIRS` | `list[str]` | `[]` | Template directories searched after template_dir and before the built-ins, in order. |
| `prompt.debug` | `HPCAGENT_BENCH_PROMPT_DEBUG` | `bool` | `false` | Bracket the prompt with markers naming every resolved template and skill file. |
| `prompt.generator` | `HPCAGENT_BENCH_PROMPT_GENERATOR` | `str \| null` | `null` | 'module:function' replacing prompt generation, called as fn(task, *, oracle, baseline, feedback) -> str; null = the built-in renderer. |
| `prompt.inline_kernel` | `HPCAGENT_BENCH_PROMPT_INLINE_KERNEL` | `bool` | `false` | Embed the NumPy reference source in the prompt. |
| `prompt.container_workdir` | `HPCAGENT_BENCH_PROMPT_CONTAINER_WORKDIR` | `str` | `"/app"` | Mount point of the per-kernel folder in the agent container. |
| `prompt.include_translation` | `HPCAGENT_BENCH_PROMPT_INCLUDE_TRANSLATION` | `bool` | `false` | Embed a NumpyToX C/C++/Fortran translation of the reference. |
| `prompt.include_reference` | `HPCAGENT_BENCH_PROMPT_INCLUDE_REFERENCE` | `bool` | `false` | Offer the original ported source (<kernel>_reference.<ext>) when one exists. |
| `prompt.strategy` | `HPCAGENT_BENCH_PROMPT_STRATEGY` | `one of default, loopnest, profile_first, language_native` | `"default"` | Named optimization strategy of the how-to section (prompts.STRATEGIES). |
| `prompt.optimization_guidance` | `HPCAGENT_BENCH_PROMPT_OPTIMIZATION_GUIDANCE` | `bool` | `true` | Include the how-to-optimize section. |
| `prompt.profiling_guidance` | `HPCAGENT_BENCH_PROMPT_PROFILING_GUIDANCE` | `bool` | `false` | Emphasize profiling in the how-to-optimize section; strategy profile_first sets it. |
| `prompt.language_track` | `HPCAGENT_BENCH_PROMPT_LANGUAGE_TRACK` | `bool` | `false` | Emphasize optimizing idiomatically in the forced language. |
| `prompt.native` | `HPCAGENT_BENCH_PROMPT_NATIVE` | `bool` | `false` | Native framing: the reference path shown is repo-relative, not /app/... |
| `prompt.sections` | `HPCAGENT_BENCH_PROMPT_SECTIONS` | `map[str, str \| bool]` | `{}` | Section key -> off (false) \| template name \| file path (prompt_sections.py); $HPCAGENT_BENCH_PROMPT_SECTIONS_<KEY> overrides one entry. |
| `prompt.variant` | `HPCAGENT_BENCH_PROMPT_VARIANT` | `str` | `"default"` | Active named prompt variant (prompts.PROMPT_VARIANTS). |
| `prompt.variants` | `HPCAGENT_BENCH_PROMPT_VARIANTS` | `map[str, map[str, any]]` | `{}` | Extra prompt variants: name -> PromptConfig field overrides. |
| `references.prefer_committed` | `HPCAGENT_BENCH_REFERENCES_PREFER_COMMITTED` | `bool` | `false` | true: a kernel with a committed hand-written <module>_reference.<ext> (no autogen marker on line 1) is graded and timed against that file instead of the NumpyToX emit; changes the denominator. |
| `references.generated_cache_dir` | `HPCAGENT_BENCH_REFERENCES_GENERATED_CACHE_DIR` | `str` | `""` | Directory of pre-emitted reference sources; empty or missing = no cache. |
| `cache.disk_results_levels` | `HPCAGENT_BENCH_CACHE_DISK_RESULTS_LEVELS` | `list[int >= 1, <= 3]` | `[]` | Kernel levels the store serves; empty with disk_results_tracks empty = store off. |
| `cache.disk_results_tracks` | `HPCAGENT_BENCH_CACHE_DISK_RESULTS_TRACKS` | `list[one of loop_level_reasoning, scientific_computing, machine_learning]` | `[]` | Tracks the store serves whatever the level. |
| `cache.disk_results_dir` | `HPCAGENT_BENCH_CACHE_DISK_RESULTS_DIR` | `str` | `""` | Store root; empty = hpcagent-bench-judge-cache under $FAST_SCRATCH (else $SCRATCH). Holds secret-seed outputs: judge role only. |
| `attempts.max_rounds` | `HPCAGENT_BENCH_ATTEMPTS_MAX_ROUNDS` | `int > 0 \| null` | `1` | Attempts per run; null = unbounded. |
| `attempts.time_budget_s` | `HPCAGENT_BENCH_ATTEMPTS_TIME_BUDGET_S` | `float >= 0 \| null` | `null` | s. Wall clock of the attempt loop, checked before each attempt; null = unbounded. |
| `attempts.token_budget` | `HPCAGENT_BENCH_ATTEMPTS_TOKEN_BUDGET` | `int > 0 \| null` | `null` | Agent tokens (input + output, reasoning included) per kernel when no narrower layer applies; null = unbounded. scoring.resolve_token_budget: token_budget_override, else token_budget_by_level[level], else token_budget. |
| `attempts.token_budget_by_level` | `HPCAGENT_BENCH_ATTEMPTS_TOKEN_BUDGET_BY_LEVEL` | `map[int >= 1, <= 3, int > 0]` | `{"1": 500000, "2": 1000000, "3": 2000000}` | Token budget per KernelBench level (JSON object in the environment). |
| `attempts.token_budget_override` | `HPCAGENT_BENCH_ATTEMPTS_TOKEN_BUDGET_OVERRIDE` | `int > 0 \| null` | `null` | Token budget of every kernel, over every other layer; null = unset. |
| `images.cpu.agent` | `HPCAGENT_BENCH_IMAGES_CPU_AGENT` | `str` | `"docker.io/spcleth/hpcagent-bench:agent-cpu-x86_64-latest"` | Agent image reference. |
| `images.cpu.verifier` | `HPCAGENT_BENCH_IMAGES_CPU_VERIFIER` | `str` | `"docker.io/spcleth/hpcagent-bench:judge-cpu-x86_64-latest"` | Verifier image reference. |
| `images.amd.agent` | `HPCAGENT_BENCH_IMAGES_AMD_AGENT` | `str` | `"docker.io/spcleth/hpcagent-bench:agent-amd-latest"` | Agent image reference. |
| `images.amd.verifier` | `HPCAGENT_BENCH_IMAGES_AMD_VERIFIER` | `str` | `"docker.io/spcleth/hpcagent-bench:judge-amd-latest"` | Verifier image reference. |
| `images.nvidia.agent` | `HPCAGENT_BENCH_IMAGES_NVIDIA_AGENT` | `str` | `"docker.io/spcleth/hpcagent-bench:agent-nvidia-latest"` | Agent image reference. |
| `images.nvidia.verifier` | `HPCAGENT_BENCH_IMAGES_NVIDIA_VERIFIER` | `str` | `"docker.io/spcleth/hpcagent-bench:judge-nvidia-latest"` | Verifier image reference. |
| `images.mpi.agent` | `HPCAGENT_BENCH_IMAGES_MPI_AGENT` | `str` | `"docker.io/spcleth/hpcagent-bench:agent-cpu-x86_64-latest"` | Agent image reference. |
| `images.mpi.verifier` | `HPCAGENT_BENCH_IMAGES_MPI_VERIFIER` | `str` | `"docker.io/spcleth/hpcagent-bench:judge-cpu-x86_64-latest"` | Verifier image reference. |
| `grading.seal` | `HPCAGENT_BENCH_GRADING_SEAL` | `bool` | `true` | Run every grading and /profile child in the seal (seal.py): own user, mount, pid, net and ipc namespaces. false only on a host without user namespaces. |
| `grading.seal_hide` | `HPCAGENT_BENCH_GRADING_SEAL_HIDE` | `list[str]` | `[]` | Extra absolute paths hidden from the sealed child. |
| `grading.allow_agent_build_flags` | `HPCAGENT_BENCH_GRADING_ALLOW_AGENT_BUILD_FLAGS` | `bool` | `false` | false: a submission's build list contributes only -I/-D/-l/-L. true: tuning and autopar flags too; FP-semantics and dialect flags stay refused. |
| `grading.distributed_libraries` | `HPCAGENT_BENCH_GRADING_DISTRIBUTED_LIBRARIES` | `list[str]` | `["mpi", "rccl"]` | Library names a distributed-track judge (mpi.grade_distributed) honours with allow_agent_build_tokens false; every other name is refused there. |
| `grading.allow_agent_build_tokens` | `HPCAGENT_BENCH_GRADING_ALLOW_AGENT_BUILD_TOKENS` | `bool` | `true` | true: a submission's -l<name> tokens and libraries requests are honoured; false: dropped, except distributed_libraries on a distributed judge. |
| `record.enabled` | `HPCAGENT_BENCH_RECORD_ENABLED` | `bool` | `false` | Write grade rows to the results DB. |
| `record.allow_memory_db` | `HPCAGENT_BENCH_RECORD_ALLOW_MEMORY_DB` | `bool` | `false` | Accept a db_path on tmpfs/ramfs. |
| `record.study` | `HPCAGENT_BENCH_RECORD_STUDY` | `str` | `""` | Study every setup row of this run belongs to. |
| `record.model` | `HPCAGENT_BENCH_RECORD_MODEL` | `str` | `""` | Short model tag (envs/display_names.yaml). |
| `record.language` | `HPCAGENT_BENCH_RECORD_LANGUAGE` | `str` | `""` | Language the setup asked for; empty = none. |
| `record.device` | `HPCAGENT_BENCH_RECORD_DEVICE` | `one of cpu, gpu, cpu-multinode, gpu-multinode \| null` | `null` | Device class the setup declares; stamped on every row (null is stamped cpu). Non-null cpu or cpu-multinode makes task.host_only_setup refuse device submissions. |
| `record.packet` | `HPCAGENT_BENCH_RECORD_PACKET` | `str` | `""` | Skill packets, any separator; stored sorted and '+'-joined. |
| `record.setup` | `HPCAGENT_BENCH_RECORD_SETUP` | `str` | `""` | Setup name (provenance only). |
| `record.harness` | `HPCAGENT_BENCH_RECORD_HARNESS` | `str` | `""` | Agent harness that drove the setup (claude, miniswe, openhands); empty = NULL. |
| `record.temperature` | `HPCAGENT_BENCH_RECORD_TEMPERATURE` | `float >= 0` | `1.0` | Sampling temperature the serving engine applied. |
| `record.commit` | `HPCAGENT_BENCH_RECORD_COMMIT` | `str` | `""` | hpcagent_bench commit the setup was submitted from; empty = unknown. |
| `record.db_path` | `HPCAGENT_BENCH_RECORD_DB_PATH` | `str` | `"results/hpcagent_bench.db"` | Results DB (SQLite); relative = under the repo root; refused on tmpfs/ramfs unless allow_memory_db. |
| `record.expensive_gates` | `HPCAGENT_BENCH_RECORD_EXPENSIVE_GATES` | `list[str]` | `[]` | Anti-cheat gates labelled expensive that run (anticheat.py); every other gate always runs. |
| `record.speedup_suspect_above_host` | `HPCAGENT_BENCH_RECORD_SPEEDUP_SUSPECT_ABOVE_HOST` | `float > 0` | `2000.0` | A host-residency speedup above this is flagged suspect (still recorded). |
| `record.speedup_suspect_above_device` | `HPCAGENT_BENCH_RECORD_SPEEDUP_SUSPECT_ABOVE_DEVICE` | `float > 0` | `16000.0` | A device-residency speedup above this is flagged suspect (still recorded). |
| `record.physical_bandwidth_gbps_device` | `HPCAGENT_BENCH_RECORD_PHYSICAL_BANDWIDTH_GBPS_DEVICE` | `float > 0` | `10600.0` | GB/s. A device-residency grade whose native_ns < declared bytes / this is flagged suspect (scoring.floor_bandwidth_gbps). |
| `record.physical_bandwidth_gbps_host` | `HPCAGENT_BENCH_RECORD_PHYSICAL_BANDWIDTH_GBPS_HOST` | `float > 0` | `10600.0` | GB/s. The same bound for a host-residency grade. |
| `record.log_attempts` | `HPCAGENT_BENCH_RECORD_LOG_ATTEMPTS` | `bool` | `true` | Log non-verified submissions to the attempts table. |
| `record.log_calls` | `HPCAGENT_BENCH_RECORD_LOG_CALLS` | `bool` | `true` | Log every served grade (/score and /submit) to the calls table. |
| `record.execution` | `HPCAGENT_BENCH_RECORD_EXECUTION` | `one of native, container` | `"native"` | Where a framework-results runtime was measured (frameworks/test.py). |
| `record.build` | `HPCAGENT_BENCH_RECORD_BUILD` | `str` | `""` | Build label stamped on every framework-results row; empty = NULL. |
| `perf_reports.vect_cost_model` | `HPCAGENT_BENCH_PERF_REPORTS_VECT_COST_MODEL` | `one of default, unlimited` | `"default"` | unlimited switches the vectorizer cost model off in the report compile only (gcc -fvect-cost-model=unlimited); the timed build never changes. |
| `flags.fp_associative` | `HPCAGENT_BENCH_FLAGS_FP_ASSOCIATIVE` | `bool` | `false` | true adds -fassociative-math to every compile column; moves the baseline, so rows graded with true and false do not pool. Read once at import of flags.py; later overrides and scoped environments do not change it. |
| `timeouts.kernel_s` | `HPCAGENT_BENCH_TIMEOUTS_KERNEL_S` | `float >= 0` | `300.0` | s. Per-kernel agent run when no narrower layer applies. scoring.resolve_kernel_timeout: kernel_s_override, else the manifest timeout_s, else kernel_s_by_level[level], else kernel_s. |
| `timeouts.kernel_s_by_level` | `HPCAGENT_BENCH_TIMEOUTS_KERNEL_S_BY_LEVEL` | `map[int >= 1, <= 3, float >= 0]` | `{"1": 180.0, "2": 300.0, "3": 600.0}` | s. Per-kernel agent run per KernelBench level (JSON object in the environment). |
| `timeouts.kernel_s_override` | `HPCAGENT_BENCH_TIMEOUTS_KERNEL_S_OVERRIDE` | `float >= 0 \| null` | `null` | s. Per-kernel agent run of every kernel, over every other layer; null = unset. |
| `timeouts.guillotine_factor` | `HPCAGENT_BENCH_TIMEOUTS_GUILLOTINE_FACTOR` | `float >= 0` | `2.0` | Candidate per-rep cap = max(guillotine_floor_s, guillotine_factor x its baseline), at most the kernel timeout; 0 = off. A stopped correct candidate is credited baseline / cap. |
| `timeouts.guillotine_floor_s` | `HPCAGENT_BENCH_TIMEOUTS_GUILLOTINE_FLOOR_S` | `float >= 0` | `1.0` | s. Lower bound of the guillotine cap. |
| `timeouts.cpf_render_s` | `HPCAGENT_BENCH_TIMEOUTS_CPF_RENDER_S` | `float >= 0` | `14400.0` | s. Wall clock of one canonical-parallel-form render. |
| `limits.oracle_cache_gb` | `HPCAGENT_BENCH_LIMITS_ORACLE_CACHE_GB` | `float > 0` | `12.0` | GB. Per-process bound on the reference-output cache (scoring.py). |
| `limits.kernel_memory_gb` | `HPCAGENT_BENCH_LIMITS_KERNEL_MEMORY_GB` | `float > 0` | `20.0` | GB. Floor of the per-kernel RLIMIT_AS budget above the harness baseline (sizing.kernel_memory_gb), and the budget of a kernel without declarative shapes; host kernels only. |
| `limits.reference_node_fraction` | `HPCAGENT_BENCH_LIMITS_REFERENCE_NODE_FRACTION` | `float > 0, <= 1` | `0.75` | Cap of a judge-owned reference as a fraction of its rank's share of node RAM, never below kernel_memory_gb (sizing.reference_memory_gb). |
| `limits.thread_stack_mb` | `HPCAGENT_BENCH_LIMITS_THREAD_STACK_MB` | `int > 0` | `512` | MiB. Stack of every OpenMP thread a timed child starts (OMP_STACKSIZE). |
| `sparse.bsr_block_sizes` | `HPCAGENT_BENCH_SPARSE_BSR_BLOCK_SIZES` | `list[int > 0]` | `[2, 4, 8]` | Square block edges a bsr request may name; sparse dimensions are drawn as multiples of their lcm. |
| `mpi.compute_hint` | `HPCAGENT_BENCH_MPI_COMPUTE_HINT` | `bool` | `false` | Add the local-compute optimization paragraph to the ML-layout prompt section (sections/mpi.j2). |
| `mpi.grade_distributed` | `HPCAGENT_BENCH_MPI_GRADE_DISTRIBUTED` | `bool` | `false` | true: /score and /submit grade a kernel with an mpi: block at residency distributed through mpi_call; a kernel without one grades single-node. |
| `mpi.launcher` | `HPCAGENT_BENCH_MPI_LAUNCHER` | `list[str]` | `["mpiexec.mpich", "-launcher", "fork", "-n"]` | argv prefix that takes the rank count next; the runner appends [ranks, program...]. |
| `mpi.compilers` | `HPCAGENT_BENCH_MPI_COMPILERS` | `map[str, str]` | `{}` | Language -> MPI compiler wrapper of the distributed build; empty = the compilers.yaml MPI blocks. |
| `mpi.ranks` | `HPCAGENT_BENCH_MPI_RANKS` | `int > 0` | `4` | Rank count of the scalar distributed speedup. |
| `mpi.ranks_per_node` | `HPCAGENT_BENCH_MPI_RANKS_PER_NODE` | `int > 0` | `4` | Ranks one node of a grade-under gang places. |
| `mpi.rank_counts` | `HPCAGENT_BENCH_MPI_RANK_COUNTS` | `list[int > 0]` | `[]` | Rank counts of the scaling curve; empty = no curve (the scalar at ranks only). |
| `mpi.leaderboard_preset` | `HPCAGENT_BENCH_MPI_LEADERBOARD_PRESET` | `one of S, M, L, XL` | `"XL"` | Single-node base preset of the distributed metric. |
| `mpi.mode` | `HPCAGENT_BENCH_MPI_MODE` | `one of strong, weak` | `"strong"` | Scaling law of the distributed metric. |
| `mpi.k_repeats` | `HPCAGENT_BENCH_MPI_K_REPEATS` | `int > 0` | `5` | Timed repeats inside the driver of a scaling-curve point. |
| `mpi.residency` | `HPCAGENT_BENCH_MPI_RESIDENCY` | `one of host, device` | `"host"` | Location of each rank's tiles at the kernel boundary for an array whose manifest sets no location. device: copied to the GPU before and back after the kernel, untimed; a c/cpp/fortran kernel is a scored config error. host: sandbox.build_mpi links no kernel library, so run_sharded refuses c/cpp/hip submissions. |
| `mpi.launch_timeout_s` | `HPCAGENT_BENCH_MPI_LAUNCH_TIMEOUT_S` | `float >= 0` | `120.0` | s. Cap on one MPI launch (a scored failure); also the --time of a gang step. |
| `mpi.env` | `HPCAGENT_BENCH_MPI_ENV` | `map[str, str]` | `{"HWLOC_COMPONENTS": "-opencl,-levelzero,-gl"}` | Extra environment of the launch process (JSON object in the environment). |
| `ml.datatype` | `HPCAGENT_BENCH_ML_DATATYPE` | `str` | `"bf16"` | Datatype of a machine_learning kernel that declares no storage precision; empty = service.datatype. |
| `ml.torch_cache_root` | `HPCAGENT_BENCH_ML_TORCH_CACHE_ROOT` | `str` | `""` | Inductor/Triton cache root; empty = $SCRATCH/hpcagent-bench-inductor-cache. |
| `ml.torch_work_root` | `HPCAGENT_BENCH_ML_TORCH_WORK_ROOT` | `str` | `""` | Node-local torch-autotune working cache; empty = <system temp>/hpcagent-bench-torch-autotune. |
| `ml.torch_archive_root` | `HPCAGENT_BENCH_ML_TORCH_ARCHIVE_ROOT` | `str` | `""` | Archive of torch-autotune working caches; empty = <torch_cache_root>/archives. |
| `ml.kernelbench_dir` | `HPCAGENT_BENCH_ML_KERNELBENCH_DIR` | `str` | `""` | KernelBench corpus root; empty = <repo>/third_party/KernelBench/KernelBench. |
| `ml.torch_baseline_timeout_s` | `HPCAGENT_BENCH_ML_TORCH_BASELINE_TIMEOUT_S` | `float >= 0` | `1800.0` | s. Cap on one torch-autotune child. |
| `ml.rank_counts` | `HPCAGENT_BENCH_ML_RANK_COUNTS` | `list[int > 0]` | `[1, 2, 4]` | ML-track rank counts when mpi.rank_counts is empty. |
| `ml.grade_rank_counts` | `HPCAGENT_BENCH_ML_GRADE_RANK_COUNTS` | `list[int > 0]` | `[1, 2, 4, 8, 16]` | Rank counts grade-under asks of an item whose task scales. |
| `runtime.backend` | `HPCAGENT_BENCH_RUNTIME_BACKEND` | `one of oci, sif, ce, native, docker, podman, apptainer` | `"oci"` | Container backend: a family (oci, sif, ce, native), resolved to its first installed member (oci: docker, then podman), or a runtime (containers.resolve_backend). |
| `runtime.mp_context` | `HPCAGENT_BENCH_RUNTIME_MP_CONTEXT` | `one of auto, fork, spawn, forkserver` | `"auto"` | multiprocessing start method; auto = fork on Linux, spawn on macOS (osinfo.mp_context). |
| `runtime.omp_context_root` | `HPCAGENT_BENCH_RUNTIME_OMP_CONTEXT_ROOT` | `str` | `"/opt/omp"` | Directory of the image's OpenMP contexts, one per toolchain family (omp_context.py). |
| `runtime.omp_catalog` | `HPCAGENT_BENCH_RUNTIME_OMP_CATALOG` | `str` | `""` | OpenMP runtime catalog file; empty = <omp_context_root>/catalog.json. |
| `build.ccache` | `HPCAGENT_BENCH_BUILD_CCACHE` | `bool` | `true` | Prefix C and C++ compile steps with ccache when it is on PATH. |
| `build.compiler.c` | `HPCAGENT_BENCH_BUILD_COMPILER_C` | `str \| null` | `null` | Family pin of C (a languages.COMPILER_FAMILIES key). |
| `build.compiler.cpp` | `HPCAGENT_BENCH_BUILD_COMPILER_CPP` | `str \| null` | `null` | Family pin of C++ (a languages.COMPILER_FAMILIES key). |
| `build.compiler.fortran` | `HPCAGENT_BENCH_BUILD_COMPILER_FORTRAN` | `str \| null` | `null` | Family pin of Fortran (a languages.COMPILER_FAMILIES key). |
| `judge.gpus_per_node` | `HPCAGENT_BENCH_JUDGE_GPUS_PER_NODE` | `int >= 0 \| null` | `null` | GPU slots; null = the local GPU count (judge_scheduler.judge_gpu_slots). |
| `judge.cpu_slots_per_node` | `HPCAGENT_BENCH_JUDGE_CPU_SLOTS_PER_NODE` | `int >= 0 \| null` | `null` | CPU slots; null = 1 when the judge has no GPU slot, else 0. |
| `agent.workers` | `HPCAGENT_BENCH_AGENT_WORKERS` | `int > 0 \| null` | `null` | Concurrent agent workers; null = max(len(vllm_urls), len(judge_urls), 1). |
| `agent.execution` | `HPCAGENT_BENCH_AGENT_EXECUTION` | `one of native, container, harbor` | `"container"` | Run mode of hpcagent-bench agent. |
| `oracle.compile_timeout_s` | `HPCAGENT_BENCH_ORACLE_COMPILE_TIMEOUT_S` | `float >= 0` | `75.0` | s. Per-backend compile cap. |
| `oracle.polycc_timeout_s` | `HPCAGENT_BENCH_ORACLE_POLYCC_TIMEOUT_S` | `float >= 0` | `360.0` | s. Cap on one polycc (Pluto) run. |
| `oracle.numba_fastmath` | `HPCAGENT_BENCH_ORACLE_NUMBA_FASTMATH` | `bool` | `false` | Compile numba backends with fastmath=True. |
| `oracle.overrides` | `HPCAGENT_BENCH_ORACLE_OVERRIDES` | `map[str, OracleOverride]` | `{"cloudsc": {"compile_timeout_s": 45.0, "numba_fastmath": null, "polycc_timeout_s": null}, "gromacs_nbnxm": {"compile_timeout_s": 30.0, "numba_fastmath": null, "polycc_timeout_s": null}, "jacobi_2d_tile_4lvl_silly": {"compile_timeout_s": 30.0, "numba_fastmath": null, "polycc_timeout_s": null}}` | Kernel short name -> per-kernel values (JSON object in the environment). |
| `cpf.cache` | `HPCAGENT_BENCH_CPF_CACHE` | `str` | `""` | Cache root an on-demand render writes to; empty = no on-demand render. |

<!-- END generated by helpers/scripts/gen_config_docs.py -->
