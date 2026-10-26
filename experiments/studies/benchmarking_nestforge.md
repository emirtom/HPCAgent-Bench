# Benchmarking NestForge: Loop Moves@14

Fourteen kernels and the experiment that runs agentic [NestForge](https://github.com/spcl/NestForge) on them. Tag:
[`nestmoves14`](../../hpcagent_bench/tags/nestmoves14.txt). NestForge ships as an HPCAgent-Bench plugin
([plugins.md](../../docs/extending/plugins.md#example-nestforge)): an agent optimizes a whole DaCe program through
tools (read the loop-nest tree, list and apply legal moves, scope kernels, rewrite and tune kernels, measure), and
every phase also has a deterministic default. This page says what the set is for, how it was chosen and organized,
what each kernel tests (with its solution and decoys), and how the experiment is run and analyzed.

**Status:** kernels, tag and this page are in place; no experiment is defined in `setups.yaml` or
`envs/studies.yaml` yet (section 5 lists what is missing). The ratios come from NestForge's own harness, not from a
judge grade (section 6).

**This page contains the solutions.** Studies are never mounted for an agent; do not hand this page to one whose task
is to find the schedule.

## 1. The question

NestForge's stage 2 reshapes loop nests with legal moves (fusion, fission, interchange). Its default,
`default_moves()`, is a fixed recipe. The set asks two things of an agent that drives the same moves:

1. **Does it produce faster code than the default?** Only where the default is not good enough is an agent worth
   its cost, so every kernel here is one where the default is measurably beaten.
2. **How does it decide?** Which moves it takes, in which order, which wrong ones it tries, and whether it notices
   and recovers. A good final time reached through a lucky or wasteful path should be visible as such.

The known best schedule of each kernel is a lower bound, not an answer key: an agent that beats it scores higher.

## 2. How the set was chosen and organized

### 2.1 What makes a kernel a test

A kernel is kept when all four hold, measured at preset M:

1. **The default is beaten** on a stated device and thread count, by a ratio recorded with its repeats.
2. **Finding the schedule takes reasoning.** The winning move is not the first greedy step: it is offered only after an earlier move, or it pays off only together with a later one, or the right choice flips with the device, the thread count or the size.
3. **A decoy exists:** a legal, plausible move or sequence that is slower than the reference. Its time is measured.
4. **The oracle checks the result.** The default and the reference match the NumPy reference at preset S, and each other at preset M.

A score from a search, or the number of moves, is not a criterion. Textbook cases are mostly not tests: stage 1 (canonicalization) already solves most single textbook moves, so kernels had to be found or written where the deterministic choice is wrong.

### 2.2 Where the kernels came from

- **The HPCAgent-Bench corpus, unedited (2):** `tsvc_2_s231`, `tsvc_2_s233`. All 438 corpus kernels were scanned at stage 1 (2026-10-21); most fail criterion 1 or 2 because stage 1 already produces the good loop order or the default takes the only useful move.
- **A corpus kernel at new sizes (1):** `dbcsr_small_blocks`, `dbcsr` with 4x4 blocks at density 0.05. The corpus presets (22-32 blocks, density 0.25) hide the gap (1.04x).
- **Corpus kernels rewritten as explicit loop nests (5):** `scattering_self_energies_m2`, `_m3`, `seissol_volume`, `conv2d_bias_relu`, `conv3d_box`. They keep the corpus math and add a hand-written DaCe program (`<kernel>_dace.py`, a hand override) in the loop order a direct C or GPU port uses. Written as NumPy (`@`, `einsum`, slices), these become library nodes or are already in a good order, and stage 2 has nothing to do.
- **Cut from the CloudSC dwarf (4):** `cloudsc_lateral_drift` (with a synthetic drift), `cloudsc_nproma`, `cloudsc_ntrac`, `cloudsc_solver_carry`. One physics section each, in the loop order the dwarf's GPU ports use, with source lines in the reference.
- **Synthetic (2):** `graupel_sedim2b`, `column_physics`, written for a mechanism no corpus or CloudSC kernel offered.

### 2.3 How the set is organized

**Tier, by the length of the known best route** (a property recorded, not a requirement): easy (1 move) 2 kernels, medium (2 to 5 moves) 9, hard (more than 5, or a long chain of moves that only enable later ones) 3.

**Lever: the resource the default wastes and the schedule recovers.** Each preset is chosen so that resource is scarce.

| Lever | What the default does | Kernels |
| --- | --- | --- |
| parallel regions and launches | opens one parallel region, or launches one GPU kernel, per iteration of an outer loop | `tsvc_2_s231`, `graupel_sedim2b`, `cloudsc_lateral_drift`, `cloudsc_ntrac` |
| parallel width | parallelizes a loop with fewer iterations than threads | `scattering_self_energies_m2`, `scattering_self_energies_m3`, `conv2d_bias_relu` |
| locality and reuse | walks memory at a large stride, or re-reads operands per output | `cloudsc_nproma`, `column_physics`, `tsvc_2_s233`, `seissol_volume`, `conv3d_box` |
| redundant work | evaluates a loop-invariant test inside the inner loop | `dbcsr_small_blocks` |
| live state | keeps a whole solver's state live in one loop body | `cloudsc_solver_carry` |

**Context: the target, and whether the answer flips with it.** Two kernels target the GPU (`cloudsc_lateral_drift`, `cloudsc_ntrac`), six a single CPU thread, seven 72 CPU threads (`tsvc_2_s233` both). In nine of the fourteen the best schedule on another device or thread count differs from the target's, and in four the target's route is 6x to 33x slower on the other target (`tsvc_2_s231`: 3.29x at 72 threads, 0.08x on one thread; `cloudsc_lateral_drift`, `cloudsc_nproma`, `cloudsc_ntrac`), so an agent that applies one rule everywhere is caught.

**Kind of reasoning required** (several per kernel; the solutions in section 4 say which):

- *enabling moves*: a move that gains nothing alone and makes a later one legal (`cloudsc_nproma`, `seissol_volume`, `conv3d_box`, `cloudsc_solver_carry`, `column_physics`, `cloudsc_ntrac`, `tsvc_2_s233`);
- *moves offered only later*: the winning move is not on stage 1's menu (`graupel_sedim2b`, `cloudsc_solver_carry`);
- *a backwards-looking first step*: `scattering_self_energies_m3`'s route starts with an interchange that pushes the parallel map in, then pulls it out;
- *stopping at the right point*: a prefix of the route already pays, and stopping there loses the rest (`graupel_sedim2b`, `conv3d_box`, `cloudsc_ntrac`), or continuing past the best point gains nothing (`conv2d_bias_relu`, `seissol_volume`);
- *context*: the same move is right or wrong by device or thread count (above).

### 2.4 What was left out

**Rejected kernels.** Kernels tried and dropped: `vadv`, `adi` (corpus form), `bout_arakawa`, `aes_graupel`, `twin_sweep`, `guarded_twin_sweep`, `layout_flip_fuse`, `rk4_ensemble`, `seissol_tensor_contraction` (corpus form), `cloudsc_satadj`, and, as one-move candidates, `tsvc_2_s275`, `tsvc_2_s1119` (after stage 1 it is the same program as `tsvc_2_s231`), `tsvc_2_s162` and `fission_scatter_2body`. In each case either the default already takes the move, stage 1 already solves the kernel, no move is offered, or the gain stays under the default's noise.

Not in the set, with folders kept in NestForge's `benchmarking_creation/`: `cloudsc_kcache_cpu` (its route gains 1.13x), `adi_column_sweep` (its 10.7x route needs a `MapFusionVertical` extension; without it the best reachable route is 1.45x), `two_pass_scope` (on a GPU target stage 4 places the scope's kernel on the CPU, and the CPU default beats it). Two slots of the original plan stay open: a one-move kernel whose move is not an interchange of a recurrence, and `adi_column_sweep` once its fusion is available.

## 3. The fourteen kernels, measured

Ratio = default time / reference time at preset M: whole-program host time, median of 5 calls after one warm-up, one exclusive GH200 node (72 CPU threads, or its GPU). Measured 2026-10-23 at DaCe `042413d` and HPCAgent-Bench `2666c06`.

| Kernel | Origin | Bottleneck in the default | Moves | Target | Ratio |
| --- | --- | --- | --- | --- | --- |
| [`tsvc_2_s231`](../../hpcagent_bench/benchmarks/loop_level_reasoning/tsvc_2_s231/) | HPCAgent-Bench, unedited | the recurrence loop sits outside the parallel map: one parallel region per row | 1 | CPU 72 threads | 3.29x |
| [`dbcsr_small_blocks`](../../hpcagent_bench/benchmarks/scientific_computing/sparse_linear_algebra/dbcsr_small_blocks/) | HPCAgent-Bench `dbcsr`, small-block presets | a loop-invariant padding test runs inside the inner block loop | 1 | CPU 1 thread | 2.53x |
| [`cloudsc_lateral_drift`](../../hpcagent_bench/benchmarks/scientific_computing/structured_grids/cloudsc_lateral_drift/) | synthetic, on CloudSC flux code | two drift sweeps launch one GPU kernel per column | 2 | GPU | 1.92x |
| [`tsvc_2_s233`](../../hpcagent_bench/benchmarks/loop_level_reasoning/tsvc_2_s233/) | HPCAgent-Bench, unedited | two statements with opposite carried axes are fused in one nest | 2 (1 thread), 3 (72 threads) | CPU | 5.47x, 2.30x |
| [`cloudsc_nproma`](../../hpcagent_bench/benchmarks/scientific_computing/structured_grids/cloudsc_nproma/) | CloudSC dwarf | GPU loop order on CPU: each column walks the levels at stride `nproma` | 2 | CPU 1 thread | 2.35x |
| [`graupel_sedim2b`](../../hpcagent_bench/benchmarks/scientific_computing/structured_grids/graupel_sedim2b/) | synthetic | 2160 parallel regions per call (substep x species x level) | 2 | CPU 72 threads | 19.7x |
| [`scattering_self_energies_m2`](../../hpcagent_bench/benchmarks/scientific_computing/dense_linear_algebra/scattering_self_energies_m2/) | HPCAgent-Bench, rewritten | only `(k, E)` is parallel: 8 points for 72 threads | 2 + collapse | CPU 72 threads | 9.78x |
| [`cloudsc_ntrac`](../../hpcagent_bench/benchmarks/scientific_computing/structured_grids/cloudsc_ntrac/) | CloudSC dwarf | 272 per-level GPU kernel launches, two recurrences in one level loop | 3 | GPU | 2.29x |
| [`seissol_volume`](../../hpcagent_bench/benchmarks/scientific_computing/dense_linear_algebra/seissol_volume/) | HPCAgent-Bench, rewritten | output-stationary order: operator rows are re-read for every output entry | 3 | CPU 1 thread | 1.62x |
| [`conv2d_bias_relu`](../../hpcagent_bench/benchmarks/machine_learning/conv2d_bias_relu/) | HPCAgent-Bench, rewritten | only `n` is parallel (8 of 72 threads), and the output row is not innermost | 4 | CPU 72 threads | 14.9x |
| [`scattering_self_energies_m3`](../../hpcagent_bench/benchmarks/scientific_computing/dense_linear_algebra/scattering_self_energies_m3/) | HPCAgent-Bench, rewritten | only `(k, E)` is parallel (16 points), and the atom map sits under 3 reduction loops | 5 + collapse | CPU 72 threads | 5.08x |
| [`conv3d_box`](../../hpcagent_bench/benchmarks/scientific_computing/structured_grids/conv3d_box/) | HPCAgent-Bench, rewritten | 125 scattered reads per grid point, no reuse along the contiguous axis | 5 | CPU 1 thread | 1.89x |
| [`cloudsc_solver_carry`](../../hpcagent_bench/benchmarks/scientific_computing/structured_grids/cloudsc_solver_carry/) | CloudSC dwarf | the whole 5x5 solver in one column body, so its state stays live across every statement | 7 | CPU 72 threads | 2.86x |
| [`column_physics`](../../hpcagent_bench/benchmarks/scientific_computing/structured_grids/column_physics/) | synthetic, on CloudSC column structure | GPU loop order on CPU: each column walks the levels at stride `KLON` | 8 | CPU 1 thread | 4.51x (9-move route; not re-timed) |

Moves counts the stage 2 moves of the reference, not the `finish_moves()` calls. "Collapse" means the reference kernel is built with `#pragma omp parallel for collapse(3)`, against the default's best build `collapse(2)`.


## 4. Solutions and decoys

Labels such as `kernel2_0` are from the measured runs and change between versions; the solutions name every move by what it reads and writes.

### `tsvc_2_s231`

- **Program:** TSVC `s231`, `aa[j, i] = aa[j-1, i] + bb[j, i]`. The recurrence runs along `j` (strided); `i` (contiguous) is independent. Stage 1 gives `for j { map i }`, one parallel region per `j`.
- **Solution (CPU, 72 threads):** `interchange-loop-map`: the `j` loop goes inside the `i` map, giving one parallel region. 3.29x.
- **Decoy:** the same move on 1 thread (0.08x), where each thread then walks a column at stride `LEN_2D`. On GPU the default already makes the interchange.

### `dbcsr_small_blocks`

- **Program:** DBCSR block-sparse multiply as a sort-merge join over block slots. The test "is this A slot padding?" sits inside the B-slot loop, so it runs `n_a * n_b` times.
- **Solution (CPU, 1 thread):** `interchange-loop-if` hoists the test above the B-slot loop. 2.53x with 4x4 blocks at density 0.05. The corpus presets (32x32 blocks, density 0.25) give 1.04x, because the block products hide the test.
- **Decoys:** the undo, `interchange-if-loop` (0.84x); `map-fission` of the block product (1.00x).

### `cloudsc_lateral_drift`

- **Program:** rain sedimentation down the levels, then an east and a west upwind drift along the columns, then a masked pick. On GPU stage 1 leaves each drift as a host loop over columns around a map over levels, so each column is one kernel launch.
- **Solution (GPU):** `interchange-loop-map` on both drift sweeps. 1.92x.
- **Decoys:** one sweep only (1.00x); the same route on CPU (0.03x at 72 threads).

### `tsvc_2_s233`

- **Program:** TSVC `s233`, one `i` loop holding two `j` loops. `aa` is carried along `j` and `bb` along `i`, so the two statements want opposite loop orders.
- **Solution:** `loop-fission`, then:
  - 1 thread: `interchange-loop-map` on the `bb` half (5.47x);
  - 72 threads: on both halves (2.30x).
  The two interchanges are offered side by side, with equal static scores.
- **Decoys:** the `aa` interchange on 1 thread (0.46x); the fission alone (0.87x and 0.59x); one interchange at 72 threads (0.88x, 0.99x).

### `cloudsc_nproma`

- **Program:** CloudSC precipitation fall and cover in the driver's blocked NPROMA layout, in the GPU loop order: `map block { map column { for level } }`.
- **Solution (CPU, 1 thread):** `map-fission` of the column map, then `interchange-map-loop` of the body map and the level loop, so each level walks a contiguous `nproma` run. 2.35x. The fission gains nothing alone (1.00x); it makes the interchange legal.
- **Decoys:** the fission alone (1.00x); the undo interchange (1.00x); the route on GPU (0.16x).

### `scattering_self_energies_m3`

- **Program:** OMEN's electron-phonon self-energy (via NPBench). The atom map `a` sits under three reduction loops (`q`, `w`, `i`), and the 10x10 products are explicit loops.
- **Solution (CPU, 72 threads):**
  - Moves: `interchange-map-loop` on `a` and `b` (it looks backwards), `interchange-loop-map` back (this round trip unlocks the pulls), then three `interchange-loop-map` pulls of `a`. `(k, E, a)` becomes one parallel map.
  - Build: the kernel with `collapse(3)`.
  - Result: 5.08x over the default's best build. Preset M has 16 `(k, E)` points, fewer than the threads, so only the route's map fills them.
- **Decoys:** the first move alone (1.16x as built); the round trip (1.12x); the full route without `collapse(3)` (1.11x).

### `graupel_sedim2b`

- **Program:** sedimentation of 3 precipitation species with per-column substeps, `for substep { for species { for level { map column } } }`. The default opens 2160 parallel regions per call.
- **Solution (CPU, 72 threads):** `interchange-loop-map` of the level loop into the column map, then of the species loop (offered only after the first). 16 regions remain. 19.7x, and 15.1x in a repeat on another node.
- **Decoy:** the first pull alone. It already gives 15x, and the route beats it 1.28x to 1.33x, so an agent that stops once a move pays off stops one move short. On 1 thread the first pull is the best schedule.

### `cloudsc_ntrac`

- **Program:** `NTRAC` tracers sediment down the levels and a precipitation cover runs down the same levels, in one level loop. No interchange is legal while that loop holds two maps.
- **Solution (GPU):** `loop-fission`, `interchange-loop-map` (cover), `finish_moves()`, which collapses the tracer maps into one 2-D map, then `interchange-loop-map` (tracers), `finish_moves()`. The 272 per-level launches become 2. 2.29x at 1031 columns.
- **Decoys:**
  - half the route (1.37x);
  - `loop-fusion` right after the fission (it undoes the fission);
  - on CPU the full route (0.14x on 1 thread). At 72 threads the half route is the best CPU schedule (1.55x).

### `scattering_self_energies_m2`

- **Program:** the same kernel as `_m3`, with `a` under two reduction loops.
- **Solution (CPU, 72 threads):**
  - Moves: two `interchange-loop-map` pulls of `a`.
  - Build: the kernel with `collapse(3)`.
  - Result: 9.78x over the default's best build (8 `(k, E)` points at preset M). With `_m3` it grades the same lever at two route depths.
- **Decoys:** one pull (1.11x as built); `interchange-map-loop` on `a` and `b`, which pushes `b` out of `a` (1.17x as built; it does not compile with `collapse(3)`).

### `conv2d_bias_relu`

- **Program:** one convolution layer with bias and ReLU, written one output element at a time: init, the `ci`, `kh`, `kw` sums, then ReLU. From the HPCAgent-Bench `machine_learning` conv2d family.
- **Solution (CPU, 72 threads):** `map-fission` of the `ow` map (init, sum and ReLU become three maps), `interchange-map-loop` of `ow` inside `ci`, then inside `kh`, then `interchange-map-map` of `n` and `co`. The default parallelises only over `n` (8 iterations). 14.9x. On 1 thread the first three moves (1.98x); on GPU the fission alone (108x).
- **Decoys:** the fission alone on CPU (0.83x); the `oh`/`ow` swap (0.89x); continuing to the 7-move route (no gain); the CPU interchanges on GPU (43x to 73x against 108x).

### `seissol_volume`

- **Program:** SeisSol's ADER-DG volume update, `upd = einsum("dkl,blq,dqp->bkp")` then `Q += dt * upd`, written one `(b, k, p)` entry at a time. From HPCAgent-Bench `seissol_tensor_contraction`, whose einsum form leaves stage 2 nothing to do.
- **Solution (CPU, 1 thread):** `map-fission` of the `p` map (init from sum), then `interchange-map-loop` of `p` inside `l`, then inside `d`. 1.62x (1.45x at 72 threads). On GPU the fission alone (54x).
- **Decoys:** the fission alone on CPU (1.00x); continuing to 6 moves (no CPU gain, 0.15x on GPU); the `b`/`k` swap (0.78x at 72 threads).

### `cloudsc_solver_carry`

- **Program:** CloudSC section 5.2 cloud solver (5x5 matrix, non-pivoting LU, substitution) and the vertical precipitation-flux carry, in one column loop.
- **Solution (CPU, 72 threads):** `loop-fission`, `subgraph-fission`, two `map-fission`, two `interchange-map-map`, then `interchange-loop-map`. The last move is not on the stage 1 menu; it appears only after the fissions. 2.86x (2.39x on 1 thread).
- **Decoys:** the two stage 1 `map-fusion` (1.02x); a repeated `subgraph-fission` that fills the graph with empty maps; the two map fissions alone (2.46x at 72 threads, close to the route). No GPU gain (1.00x).

### `conv3d_box`

- **Program:** a 125-tap 3-D box stencil on an edge-padded grid, written one grid point at a time. From HPCAgent-Bench `structured_grids/conv_3d`.
- **Solution (CPU, 1 thread):** `map-fission` of `k`, `interchange-map-loop` of `k` inside `di`, `map-fission` of `j`, `interchange-map-loop` of `j` inside `di`, then of `k` inside `dj`. Each fission is offered only before its pushes and gains nothing alone (0.99x). 1.89x (1.72x at 72 threads). On GPU one `subgraph-fission` (34x).
- **Decoys:** stopping after the `k` pushes (1.53x); the map swaps (0.96x to 1.00x); the CPU route on GPU (0.84x).

### `column_physics`

- **Program:** a column model in the GPU order, column loop outermost: a cloud-cover recurrence and a tracer recurrence down the levels, then a carried diagnostic pass.
- **Solution (CPU, 1 thread):** eight moves in dependent order. `subgraph-fission`, `interchange-map-loop` (tracers), `map-fission`, `interchange-map-loop` (cover), `loop-fusion`, `interchange-map-map`, then two `map-fusion` (at DaCe `042413d` a third `map-fusion` was offered and taken; from `8a4d824` on, none is offered after the second and the eight-move program matches the reference). Four of the first six gain nothing alone and only make a later move legal. 4.51x with the nine-move route of 2026-10-23.
- **Decoys:** re-fusing cover and tracers, which is what the default does (1.07x); the tracer GPU order (0.55x); stopping after four moves (3.39x); the undo after the route (3.43x).

## 5. The experiment: agentic NestForge on the set

### 5.1 Arms

Each arm runs every kernel at its target (section 3): device, thread count and preset.

| Arm | Who decides stage 2 | Tools | Role |
| --- | --- | --- | --- |
| **D**: default | nobody (no LLM): `run_defaults`, then `finalize` | - | the denominator: what NestForge does without an agent |
| **R**: reference | nobody: the route of section 7 replayed, then `finalize` | - | the known best under the judge's clock; checks that each gap survives grading |
| **A-moves**: agent, stage 2 only | an LLM agent | `get_cpf`, `view_tree`, `list_moves`, `apply_move`, `run_defaults`, `get_performance`, `finalize` | the main treatment: decisions over the same moves the default chooses from |
| **A-blind**: agent, stage 2, no feedback | an LLM agent | as A-moves, without `get_performance` and without the performance `apply_move` returns | does measuring each move lead to better decisions than reasoning from the tree alone |
| **A-full**: agent, every phase | an LLM agent | the full `nestforge` packet (adds `set_scope`, `unset_scope`, `list_kernels`, `read_kernel`, `write_kernel`, `tune_kernel`) | do later phases help; the only arm that can reach `collapse(3)` (by rewriting the kernel) |

Comparisons: A-moves against D (does the agent add value), against R (how close, or better), against A-blind
(value of feedback), and A-full against A-moves (value of the other phases). An agent without NestForge (writing C
directly, as in `scicomp40`) is not an arm: it answers a different question.

### 5.2 Protocol

- **Start:** every episode starts from stage 1's output of the kernel's DaCe program, as the plugin does
  (`nestforge.server.start`), on the target device.
- **Prompt:** the packet's method text and the kernel's task. No hint about the lever, the route or the decoys.
- **Size:** preset M, the size the gaps were built and measured at (requirement R1).
- **Submission:** single mode: one graded `/submit` of the `finalize` library. `get_performance` is the agent's
  measurement; `/score` is not needed and is off.
- **Repeats:** 5 agents per kernel and setup, so a decision can be told apart from luck (the bench's `repeat5` study
  measures that spread for other agents).
- **Models:** the study's served models (for example `oss120b`, `qwen38`, `glm53`), each with the same budget.
- **Budget per episode:** a token budget and a wall-time limit large enough for about 30 measured moves (a build of
  one variant took 8 s to 3 min at preset S on 2026-10-26, more at M; NestForge's `TOOL_TIMEOUT_S` is 30 min). Proposed: 4M tokens, 4 h. The budget
  is part of the cost, not a hidden constraint: an agent that measures every legal move pays for it.

### 5.3 What is recorded and measured

Per episode:

| Measure | From | Tells |
| --- | --- | --- |
| correct | the judge (`mw4x5`) | a wrong program scores zero, however fast |
| speedup over D | judge grades of the episode and of D | whether the agent adds value |
| ratio to R | judge grades | how close to the known best, or whether it beat it |
| decision trace | the plugin's tool-call log: each call, its arguments (kind, labels, epoch), its answer and its performance | the sequence of decisions |
| decoys taken | the trace against section 4's decoys, matched by move kind and the arrays the move reads and writes (labels change between versions) | which tempting wrong moves the agent fell for |
| wasted moves | moves undone later, moves that changed nothing, moves the agent measured and then discarded | how directly it got there |
| stopping | stopped on a strict prefix of the reference (too early), or went past the best measured point (too late) | whether it knows when to stop |
| cost | billed tokens, tool calls, builds (calls that time the program), wall time | what the answer cost |

### 5.4 Analysis

- Per kernel, per arm: speedup over D and ratio to R, median and spread over the 5 repeats.
- Grouped by tier, lever, context (target device; whether the answer flips) and kind of reasoning (section 2.3):
  where do agents beat the default, and where not.
- The decision traces: which decoys are taken, which enabling moves are missed, how often agents stop early or
  overshoot, and whether the reason an agent states for a move matches what `get_performance` showed.
- A-moves against A-blind for the value of feedback; A-full against A-moves for the value of the other phases, and
  whether A-full finds `collapse` on the two scattering kernels.

### 5.5 What is missing before the first run

| | Requirement | Why |
| --- | --- | --- |
| R1 | grade at preset M (a setup knob for the timed preset), or presets whose XL keeps the scarcity | the judge times `fuzzed` sizes around XL; some gaps exist only at M (`scattering_self_energies_m2` has 8 `(k, E)` points at M and 1224 at XL) |
| R2 | thread count per kernel: one setup per target (CPU 1 thread, CPU 72 threads, GPU), each over its sub-list of the tag | the judge runs with every core; seven kernels target one thread |
| R3 | a `nestforge-moves` packet in NestForge's `plugin/nestforge/plugin.toml` with the A-moves tools, and a switch that withholds performance for A-blind | today's packet serves every tool |
| R4 | GH200 nodes (NVIDIA GPU, Grace CPU) | the ratios were measured there; the two GPU kernels have only NVIDIA numbers |
| R5 | the plugin's tool-call log in the results DB (or the episode folder), with epochs and performance | the decision trace is half of the result |
| R6 | D and R as runnable optimizers (a flavor, or a scripted agent that calls the plugin) | they are the baselines every agent arm is compared with |
| R7 | the two kernels HPCAgent-Bench's own smoke run still fails: `cloudsc_lateral_drift` (its C translation disagrees with numba) and `conv2d_bias_relu` (no PyTorch reference in `kernelbench_map.yaml` for the machine-learning track) | every arm's grade needs the bench's reference and baseline |

Already checked: NestForge loads all fourteen kernels from this layout, and the default and the reference route of
each build and match the NumPy reference at S on the new pins (2026-10-26).

### 5.6 Order of work

1. **Pilot P1:** arms D and R through the judge on every kernel at its target, 3 repeats. Keep a kernel only if R
   still beats D under `mw4x5`; a gap that disappears is reported, not hidden.
2. **Pilot P2:** A-moves with one model on one kernel per tier (`tsvc_2_s231`, `graupel_sedim2b`,
   `cloudsc_solver_carry`), 1 agent each: checks the tools, the budget and the trace log end to end.
3. **Main run:** every arm, every kernel, 5 agents, every model.
4. **Analysis** as in 5.4.

Proposed launch once R1 to R6 exist (a new experiment `nestforge14` in `setups.yaml`, one per target):

```bash
BASE=nestforge14-cpu72 TAG=nestmoves14-cpu72 EXPERIMENT=nestforge14 RECORD_STUDY=nestforge14 \
    MODELS="oss120b qwen38 glm53" LANGUAGES=c PACKETS="nestforge-moves nestforge-blind nestforge" REPEAT=5 \
    ../hpcagent_bench/cluster/submit.sh $J
```

## 6. How the ratios were measured

The ratios are NestForge's replay, not an HPCAgent-Bench grade: stage 1, the reference moves (or `default_moves()`),
then stages 3 to 5 with the default kernel build; the program is checked against the NumPy reference at preset S and
against the default's output at preset M, and timed at preset M (median of 5 calls after one warm-up, host time, one
exclusive GH200 node). A judge grade runs at `fuzzed` sizes around XL under `mw4x5` and can differ. Kernels whose
DaCe program is hand-written commit it as a hand override (`.gitignore` exception), so the program NestForge starts
from is the one measured; the other five use the generated `<kernel>_dace.py`.

## 7. Reference routes

A step is `"finish"` (`finish_moves()`), `[kind, labels]` for a move (labels `null`: the first offer of that kind), or
`["set_scope", labels]`. Labels are from the 2026-10-23 runs at DaCe `042413d` and change between versions; the
solutions above name every move by what it reads and writes. `collapse`: the CPU kernels are built with
`#pragma omp parallel for collapse(N)`.

```json
{
  "tsvc_2_s231": {"moves": 1, "device": "cpu", "threads": 72, "collapse": 0, "ratio": 3.29, "reference": [["interchange-loop-map", ["for0_0", "kernel2_0"]], "finish"]},
  "cloudsc_lateral_drift": {"moves": 2, "device": "gpu", "threads": 0, "collapse": 0, "ratio": 1.92, "reference": [["interchange-loop-map", null], ["interchange-loop-map", null], "finish"]},
  "tsvc_2_s233": {"moves": 2, "device": "cpu", "threads": 1, "collapse": 0, "ratio": 5.47, "reference": [["loop-fission", ["for0_0"]], ["interchange-loop-map", ["for0_1", "kernel2_1"]], "finish"]},
  "cloudsc_nproma": {"moves": 2, "device": "cpu", "threads": 1, "collapse": 0, "ratio": 2.35, "reference": [["map-fission", ["kernel2_0"]], ["interchange-map-loop", ["kernel3_6", "for4_0"]], "finish"]},
  "scattering_self_energies_m3": {"moves": 5, "device": "cpu", "threads": 72, "collapse": 3, "ratio": 5.08, "baseline_collapse": 2, "reference": [["interchange-map-loop", ["kernel7_0", "for8_0"]], ["interchange-loop-map", ["for6_0", "kernel8_0"]], ["interchange-loop-map", ["for5_0", "kernel7_0"]], ["interchange-loop-map", ["for4_0", "kernel6_0"]], ["interchange-loop-map", ["for3_0", "kernel5_0"]], "finish"]},
  "graupel_sedim2b": {"moves": 2, "device": "cpu", "threads": 72, "collapse": 0, "ratio": 19.7, "reference": [["interchange-loop-map", ["for2_0", "kernel4_0"]], ["interchange-loop-map", ["for1_0", "kernel3_0"]], "finish"]},
  "cloudsc_ntrac": {"moves": 3, "device": "gpu", "threads": 0, "collapse": 0, "ratio": 2.29, "reference": [["loop-fission", ["for0_0"]], ["interchange-loop-map", ["for0_1", "kernel2_1"]], "finish", ["interchange-loop-map", ["for0_0", "kernel2_0"]], "finish"]},
  "scattering_self_energies_m2": {"moves": 2, "device": "cpu", "threads": 72, "collapse": 3, "ratio": 9.78, "baseline_collapse": 2, "reference": [["interchange-loop-map", ["for4_0", "kernel6_0"]], ["interchange-loop-map", ["for3_0", "kernel5_0"]], "finish"]},
  "conv2d_bias_relu": {"moves": 4, "device": "cpu", "threads": 72, "collapse": 0, "ratio": 14.9, "reference": [["map-fission", ["kernel4_0"]], ["interchange-map-loop", ["kernel5_2", "for6_0"]], ["interchange-map-loop", ["kernel6_0", "for7_0"]], ["interchange-map-map", ["kernel1_0", "kernel2_0"]], "finish"]},
  "seissol_volume": {"moves": 3, "device": "cpu", "threads": 1, "collapse": 0, "ratio": 1.62, "reference": [["map-fission", ["kernel3_0"]], ["interchange-map-loop", ["kernel4_1", "for5_0"]], ["interchange-map-loop", ["kernel5_0", "for6_0"]], "finish"]},
  "cloudsc_solver_carry": {"moves": 7, "device": "cpu", "threads": 72, "collapse": 0, "ratio": 2.86, "reference": [["loop-fission", ["for2_0"]], ["subgraph-fission", ["kernel1_0", "state2_0"]], ["map-fission", ["kernel1_2"]], ["map-fission", ["kernel2_4"]], ["interchange-map-map", ["kernel2_0", "kernel3_1"]], ["interchange-map-map", ["kernel2_1", "kernel3_2"]], ["interchange-loop-map", ["for2_0", "kernel4_0"]], "finish"]},
  "conv3d_box": {"moves": 5, "device": "cpu", "threads": 1, "collapse": 0, "ratio": 1.89, "reference": [["map-fission", ["kernel3_0"]], ["interchange-map-loop", ["kernel4_1", "for5_0"]], ["map-fission", ["kernel2_0"]], ["interchange-map-loop", ["kernel3_1", "for4_0"]], ["interchange-map-loop", ["kernel6_0", "for7_0"]], "finish"]},
  "dbcsr_small_blocks": {"moves": 1, "device": "cpu", "threads": 1, "collapse": 0, "ratio": 2.53, "reference": [["interchange-loop-if", ["for1_0", "if2_0"]], "finish"]},
  "column_physics": {"moves": 8, "device": "cpu", "threads": 1, "collapse": 0, "ratio": 4.51, "reference": [["subgraph-fission", ["kernel1_0", "state2_0"]], ["interchange-map-loop", ["kernel1_2", "for2_1"]], ["map-fission", ["kernel2_0"]], ["interchange-map-loop", ["kernel1_1", "for2_0"]], ["loop-fusion", ["for0_0", "for0_1"]], ["interchange-map-map", ["kernel3_0", "kernel4_0"]], ["map-fusion", ["kernel2_1", "kernel2_2"]], ["map-fusion", ["kernel2_1", "kernel2_2"]], "finish"]}
}
```
