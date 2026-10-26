# NestForge Loop Moves@14

Fourteen kernels for the stage 2 (loop-nest moves) of [NestForge](https://github.com/spcl/NestForge), the SDFG optimizer
that ships as an HPCAgent-Bench plugin ([plugins.md](../../docs/extending/plugins.md#example-nestforge)). Tag:
[`nestmoves14`](../../hpcagent_bench/tags/nestmoves14.txt). Each kernel is a fixed DaCe program, a NumPy reference, and a
measured schedule that beats NestForge's default stage 2 (`default_moves()`) on a stated device. The question the set
asks: can an agent driving NestForge's moves find a schedule the deterministic default misses, when the right move is
not the first greedy step and a plausible wrong one is on the menu?

**No experiment yet.** The set has no setup in `setups.yaml` and no entry in `envs/studies.yaml`; the ratios below
come from NestForge's own harness, not from a judge grade (see "Measurement").

**This page contains the solutions.** Studies are never mounted for an agent; do not hand this page to one whose task
is to find the schedule.

## How the kernels were selected

**The criteria.** A kernel is kept when all four hold, measured at preset M:

1. **The default is beaten** on a stated device and thread count, by a ratio recorded with its repeats.
2. **Finding the schedule takes reasoning.** The winning move is not the first greedy step: either it is offered only after an earlier move, or it pays off only together with a later one, or the right choice flips with the device or thread count.
3. **A decoy exists:** a legal, plausible move or sequence that is slower than the reference. Its time is measured.
4. **The oracle checks the result.** The baseline and the reference match the NumPy oracle at preset S, and their outputs match each other at preset M.

A score from a search, or the number of moves, is not a criterion.

**Where the kernels came from.**

- **The HPCAgent-Bench corpus.** All 438 kernels were scanned at stage 1 on 2026-10-21 (DaCe `801cafd`), after a first probe of candidates on 2026-10-16. For each, the scan recorded what `list_moves` offers after stage 1 and what `default_moves()` already takes. Most corpus kernels fail criterion 1 or 2: stage 1 already produces the good loop order, or the default takes the only useful move. Three kernels passed unedited (`tsvc_2_s231`, `tsvc_2_s233`, and `dbcsr`, which needed small-block presets: `dbcsr_small_blocks`).
- **Corpus kernels rewritten as explicit loop nests.** Five kernels keep the corpus kernel's NumPy math and arguments, and use a hand-written DaCe program (`<kernel>_dace.py`, a hand override) in the loop order a direct C or GPU port uses. Written as NumPy (`@`, `einsum`, slices), these kernels become library nodes or are already in a good order, so stage 2 has nothing to do.
- **Cut from the CloudSC dwarf** (`ecmwf-ifs/dwarf-p-cloudsc`), which is not in the corpus in this form. Each takes one physics section in the GPU loop order the dwarf's GPU ports use, with source line references in its oracle.
- **Synthetic.** Three kernels (two in the list) were written to test a mechanism no corpus or CloudSC kernel offered.

**Targets.** Each kernel was timed on CPU at 1 and 72 threads and on GPU. The target is where the gap is measured; the solutions below also state the other devices' answers.

**Presets.** The gap exists only where the default wastes a resource that the schedule recovers: idle threads, kernel launches, cache reuse or padded tests. Preset M is chosen so that resource is scarce. For four kernels (`dbcsr_small_blocks`, `cloudsc_ntrac`, `scattering_self_energies_m2`, `_m3`) the corpus or first preset hid the gap; the sweep that set the new one is in each kernel's record in NestForge (`benchmarking_creation/<kernel>/<kernel>.md`).

**Rejected.** Kernels tried and dropped: `vadv`, `adi` (corpus form), `bout_arakawa`, `aes_graupel`, `twin_sweep`, `guarded_twin_sweep`, `layout_flip_fuse`, `rk4_ensemble`, `seissol_tensor_contraction` (corpus form), `cloudsc_satadj`, and, as one-move candidates, `tsvc_2_s275`, `tsvc_2_s1119` (after stage 1 it is the same program as `tsvc_2_s231`), `tsvc_2_s162` and `fission_scatter_2body`. In each case either the default already takes the move, stage 1 already solves the kernel, no move is offered, or the gain stays under the default's noise.

## The benchmarks

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
| [`column_physics`](../../hpcagent_bench/benchmarks/scientific_computing/structured_grids/column_physics/) | synthetic, on CloudSC column structure | GPU loop order on CPU: each column walks the levels at stride `KLON` | 9 | CPU 1 thread | 4.51x |

Moves counts the stage 2 moves of the reference, not the `finish_moves()` calls. "Collapse" means the reference kernel is built with `#pragma omp parallel for collapse(3)`, against the default's best build `collapse(2)`.

Not in the list: `cloudsc_kcache_cpu` (CloudSC; its route gains 1.13x) and `adi_column_sweep` (its 10.7x route needs `MapFusionVertical` to fuse recurrence rows; without that the best reachable route is 1.45x). `two_pass_scope` (synthetic, `set_scope`): on a GPU target stage 4 places the scope's kernel on the CPU, and the CPU default beats it. `cpu_gpu_flip` and `sweep_scale` are templates.

## Solutions and decoys

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
- **Solution (CPU, 1 thread):** nine moves in dependent order. `subgraph-fission`, `interchange-map-loop` (tracers), `map-fission`, `interchange-map-loop` (cover), `loop-fusion`, `interchange-map-map`, then three `map-fusion`. Four of the first six gain nothing alone and only make a later move legal. 4.51x.
- **Decoys:** re-fusing cover and tracers, which is what the default does (1.07x); the tracer GPU order (0.55x); stopping after four moves (3.39x); the undo after the route (3.43x).

## Measurement

The ratios are NestForge's replay, not an HPCAgent-Bench grade: stage 1, the reference moves (or `default_moves()`),
then stages 3 to 5 with the default kernel build; the program is checked against the NumPy reference at preset S and
against the default's output at preset M, and timed at preset M (median of 5 calls after one warm-up, host time, one
exclusive GH200 node). A judge grade runs at `fuzzed` sizes around XL under `mw4x5` and can differ. Kernels whose
DaCe program is hand-written commit it as a hand override (`.gitignore` exception), so the program NestForge starts
from is the one measured; the other five use the generated `<kernel>_dace.py`.

## Reference routes

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
  "column_physics": {"moves": 9, "device": "cpu", "threads": 1, "collapse": 0, "ratio": 4.51, "reference": [["subgraph-fission", ["kernel1_0", "state2_0"]], ["interchange-map-loop", ["kernel1_2", "for2_1"]], ["map-fission", ["kernel2_0"]], ["interchange-map-loop", ["kernel1_1", "for2_0"]], ["loop-fusion", ["for0_0", "for0_1"]], ["interchange-map-map", ["kernel3_0", "kernel4_0"]], ["map-fusion", ["kernel2_1", "kernel2_2"]], ["map-fusion", ["kernel2_1", "kernel2_2"]], ["map-fusion", ["kernel2_1", "kernel2_2"]], "finish"]}
}
```
