# The requests API

**Status.** Implemented: `libraries`, `precision`, `sparse`, and MPI distribution as layout op lists
(`grid` + `layout` with `block` / `unblock` / `permute` / `distribute`, and `location`), with the
results schema at version 7. Parsed and refused as not implemented yet: `pad` / `unpad`, `shuffle`,
`linear_shuffle`, `zip` / `unzip`, single-node layout changes and distributing sparse formats (§4.6).
The layout ops, their JSON, inverses and lowering live in relayout (`relayout.ops`,
github.com/ThrudPrimrose/relayout, a pinned dependency); hb keeps the block
(`hpcagent_bench/harness/requests_api.py`) and the casts (`hpcagent_bench/harness/precision_casts.py`).

A submission asks the judge to change how its data reaches the kernel through one `requests`
object in the envelope. Four kinds of request exist:

| Request | Changes | Conversion charged to | Status |
| --- | --- | --- | --- |
| `precision` | the element type of named arrays | separate timer, not scored | implemented |
| `libraries` | which catalog libraries the judge links | nothing (build time) | implemented (moved from top level) |
| `sparse` | the format of each sparse array | separate timer, not scored | implemented (moved from `sparse_config`) |
| `layout` | how an array's elements are placed in memory | separate timer, not scored | the distribution subset (§4.5) |
| `distribute` (an op inside `layout`) | which rank owns which elements | untimed scatter, as before | implemented |

The old top-level `libraries`, `distribution` and `sparse_config` fields are refused with a 400
that names their new spelling.

```json
{
  "language": "c",
  "source": "...",
  "requests": {
    "libraries": ["openblas"],
    "precision": {"A": "fp32", "x": "bf16"},
    "grid": [4],
    "layout": {
      "A": [{"op": "block", "dim": 0, "factor": "32*P0"}, {"op": "block", "dim": 0, "factor": 32},
            {"op": "permute", "perm": [0, 2, 3, 1]}, {"op": "distribute", "digit": 1, "grid_dim": 0}],
      "x": []
    }
  }
}
```

Requests are checked before anything is built. A request the kernel cannot honour is a **request
fault**: the judge answers 400 and the submission is not spent, the same as an unoffered library
or a bad sparse format today.

## 1. Precision

`precision` maps an array name to a `Precision` value (`fp64`, `fp32`, `fp16`, `bf16`,
`fp8_e4m3`, `fp8_e5m2`). Every array not named stays in the **problem precision**, the sweep's
`--datatype`.

Refused:
- a name that is not a floating array argument of the binding (scalars, integer and index
  arrays, sparse index arrays);
- a complex array (no low-precision complex format exists);
- a request equal to the array's problem dtype (`fp64` on an fp64 problem). A no-op request is
  refused rather than ignored.

Per timed rep:

1. The inputs are staged at the problem precision (untimed, as today).
2. **Cast-in:** every requested array is converted into a buffer of the requested type,
   allocated untimed. Outputs too: the ABI declares every output `intent(inout)`, so a kernel may
   read one.
3. **Kernel:** the bracket timed today. Only this is the score.
4. **Cast-out:** each requested output or inout array is converted back to the problem precision.
5. Readback and grading as today, at the **problem precision's tolerance band**. The agent
   carries the accuracy risk of narrowing.

Steps 2 and 4 are timed with their own brackets and recorded (`convert_in_ns`, `convert_out_ns`
in §5). They never enter the speedup.

**Cast kernels** are harness-owned, one kernel per array:
- **CPU:** a C library with one `#pragma omp parallel for simd schedule(static)` loop per type
  pair, built once **per OpenMP context** (gnu / llvm, `omp_context`) with that family's C compiler,
  cached by content hash, and copied beside the submission's library so the sealed child can load
  it. The timed process therefore maps the submission's own OpenMP runtime and no second one.
- **GPU:** a `cupy.ElementwiseKernel` per type pair. fp16 uses cupy `float16`; bf16 and fp8 ride
  in `uint16` and `uint8` buffers.
- **Conversions:** one expression table (`precision_casts.cast_expr`) for both, through the
  emitter's `FP8_HELPERS` for bf16 and the fp8 pair (bit-exact against ml_dtypes); a storage format
  converts through float32.

Measured on one MI300A slot (24 cores, 16M-128M elements, fp64<->fp32): threaded numpy 26-32 GB/s,
the OpenMP loop 150-200 GB/s; on the GPU cupy reaches 2.3-2.5 TB/s. Hence OpenMP on the host and
cupy on the device.

**MPI:** the generated C driver inlines the same loops and converts each rank's host tiles around
every repeat, in MAX-over-ranks `MPI_Wtime` brackets of their own; the outfile (wire version 2)
carries the per-repeat cast times beside the kernel's, and the grade records those of the credited
repeat. Not covered yet, and refused before the build: python deliveries, the ML track (its ranks
generate their own shards) and device-resident MPI tiles.

**The C ABI follows the request.** A requested array's pointer is the requested C type
(`float *`, `_Float16 *`, `__npb_bf16 *`, ...); the entry keeps its name. The prompt's requests
section tables the C type of every precision.

## 2. Libraries

`requests.libraries` replaces the top-level `libraries` field, which is deleted. The semantics
are unchanged: each name must be one `languages.library_offered` accepts for the submission's
language, else the request is refused (`sandbox.catalog_refusal`).

## 3. Layout primitives

The five primitives of the SC'26 layout paper (pad, permute, block, shuffle, zip). Each is
injective, so any composition keeps physical addresses distinct. The algebra is the one in dace
extended `dace/libraries/layout/algebra.py`; the request schema and its lowering are relayout's
`relayout.ops`, which depends on dace and which hb depends on.

**One registered class per op.** A layout op is a frozen dataclass registered under its JSON name
(`@layout_op`); it declares its fields, its inverse and its effect on the digit tuple. A field's type
names its JSON codec, so parsing, writing back and the "not implemented yet" refusal all come from the
registry, and adding an op is one class. `zip` and `unzip` change how many arrays there are, so they
are group ops of their own registry (`@group_op`), listed under their own request keys, with no
inverse link between them.

**Dense arrays only.** A `layout` sequence on a sparse array is refused; sparse arrays change
layout through their format (`requests.sparse`) and are distributed per format (§4.6).

An array starts as its row-major **digit tuple**, one digit `(dim, stride, extent)` per
dimension. A digit's value is `(i[dim] // stride) % extent`. Ops rewrite the tuple, and the final
tuple, read left to right, is the storage order.

Dimensions are numpy's: every buffer the harness hands a kernel is C-contiguous, and the ops act in
that row-major index space. A Fortran kernel declares the same buffer with reversed extents (numpy
`A[M, N]` is Fortran `A(N, M)`), so its column-major view reads the same memory; reversing the
dimensions and the accesses is a codegen step after the transformations, never an op.

| Op | JSON | Effect on the digit tuple | Inverse |
| --- | --- | --- | --- |
| pad | `{"op": "pad", "dim": d, "amount": a}` | grows the coarsest digit of `d` by `a` dead cells | `unpad` |
| unpad | `{"op": "unpad", "dim": d, "amount": a}` | drops `a` dead cells from the coarsest digit of `d` | `pad` |
| permute | `{"op": "permute", "perm": [...]}` | `new[k] = old[perm[k]]`, over digit positions | permute by `perm⁻¹` |
| block | `{"op": "block", "dim": d, "factor": b}` | splits the finest digit of `d` into outer `(d, s·b, ⌈e/b⌉)` in place and inner `(d, s, b)` appended | `unblock` |
| unblock | `{"op": "unblock", "dim": d, "factor": b}` | merges `d`'s inner digit of extent `b` back into its outer partner | `block` |
| shuffle | `{"op": "shuffle", "dims": [d, ...], "f": [...], "f_inv": [...]}` | renumbers the indices of `dims` by the bijection `f`, one expression per dimension over `i0, i1, ...` | the same op with `f` and `f_inv` swapped |
| linear_shuffle | `{"op": "linear_shuffle", "dims": [d, ...], "bits": [b, ...], "matrix": [[0/1, ...], ...]}` | renumbers the indices of `dims` (extents `2^b`) by an invertible linear map over GF(2) on their bits | the inverse matrix, computed by the judge |
| distribute | `{"op": "distribute", "digit": k, "grid_dim": g}` | §4 | undone by the gather |

| Group op | JSON (top level of `requests`) | Effect |
| --- | --- | --- |
| zip | `"zip": [{"name": n, "fields": [a, b, ...]}]` | fuses arrays of one logical shape into one array `n` of C structs, one member per field in order |
| unzip | `"unzip": [{"name": n, "fields": [a, b, ...]}]` | splits the struct array `n` into one array per member; a complex array is the struct `(re, im)` |

Unzips run before zips. Arrays an unzip creates are ordinary arrays: they take a `precision`
(so `complex128` can become two fp32 arrays) and a `layout`; a new name that clashes with a kernel
argument is refused. `zip(re, im)` gives back the complex interleave, and `zip(im, re)` swaps it.

The agent writes only the forward sequence. The judge derives the inverse sequence (reversed,
each op inverted) for the way back.

**Shuffle.** The paper's shuffle maps the whole index vector, so each output index may read every
shuffled index: an XOR swizzle `col' = col ^ (row % 8)` is `"dims": [0, 1], "f": ["i0", "i1 ^ (i0 % 8)"]`.
`f` and `f_inv` are integer expressions over `i0, i1, ...`, the size symbols and `P0, P1, ...`, using
`+ - * // % ^ & | << >> min max` with Python's floor semantics; relayout parses and evaluates them
itself, never with `eval`. The agent supplies `f_inv`, and before the build the judge checks
`f_inv(f(i)) == i` over the padded extent at every measured rank count (O(N), untimed; a failure is a
request fault). `linear_shuffle` is the form of Linear Layouts (Zhou et al., arXiv:2505.23819): the
input bit vector is each dimension's bits, least significant first, in `dims` order, and row `r` of
the matrix is output bit `r` as the XOR of the input bits it marks. Its inverse is the inverse
matrix, and a singular matrix is refused when the request is parsed, so no O(N) check is needed.

`factor`, `amount` and every extent may be expressions over size symbols and grid symbols, so one
request stays valid at every rank count a scaling run measures. The judge checks every measured
`P` before building.

**Dead cells.** `pad`, and a `block` whose factor does not divide its extent, create cells that
hold no element. Inputs see them zero-filled. Outputs in them are ignored. The kernel still
receives each array's live extents as the size symbols it gets today, so it can skip dead cells
where zero is not neutral (max, division).

### 3.1 Zip and the C ABI

Zip emits a plain C struct with data members only: no methods and no classes. The kernel gets one
pointer to it in place of the field pointers. An unzip does the reverse: the kernel gets one pointer
per member in place of the struct array's.

```c
typedef struct { float u; double v; } hb_zip_uv;   /* field order = request order */
void kernel(const hb_zip_uv *restrict uv, ..., void *workspace, int64_t workspace_bytes);
```

- **Field types** are the per-array precisions after `precision` (§1).
- **Padding and alignment** follow the C rules for the struct. The judge allocates it as a numpy
  structured dtype with `align=True`, which has the same `offsetof` for every field.
- **Fortran** sees `type, bind(C) :: hb_zip_uv; real(c_float) :: u; real(c_double) :: v; end type`.
- **Signature position:** the struct pointer takes the position of its first field, and the other
  fields' parameters are dropped.
- **Zip group rules:** the fields take no layout of their own; the zipped array's `layout` (keyed by
  its `name`) is the group's, and its distribution.
- **Roles:** an input field is cast and relaid in. An output field is cast and relaid out.

## 4. Distribution as a layout op

MPI distribution is one more op in the same sequence, not a separate request.

| Op | JSON | Effect |
| --- | --- | --- |
| distribute | `{"op": "distribute", "digit": k, "grid_dim": g}` | digit `k` (extent `P_g`) becomes rank coordinate `g` and leaves the local digit tuple; a whole undivided dimension is split into `P_g` load-balanced bands instead |

- Ops **before** a `distribute` shape ownership: they decide which global elements each rank
  gets.
- Ops **after** a `distribute` shape each rank's local storage.
- **Replication:** a grid dimension that no `distribute` of an array names replicates that array
  across it. An array with no `distribute` is fully replicated, as `{"replicated": true}` is today.
- **Scatter and gather (design target):** the judge evaluates the composed map once per element.
  Rank `r` receives the elements whose distributed digits equal `r`'s grid coordinates, already in
  its local layout and its requested precision. One fused pack kernel runs per array per rank, and
  the gather uses the inverse. Today the lowered distribution is scattered by the descriptor and the
  precision casts run on each rank's tiles after it (§1).

### 4.1 Today's schemes, lowered

For axis `d` of extent `N` on grid dimension `g` of size `P`:

| Scheme (today's `distribution`) | Op sequence |
| --- | --- |
| block, ScaLAPACK style (`⌈N/P⌉` per rank, last rank short; not lowered yet) | `block(d, ⌈N/P⌉)` then `distribute(outer, g)` |
| block, load-balanced (hb's `block`, `_block_bounds`: parts differ by one) | `distribute(d's whole digit, g)`: a whole undivided dimension is split into load-balanced bands. (In the pure algebra it is `pad`, a `shuffle` that spreads the dead cells, `block(d, ⌈N/P⌉)`, `distribute(outer, g)`; the shorthand lowers to the same tiles.) |
| cyclic | `block(d, P)` then `distribute(inner, g)` |
| block-cyclic `b` | `block(d, b·P)`, then `block(d, b)` (splits the finest `(d, 1, b·P)` into `(d, b, P)` and `(d, 1, b)`), then `distribute((d, b, P), g)`. The local tuple `[(d, bP, ⌈N/bP⌉), (d, 1, b)]` is exactly ScaLAPACK's local storage. |
| replicated axis | no `distribute` on `d` |

The old `distribution` JSON is not accepted any more. A kernel's default distribution is shown to
the agent as these op sequences (`relayout.ops.ops_for_axes`), and a schema-6 grade's recorded
distribution is folded into them (`results_db.fold_legacy_grade`). The refusals keep their meaning
on the lowered ops:
- one array dimension per distributed grid dimension (`DigitTuple.distribute`);
- the equal-edge hypercube rule for cyclic;
- `layout_flexible_allowlist` and `replicatable_allowlist`.

The one quasi-affine shuffle of the load-balanced block:

```
base, rem = N // P, N % P;  m = ceil(N / P)
f(i)     = i + max(0, (i - rem*m) // base)      live i < N
f_inv(j) = j - max(0, j // m - rem)             live slots
dead i in [N, P*m) -> slot (rem + i - N)*m + base
```

### 4.2 ScaLAPACK coverage

`distribute` covers every distribution ScaLAPACK supports for dense descriptors (DTYPE=1). For one dimension with `MB` (or `NB`), source process
`RSRC` (or `CSRC`) and leading dimension `LLD`:

1. `block(d, MB·P)` then `block(d, MB)`: block-cyclic as above.
2. `RSRC ≠ 0`: ScaLAPACK gives global block `g` to process `(g + RSRC) % P` and keeps its local
   index `g // P`. A shift `i + RSRC·MB` would move the local index, so it is wrong. The right
   shuffle rotates within each group of `P` blocks:

   ```
   f(i)     = ((g // P)·P + (g + RSRC) % P)·MB + i % MB,  g = i // MB
   f_inv(j) = the same with -RSRC (mod P)
   ```

   This is a bijection on the padded extent, quasi-affine (floor and mod only), and is applied
   before the blocks.
3. `distribute(P-digit, g)`.
4. ScaLAPACK's column-major local storage of the numpy matrix is `permute` after the distribute.
   A Fortran kernel needs none: its reversed-extent declaration of the row-major tile is already
   column-major, over the transposed matrix (so it calls ScaLAPACK with the transposed operation).
5. `LLD > LOCr` is `pad` on the local row digit after the distribute.
6. BLACS row-major or column-major process ordering is a property of the grid, not of an array:
   today's grid is row-major (`Grid.coords_of`); a `grid_order` field is an open item (§6).

**Completeness beyond ScaLAPACK.** Take any partition of `[N]` into `P` parts whose largest part
has `m` elements. It is `pad(d, P·m − N)`, then `shuffle(d, σ)`, then `block(d, m)`, then
`distribute(outer, g)`, where `σ` sends each element to slot `owner·m + local` and fills the
remaining slots with the dead cells. So any ownership function the agent can write as an integer
expression with its inverse is expressible. The cost is O(N) σ evaluations in the untimed scatter.

**Not expressible:**
- **halos / ghost cells:** an element on two ranks is not injective. They are excluded, as today
  ("no ghost cells"). A later non-injective `halo(d, w)` op would sit outside this algebra.
- **sparse descriptors** (ScaLAPACK DTYPE 501/502 band and out-of-core), which hb has no kernels
  for.

### 4.3 Rules the validator enforces

1. Every op type-checks against the current tuple: the dim or digit exists, a permutation has
   the tuple's length, unblock has a partner digit.
2. A `distribute` digit's extent equals `P_g` at every measured rank count.
3. Each grid dimension is distributed at most once per array.
4. Every shuffle passes the exhaustive inverse check.
5. All extents are positive at every measured `P`.
6. Zip fields share one shape (after padding to the maximum, as in the paper).
7. `layout` ops apply to dense arrays only; a sparse array is distributed through §4.6.
8. `unzip` applies to struct (today: complex) arrays only.
9. MPI collectives inside the kernel are the agent's: the judge relays only scatter and gather.
   (DaCe's `MpiPackUnpack` refuses shuffled halo buffers, but that limits generated halo
   exchanges, not the judge's scatter.)

### 4.4 Ordering of the three conversions

Distribution and layout form one ordered sequence (§4), so the only free question is where the
cast goes:
- precision is elementwise and commutes with every op except zip;
- precision is therefore defined on the logical arrays, before zip, and a zip field's type is
  that array's requested precision;
- the cast is fused into the relayout pack and unpack kernels: one kernel per array (per zip
  group) each way, `dst[p] = convert(src[L⁻¹(p)])`, writes contiguous, dead cells zero.

### 4.5 What lowers today

`block`, `unblock`, `permute` and `distribute` (the ops whose class defines `apply`), where every distributed digit is a whole undivided
dimension (load-balanced block) or a digit of extent `P_g` sitting between a coarse digit of stride
`s·P_g` and, for `s > 1`, a fine digit `(d, 1, s)` (cyclic, block-cyclic of width `s`), and the
rank-local digits end row-major (otherwise the refusal names the `permute` to append). That is
every distribution hb graded before plus ScaLAPACK's without source offsets. `pad`, `unpad`,
`shuffle`, `linear_shuffle`, `zip`, `unzip`, a local relayout and arbitrary shaping sequences are
refused as not implemented yet; they follow once relayout generates the pack kernels.

### 4.6 Sparse arrays: distribution only, one converter per format

A sparse array takes no layout ops. It is distributed along the one axis its format stores
row-wise, through a per-format converter. The converter goes from the stored canonical CSR to
each rank's local array in the requested format.

| Format | Split axis | Each rank receives |
| --- | --- | --- |
| csr | 0 (rows) | CSR of its owned rows: local `indptr` from 0, column indices global |
| csc | 1 (columns) | CSC of its owned columns: local `indptr` from 0, row indices global |
| coo | 0 or 1 (the request names it) | the entries whose row (or column) it owns, in canonical order |
| bsr:b | 0 (block rows) | BSR of its owned block rows. The split block width must be a multiple of `b`, so blocks are never cut. |
| ell | 0 (rows) | ELL of its owned rows, width = the local maximum row length, passed as a per-rank symbol |
| dia | 1 (columns, scipy's `data[k, j]` convention) | the owned columns of every stored diagonal; the offsets are global and the same on every rank |

- **Owner function:** any of today's 1-D schemes on the split axis (block, load-balanced block,
  cyclic, block-cyclic), requested as the matrix's `requests.layout` op list with its one
  `distribute` on the split axis.
- **The other axis is never split**, so indices along it stay global. A 2-D split of a sparse
  array, or a split on the wrong axis for the format, is refused. A replicated sparse array is
  allowed, with a whole copy on every rank.
- **Index base:** per-language rebasing (Fortran 1-based) applies to the local arrays as it does
  today.
- **Live extents:** each rank gets its local row (or column) count and local nnz as size symbols.
- **Cost:** the conversion's time goes into `convert_in_ns`, and the gather of a sparse output
  into `convert_out_ns`.
- **Precision:** a request on the values array of a sparse matrix is an open item (§6). Index
  arrays are never narrowed.

### 4.7 Preview route

`POST /requests/preview` is free and does not spend the submission. It returns, for a kernel and
a `requests` object:
- the resolved digit tuple after every op;
- each rank's local shape and live extents;
- the C signature with the zip structs.

Agents then check a sequence before paying for a build.

### 4.8 Starter code

The judge renders the starter stub in the requested layout. A Python-AST pass over the numpy
reference rewrites each subscript `A[i, j]` to the laid-out index (the digit expressions and
`shuffle_*` calls). This is the hb counterpart of the SDFG subscript rewrite in
`dace/transformation/layout/subscript_rewrite.py` (which rewrites SDFG subscripts, not Python), and
a starter is all it produces. It belongs in relayout beside the ops. Grading never runs rewritten
code: the judge moves data, and the reference runs on canonical arrays.

## 5. Database schema (version 7)

| Change | Column(s) | Notes |
| --- | --- | --- |
| add | `grades.requests TEXT NOT NULL DEFAULT ''` | the `requests` object as sent, normalized JSON (sorted keys); `''` = none |
| add | `grades.convert_in_ns INTEGER NOT NULL DEFAULT 0` | sparse format conversion + precision cast-in of the credited rep; 0 = no conversion |
| add | `grades.convert_out_ns INTEGER NOT NULL DEFAULT 0` | precision cast-out of the credited rep |
| fold into `requests` and drop | `requested_libraries`, `distribution`, `layout_request` | migration copies them into `requests` (`libraries`, `distribution`, `sparse`) |
| fold into `convert_in_ns` and drop | `layout_prep_ns` | the untimed sparse conversion is one more conversion |
| keep | `layout` (sparse format label), `size_scale`, `scale_axes`, `workspace_bytes` | unchanged |

- `grades_flat` (`g.*`) carries the new columns as is.
- The schema stays one JSON value per request and no child table, the rule `schema.sql` states.
  A per-array question ("how often is `A` narrowed to fp32") is a `json_each` query over
  `json_extract(requests, '$.precision')`.
- A per-array `resolved_layout` column (the lowered digit tuple) waits for the layout ops it would
  describe.

Version 6 stays readable, not just convertible: `open_ro` accepts it, a query written for both
versions selects through `results_db.grade_columns` (a missing column reads its default), and
`results_db.requests_of_row` reads either version's row. `merge` and `copy_grade` fold a v6 row into
v7 (`fold_legacy_grade`), and `open_db` on a v6 file upgrades it in place (`upgrade_legacy`: add the
v7 columns, fold every grade, drop the folded columns). New files are created at version 7.

## 6. Open items

- **Python delivery** (precision and layout): the callable would get arrays already converted, with
  zip as a numpy structured array. Refused today.
- **Precision on the ML track and on device-resident MPI tiles:** refused today; the ML ranks
  generate their own shards, and the device driver would need the casts as device kernels.
- **Sparse values precision:** whether a sparse matrix's values array may take a `precision`
  request (the converter would cast it).
- **Grid ordering:** whether `grid_order` (BLACS row-major or column-major) is needed before any
  kernel asks for it.
- **One algebra:** `relayout.ops` simulates its own digit tuple over concrete grids; moving it onto
  dace's symbolic `LayoutMap` (`dace/libraries/layout/algebra.py`) gives one algebra and the symbolic
  extents `pad` and dead cells need.
- **AoSoA:** fields interleaved per block (`u0..u63, v0..v63, u64..`), the middle ground between a
  struct (zip) and separate arrays, as an option of `zip`.
- **Compact op syntax:** `"block(0, 32*P0); distribute(1, 0)"` beside the JSON objects, parsed into the
  same registered ops (fewer agent tokens; JSON stays the stored form).
