# The requests API

A submission asks the judge to change how its data reaches the kernel through one `requests` object in
the envelope. The block is parsed in `hpcagent_bench/harness/requests_api.py`; the layout ops, the grid
expressions and their lowering are relayout's (`relayout.ops`, a pinned dependency), documented in
relayout's [`docs/layout.md`](https://github.com/ThrudPrimrose/relayout/blob/main/docs/layout.md).

| Request | Changes | Conversion charged to | Status |
| --- | --- | --- | --- |
| `libraries` | which catalog libraries the judge links | nothing (build time) | implemented |
| `precision` | the element type of named arrays | separate timer, not scored | implemented |
| `sparse` | the format of each sparse array | separate timer, not scored | implemented |
| `grid` + `layout` + `location` | which rank owns which elements, and where its tiles live | untimed scatter | the distribution subset (§3) |
| `zip` / `unzip` | how many arrays there are (struct of fields) | separate timer, not scored | parsed, refused |

```json
{
  "language": "c",
  "source": "...",
  "requests": {
    "libraries": ["openblas"],
    "precision": {"A": "fp32", "x": "bf16"},
    "grid": ["P"],
    "layout": {
      "A": [{"op": "block", "digit": "i0", "factor": "32*P0", "into": ["i0", "c"]},
            {"op": "block", "digit": "c", "factor": 32, "into": ["r", "b"]},
            {"op": "distribute", "digit": "r", "grid_dim": 0}],
      "x": []
    }
  }
}
```

Requests are checked before anything is built. A request the kernel cannot honour is a **request
fault**: the judge answers 400 and the submission is not spent. The old top-level `libraries`,
`distribution` and `sparse_config` fields are refused with a 400 that names their new spelling.

## 1. Precision

`precision` maps an array name to a `Precision` (`fp64`, `fp32`, `fp16`, `bf16`, `fp8_e4m3`,
`fp8_e5m2`). Every array not named stays in the **problem precision**, the sweep's `--datatype`.

Refused: a name that is not a floating dense array argument (scalars, integer, index and sparse
arrays); a request equal to the array's problem dtype (a no-op is refused, not ignored); python
deliveries, the ML track (its ranks generate their own shards) and device-resident MPI tiles.

Per timed rep the inputs are staged at the problem precision, every requested array is **cast in**
to a buffer of the requested type (outputs too: the ABI makes every output `intent(inout)`), the
kernel runs in the timed bracket, and each requested output is **cast out** to the problem precision.
The casts have their own brackets (`convert_in_ns`, `convert_out_ns`) and never enter the speedup.

**Grading stays at the problem precision's tolerance band.** Narrowing is the agent's accuracy risk:
an output narrowed to bf16 or fp8 is graded against the fp64 band like any other, and fails where its
error exceeds it.

**Cast kernels** are harness-owned, one table of conversions (`precision_casts.cast_expr`, through the
emitter's `FP8_HELPERS` for bf16 and fp8, bit-exact against ml_dtypes):
- host: one `#pragma omp parallel for simd` loop per type pair, built once per OpenMP context with
  that family's C compiler and loaded beside the submission, so the timed process maps one OpenMP
  runtime (150-200 GB/s on an MI300A slot, against 26-32 GB/s for threaded numpy);
- device: one `cupy.ElementwiseKernel` per pair (2.3-2.5 TB/s); bf16 and fp8 ride in unsigned buffers;
- MPI: the generated driver inlines the same loops around every repeat on each rank's host tiles,
  in MAX-over-ranks brackets; the credited repeat's cast times are recorded.

**The C ABI follows the request:** a requested array's pointer is the requested C type (`float *`,
`_Float16 *`, `__npb_bf16 *`, ...). The prompt's requests section tables the C type of each precision.

## 2. Libraries

Each name in `requests.libraries` must be one `languages.library_offered` accepts for the submission's
language, else the request is refused (`sandbox.catalog_refusal`).

## 3. Grid, layout and location

`grid` lists the processor grid's extents as expressions over the rank count `P`: `["P"]` is 1-D,
`["root(P, 2)", "root(P, 2)"]` is square. The grade at `mpi.ranks` and every point of a scaling run
evaluate it at their `P`; it must span the leaderboard's `mpi.ranks`, and a scaling point it does not
span is skipped with a note.

`layout` maps each dense array to an op list over its digits: dimension `d` starts as one digit named
`i<d>`, and `split` / `block` cut digits, `permute` reorders them and `distribute` hands a digit of
extent `P<g>` to grid dimension `g` (relayout's `docs/layout.md`). An empty list, or an array not
listed, is replicated. Dimensions are numpy's: a Fortran kernel declares each tile with its extents
reversed. The judge lowers each sequence at the rank count (`Requests.plan`) to the plan the scatter
descriptor reads; a sequence that does not lower is refused with the reason, and a kernel's default
distribution is shown to the agent as op lists (`Requests.from_plan`). The refusals on the lowered
plan: a cyclic layout needs an equal-edge grid, `replicatable_allowlist` and
`layout_flexible_allowlist`.

`location` maps an array of a distributed run to `"host"` or `"device"` (default: `mpi.residency`);
it needs a `grid`.

Without a `grid`, a `layout`, `location`, `zip` or `unzip` request is refused: single-node layout
changes are not implemented yet.

### 3.1 Zip and the C ABI

`zip` (`{"uv": ["u", "v"]}`) gives the kernel one pointer to a plain C struct in place of the fields'
pointers, at the position of the first field; `unzip` (`{"z": ["z_re", "z_im"]}`) gives one pointer
per member in place of the struct array's.

```c
typedef struct { float u; double v; } hb_zip_uv;   /* field order = request order */
void kernel(const hb_zip_uv *restrict uv, ..., void *workspace, int64_t workspace_bytes);
```

Field types are the per-array precisions; padding and alignment follow C (the judge allocates a numpy
structured dtype with `align=True`); Fortran sees `type, bind(C) :: hb_zip_uv`. An input field is cast
and relaid in, an output field cast and relaid out.

### 3.2 Sparse arrays

A sparse array takes no layout ops: it changes layout through `requests.sparse`. Distributing a sparse
format is not implemented yet (a sparse kernel with a `grid` is refused); the design is one converter
per format from the canonical CSR to each rank's local array, split along the axis the format stores
row-wise (csr rows, csc columns, coo either, bsr block rows, ell rows, dia columns), the other axis
never split and its indices global, with the conversion time in `convert_in_ns`.

## 4. Database schema (version 7)

| Change | Column(s) | Notes |
| --- | --- | --- |
| add | `grades.requests TEXT NOT NULL DEFAULT ''` | the `requests` object as sent, normalized JSON (sorted keys); `''` = none |
| add | `grades.convert_in_ns INTEGER NOT NULL DEFAULT 0` | sparse conversion + precision cast-in of the credited rep |
| add | `grades.convert_out_ns INTEGER NOT NULL DEFAULT 0` | precision cast-out of the credited rep |
| fold into `requests` and drop | `requested_libraries`, `distribution`, `layout_request` | `libraries`, `grid` + `layout`, `sparse` |
| fold into `convert_in_ns` and drop | `layout_prep_ns` | the sparse conversion is one more conversion |

One JSON value per request and no child table, as `schema.sql` states: a per-array question is a
`json_each` query over `json_extract(requests, '$.precision')`.

Version 6 stays readable: `open_ro` accepts it, `results_db.grade_columns` selects across versions (a
missing column reads its default) and `results_db.requests_of_row` reads either version's row. `merge`
and `copy_grade` fold a v6 row into v7 (`fold_legacy_grade`); `open_db` upgrades a v6 file in place.

## 5. Open items

- **Python delivery** of precision and layout: the callable would get converted arrays, with zip as a
  numpy structured array.
- **Precision on the ML track and on device-resident MPI tiles.**
- **Sparse values precision:** whether a sparse matrix's values array may take a `precision`.
- **Preview route:** a free `POST /requests/preview` returning the digit tuple after every op, each
  rank's local shape and the C signature with the zip structs.
- **Compact op syntax:** `"block(i0, 32*P0 -> i0, c); distribute(c, 0)"` beside the JSON objects,
  parsed into the same registered ops (fewer agent tokens; JSON stays the stored form).
