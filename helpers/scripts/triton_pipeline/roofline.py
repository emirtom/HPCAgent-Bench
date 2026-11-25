# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hardware roofline bookkeeping for the Triton pipeline (the tuning tool).

  roofline.py ncu-csv <kernel> <sha> <preset> < ncu --csv output (with probe.py's RESULT line mixed in)
      Sum executed FLOPs, DRAM and L2 bytes and GPU time over every kernel launched by one call; print one line per
      launch with its utilizations and RESULT kind=ncu.
  roofline.py report <ledger.jsonl> <kernel> [--flops '<python expr over the preset params>'] [--preset M]
      One row per SHA: harness medians joined with the latest ncu row at the preset, placed on the hardware roofline.

Everything on the roofline is MEASURED for the version itself: FLOPs are the FP instructions the GPU executed (an FMA
counts 2; tensor-core FLOPs come from sm__ops_path_tensor_*), bytes are DRAM and L2 traffic, time is the ncu kernel
time. Utilization of a resource = its rate / its ceiling (gpu.json, measured by peaks.py and bw.py). Compute
utilization sums over precisions (fp32, fp64, fp16, tensor), since they share issue slots. The % of roof is the
HIGHEST utilization, which equals achieved / attainable FLOP/s on a roofline with compute, L2 and DRAM ceilings; the
resource that sets it is the bound. It also works for kernels with no floating point at all.

--flops is optional: the useful work counted from the NumPy reference. It adds "ref GFLOP/s" (over the harness M
median, comparable across versions) and "exec/ref" (executed over useful FLOPs: above 1 means recomputation).
"""

import argparse
import csv
import io
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
GPU = json.loads((HERE / "gpu.json").read_text())

#: ncu metric -> (FLOP class, FLOPs per count). Calibrated by peaks.py --ncu.
FLOP_METRICS = {
    **{f"sm__sass_thread_inst_executed_op_{p}fma_pred_on.sum": (c, 2) for p, c in (("f", "fp32"), ("d", "fp64"))},
    **{
        f"sm__sass_thread_inst_executed_op_{p}{o}_pred_on.sum": (c, 1)
        for p, c in (("f", "fp32"), ("d", "fp64"))
        for o in ("add", "mul")
    },
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum": ("fp16", 2),
    "sm__sass_thread_inst_executed_op_hadd_pred_on.sum": ("fp16", 1),
    "sm__sass_thread_inst_executed_op_hmul_pred_on.sum": ("fp16", 1),
    "sm__ops_path_tensor_src_fp16_dst_fp32.sum": ("tensor_fp16", 1),
    "sm__ops_path_tensor_src_bf16_dst_fp32.sum": ("tensor_fp16", 1),
    "sm__ops_path_tensor_src_fp16_dst_fp16.sum": ("tensor_fp16", 1),
    "sm__ops_path_tensor_src_tf32_dst_fp32.sum": ("tensor_tf32", 1),
}
#: FLOP class -> ceiling in GFLOP/s. Scalar fp16 runs on the fp32 pipes at the fp32 rate.
PEAK_GFLOPS = {
    "fp32": GPU["fp32_TFLOPs"] * 1e3,
    "fp64": GPU["fp64_TFLOPs"] * 1e3,
    "fp16": GPU["fp32_TFLOPs"] * 1e3,
    "tensor_fp16": GPU["tensor_fp16_TFLOPs"] * 1e3,
    "tensor_tf32": GPU["tensor_tf32_TFLOPs"] * 1e3,
}
DRAM_GBPS = GPU["bw_measured_GBps"]
L2_GBPS = GPU["l2_GBps"]

#: fp32-if-allowed resolves to one of these per kernel; low-precision rows (bf16 ...) stay out of the tables.
WORKING = ("float32", "float64")


def num(text: str) -> float:
    return float(text.replace(",", "")) if text not in ("", "n/a") else 0.0


def utilization(
    flops: dict[str, float] | None, dram_b: float, l2_b: float | None, ms: float
) -> dict[str, float | None]:
    """Fraction of each ceiling used: compute (summed over FLOP classes), L2 and DRAM. None where not measured."""
    s = ms * 1e-3
    if s <= 0:
        return {"compute": None, "L2": None, "DRAM": None}
    comp = None if flops is None else sum(f / (PEAK_GFLOPS[c] * 1e9) for c, f in flops.items()) / s
    return {
        "compute": comp,
        "L2": None if l2_b is None else l2_b / s / (L2_GBPS * 1e9),
        "DRAM": dram_b / s / (DRAM_GBPS * 1e9),
    }


def bound(util: dict[str, float | None]) -> tuple[str, float]:
    known = {k: v for k, v in util.items() if v is not None}
    k = max(known, key=lambda x: known[x])
    return k, known[k]


def pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.0f}%"


def ncu_csv(kernel: str, sha: str, preset: str) -> None:
    raw = sys.stdin.read()
    if "MODULE STATE ERROR" in raw:
        # probe.py found state kept between calls: the profile is of a kernel that breaks the rule, so no row.
        print("\n".join(line for line in raw.splitlines() if "MODULE STATE ERROR" in line))
        sys.exit(1)
    probe = next((json.loads(m) for m in re.findall(r"^RESULT (\{.*\})$", raw, re.M)), {})
    body = "\n".join(line for line in raw.splitlines() if line.startswith('"'))
    rows = list(csv.DictReader(io.StringIO(body)))
    per = {}
    for r in rows:
        k = (r["ID"], r["Kernel Name"])
        # gpu_run.sh runs ncu with --print-units base: bytes in byte, durations in nanoseconds.
        unit = r.get("Metric Unit", "")
        value = num(r["Metric Value"])
        name = r["Metric Name"]
        if name.startswith("gpu__time") and unit not in ("ns", "nsecond"):
            raise SystemExit(f"unexpected time unit {unit!r}; run ncu with --print-units base")
        if (name.startswith("dram__bytes") or name.startswith("lts__t_bytes")) and unit not in ("byte", "B"):
            raise SystemExit(f"unexpected byte unit {unit!r}; run ncu with --print-units base")
        per.setdefault(k, {})[name] = value * 1e-9 if name.startswith("gpu__time") else value
    launches = []
    for k, v in per.items():
        flops: dict[str, float] = {}
        for metric, (cls, mult) in FLOP_METRICS.items():
            if metric in v:
                flops[cls] = flops.get(cls, 0.0) + mult * v[metric]
        launches.append(
            {
                "name": k[1][:80],
                "time_ms": v.get("gpu__time_duration.sum", 0) * 1e3,
                "read_MB": v.get("dram__bytes_op_read.sum", 0) / 1e6,
                "write_MB": v.get("dram__bytes_op_write.sum", 0) / 1e6,
                "l2_MB": v["lts__t_bytes.sum"] / 1e6 if "lts__t_bytes.sum" in v else None,
                "flops": flops or None,
                "dram_pct": v.get("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
                "sm_pct": v.get("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                "regs": v.get("launch__registers_per_thread"),
                "warps_active_pct": v.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
            }
        )
    flops_total: dict[str, float] = {}
    for x in launches:
        for c, f in (x["flops"] or {}).items():
            flops_total[c] = flops_total.get(c, 0.0) + f
    l2 = [x["l2_MB"] for x in launches]
    row = {
        "kind": "ncu",
        "kernel": kernel,
        "sha": sha,
        "preset": preset,
        "ncu_settings": "--clock-control none --cache-control all",
        "params": probe.get("params", {}),
        "kernel_time_ms": sum(x["time_ms"] for x in launches),
        "dram_read_MB": sum(x["read_MB"] for x in launches),
        "dram_write_MB": sum(x["write_MB"] for x in launches),
        "l2_MB": None if None in l2 else sum(l2),
        "flops": flops_total if any(x["flops"] for x in launches) else None,
        "launches": launches,
    }
    for x in launches:
        u = utilization(
            x["flops"],
            (x["read_MB"] + x["write_MB"]) * 1e6,
            None if x["l2_MB"] is None else x["l2_MB"] * 1e6,
            x["time_ms"],
        )
        b, v = bound(u) if x["time_ms"] else ("-", 0.0)
        print(
            f"  {x['name'][:40]:40s} {x['time_ms']:8.3f} ms  R {x['read_MB']:8.2f} MB  W {x['write_MB']:8.2f} MB  "
            f"L2 {x['l2_MB'] or 0:9.2f} MB  util compute {pct(u['compute'])} L2 {pct(u['L2'])} DRAM {pct(u['DRAM'])}"
            f"  -> {b} {pct(v)}  regs {x['regs']}  warps {x['warps_active_pct']}%"
        )
    if not launches:
        # No RESULT row: an empty profile must not reach the ledger as a zero-byte measurement.
        print("NCU ERROR: no kernels captured; output tail:\n" + raw[-1500:])
        sys.exit(1)
    print("RESULT " + json.dumps(row))


def report(ledger: str, kernel: str, flops_expr: str | None, preset: str) -> None:
    rows = [json.loads(line) for line in pathlib.Path(ledger).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("kernel") == kernel]
    shas = list(dict.fromkeys(r["sha"] for r in rows))
    print(
        f"Hardware roofline at preset {preset}, measured per version (executed FLOPs, DRAM and L2 bytes, ncu kernel "
        f"time). Ceilings: fp32 {GPU['fp32_TFLOPs']} / fp64 {GPU['fp64_TFLOPs']} / tensor fp16 "
        f"{GPU['tensor_fp16_TFLOPs']} / tensor tf32 {GPU['tensor_tf32_TFLOPs']} TFLOP/s; L2 {L2_GBPS} GB/s; DRAM "
        f"{DRAM_GBPS} GB/s. % of roof = the highest utilization (= achieved / attainable); bound = that resource."
        + (f"\nUseful FLOPs (from the reference) = {flops_expr}" if flops_expr else "")
        + "\n"
    )
    cols = [
        "SHA",
        "harness S",
        "harness M",
        "GPU ms",
        "GPU share M",
        "exec GFLOP",
        "DRAM MB",
        "L2 MB",
        "FLOP/B DRAM",
        "FLOP/B L2",
        "compute",
        "L2",
        "DRAM",
        "bound",
        "% of roof",
    ]
    if flops_expr:
        cols += ["ref GFLOP/s (harness)", "exec/ref"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    # Old ledgers also hold CPU rows (numba/cc, from a removed cpu:<P> step); rows without "framework" are Triton.
    tri = [r for r in rows if r["kind"] != "harness" or r.get("framework", "triton") == "triton"]
    for sha in shas:
        mine = [r for r in tri if r["sha"] == sha]
        if not mine:
            continue
        hs = [r for r in mine if r["kind"] == "harness" and r["dtype"] in WORKING]

        def med(p: str, hs: list[dict] = hs) -> str:
            vals = [f"{r['median_ms']:.3f}" if r["median_ms"] else r["status"] for r in hs if r["preset"] == p]
            return " / ".join(vals) or "-"

        def last_ms(p: str, hs: list[dict] = hs) -> float | None:
            ok = [r["median_ms"] for r in hs if r["preset"] == p and r["median_ms"]]
            return ok[-1] if ok else None

        cells = [sha[:9], med("S"), med("M")]
        nc = [r for r in mine if r["kind"] == "ncu" and r["preset"] == preset]
        if not nc:
            print("| " + " | ".join(cells + ["-"] * (len(cols) - 3)) + " |")
            continue
        n = nc[-1]
        ms = n["kernel_time_ms"]
        dram = (n["dram_read_MB"] + n["dram_write_MB"]) * 1e6
        l2 = None if n.get("l2_MB") is None else n["l2_MB"] * 1e6
        flops = n.get("flops")
        exec_f = None if flops is None else sum(flops.values())
        u = utilization(flops, dram, l2, ms)
        b, v = bound(u)
        hm = last_ms(preset)
        cells += [
            f"{ms:.3f}",
            "-" if not hm else pct(ms / hm),
            "-" if exec_f is None else f"{exec_f / 1e9:.2f}",
            f"{dram / 1e6:.1f}",
            "-" if l2 is None else f"{l2 / 1e6:.1f}",
            "-" if exec_f is None or not dram else f"{exec_f / dram:.2f}",
            "-" if exec_f is None or not l2 else f"{exec_f / l2:.2f}",
            pct(u["compute"]),
            pct(u["L2"]),
            pct(u["DRAM"]),
            b if len([x for x in u.values() if x is not None]) > 1 else f"{b} (only DRAM measured)",
            pct(v),
        ]
        if flops_expr:
            ref = eval(flops_expr, {}, dict(n["params"]))  # noqa: S307 - a trusted formula from the agent
            cells += [
                "-" if not hm else f"{ref / (hm * 1e-3) / 1e9:.0f}",
                "-" if exec_f is None or not ref else f"{exec_f / ref:.2f}",
            ]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ncu-csv":
        ncu_csv(*sys.argv[2:5])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd", choices=["report"])
        ap.add_argument("ledger")
        ap.add_argument("kernel")
        ap.add_argument("--flops")
        ap.add_argument("--preset", default="M")
        a = ap.parse_args()
        report(a.ledger, a.kernel, a.flops, a.preset)
