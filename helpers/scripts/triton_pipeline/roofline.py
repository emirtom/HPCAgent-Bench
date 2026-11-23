# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Roofline bookkeeping for the Triton pipeline.

  roofline.py ncu-csv <kernel> <sha> <preset> < ncu --csv output (with probe.py's RESULT line mixed in)
      Sum DRAM read/write bytes and GPU time over every kernel launched by one call; print RESULT kind=ncu.
  roofline.py report <ledger.jsonl> <kernel> --flops '<python expr over the preset params>' [--preset M]
      Join the latest ncu row per SHA with harness medians and the GPU ceilings; print a markdown table.

Ceilings come from gpu.json beside this script (measured once per GPU by bw.py; the peak is the spec value).
The bandwidth ceiling is the HIGHEST measured streaming bandwidth (an upper bound), and the table also shows
the theoretical peak, so no kernel can appear to beat its own ceiling.
"""

import csv
import io
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent


def num(text: str) -> float:
    return float(text.replace(",", "")) if text not in ("", "n/a") else 0.0


def ncu_csv(kernel: str, sha: str, preset: str) -> None:
    raw = sys.stdin.read()
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
        if name.startswith("dram__bytes") and unit not in ("byte", "B"):
            raise SystemExit(f"unexpected byte unit {unit!r}; run ncu with --print-units base")
        per.setdefault(k, {})[name] = value * 1e-9 if name.startswith("gpu__time") else value
    launches = [
        {
            "name": k[1][:80],
            "time_ms": v.get("gpu__time_duration.sum", 0) * 1e3,
            "read_MB": v.get("dram__bytes_op_read.sum", 0) / 1e6,
            "write_MB": v.get("dram__bytes_op_write.sum", 0) / 1e6,
            "dram_pct": v.get("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
            "sm_pct": v.get("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
            "regs": v.get("launch__registers_per_thread"),
            "warps_active_pct": v.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
        }
        for k, v in per.items()
    ]
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
        "launches": launches,
    }
    for x in launches:
        print(
            f"  {x['name'][:50]:50s} {x['time_ms']:8.3f} ms  R {x['read_MB']:8.2f} MB  W {x['write_MB']:8.2f} MB  "
            f"dram {x['dram_pct']}%  sm {x['sm_pct']}%  regs {x['regs']}  warps {x['warps_active_pct']}%"
        )
    if not launches:
        # No RESULT row: an empty profile must not reach the ledger as a zero-byte measurement.
        print("NCU ERROR: no kernels captured; output tail:\n" + raw[-1500:])
        sys.exit(1)
    print("RESULT " + json.dumps(row))


def report(ledger: str, kernel: str, flops_expr: str, preset: str) -> None:
    gpu = json.loads((HERE / "gpu.json").read_text())
    rows = [json.loads(line) for line in pathlib.Path(ledger).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("kernel") == kernel]
    shas = list(dict.fromkeys(r["sha"] for r in rows))
    bw = gpu["bw_measured_GBps"]
    print(
        f"Ceilings: FP32 CUDA cores {gpu['fp32_TFLOPs']} TFLOP/s; DRAM measured {bw} GB/s "
        f"(theoretical {gpu['bw_peak_GBps']} GB/s). Preset {preset}. FLOPs = {flops_expr}\n"
    )
    print("| SHA | harness S | harness default | DRAM R+W MB | FLOP/B | ceiling GFLOP/s | kernel GFLOP/s | % ceiling |")
    print("|---|---|---|---|---|---|---|---|")
    for sha in shas:
        mine = [r for r in rows if r["sha"] == sha]
        hs = [r for r in mine if r["kind"] == "harness"]

        def med(p: str, hs: list[dict] = hs) -> str:
            vals = [f"{r['median_ms']:.3f}" if r["median_ms"] else r["status"] for r in hs if r["preset"] == p]
            return " / ".join(vals) or "-"

        nc = [r for r in mine if r["kind"] == "ncu" and r["preset"] == preset]
        if nc:
            n = nc[-1]
            flops = eval(flops_expr, {}, dict(n["params"]))  # noqa: S307 - a trusted formula from the agent
            byts = (n["dram_read_MB"] + n["dram_write_MB"]) * 1e6
            ai = flops / byts if byts else float("inf")
            ceil = min(gpu["fp32_TFLOPs"] * 1e3, bw * ai)
            got = flops / (n["kernel_time_ms"] * 1e-3) / 1e9 if n["kernel_time_ms"] else 0
            print(
                f"| {sha[:9]} | {med('S')} | {med('fuzzed')} | {n['dram_read_MB']:.1f}+{n['dram_write_MB']:.1f} | "
                f"{ai:.2f} | {ceil:.0f} | {got:.0f} | {100 * got / ceil:.0f}% |"
            )
        else:
            print(f"| {sha[:9]} | {med('S')} | {med('fuzzed')} | - | - | - | - | - |")


if __name__ == "__main__":
    if sys.argv[1] == "ncu-csv":
        ncu_csv(*sys.argv[2:5])
    elif sys.argv[1] == "report":
        args = sys.argv[2:]
        pre = args[args.index("--preset") + 1] if "--preset" in args else "M"
        report(args[0], args[1], args[args.index("--flops") + 1], pre)
