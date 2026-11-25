# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A short tuning digest of a full Nsight Compute report (gpu_run.sh's ncu-full:<P> step).

Usage: ncu_digest.py <report.ncu-repz> [--ncu PATH] [--lines N]

Reads the report with ``ncu --import`` (no GPU needed, so it also runs on the login node) and prints, per profiled
kernel: speed-of-light throughputs, occupancy and what limits it, the top warp stall reasons, ncu's own OPT rules
sorted by estimated speedup, and the Triton source lines with the most stall samples together with their excess L2
sectors (uncoalesced or misaligned accesses) and shared-memory bank conflicts.
"""

import argparse
import csv
import io
import re
import subprocess

STALL = re.compile(r"^smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio$")


def ncu_csv(ncu: str, rep: str, *args: str) -> list[list[str]]:
    out = subprocess.run([ncu, "--import", rep, "--csv", *args], capture_output=True, text=True, check=True).stdout
    return list(csv.reader(io.StringIO(out)))


def fnum(text: str) -> float:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return 0.0


def details(ncu: str, rep: str) -> dict[str, dict]:
    rows = ncu_csv(ncu, rep, "--page", "details", "--print-rule-details")
    head = rows[0]
    kernels: dict[str, dict] = {}
    for raw in rows[1:]:
        r = dict(zip(head, raw, strict=False))
        k = kernels.setdefault(
            r["ID"], {"name": r["Kernel Name"], "grid": r["Grid Size"], "block": r["Block Size"], "m": {}, "rules": []}
        )
        if r.get("Metric Name"):
            k["m"][(r["Section Name"], r["Metric Name"])] = (r["Metric Value"], r["Metric Unit"])
        if r.get("Rule Name") and r.get("Rule Type") == "OPT":
            k["rules"].append((fnum(r.get("Estimated Speedup", "")), r["Rule Name"], r["Rule Description"]))
    return kernels


def stalls(ncu: str, rep: str) -> dict[str, list[tuple[str, float]]]:
    rows = ncu_csv(ncu, rep, "--page", "raw")
    head = rows[0]
    out = {}
    for raw in rows[2:]:  # row 1 holds the units
        r = dict(zip(head, raw, strict=False))
        vals = [(m.group(1), fnum(v)) for c, v in r.items() if (m := STALL.match(c))]
        out[r["ID"]] = sorted(((n, v) for n, v in vals if n not in ("selected",) and v > 0), key=lambda x: -x[1])
    return out


def hot_lines(ncu: str, rep: str, n: int) -> list[str]:
    """Source lines (Triton Python) ranked by warp stall samples, over every profiled kernel."""
    rows = ncu_csv(ncu, rep, "--page", "source", "--print-source", "cuda,sass")
    out: list[str] = []
    head: list[str] = []
    lines: list[tuple[float, str]] = []
    total = 0.0
    fname = func = ""

    def flush() -> None:
        if lines and total:
            out.append(f"  {fname.split('/')[-1]} :: {func}")
            for s, text in sorted(lines, key=lambda x: -x[0])[:n]:
                out.append(f"    {100 * s / total:5.1f}% of stall samples  {text}")
        lines.clear()

    for r in rows:
        if not r:
            continue
        if r[0] in ("File Path", "File Name"):
            flush()
            fname, total = r[1], 0.0
        elif r[0] == "Function Name":
            flush()
            func, total = r[1], 0.0
        elif r[0] == "Line No":
            head = r
        elif head and r[0].isdigit() and len(r) > 3 and r[2] == "-":
            c = dict(zip(head[2:], r[2:], strict=False))  # head has "Source" twice; the first is the line text
            s = fnum(c.get("Warp Stall Sampling (All Samples)", "0"))
            if not s:
                continue
            total += s
            exc, l2 = (
                fnum(c.get("L2 Theoretical Sectors Global Excessive", "0")),
                fnum(c.get("L2 Theoretical Sectors Global", "0")),
            )
            bank = fnum(c.get("L1 Wavefronts Shared Excessive", "0"))
            note = []
            if l2 and exc / l2 > 0.1:
                note.append(f"{100 * exc / l2:.0f}% excess L2 sectors")
            if bank:
                note.append(f"{bank:.0f} excess shared wavefronts (bank conflicts)")
            lines.append((s, f"L{r[0]:>4} {r[1].strip()[:90]}" + (f"   [{'; '.join(note)}]" if note else "")))
    flush()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--ncu", default="ncu")
    ap.add_argument("--lines", type=int, default=5)
    a = ap.parse_args()
    ks = details(a.ncu, a.report)
    st = stalls(a.ncu, a.report)
    for kid, k in ks.items():
        m = k["m"]

        def g(sec: str, name: str, m: dict = m) -> str:
            v = m.get((sec, name))
            return "-" if v is None else f"{v[0]}{'' if v[1] in ('', 'block', 'warp') else ' ' + v[1]}"

        sol = "GPU Speed Of Light Throughput"
        occ = "Occupancy"
        print(f"== {k['name']}  grid {k['grid']} block {k['block']}  (ncu ID {kid})")
        print(
            f"  SOL: duration {g(sol, 'Duration')}, SM {g(sol, 'SM Frequency')}; compute {g(sol, 'Compute (SM) Throughput')}"
            f", memory {g(sol, 'Memory Throughput')} (L1 {g(sol, 'L1/TEX Cache Throughput')}, L2 "
            f"{g(sol, 'L2 Cache Throughput')}, DRAM {g(sol, 'DRAM Throughput')})"
        )
        limits = {
            n.replace("Block Limit ", ""): fnum(v[0])
            for (s, n), v in m.items()
            if s == occ and n.startswith("Block Limit")
        }
        lim = min(limits, key=lambda x: limits[x]) if limits else "-"
        print(
            f"  Occupancy: theoretical {g(occ, 'Theoretical Occupancy')}, achieved {g(occ, 'Achieved Occupancy')}; "
            f"limited by {lim} ({limits.get(lim, 0):.0f} blocks/SM); "
            f"{g('Launch Statistics', 'Registers Per Thread')}, waves/SM {g('Launch Statistics', 'Waves Per SM')}"
        )
        print(
            f"  Issue: {g('Scheduler Statistics', 'Issued Warp Per Scheduler')} warps issued per scheduler cycle, "
            f"{g('Warp State Statistics', 'Warp Cycles Per Issued Instruction')} between issues; L1 hit "
            f"{g('Memory Workload Analysis', 'L1/TEX Hit Rate')}, L2 hit {g('Memory Workload Analysis', 'L2 Hit Rate')}; "
            f"spills {g('Instruction Statistics', 'Local Memory Spilling Requests')}"
        )
        top = st.get(kid, [])
        tot = sum(v for _, v in top) or 1.0
        print(
            "  Stalls (cycles per issued instruction): "
            + ", ".join(f"{n} {v:.2f} ({100 * v / tot:.0f}%)" for n, v in top[:5])
        )
        for sp, name, desc in sorted(k["rules"], key=lambda x: -x[0]):
            text = re.sub(r"\s+", " ", desc).strip()
            print(f"  OPT {name} (est. speedup {sp:.0f}%): {text[:400]}")
    print("Hot source lines:")
    for line in hot_lines(a.ncu, a.report, a.lines):
        print(line)


if __name__ == "__main__":
    main()
