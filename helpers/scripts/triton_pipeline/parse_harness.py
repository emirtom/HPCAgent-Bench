# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Turn one ``hpcagent-bench run-framework`` log (stdin) into a ``RESULT {...}`` JSON line.

Usage: parse_harness.py <kernel> <sha> <preset> <dtype> [<framework>] < log

<framework> is the ``-f`` short name the log came from (default ``triton``); CPU rows (``numba``, ``cc``) also
record how many cores the process could use, since the GPU allocation caps that.
"""

import json
import os
import re
import sys

kernel, sha, preset, dtype = sys.argv[1:5]
framework = sys.argv[5] if len(sys.argv) > 5 else "triton"
# The harness prints "<column full name> - <impl> - median: X ms"; match only this framework's lines.
LABELS = {"triton": r"Triton", "numba": r"Numba", "cc": r"C \(gcc\)"}
log = sys.stdin.read()
if dtype == "fp32-if-allowed":
    # Record what the harness resolved it to from the manifest (float32 or float64).
    hit = re.search(r"\bdatatype (float32|float64)\b", log)
    dtype = hit.group(1) if hit else dtype
medians = [float(m) for m in re.findall(LABELS[framework] + r"(?!\w).*?median:\s*([0-9.]+)\s*ms", log)]
if "VALIDATION FAILED" in log or "FAIL" in log.replace("FAILED", "FAIL") and "SUCCESS" not in log:
    status = "FAILED"
elif "SUCCESS" in log:
    status = "SUCCESS"
elif medians:
    status = "UNVALIDATED"
else:
    status = "ERROR"
reason = ""
if status != "SUCCESS":
    hit = re.search(r"(VALIDATION FAILED.*|[A-Za-z]*Error:.*)", log)
    reason = hit.group(1)[:300] if hit else log.strip().splitlines()[-1][:300] if log.strip() else "no output"
row = {
    "kind": "harness",
    "kernel": kernel,
    "framework": framework,
    "sha": sha,
    "preset": preset,
    "dtype": dtype,
    "status": status,
    "median_ms": medians[-1] if medians else None,
    "reason": reason,
}
if framework != "triton":
    row["cores"] = len(os.sched_getaffinity(0))
print("RESULT " + json.dumps(row))
