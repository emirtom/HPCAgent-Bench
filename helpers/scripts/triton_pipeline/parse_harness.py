# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Turn one ``hpcagent-bench run-framework`` log (stdin) into a ``RESULT {...}`` JSON line.

Usage: parse_harness.py <kernel> <sha> <preset> <dtype> < log
"""

import json
import re
import sys

kernel, sha, preset, dtype = sys.argv[1:5]
log = sys.stdin.read()
medians = [float(m) for m in re.findall(r"Triton\b.*?median:\s*([0-9.]+)\s*ms", log)]
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
    "sha": sha,
    "preset": preset,
    "dtype": dtype,
    "status": status,
    "median_ms": medians[-1] if medians else None,
    "reason": reason,
}
print("RESULT " + json.dumps(row))
