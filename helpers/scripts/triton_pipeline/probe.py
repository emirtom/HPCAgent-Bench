# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Call one kernel's ``<kernel>_triton.py`` entry point on the harness's own inputs, outside the harness.

Usage: probe.py <kernel> <preset S|M|L|XL> [--dtype float32] [--sha SHA] [--ncu]

Inputs come from ``Benchmark(kernel).get_data`` (the harness's initializer at the preset's sizes) and are moved
to the GPU. Three warm-up calls cover JIT compilation and autotuning; then either one call between
cudaProfilerStart/Stop (``--ncu``, run under ``ncu --profile-from-start off``) or 20 timed calls (CUDA events).
Every call restores the inputs from a pristine device copy OUTSIDE the timed bracket, so in-place and
accumulating kernels see the same data each time.
Prints ``RESULT {...}`` with the preset's parameters, so roofline.py can evaluate a FLOP formula over them.
"""

import argparse
import importlib
import json
import statistics

import numpy as np
import torch

from hpcagent_bench.categories import Preset
from hpcagent_bench.frameworks.benchmark import Benchmark

ap = argparse.ArgumentParser()
ap.add_argument("kernel")
ap.add_argument("preset")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--sha", default="")
ap.add_argument("--ncu", action="store_true")
ns = ap.parse_args()

bench = Benchmark(ns.kernel)
spec = bench.spec
if spec.sparse_layouts:
    raise SystemExit(f"probe.py: {ns.kernel} has sparse inputs; write a kernel-specific probe instead")
data = bench.get_data(Preset[ns.preset.upper()], ns.dtype)
params = {k: v for k, v in data.items() if isinstance(v, (int, float, np.integer, np.floating)) and k != "datatype"}
mod = importlib.import_module(bench.impl_module("triton"))
fn = getattr(mod, spec.func_name)

pristine = {}
for name in spec.input_args:
    v = data[name]
    pristine[name] = torch.from_numpy(np.ascontiguousarray(v)).cuda() if isinstance(v, np.ndarray) else v
work = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in pristine.items()}


def restore() -> None:
    for k, v in pristine.items():
        if isinstance(v, torch.Tensor):
            work[k].copy_(v)


def call() -> object:
    return fn(*[work[a] for a in spec.input_args])


for warmup in range(3):
    restore()
    call()
torch.cuda.synchronize()

row = {
    "kind": "probe",
    "kernel": ns.kernel,
    "sha": ns.sha,
    "preset": ns.preset,
    "dtype": ns.dtype,
    "params": {k: float(v) if isinstance(v, (float, np.floating)) else int(v) for k, v in params.items()},
    "input_bytes": int(sum(v.numel() * v.element_size() for v in pristine.values() if isinstance(v, torch.Tensor))),
}
if ns.ncu:
    restore()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    call()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
else:
    times = []
    for rep in range(20):
        restore()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        call()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    row["median_ms"] = statistics.median(times)
    row["min_ms"] = min(times)
print("RESULT " + json.dumps(row))
