# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Streaming DRAM bandwidth on this GPU: copy (read+write), read-only and write-only, 1 GiB fp32, CUDA events."""

from collections.abc import Callable

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def read_kernel(x, out, n, BLOCK: tl.constexpr, ITERS: tl.constexpr) -> None:  # noqa: ANN001
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    base = pid.to(tl.int64) * BLOCK * ITERS
    for i in range(ITERS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(x + offs, mask=offs < n, other=0.0)
    tl.store(out + pid, tl.sum(acc))


def time_ms(fn: Callable[[], object], reps: int = 30) -> tuple[float, float]:
    for warmup in range(3):
        fn()
    ts = []
    for rep in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts)), float(np.min(ts))


n = 256 * 1024 * 1024
nbytes = n * 4
src = torch.randn(n, device="cuda")
dst = torch.empty_like(src)
results = {}
results["copy (R+W)"] = (2 * nbytes, time_ms(lambda: dst.copy_(src)))
results["torch.sum (R)"] = (nbytes, time_ms(lambda: src.sum()))
results["fill_ (W)"] = (nbytes, time_ms(lambda: dst.fill_(1.0)))
for block, iters, warps in [(4096, 8, 8), (2048, 16, 8), (8192, 4, 16)]:
    out = torch.empty(triton.cdiv(n, block * iters), device="cuda")
    grid = (triton.cdiv(n, block * iters),)
    results[f"triton read B{block} I{iters} w{warps} (R)"] = (
        nbytes,
        time_ms(
            lambda grid=grid, out=out, block=block, iters=iters, warps=warps: read_kernel[grid](
                src, out, n, BLOCK=block, ITERS=iters, num_warps=warps
            )
        ),
    )
for name, (b, (med, best)) in results.items():
    print(f"{name:40s} median {med:.3f} ms {b / med / 1e6:7.1f} GB/s   best {best:.3f} ms {b / best / 1e6:7.1f} GB/s")
