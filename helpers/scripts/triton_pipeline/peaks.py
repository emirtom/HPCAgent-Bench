# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measured compute and L2 ceilings of this GPU for the hardware roofline (DRAM is bw.py's job).

Usage: peaks.py            time each microbenchmark (CUDA events, median and best of 30) and print GFLOP/s or GB/s
       peaks.py --ncu      run each microbenchmark once between cudaProfilerStart/Stop, for ncu calibration: the
                           counters roofline.py reads must give back the analytic FLOP and byte counts printed here

Microbenchmarks: fp32 and fp64 FMA chains (8 independent chains per lane, so latency is hidden), tl.dot fp16->fp32
and tf32 on register-resident tiles, and an L2 read loop over a buffer of half the L2 size loaded with .cg
(bypasses L1, stays resident in L2).
"""

import statistics
import sys
from collections.abc import Callable

import torch
import triton
import triton.language as tl

CHAINS = 8


@triton.jit
def fma_kernel(out, ITERS: tl.constexpr, BLOCK: tl.constexpr, DT: tl.constexpr) -> None:  # noqa: ANN001
    pid = tl.program_id(0)
    x = (tl.arange(0, BLOCK) + pid).to(DT) * 1e-7
    b = x * 0.999 + 0.5
    c = x * 1e-3
    a0 = x
    a1 = x + 1
    a2 = x + 2
    a3 = x + 3
    a4 = x + 4
    a5 = x + 5
    a6 = x + 6
    a7 = x + 7
    for i in range(ITERS):
        a0 = a0 * b + c
        a1 = a1 * b + c
        a2 = a2 * b + c
        a3 = a3 * b + c
        a4 = a4 * b + c
        a5 = a5 * b + c
        a6 = a6 * b + c
        a7 = a7 * b + c
    s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
    tl.store(out + pid * BLOCK + tl.arange(0, BLOCK), s)


@triton.jit
def dot_kernel(
    a_ptr,  # noqa: ANN001
    b_ptr,  # noqa: ANN001
    out,  # noqa: ANN001
    ITERS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PREC: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(a_ptr + rm[:, None] * BK + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * BN + rn[None, :])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for i in range(ITERS):
        acc = tl.dot(a, b, acc, input_precision=PREC)
    tl.store(out + pid * BM * BN + rm[:, None] * BN + rn[None, :], acc)


@triton.jit
def l2_read_kernel(x, out, n_per_prog, REPS: tl.constexpr, BLOCK: tl.constexpr) -> None:  # noqa: ANN001
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    base = pid.to(tl.int64) * n_per_prog
    for r in range(REPS):
        for off in range(0, n_per_prog, BLOCK):
            acc += tl.load(x + base + off + tl.arange(0, BLOCK), cache_modifier=".cg")
    tl.store(out + pid, tl.sum(acc))


def time_ms(fn: Callable[[], object], reps: int = 30) -> tuple[float, float]:
    for warmup in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for rep in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts), min(ts)


SMS = torch.cuda.get_device_properties(0).multi_processor_count
benches: dict[str, tuple[str, float, Callable[[], object]]] = {}

# Every microbenchmark runs at several programs-per-SM counts: a grid that leaves a partly full last wave (more
# programs per SM than the registers allow at once) under-reports the peak. The best variant is the ceiling.
# FMA: 2 FLOPs per FMA, CHAINS chains, ITERS iterations, BLOCK lanes per program.
for name, dt, tdt, iters in [
    ("fp32 FMA", tl.float32, torch.float32, 8192),
    ("fp64 FMA", tl.float64, torch.float64, 512),
]:
    for mult in (2, 3, 4, 6):
        block, progs = 1024, SMS * mult
        out = torch.empty(progs * block, device="cuda", dtype=tdt)
        flops = 2 * CHAINS * iters * block * progs
        benches[f"{name} x{mult}/SM"] = (
            "GFLOP/s",
            flops,
            lambda out=out, it=iters, b=block, p=progs, d=dt: fma_kernel[(p,)](
                out, ITERS=it, BLOCK=b, DT=d, num_warps=8
            ),
        )

# Tensor cores: 2*BM*BN*BK FLOPs per tl.dot.
for name, tdt, prec in [("tensor fp16->fp32", torch.float16, "ieee"), ("tensor tf32->fp32", torch.float32, "tf32")]:
    for mult in (1, 2, 4):
        bm, bn, bk, iters, progs = 128, 128, 32, 8192 // mult, SMS * mult
        a = torch.randn(bm, bk, device="cuda", dtype=tdt)
        b = torch.randn(bk, bn, device="cuda", dtype=tdt)
        out = torch.empty(progs * bm * bn, device="cuda", dtype=torch.float32)
        flops = 2 * bm * bn * bk * iters * progs
        benches[f"{name} x{mult}/SM"] = (
            "GFLOP/s",
            flops,
            lambda a=a, b=b, out=out, p=progs, it=iters, bm=bm, bn=bn, bk=bk, pr=prec: dot_kernel[(p,)](
                a, b, out, ITERS=it, BM=bm, BN=bn, BK=bk, PREC=pr, num_warps=8
            ),
        )

# L2: an fp32 buffer of half the L2 split across programs, each reads its slice REPS times with .cg loads.
L2_BYTES = torch.cuda.get_device_properties(0).L2_cache_size
n = L2_BYTES // 2 // 4
for mult in (4, 8, 16):
    progs, blk = SMS * mult, 1024
    per = n // progs // blk * blk
    src = torch.randn(per * progs, device="cuda")
    l2out = torch.empty(progs, device="cuda")
    reps = 64
    benches[f"L2 read ({4 * per * progs >> 20} MiB) x{mult}/SM"] = (
        "GB/s",
        4 * per * progs * reps,
        lambda src=src, o=l2out, p=progs, per=per, r=reps, blk=blk: l2_read_kernel[(p,)](
            src, o, per, REPS=r, BLOCK=blk, num_warps=8
        ),
    )

print(f"GPU: {torch.cuda.get_device_name(0)}, {SMS} SMs; torch {torch.__version__}, triton {triton.__version__}")
if "--ncu" in sys.argv:
    for name, (unit, work, fn) in benches.items():
        fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print(f"ANALYTIC {name}: {work:.6e} {'FLOP' if unit == 'GFLOP/s' else 'byte'}")
else:
    for name, (unit, work, fn) in benches.items():
        med, best = time_ms(fn)
        print(
            f"{name:30s} median {med:8.3f} ms {work / med / 1e6:9.1f} {unit}   best {best:8.3f} ms "
            f"{work / best / 1e6:9.1f} {unit}"
        )
