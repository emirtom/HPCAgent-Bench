# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fixed-step RK4 over an ensemble of Brusselator systems: one system per lane, the whole NSTEPS loop in
registers, so each system reads y0 and writes y once."""

import torch
import triton
import triton.language as tl


def get_configs():
    # Compute-bound with uniform control flow: modest blocks give enough programs to fill all SMs
    # even at the smallest fuzzed NSYS.
    return [
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 64}, num_warps=2),
        triton.Config({"BLOCK": 256}, num_warps=8),
        triton.Config({"BLOCK": 32}, num_warps=1),
    ]


@triton.jit
def brusselator_rhs(u, v, w, a, b, ep):
    return a - (w + 1.0) * u + v * u * u, w * u - v * u * u, (b - w) / ep - w * u


@triton.autotune(configs=get_configs(), key=["nsys", "nsteps"], cache_results=True)
@triton.jit
def rk4_kernel(
    y0,
    y,
    nsys,
    nsteps,
    row_stride0,
    row_stride,
    col_stride0,
    col_stride,
    a: tl.float64,
    b: tl.float64,
    ep: tl.float64,
    t_end: tl.float64,
    BLOCK: tl.constexpr,
):
    systems = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = systems < nsys
    source = y0 + systems.to(tl.int64) * row_stride0
    u = tl.load(source, mask=valid, other=0.0)
    v = tl.load(source + col_stride0, mask=valid, other=0.0)
    w = tl.load(source + 2 * col_stride0, mask=valid, other=0.0)
    # h = t_end / NSTEPS rounded once in fp64, as the reference computes it.
    h = t_end / nsteps.to(tl.float64)
    half_h = 0.5 * h
    sixth_h = h / 6.0
    for step in range(nsteps):  # noqa: B007 -- the loop count is the time integration
        k1u, k1v, k1w = brusselator_rhs(u, v, w, a, b, ep)
        k2u, k2v, k2w = brusselator_rhs(u + half_h * k1u, v + half_h * k1v, w + half_h * k1w, a, b, ep)
        k3u, k3v, k3w = brusselator_rhs(u + half_h * k2u, v + half_h * k2v, w + half_h * k2w, a, b, ep)
        k4u, k4v, k4w = brusselator_rhs(u + h * k3u, v + h * k3v, w + h * k3w, a, b, ep)
        u = u + sixth_h * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        v = v + sixth_h * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        w = w + sixth_h * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
    target = y + systems.to(tl.int64) * row_stride
    tl.store(target, u, mask=valid)
    tl.store(target + col_stride, v, mask=valid)
    tl.store(target + 2 * col_stride, w, mask=valid)


def rk4_ensemble(y0, y, NSYS, NSTEPS, a, b, ep, t_end):
    nsys = int(NSYS)
    nsteps = int(NSTEPS)
    if nsys == 0:
        return
    # a, b, ep and t_end are annotated tl.float64: an unannotated Python float reaches the kernel as fp32.
    # The manifest allows fp64 only.
    if y.dtype != torch.float64:
        raise TypeError(f"rk4_ensemble is fp64-only, got {y.dtype}")
    grid = lambda meta: (triton.cdiv(nsys, meta["BLOCK"]),)
    rk4_kernel[grid](
        y0,
        y,
        nsys,
        nsteps,
        y0.stride(0),
        y.stride(0),
        y0.stride(1),
        y.stride(1),
        float(a),
        float(b),
        float(ep),
        float(t_end),
    )
