# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Adaptive Dormand-Prince over an ensemble of Robertson systems: one system per lane, each with its own t, h
and counters. A program loops until all of its lanes reach t_end; finished lanes are frozen by masks, so the
per-system accept/reject sequence is the reference's."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

MAX_STEPS = 200000
H0 = 1.0e-4


def get_configs():
    # A program runs as long as its slowest lane, so small blocks waste the least on divergent step counts.
    return [
        triton.Config({"BLOCK": 32}, num_warps=1),
        triton.Config({"BLOCK": 64}, num_warps=2),
        triton.Config({"BLOCK": 64}, num_warps=1),
        triton.Config({"BLOCK": 128}, num_warps=4),
    ]


@triton.jit
def robertson_rhs(y1, y2, y3):
    return (
        -0.04 * y1 + 1.0e4 * y2 * y3,
        0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
        3.0e7 * y2 * y2,
    )


@triton.autotune(configs=get_configs(), key=["nsys"], cache_results=True)
@triton.jit
def rk45_kernel(
    y0,
    y,
    n_accept,
    n_reject,
    nsys,
    row_stride0,
    row_stride,
    col_stride0,
    col_stride,
    accept_stride,
    reject_stride,
    RTOL: tl.float64,
    ATOL: tl.float64,
    T_END: tl.float64,
    MAX_STEPS: tl.constexpr,
    H0: tl.float64,
    BLOCK: tl.constexpr,
):
    systems = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = systems < nsys
    source = y0 + systems.to(tl.int64) * row_stride0
    y1 = tl.load(source, mask=valid, other=0.0)
    y2 = tl.load(source + col_stride0, mask=valid, other=0.0)
    y3 = tl.load(source + 2 * col_stride0, mask=valid, other=0.0)
    t = tl.zeros((BLOCK,), dtype=tl.float64)
    h = tl.zeros((BLOCK,), dtype=tl.float64) + H0
    accepts = tl.zeros((BLOCK,), dtype=tl.int64)
    rejects = tl.zeros((BLOCK,), dtype=tl.int64)
    steps = tl.zeros((BLOCK,), dtype=tl.int32)
    active = valid & (t < T_END)
    any_active = tl.max(active.to(tl.int32), axis=0)
    while any_active > 0:
        h = tl.where(active & (t + h > T_END), T_END - t, h)
        k1_1, k1_2, k1_3 = robertson_rhs(y1, y2, y3)
        k2_1, k2_2, k2_3 = robertson_rhs(
            y1 + h * (1.0 / 5.0) * k1_1, y2 + h * (1.0 / 5.0) * k1_2, y3 + h * (1.0 / 5.0) * k1_3
        )
        k3_1, k3_2, k3_3 = robertson_rhs(
            y1 + h * (3.0 / 40.0 * k1_1 + 9.0 / 40.0 * k2_1),
            y2 + h * (3.0 / 40.0 * k1_2 + 9.0 / 40.0 * k2_2),
            y3 + h * (3.0 / 40.0 * k1_3 + 9.0 / 40.0 * k2_3),
        )
        k4_1, k4_2, k4_3 = robertson_rhs(
            y1 + h * (44.0 / 45.0 * k1_1 - 56.0 / 15.0 * k2_1 + 32.0 / 9.0 * k3_1),
            y2 + h * (44.0 / 45.0 * k1_2 - 56.0 / 15.0 * k2_2 + 32.0 / 9.0 * k3_2),
            y3 + h * (44.0 / 45.0 * k1_3 - 56.0 / 15.0 * k2_3 + 32.0 / 9.0 * k3_3),
        )
        k5_1, k5_2, k5_3 = robertson_rhs(
            y1
            + h * (19372.0 / 6561.0 * k1_1 - 25360.0 / 2187.0 * k2_1 + 64448.0 / 6561.0 * k3_1 - 212.0 / 729.0 * k4_1),
            y2
            + h * (19372.0 / 6561.0 * k1_2 - 25360.0 / 2187.0 * k2_2 + 64448.0 / 6561.0 * k3_2 - 212.0 / 729.0 * k4_2),
            y3
            + h * (19372.0 / 6561.0 * k1_3 - 25360.0 / 2187.0 * k2_3 + 64448.0 / 6561.0 * k3_3 - 212.0 / 729.0 * k4_3),
        )
        k6_1, k6_2, k6_3 = robertson_rhs(
            y1
            + h
            * (
                9017.0 / 3168.0 * k1_1
                - 355.0 / 33.0 * k2_1
                + 46732.0 / 5247.0 * k3_1
                + 49.0 / 176.0 * k4_1
                - 5103.0 / 18656.0 * k5_1
            ),
            y2
            + h
            * (
                9017.0 / 3168.0 * k1_2
                - 355.0 / 33.0 * k2_2
                + 46732.0 / 5247.0 * k3_2
                + 49.0 / 176.0 * k4_2
                - 5103.0 / 18656.0 * k5_2
            ),
            y3
            + h
            * (
                9017.0 / 3168.0 * k1_3
                - 355.0 / 33.0 * k2_3
                + 46732.0 / 5247.0 * k3_3
                + 49.0 / 176.0 * k4_3
                - 5103.0 / 18656.0 * k5_3
            ),
        )
        s1 = y1 + h * (
            35.0 / 384.0 * k1_1
            + 500.0 / 1113.0 * k3_1
            + 125.0 / 192.0 * k4_1
            - 2187.0 / 6784.0 * k5_1
            + 11.0 / 84.0 * k6_1
        )
        s2 = y2 + h * (
            35.0 / 384.0 * k1_2
            + 500.0 / 1113.0 * k3_2
            + 125.0 / 192.0 * k4_2
            - 2187.0 / 6784.0 * k5_2
            + 11.0 / 84.0 * k6_2
        )
        s3 = y3 + h * (
            35.0 / 384.0 * k1_3
            + 500.0 / 1113.0 * k3_3
            + 125.0 / 192.0 * k4_3
            - 2187.0 / 6784.0 * k5_3
            + 11.0 / 84.0 * k6_3
        )
        # FSAL: the stage-7 point is the fifth-order solution.
        k7_1, k7_2, k7_3 = robertson_rhs(s1, s2, s3)
        y4_1 = y1 + h * (
            5179.0 / 57600.0 * k1_1
            + 7571.0 / 16695.0 * k3_1
            + 393.0 / 640.0 * k4_1
            - 92097.0 / 339200.0 * k5_1
            + 187.0 / 2100.0 * k6_1
            + 1.0 / 40.0 * k7_1
        )
        y4_2 = y2 + h * (
            5179.0 / 57600.0 * k1_2
            + 7571.0 / 16695.0 * k3_2
            + 393.0 / 640.0 * k4_2
            - 92097.0 / 339200.0 * k5_2
            + 187.0 / 2100.0 * k6_2
            + 1.0 / 40.0 * k7_2
        )
        y4_3 = y3 + h * (
            5179.0 / 57600.0 * k1_3
            + 7571.0 / 16695.0 * k3_3
            + 393.0 / 640.0 * k4_3
            - 92097.0 / 339200.0 * k5_3
            + 187.0 / 2100.0 * k6_3
            + 1.0 / 40.0 * k7_3
        )
        e1 = s1 - y4_1
        e2 = s2 - y4_2
        e3 = s3 - y4_3
        enorm = tl.sqrt(e1 * e1 + e2 * e2 + e3 * e3)
        ynorm = tl.sqrt(s1 * s1 + s2 * s2 + s3 * s3)
        err = enorm / (ATOL + RTOL * ynorm)
        steps += active.to(tl.int32)
        accept = active & (err <= 1.0)
        reject = active & (err > 1.0)
        y1 = tl.where(accept, s1, y1)
        y2 = tl.where(accept, s2, y2)
        y3 = tl.where(accept, s3, y3)
        t = tl.where(accept, t + h, t)
        accepts += accept.to(tl.int64)
        rejects += reject.to(tl.int64)
        factor = 0.9 * libdevice.pow(tl.where(err > 0.0, err, 1.0), tl.full((BLOCK,), -0.2, tl.float64))
        factor = tl.minimum(tl.maximum(factor, 0.2), 5.0)
        factor = tl.where(err > 0.0, factor, 5.0)
        h = tl.where(active, h * factor, h)
        active = active & (t < T_END) & (steps < MAX_STEPS)
        any_active = tl.max(active.to(tl.int32), axis=0)
    target = y + systems.to(tl.int64) * row_stride
    tl.store(target, y1, mask=valid)
    tl.store(target + col_stride, y2, mask=valid)
    tl.store(target + 2 * col_stride, y3, mask=valid)
    tl.store(n_accept + systems.to(tl.int64) * accept_stride, accepts.to(n_accept.dtype.element_ty), mask=valid)
    tl.store(n_reject + systems.to(tl.int64) * reject_stride, rejects.to(n_reject.dtype.element_ty), mask=valid)


def rk45_ensemble(y0, y, n_accept, n_reject, NSYS, rtol, atol, t_end):
    nsys = int(NSYS)
    if nsys == 0:
        return
    if y.dtype != torch.float64:
        raise TypeError(f"rk45_ensemble is fp64-only, got {y.dtype}")
    # The float scalars are annotated tl.float64: an unannotated Python float reaches the kernel as fp32, and a
    # constexpr float compares against its fp32 rounding (t < 0.05 then never ends).
    grid = lambda meta: (triton.cdiv(nsys, meta["BLOCK"]),)
    rk45_kernel[grid](
        y0,
        y,
        n_accept,
        n_reject,
        nsys,
        y0.stride(0),
        y.stride(0),
        y0.stride(1),
        y.stride(1),
        n_accept.stride(0),
        n_reject.stride(0),
        float(rtol),
        float(atol),
        float(t_end),
        MAX_STEPS,
        H0,
    )
