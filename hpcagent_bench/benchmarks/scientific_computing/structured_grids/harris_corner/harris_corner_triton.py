# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import triton
import triton.language as tl


@triton.jit
def sobel_products(img, ixx, iyy, ixy, row_stride, col_stride, height, width, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < height * width
    # Each compact index names the top-left image pixel of its Sobel window.
    top = (index // width).to(tl.int64) * row_stride + (index % width).to(tl.int64) * col_stride
    middle = top + row_stride
    bottom = middle + row_stride

    top_left = tl.load(img + top, valid, other=0)
    top_center = tl.load(img + top + col_stride, valid, other=0)
    top_right = tl.load(img + top + 2 * col_stride, valid, other=0)
    middle_left = tl.load(img + middle, valid, other=0)
    middle_right = tl.load(img + middle + 2 * col_stride, valid, other=0)
    bottom_left = tl.load(img + bottom, valid, other=0)
    bottom_center = tl.load(img + bottom + col_stride, valid, other=0)
    bottom_right = tl.load(img + bottom + 2 * col_stride, valid, other=0)

    ix = ((top_right - top_left) + 2.0 * (middle_right - middle_left) + (bottom_right - bottom_left)) * 0.125
    iy = ((bottom_left - top_left) + 2.0 * (bottom_center - top_center) + (bottom_right - top_right)) * 0.125
    tl.store(ixx + index, ix * ix, valid)
    tl.store(iyy + index, iy * iy, valid)
    tl.store(ixy + index, ix * iy, valid)


@triton.jit
def box_sum(product, top_left, width, valid):
    total = tl.load(product + top_left, valid, other=0)
    for neighbor in tl.static_range(1, 9):
        offset = (neighbor // 3) * width + neighbor % 3
        total += tl.load(product + top_left + offset, valid, other=0)
    return total


@triton.jit
def box_response(ixx, iyy, ixy, response, row_stride, col_stride, height, width, K: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < height * width
    row = index // width
    col = index % width
    product_width = width + 2
    top_left = row * product_width + col

    sxx = box_sum(ixx, top_left, product_width, valid)
    syy = box_sum(iyy, top_left, product_width, valid)
    sxy = box_sum(ixy, top_left, product_width, valid)
    det = sxx * syy - sxy * sxy
    trace = sxx + syy
    offset = (row.to(tl.int64) + 2) * row_stride + (col.to(tl.int64) + 2) * col_stride
    tl.store(response + offset, det - K * (trace * trace), valid)


def kernel(img, R, k=0.04):
    """Write the Harris response on the two-pixel interior of R."""
    height, width = img.shape
    if height < 5 or width < 5:
        return

    gradient_height, gradient_width = height - 2, width - 2
    output_height, output_width = height - 4, width - 4
    products = torch.empty((3, gradient_height, gradient_width), dtype=img.dtype, device=img.device)
    ixx, iyy, ixy = products.unbind()
    block = 256
    sobel_products[(triton.cdiv(gradient_height * gradient_width, block),)](
        img, ixx, iyy, ixy, *img.stride(), gradient_height, gradient_width, block, enable_fp_fusion=False
    )
    box_response[(triton.cdiv(output_height * output_width, block),)](
        ixx, iyy, ixy, R, *R.stride(), output_height, output_width, float(k), block, enable_fp_fusion=False
    )
