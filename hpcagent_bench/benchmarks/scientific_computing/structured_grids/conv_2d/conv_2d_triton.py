import torch
import triton
import triton.language as tl


@triton.jit
def conv_2d_fused(src, weights, dst, N, R, BLOCK_SIZE: tl.constexpr):
    cols = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row = tl.program_id(1)
    mask = cols < N
    k = 2 * R + 1

    acc = tl.zeros((BLOCK_SIZE,), dtype=dst.dtype.element_ty)

    for dy in range(k):
        src_row = tl.minimum(tl.maximum(row + dy - R, 0), N - 1)
        for dx in range(k):
            src_cols = tl.minimum(tl.maximum(cols + dx - R, 0), N - 1)
            values = tl.load(src + src_row * N + src_cols, mask=mask, other=0.0)
            weight = tl.load(weights + dy * k + dx)
            acc += values * weight

    tl.store(dst + row * N + cols, acc, mask=mask)


def conv_2d(in_grid, out_grid, w_box, N, R):
    block_size = 256
    grid = (triton.cdiv(N, block_size), N)
    conv_2d_fused[grid](in_grid, w_box, out_grid, N, R, block_size)
