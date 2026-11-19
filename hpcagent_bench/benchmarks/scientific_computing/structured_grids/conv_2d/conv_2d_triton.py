import torch
import triton
import triton.language as tl

@triton.jit
def conv_2d_kernel(input, kernel, output,
                   input_rows, input_cols,
                   kernel_rows, kernel_cols,
                   output_cols, BLOCK_SIZE: tl.constexpr):
    # One program = one output row (p_y) x BLOCK_SIZE consecutive columns (p_x).
    # input is the edge-padded grid, so no clamping is needed.
    p_x = tl.program_id(0)
    p_y = tl.program_id(1)  

    cols = p_x*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = p_y*input_cols + cols
    mask = cols < output_cols

    # Accumulate in the output's dtype (float64 by default), float32 would fail the 1e-9 check
    acc = tl.zeros((BLOCK_SIZE,), dtype=output.dtype.element_ty)

    # Iterate over kernel and compute stencil
    for dy in range(kernel_rows):
        for dx in range(kernel_cols):
            values = tl.load(input + offsets + dy*input_cols + dx, mask=mask, other=0.0)
            weight = tl.load(kernel + dy*kernel_cols + dx)
            acc += values * weight

    tl.store(output + p_y*output_cols + cols, acc, mask=mask)

def conv_2d(in_grid, out_grid, w_box, N, R):
    # in_grid / out_grid / w_box arrive as CUDA torch tensors (bench.py moves them to the GPU),
    # so the padding is built with torch on the same device instead of numpy.
    # Construct padded matrix. This will later be done in another kernel!
    padded = torch.empty((N + 2 * R, N + 2 * R), dtype=in_grid.dtype, device=in_grid.device)
    padded[R : R + N, R : R + N] = in_grid
    for k in range(R):
        padded[R : R + N, k] = in_grid[:, 0]
        padded[R : R + N, R + N + k] = in_grid[:, N - 1]
    for k in range(R):
        padded[k, :] = padded[R, :]
        padded[R + N + k, :] = padded[R + N - 1, :]

    # The length of an output block. Modifiable!
    BLOCK_SIZE = 256
    K = 2 * R + 1
    # axis 0: column tiles of one output row, axis 1: one program per output row
    grid = (triton.cdiv(N, BLOCK_SIZE), N)

    conv_2d_kernel[grid](
        padded, w_box, out_grid,
        N + 2 * R, N + 2 * R,  # input_rows, input_cols (padded grid)
        K, K,                  # kernel_rows, kernel_cols
        N,                     # output_cols
        BLOCK_SIZE,
    )
