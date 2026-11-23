# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

import os

import numpy as np
import pytest

from tests.optional_imports import import_or_skip


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(("shape", "k"), [((5, 5), None), ((17, 29), 0.06), ((4, 9), None), ((9, 4), None)])
def test_harris_corner_triton_matches_numpy(
    dtype: type[np.float32] | type[np.float64], shape: tuple[int, int], k: float | None
) -> None:
    torch = import_or_skip("torch")
    import_or_skip("triton")
    if os.environ.get("TRITON_INTERPRET") == "1":
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        pytest.skip("Triton GPU or interpreter required")

    from hpcagent_bench.benchmarks.scientific_computing.structured_grids.harris_corner import (
        harris_corner_numpy,
        harris_corner_triton,
    )

    if shape == (5, 5):
        image = np.zeros(shape, dtype=dtype)
        image[2:, 2:] = 8
    else:
        image = np.random.default_rng(42).random(shape).astype(dtype)

    expected = np.full(shape, -17, dtype=dtype)
    args = () if k is None else (k,)
    harris_corner_numpy.kernel(image, expected, *args)

    # Step over every other column in the input and transpose the output.
    torch_dtype = torch.from_numpy(image).dtype
    height, width = shape
    img = torch.empty((height, 2 * width), device=device, dtype=torch_dtype)[:, ::2]
    img.copy_(torch.from_numpy(image))
    response = torch.full((width, height), -17, device=device, dtype=torch_dtype).T
    harris_corner_triton.kernel(img, response, *args)

    actual = response.cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6 if dtype == np.float32 else 1e-12, atol=1e-9)
    border = np.ones(shape, dtype=bool)
    border[2:-2, 2:-2] = False
    np.testing.assert_array_equal(actual[border], -17)
