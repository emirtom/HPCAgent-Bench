# Copyright 2026 the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""histogram_equalization at each ``nbins`` its manifest declares: the image is uint16 below nbins,
the equalized output spans the whole [0, nbins - 1] range, and the resolution changes the result."""

import numpy as np

from hpcagent_bench.benchmarks.scientific_computing.map_reduce.histogram_equalization.histogram_equalization import (
    initialize,
)
from hpcagent_bench.benchmarks.scientific_computing.map_reduce.histogram_equalization.histogram_equalization_numpy import (
    histogram_equalization,
)
from hpcagent_bench.spec import BenchSpec

#: The S preset's image edge.
EDGE = 256


def equalized(nbins: int) -> tuple[np.ndarray, np.ndarray]:
    """``(img, out)`` of one S-sized fp64 run at ``nbins``."""
    img, out = initialize(EDGE, EDGE, nbins, datatype=np.float64)
    histogram_equalization(img, out, EDGE, EDGE, nbins)
    return img, out


def declared_nbins() -> list[int]:
    return [int(row["nbins"]) for row in BenchSpec.load("histogram_equalization").config_space]


def test_the_manifest_declares_an_8_and_a_12_bit_resolution() -> None:
    assert declared_nbins() == [256, 4096]


def test_every_resolution_draws_a_uint16_image_below_nbins_and_spans_the_output_range() -> None:
    for nbins in declared_nbins():
        img, out = equalized(nbins)
        assert img.dtype == np.uint16, nbins
        assert int(img.max()) < nbins, nbins
        assert (out.min(), out.max()) == (0.0, float(nbins - 1)), nbins


def test_the_resolution_changes_the_result() -> None:
    low = equalized(256)[1]
    high = equalized(4096)[1]
    assert not np.allclose(low, high)


if __name__ == "__main__":
    test_the_manifest_declares_an_8_and_a_12_bit_resolution()
    test_every_resolution_draws_a_uint16_image_below_nbins_and_spans_the_output_range()
    test_the_resolution_changes_the_result()
