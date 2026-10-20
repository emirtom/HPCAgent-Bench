# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A baseline sweep must know which columns build for a device, or it runs one with no GPU behind it.

The sweep asks the column's registered device (``arch``). The framework decides by its own table
(``native_runtime.FRAMEWORK_LANG``) for the native columns, and the two had drifted before: a name rule (``*gpu*``)
matched ``dace_gpu*`` and missed every PPCG column, so ``ppcg_hip``, the AMD CUDA->HIP column, was submitted with no
GPU at all.
"""

import pytest

from hpcagent_bench.categories import Language
from hpcagent_bench.cluster import baseline
from hpcagent_bench.frameworks.native_runtime import FRAMEWORK_LANG

DEVICE_LANGUAGES = (Language.HIP, Language.CUDA)


@pytest.mark.parametrize("column", sorted(c for c, lang in FRAMEWORK_LANG.items() if lang in DEVICE_LANGUAGES))
def test_every_device_column_the_framework_builds_is_known_to_the_sweep(column: str) -> None:
    assert baseline.is_device_column(column), (
        f"{column} builds {FRAMEWORK_LANG[column]} but the sweep takes it for a CPU one"
    )


@pytest.mark.parametrize("column", ["dace_gpu_parallel", "dace_gpu_canon", "dace_gpu_autoopt", "tvm_gpu"])
def test_the_gpu_flavors_are_known_to_the_sweep(column: str) -> None:
    """dace and tvm are not cpp columns, so they are absent from FRAMEWORK_LANG and need their own check."""
    assert baseline.is_device_column(column)


@pytest.mark.parametrize(
    "column", ["numba", "cc", "cpp", "fortran", "pluto", "dace_cpu_parallel", "dace_cpu_canon", "tvm_cpu"]
)
def test_a_cpu_column_is_not_a_device_column(column: str) -> None:
    assert not baseline.is_device_column(column), f"{column} would be reported as missing GPUs it never uses"
