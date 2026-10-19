# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A pytest plugin that makes the run the outermost launch grading needs (:func:`flags.enter_openmp_launch_env`).

``tests/conftest.py`` does this for the suite; the per-kernel reference tests under ``hpcagent_bench/benchmarks/``
sit outside ``tests/`` and load it with ``-p tests.openmp_launch_plugin``.
"""

from hpcagent_bench import flags

flags.enter_openmp_launch_env()
