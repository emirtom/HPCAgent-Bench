# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A pytest plugin making the run the outermost OpenMP launch (:func:`flags.enter_openmp_launch_env`), for the
per-kernel reference tests outside ``tests/`` (``-p tests.openmp_launch_plugin``)."""

from hpcagent_bench import flags

flags.enter_openmp_launch_env()
