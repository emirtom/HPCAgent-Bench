# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-kernel reference tests' launch: the OpenMP environment grading needs, as ``tests/conftest.py`` sets it.

These tests sit outside ``tests/``, so that conftest never runs for them; a test that grades a native emission
against the compiled numba/C reference refuses to launch it without this (``OpenMPLaunchEnvError``).
"""

import os
import resource

from hpcagent_bench import flags

# Before anything imports numpy, whose OpenBLAS loads the runtime that reads it once (flags.openmp_launch_env).
os.environ.update(flags.openmp_launch_env())
resource.setrlimit(resource.RLIMIT_STACK, (resource.getrlimit(resource.RLIMIT_STACK)[1],) * 2)
