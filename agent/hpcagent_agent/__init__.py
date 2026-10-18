# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The agent-side runtime: everything that runs inside the agent container. Standard library only, and
nothing from hpcagent_bench, which the agent never sees.

So every file and wire format this package reads or writes (the judge's answers, ``harness-end.json``,
``usage.jsonl``, the driver's records) is bare :mod:`json`, not a pydantic model: the one exemption, by
design, from the rule that the repo's formats are pydantic models (docs/results_db.md#formats)."""
