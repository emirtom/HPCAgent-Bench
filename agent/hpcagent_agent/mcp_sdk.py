# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which MCP SDK major is installed. Both 1.x and 2.x are supported: openhands-sdk pins fastmcp 3 (mcp<2) while
every other harness venv takes 2.x. They differ in the low-level server's handler registration and the client's
read timeout; models are built from their wire (camelCase) names, which both accept."""

import datetime
import importlib.metadata

__all__ = ["V2", "read_timeout"]

#: Whether the installed ``mcp`` is 2.x or later.
V2 = int(importlib.metadata.version("mcp").split(".", 1)[0]) >= 2


def read_timeout(seconds: float) -> float | datetime.timedelta:
    """``ClientSession``'s ``read_timeout_seconds``: seconds on 2.x, a ``timedelta`` on 1.x."""
    return seconds if V2 else datetime.timedelta(seconds=seconds)
