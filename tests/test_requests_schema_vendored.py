# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The agent tool's ``requests`` schema is relayout's, regenerated whenever relayout's pin moves.

The agent runtime is stdlib-only and cannot import relayout, so it ships the schema as JSON; this is the
check that the copy is the schema of the relayout the judge parses requests with.
"""

import json

from relayout.requests import requests_schema

from hpcagent_bench import paths

VENDORED = paths.ROOT / "agent/hpcagent_agent/tools/requests_schema.json"


def test_the_agent_tool_ships_relayouts_request_schema() -> None:
    """Regenerate with ``python -c "import json, relayout.requests as r; print(json.dumps(r.requests_schema(),
    indent=1, sort_keys=True))" > agent/hpcagent_agent/tools/requests_schema.json``."""
    assert json.loads(VENDORED.read_text()) == requests_schema()


if __name__ == "__main__":
    test_the_agent_tool_ships_relayouts_request_schema()
