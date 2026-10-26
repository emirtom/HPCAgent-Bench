# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""run_anthropic, as a real process against a stub Messages server: a reply whose stream dies ends the leg as a
resumable ``api_timeout``, and ``--resume-prompt`` re-sends exactly the cut-off request plus the resume message.

The stub answers the first request with a ``bash`` tool call, drops the second mid-stream (headers and
``message_start`` sent, then the chunked body abandoned) and finishes the third with a text reply.
"""

import http.server
import json
import os
import pathlib
import subprocess
import sys
import threading
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest


pytest.importorskip("anthropic")

RESUME = "The connection dropped; continue."
MCP_STUB = """
import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server


async def list_tools(context, params):
    return types.ListToolsResult(tools=[])


async def main():
    server = Server("stub", on_list_tools=list_tools)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


anyio.run(main)
"""


def sse(kind: str, data: dict[str, Any]) -> bytes:
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode()


def message_start(index: int, prompt: int) -> bytes:
    usage = {"input_tokens": prompt, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    message = {"id": f"msg_{index}", "type": "message", "role": "assistant", "model": "stub", "content": []}
    return sse("message_start", {"type": "message_start", "message": {**message, "stop_reason": None, "usage": usage}})


def block(start: dict[str, Any], delta: dict[str, Any]) -> list[bytes]:
    return [
        sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": start}),
        sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
        sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ]


def message_end(stop_reason: str, output: int) -> list[bytes]:
    delta = {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": output}}
    return [sse("message_delta", delta), sse("message_stop", {"type": "message_stop"})]


TOOL_CALL = [
    message_start(1, 100),
    *block(
        {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}},
        {"type": "input_json_delta", "partial_json": json.dumps({"command": "echo hi > note.txt"})},
    ),
    *message_end("tool_use", 10),
]
FINISH = [
    message_start(3, 300),
    *block({"type": "text", "text": ""}, {"type": "text_delta", "text": "done"}),
    *message_end("end_turn", 5),
]


class StubMessages(http.server.BaseHTTPRequestHandler):
    """Request n gets ``SCRIPT[n]``; ``None`` is a stream that dies after its ``message_start``."""

    protocol_version = "HTTP/1.1"
    SCRIPT: ClassVar[list[list[bytes] | None]] = [TOOL_CALL, None, FINISH]
    requests: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        events = self.SCRIPT[len(self.requests) - 1]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for event in events if events is not None else [message_start(2, 200)]:
            self.wfile.write(f"{len(event):x}\r\n".encode() + event + b"\r\n")
        if events is None:
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(b"0\r\n\r\n")

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture(name="stub")
def stub_fixture() -> Iterator[str]:
    StubMessages.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StubMessages)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def run_leg(workdir: pathlib.Path, base_url: str, *extra: str) -> int:
    env = {**os.environ, "ANTHROPIC_API_KEY": "stub"}
    argv = [
        sys.executable,
        "-m",
        "hpcagent_agent.harness.run_anthropic",
        *("--workdir", str(workdir), "--prompt", str(workdir / "prompt.txt"), "--base-url", base_url),
        *("--model", "stub", "--usage", str(workdir / "usage.jsonl"), "--mcp-config", str(workdir / "mcp.json")),
        *extra,
    ]
    return subprocess.run(argv, env=env, cwd=workdir, capture_output=True, timeout=120, check=False).returncode


def test_a_dropped_stream_resumes_with_the_cut_off_request_and_counts_each_call_once(
    stub: str, tmp_path: pathlib.Path
) -> None:
    (tmp_path / "prompt.txt").write_text("the task", encoding="utf-8")
    (tmp_path / "mcp_stub.py").write_text(MCP_STUB, encoding="utf-8")
    servers = {"stub": {"command": sys.executable, "args": [str(tmp_path / "mcp_stub.py")]}}
    (tmp_path / "mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")

    assert run_leg(tmp_path, stub) == 1
    end = json.loads((tmp_path / "harness-end.json").read_text(encoding="utf-8"))
    assert (end["reason"], end["turns"]) == ("api_timeout", 1), end
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "hi\n"

    assert run_leg(tmp_path, stub, "--resume-prompt", RESUME) == 0
    first, cut, resumed = StubMessages.requests
    assert cut["messages"][: len(first["messages"])] == first["messages"]
    assert resumed["messages"] == [*cut["messages"], {"role": "user", "content": RESUME}]
    usage = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(line["input"], line["output"]) for line in usage] == [(100, 10), (300, 5)]
