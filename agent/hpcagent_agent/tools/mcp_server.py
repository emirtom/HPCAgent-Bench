"""The ``hpcagent_bench`` MCP server: every tool registered with ``@tool`` (:mod:`hpcagent_agent.tools.registry`)
that this setup is offered, served with the official MCP SDK over stdio (``--http <port>``: streamable HTTP).

One tool per judge route the agent needs, plus ``syntax_check`` (the one tool that talks to no service: THIS
process runs inside the agent's container next to the compilers) and the tools a packet brings. ``score`` and
``submit`` are deliberately separate tools because they are separate grades: the public iteration signal and the
terminal, hidden-seed, recorded one. Whether the agent also has a shell is the launcher's decision, not this
server's, so no tool may assume the absence of a shell.

Every MCP client consumes this one server: Claude Code, Codex and OpenHands, and the native Anthropic and OpenAI
harnesses (``hpcagent_agent.harness``). mini-SWE has only a shell and reaches the same tools through the
``hpcagent-bench-tool`` CLI.
"""

import contextlib
import importlib
import importlib.metadata
import json
import os
import pathlib
import pkgutil
import re
import sys
from collections.abc import AsyncIterator
from typing import Any

import anyio
import anyio.to_thread
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from hpcagent_agent import submission_mode, tools
from hpcagent_agent.tools.registry import REGISTRY, Tool, offered

__all__ = [
    "AGENT_PAYLOAD_DIR",
    "ALLOWED_ORDER",
    "ALLOWED_TOOLS",
    "BULLET_HEAD",
    "MCP_HEAD",
    "NOT_TOOLS",
    "LIST_ORDER",
    "PACKET",
    "PROMPT_ORDER",
    "SERVER_NAME",
    "TOOLS",
    "answer",
    "build_server",
    "call",
    "describe",
    "in_order",
    "main",
    "prompt_tool_list",
    "server_instructions",
    "tool_definitions",
]

#: The server's name: Claude Code shows a tool as ``mcp__hpcagent_bench__<name>``.
SERVER_NAME = "hpcagent_bench"

#: The orders recorded setups saw in ``--allowedTools`` and in the prompt. A tool missing from one follows
#: the listed tools in registration order.
ALLOWED_ORDER = ("search", "score", "profile", "submit", "syntax_check", "canonical_parallel_form")
PROMPT_ORDER = ("profile", "score", "submit", "search", "syntax_check")

#: The modules of ``hpcagent_agent.tools`` that are not tools: this server, its CLI front end and their libraries.
NOT_TOOLS = frozenset({"mcp_server", "hpcagent_bench_tool", "registry", "http_json"})

# Every other module of the package is a tool: importing it registers its @tool, so a new tool is one file.
for tool_module in pkgutil.iter_modules(tools.__path__):
    if tool_module.name not in NOT_TOOLS:
        importlib.import_module(f"{tools.__name__}.{tool_module.name}")

#: ``AGENT_PACKET=<name>`` imports the tool modules of ``hpcagent_agent.packets.<name>``; each registers itself.
PACKET: str = os.environ.get("AGENT_PACKET", "").strip()
if PACKET:
    PACKET_DIR = pathlib.Path(__file__).resolve().parents[1] / "packets" / PACKET
    if not (PACKET_DIR / "packet.md").is_file():
        raise SystemExit(f"AGENT_PACKET={PACKET}: {PACKET_DIR / 'packet.md'} does not exist")
    for packet_module in sorted(PACKET_DIR.glob("*.py")):
        if packet_module.stem == "__init__":
            continue
        if packet_module.stem in REGISTRY:
            raise SystemExit(f"packet {PACKET} tool {packet_module.stem} collides with a core tool")
        importlib.import_module(f"hpcagent_agent.packets.{PACKET}.{packet_module.stem}")

#: The ``tools/list`` order recorded setups saw; a tool not named here (a packet's) follows in registration order.
LIST_ORDER = ("score", "submit", "profile", "canonical_parallel_form", "search", "syntax_check")

#: The tools this process serves: every tool this setup is offered, in :data:`LIST_ORDER`.
TOOLS: dict[str, Tool] = {
    name: REGISTRY[name]
    for name in (
        *(name for name in LIST_ORDER if name in REGISTRY),
        *(name for name in REGISTRY if name not in LIST_ORDER),
    )
    if offered(REGISTRY[name])
}


def in_order(first: tuple[str, ...]) -> tuple[str, ...]:
    """Every tool this setup is served, those in ``first`` leading in its order."""
    return (*(name for name in first if name in TOOLS), *(name for name in TOOLS if name not in first))


#: Claude Code's ``--allowedTools``, without the ``mcp__hpcagent_bench__`` prefix: exactly the served tools, so
#: the model is never offered a tool whose only answer is a refusal.
ALLOWED_TOOLS: tuple[str, ...] = in_order(ALLOWED_ORDER)

#: A prompt bullet's head, ``- `<tool>` --``.
BULLET_HEAD = re.compile(r"^- `([a-z_]+)` --", re.MULTILINE)

#: The line over the MCP tool list: the names the model's tool list shows, which the bullets abbreviate.
MCP_HEAD = (
    "They are the MCP tools of the server `hpcagent_bench`: your tool list names each one "
    "`mcp__hpcagent_bench__<name>`, and the bullets below say when to call it."
)


def prompt_tool_list(cli: bool = False) -> str:
    """The prompt's tool list: every served tool's non-empty bullet, in PROMPT_ORDER, under :data:`MCP_HEAD`.
    ``cli`` names each tool as its ``hpcagent-bench-tool`` shell command instead."""
    text = "\n".join(TOOLS[name].prompt for name in in_order(PROMPT_ORDER) if TOOLS[name].prompt)
    if cli:
        return BULLET_HEAD.sub(r"- `hpcagent-bench-tool \1 '<json>'` --", text)
    return f"{MCP_HEAD}\n\n{text}"


#: The agent payload directory that holds the submission-mode templates (``agent/submission-<mode>.md``).
AGENT_PAYLOAD_DIR = pathlib.Path(__file__).resolve().parents[2]


def server_instructions() -> str:
    """The ``initialize`` instructions: the prompt's tool list with the submission mode's slot filled.

    The raw list carries ``{{MODE:tool}}`` (the ``submit`` bullet, which the mode decides), and Claude Code
    injects these instructions into the system prompt verbatim, so the slot is filled from the payload's
    template, the copy the launcher stages the shared one from."""
    template = submission_mode.current().template
    sections = submission_mode.template_sections(AGENT_PAYLOAD_DIR / template)
    return submission_mode.fill_slots(prompt_tool_list(), sections, template)


def tool_definitions() -> list[types.Tool]:
    """What ``tools/list`` answers."""
    return [
        types.Tool(name=name, description=entry.description, inputSchema=entry.input_schema)
        for name, entry in TOOLS.items()
    ]


def call(entry: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one tool. A tool fault is content the model must READ (a bad ``$JUDGE_RANK``, an unreachable judge),
    so it comes back as ``ok: false`` rather than killing the request."""
    try:
        return entry.run(arguments)
    except Exception as exc:  # noqa: BLE001 -- the model reads this; a dead server tells it nothing
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def answer(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
    """The ``tools/call`` result: the tool's JSON answer as text, an error exactly when it says ``ok: false``."""
    entry = TOOLS.get(name)
    response = {"ok": False, "error": f"unknown tool: {name}"} if entry is None else call(entry, arguments)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(response, indent=2, sort_keys=True))],
        isError=response.get("ok") is False,
    )


def build_server() -> Server:
    """The SDK server. Inputs are validated by each tool (its pydantic model), not by the SDK, so a malformed
    call answers with the tool's own reason. A tool blocks on the judge, so it runs in a worker thread."""
    version = importlib.metadata.version("hpcagent-agent")
    server: Server = Server(SERVER_NAME, version=version, instructions=server_instructions())

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return tool_definitions()

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        return await anyio.to_thread.run_sync(answer, name, arguments)

    return server


async def serve_stdio(server: Server) -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def serve_http(server: Server, port: int) -> None:
    """Streamable HTTP on ``127.0.0.1:<port>/mcp``, for a harness that connects by URL."""
    import uvicorn  # noqa: PLC0415 -- only the HTTP transport needs the ASGI stack
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager  # noqa: PLC0415
    from starlette.applications import Starlette  # noqa: PLC0415
    from starlette.routing import Mount  # noqa: PLC0415

    manager = StreamableHTTPSessionManager(app=server, stateless=True)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with manager.run():
            yield

    app = Starlette(routes=[Mount("/mcp", app=manager.handle_request)], lifespan=lifespan)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def describe() -> dict[str, Any]:
    """What this setup is offered, for the launcher: ``--allowedTools``, the tools this process serves and the
    prompt's tool lists."""
    return {
        "allowed_tools": list(ALLOWED_TOOLS),
        "served_tools": list(TOOLS),
        "prompt": prompt_tool_list(),
        "prompt_cli": prompt_tool_list(cli=True),
    }


def main(argv: list[str]) -> int:
    if argv == ["--describe"]:
        print(json.dumps(describe()))
        return 0
    if len(argv) == 2 and argv[0] == "--http" and argv[1].isdigit():
        serve_http(build_server(), int(argv[1]))
        return 0
    if argv:
        print("usage: python -m hpcagent_agent.tools.mcp_server [--describe | --http <port>]", file=sys.stderr)
        return 2
    anyio.run(serve_stdio, build_server())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
