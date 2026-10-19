# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One declaration per tool: ``@tool(...)`` on a module's ``run()`` registers it for the MCP server, the
``hpcagent-bench-tool`` CLI and the prompt's tool list.

    @tool(description=DESCRIPTION, input=SearchInput, prompt=PROMPT, core=False)
    def run(payload: dict[str, Any]) -> dict[str, Any]: ...

The tool's name is its module's stem unless ``name`` says otherwise. ``input`` is a pydantic model (its JSON schema is what ``tools/list``
shows, and a call is validated against it before ``run`` sees the payload) or, for a schema assembled at
import time from the setup (``score``, ``submit``, ``profile``), the JSON schema itself.

Which tools a setup is served (:func:`offered`):

* ``core`` tools always, except a ``preview`` tool (one that measures a version before it is submitted) in
  the blind submission mode, where the judge router refuses its route too;
* a non-core tool only when the setup's packet carries it: ``hpcagent_bench.packets`` exports the packet's
  ``tools`` as :data:`PACKET_TOOLS_ENV`, so a tool whose only answer elsewhere would be ``unavailable`` (or
  one that reaches the real internet, ``search``) never shows up in ``tools/list``, ``--allowedTools`` or
  the prompt.
"""

import dataclasses
import functools
import os
from collections.abc import Callable, Mapping
from typing import Any, cast

import pydantic
import pydantic.json_schema
import pydantic_core
from pydantic.json_schema import JsonSchemaValue

from hpcagent_agent import submission_mode

__all__ = [
    "KEY_ORDER",
    "NAMED_SCHEMAS",
    "PACKET_TOOLS_ENV",
    "REGISTRY",
    "Run",
    "InputSchema",
    "Tool",
    "offered",
    "packet_tools",
    "schema_of",
    "tool",
    "wire_form",
]

#: A tool's body: the call's arguments in, the JSON answer the model reads out.
Run = Callable[[dict[str, Any]], dict[str, Any]]

#: The tools the setup's packet carries, comma-separated (``hpcagent_bench.packets`` exports it).
PACKET_TOOLS_ENV = "HPCAGENT_BENCH_PACKET_TOOLS"


@dataclasses.dataclass(frozen=True, slots=True)
class Tool:
    """A registered tool: what ``tools/list`` shows, the prompt bullet, and the body."""

    name: str
    description: str
    input_schema: dict[str, Any]
    prompt: str
    run: Run
    core: bool
    preview: bool


#: Every tool that exists, by name, in registration (import) order.
REGISTRY: dict[str, Tool] = {}


#: The key order of the hand-written input schemas the recorded ``tools/list`` answers carry: a model's schema is
#: rendered in it, so a tool moved onto a pydantic model shows the agent the same bytes.
KEY_ORDER = ("type", "description", "enum", "properties", "required")
#: Schema keys whose value maps names to schemas: the names keep their declared order.
NAMED_SCHEMAS = frozenset({"properties", "$defs"})


class InputSchema(pydantic.json_schema.GenerateJsonSchema):
    """A tool's input schema as the hand-written ones read: an optional field is its plain type, with no
    ``null`` branch and no ``default`` (leaving the field out is how a caller says "none")."""

    def nullable_schema(self, schema: pydantic_core.core_schema.NullableSchema) -> JsonSchemaValue:
        return self.generate_inner(schema["schema"])

    def default_schema(self, schema: pydantic_core.core_schema.WithDefaultSchema) -> JsonSchemaValue:
        if self.get_default_value(schema) is None:
            return self.generate_inner(schema["schema"])
        return super().default_schema(schema)


def wire_form(schema: Any) -> Any:
    """``schema`` in :data:`KEY_ORDER` and without pydantic's generated ``title`` keys: the field descriptions
    are the prompt text, the titles only repeat the field names."""
    if isinstance(schema, list):
        return [wire_form(value) for value in cast("list[Any]", schema)]
    if not isinstance(schema, dict):
        return schema
    node = cast("dict[str, Any]", schema)
    keys = [key for key in KEY_ORDER if key in node] + [key for key in node if key not in {*KEY_ORDER, "title"}]
    return {
        key: {name: wire_form(value) for name, value in node[key].items()}
        if key in NAMED_SCHEMAS
        else wire_form(node[key])
        for key in keys
    }


def schema_of(model: type[pydantic.BaseModel]) -> dict[str, Any]:
    """The input schema ``tools/list`` shows for ``model`` (:class:`InputSchema`, :func:`wire_form`)."""
    schema = model.model_json_schema(schema_generator=InputSchema)
    schema.setdefault("required", [])
    return wire_form(schema)


def validated(model: type[pydantic.BaseModel], run: Run) -> Run:
    """``run`` behind ``model``'s validation: a malformed call answers ``ok: false`` with pydantic's reason
    (the model reads it and fixes the call) and ``run`` sees only the fields the caller set."""

    @functools.wraps(run)
    def call(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = model.model_validate(payload)
        except pydantic.ValidationError as exc:
            reasons = "; ".join(
                f"{'.'.join(map(str, error['loc'])) or 'arguments'}: {error['msg']}" for error in exc.errors()
            )
            return {"ok": False, "error": f"invalid arguments: {reasons}"}
        return run(parsed.model_dump(exclude_unset=True))

    return call


def tool(
    *,
    description: str,
    input: type[pydantic.BaseModel] | Mapping[str, Any],  # noqa: A002 -- the MCP name for a tool's schema
    prompt: str = "",
    core: bool = True,
    preview: bool = False,
    name: str = "",
) -> Callable[[Run], Run]:
    """Register the decorated ``run`` as the tool ``name``, by default its module's stem (see the module
    docstring)."""

    def register(run: Run) -> Run:
        tool_name = name or run.__module__.rpartition(".")[2]
        if isinstance(input, Mapping):
            schema, body = dict(input), run
        else:
            schema, body = schema_of(input), validated(input, run)
        REGISTRY[tool_name] = Tool(tool_name, description, schema, prompt, body, core, preview)
        return run

    return register


def packet_tools(environ: Mapping[str, str] = os.environ) -> frozenset[str]:
    """The tools the setup's packet carries (:data:`PACKET_TOOLS_ENV`)."""
    return frozenset(name.strip() for name in environ.get(PACKET_TOOLS_ENV, "").split(",") if name.strip())


def offered(entry: Tool, environ: Mapping[str, str] = os.environ) -> bool:
    """Whether this setup is served ``entry`` at all (see the module docstring)."""
    if not entry.core:
        return entry.name in packet_tools(environ)
    return not entry.preview or submission_mode.current().preview_served
