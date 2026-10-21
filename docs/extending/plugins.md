# Plugins: extending the benchmark without editing it

Every plugin adds optimizers, of two kinds:

- a **method** an LLM agent runs: a packet with its skill pages, method text, agent tools and judge routes
  (`plugins/autokernel`, the NestForge example below);
- a **tuner** that runs without an LLM: a `TunerOptimizer` and the DaCe stages it tunes, each a framework column
  run as `hpcagent-bench agent <column>` ([Tuners](#tuners); Alloy ships one).

A plugin lives in a folder of its own. The plugins
shipped with the repository live in `plugins/`; a plugin of your own can be dropped in beside them, kept anywhere
else and named in `HPCAGENT_BENCH_USER_PLUGIN_DIRS`, or shipped inside an installed package.

## Layout

```
my_plugin/
  plugin.toml              # the manifest: name, description, packets, tuners
  __init__.py              # optional; the folder is imported as a package either way
  tools/<tool>.py          # agent-side @tool modules
  routes/<route>.py        # judge-side @route modules
  skills/<page>/SKILL.md   # skill pages; a packet: frontmatter block makes one a packet
  methods/<method>/method.md   # a method's prompt text (the prompt's {{HINTS}})
```

The folder is imported as the package `hpcagent_bench_plugins.<name>`, so a route imports a helper next to it as
`hpcagent_bench_plugins.<name>.routes.helper` (or relatively). `plugins/autokernel` is the shipped example: a
method, its ledger tool and the packet that runs both.

```toml
# plugin.toml
name = "my_plugin"
description = "One line on what the plugin adds."

[packets.my_packet]             # a packet key; it takes the next colour slot
name = "My Packet"
method = "my_method"            # methods/my_method/method.md
tools = ["my_tool"]
```

A packet's fields are those of [packets.md](packets.md). A plugin's packets take the next slots when it is
loaded. Names are one namespace: a packet, page or tool another plugin (or the repository) already registered is
refused.

## Agent tools

A tool is a `@tool` module, as in [skills-and-tools.md](skills-and-tools.md), with `core=False`: it is served only to
the setups whose packet lists it under `tools`.

```python
# tools/my_tool.py
from typing import Any

import pydantic

from hpcagent_agent.tools import http_json
from hpcagent_agent.tools.registry import Spec, tool


class MyToolInput(pydantic.BaseModel):
    step: str = pydantic.Field(description="Which step to take.")


@tool(spec=Spec("Take one step and return the new state.", MyToolInput, "- `my_tool` -- take one step."), core=False)
def run(payload: dict[str, Any]) -> dict[str, Any]:
    return http_json.post_judge("/my_tool", payload)
```

The input is a pydantic model, whose generated schema is what `tools/list` shows; no schema is written by hand. A
tool whose description or input depends on the setup passes a function of `registry.Setup` as `spec`. The agent
image carries no `hpcagent_bench` and no plugin dependencies, so an agent-side tool stays a thin call to its route.

## Judge routes

A route is a `@route` handler (`hpcagent_bench.harness.routes`). A route that changes state answers `POST` and keeps
its state per episode in an `EpisodeStore`, so no two agents share it:

```python
# routes/my_tool.py
from http import HTTPMethod, HTTPStatus

from hpcagent_bench.harness.routes import EpisodeStore, Judge, RouteRequest, route

STATES: EpisodeStore[MyState] = EpisodeStore(MyState)   # built from the episode id on first use


@route("my_tool", methods=(HTTPMethod.POST,))
def my_tool(judge: Judge, request: RouteRequest) -> None:
    with judge.device_slot() as slot:      # never beside a timed grade
        if slot is None:
            return
        state = STATES.get(request.episode_id)
        ...
        judge.reply(HTTPStatus.OK, answer)
```

The router relays every request it does not answer itself. A `POST` reaches the handler only from an episode the
run started, carrying that episode's token, as on `/score`; the judge has already checked the rank.

## Tuners

A `[tuners.<key>]` block registers a tuner and one column `dace_<arch>_<stage>_<key>` per DaCe stage it tunes:

```toml
[tuners.alloy_profile]
name = "Alloy profile"                    # printed beside the stage on a figure
full_name = "Alloy (profiled)"            # printed by a table
optimizer = "alloy.hb_plugin.tuners:AlloyProfileOptimizer"
dace = ["cpu_canon", "cpu_parallel", "cpu_autoopt"]
```

`optimizer` names a subclass of `hpcagent_bench.harness.optimizers.TunerOptimizer`. It implements
`tune(task, pipeline)`: tune the SDFG the column's pipeline (`canon_cpu`, ...) produces and return
`self.deliver(task, code, link)`, the library the judge grades. A folder plugin's module imports as
`hpcagent_bench_plugins.<name>.<module>`. The columns register after every shipped one; a key the repository retired
(Alloy's) takes back its slot, so its figures keep their colours.

## Where plugins come from

| Source | How |
|---|---|
| `plugins/` | the shipped plugins, and any folder dropped in beside them |
| `HPCAGENT_BENCH_USER_PLUGIN_DIRS` | `:`-separated folders, each a plugin or a folder of plugins |
| an installed package | `[project.entry-points."hpcagent_bench.plugins"] name = "package.with.the.plugin"`; read where the package is installed (a judge run outside the cluster launcher); on the cluster, name its folder in `HPCAGENT_BENCH_USER_PLUGIN_DIRS` |

The cluster launcher binds every plugin folder read-only into every role's container at
`/opt/hpcagent-bench-plugins/<folder>` and exports the list as `HPCAGENT_BENCH_PLUGIN_DIRS`, so the agent serves the
tools and the judge the routes of the same plugins. `python -m hpcagent_bench.plugins roots` prints what a process
finds.

## Example: NestForge

[NestForge](https://github.com/spcl/NestForge) ships as a plugin in its own repository (`plugin/nestforge`): one
agent-side module registers a tool per optimization phase, each a thin `POST /nestforge/<tool>`, and one judge route
serves each episode's kernel from a session process of its own, answering with whole-program, offloading and
per-kernel performance. Its packet lists only the tools the session serves; a later phase's tool answers 501.
