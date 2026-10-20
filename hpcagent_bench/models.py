# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The LLMs, standalone optimizers, agent harnesses, languages and devices a figure names.

``order`` is the slot each entry's colour and marker are read from (:mod:`hpcagent_bench.vocabulary`,
``docs/extending/registry.md``). Never renumber one: a new entry takes the next free slot, so no entry already
drawn changes colour or shape.
"""

from hpcagent_bench.columns import FRAMEWORKS
from hpcagent_bench.vocabulary import DEVICES, HARNESSES, LANGUAGES, MODELS, OPTIMIZERS, DaceStage, ModelEntry

__all__ = ["dace_flavors"]

# LLMs
# A tag, its display name and the checkpoint the tag is expected to serve. The serving engine and the
# quantisation are deliberately NOT in a name: `kimi27sglang` names SGLang only because the runner had
# to tell two setups apart, and `-FP8` is a precision; neither is the model. Where the engine or the
# precision IS the variable, the caption says so once.


MODELS.add("qwen38", ModelEntry("Qwen3.8-27B", "Qwen/Qwen3.8-27B-FP8"), order=0)


MODELS.add("oss120b", ModelEntry("OSS-120B", "openai/gpt-oss-120b"), order=1)


MODELS.add("kimi27sglang", ModelEntry("Kimi-K2.7", "moonshotai/Kimi-K2.7-Code"), order=2)


MODELS.add("glm53", ModelEntry("GLM-5.3", "zai-org/GLM-5.3"), order=3)


def dace_flavors(stage: DaceStage) -> tuple[str, ...]:
    """Every DaCe flavor on ``stage``, tuned or not: one optimizer on either device."""
    return tuple(key for key, column in FRAMEWORKS.entries.items() if column.stage is stage)


# Standalone optimizers
# A compiler or a pipeline that stands where an LLM stands on a figure. Their markers count from the
# end of the pool, so a new LLM never repaints them. A skill packet an agent is given (cpf, cpf-src) is
# a packet and a colour, never an optimizer. A standalone optimizer's row keeps its framework's colour;
# device variants are aliases, since the optimizer is the same one on either device.


OPTIMIZERS.add("dace", "DaCe+AOpt", order=0, aliases=(*dace_flavors(DaceStage.PARALLEL), "dace_cpu", "dace_gpu"))


OPTIMIZERS.add("cpf", "DaCe+CPF", order=1, aliases=dace_flavors(DaceStage.CANON))


OPTIMIZERS.add("pluto", "Pluto", order=2)


OPTIMIZERS.add("ppcg_hip", "PPCG (CUDA via hipify)", order=3)


# Agent harnesses
# The ``harness`` column value. The model is a separate column: a setup that swaps the harness keeps
# its model.


HARNESSES.add("claude", "Claude Code", order=0)


HARNESSES.add("miniswe", "mini-SWE-agent", order=1)


HARNESSES.add("openhands", "OpenHands", order=2)


# Slot 3 is Optimas', which it held before it was dropped (fe5373d2c) and holds again.
HARNESSES.add("optimas", "Optimas", order=3)


HARNESSES.add("anthropic", "Anthropic Messages API", order=4)


HARNESSES.add("openai", "OpenAI Agents SDK", order=5)


# Languages
# The language the SETUP asked for, in proper names: str.title() would give "Cpp" and "Hip". A GPU-only
# language (hip, triton, cuda) is a language here, not a framework: it is what the agent was asked to
# write.


LANGUAGES.add("c", "C", order=0)


LANGUAGES.add("cpp", "C++", order=1)


LANGUAGES.add("fortran", "Fortran", order=2)


LANGUAGES.add("python", "Python", order=3)


LANGUAGES.add("cuda", "CUDA", order=4)


LANGUAGES.add("hip", "HIP", order=5)


LANGUAGES.add("triton", "Triton", order=6)


LANGUAGES.add("omp", "OpenMP", order=7)


# Devices
# Where the kernel was TIMED. `cpu` is the default a launcher stamps when it says nothing.


DEVICES.add("cpu", "CPU", order=0)


DEVICES.add("gpu", "GPU", order=1)


DEVICES.add("cpu-multinode", "CPU, Multi-Node", order=2)


DEVICES.add("gpu-multinode", "GPU, Multi-Node", order=3)
