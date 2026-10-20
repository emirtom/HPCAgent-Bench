# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Closed categories as enums whose members are ``enum.auto()``, spelled as text only at a boundary.

Code passes members (``InputMode.PY_BINDING``), never their spelling: a member is not a string, so comparing one
to a string literal is always False and a type checker flags it. The spelling (``"py-binding"``) exists where a
value leaves the process or enters it -- a YAML config, a JSON wire model, a results-database column, a CLI flag --
and is read and written in exactly one way, :attr:`Coded.text` / :meth:`Coded.parse`. A pydantic field typed with a
:class:`Coded` enum validates the spelling (or a member) and serializes back to the spelling, so a model is the
boundary codec too.

A member's spelling is its name in lower case with ``_`` as the separator, unless the class sets another
``SEPARATOR`` or spells a member differently in ``SPELLINGS`` (both ``enum.nonmember``)::

    class InputMode(Coded):
        SEPARATOR = enum.nonmember("-")

        PY_BINDING = enum.auto()   # "py-binding"
        SOURCE = enum.auto()       # "source"

    class Language(Coded):
        SPELLINGS = enum.nonmember({"CPP": "c++"})

The spellings of one class are distinct (checked when a class is first spelled). This module lives in the agent
package because both sides of the judge-agent wire use it; it imports only the standard library and pydantic-core.

``Coded`` itself declares neither name: mypy reads every attribute an enum body assigns as final, so a subclass
redefining a base's ``SEPARATOR`` would be refused. A class's own definitions are read from its namespace, with
:data:`SEPARATOR` and no overrides as the defaults; the class methods are typed by a type parameter for the same
checker, which resolves ``Self`` on an enum class method to the base.
"""

import enum
import functools
from collections.abc import Mapping
from typing import Any, Self, cast

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

__all__ = ["SEPARATOR", "Coded", "spellings"]

#: What joins the words of a member's name in its spelling when the class sets no ``SEPARATOR``.
SEPARATOR = "_"


@functools.lru_cache(maxsize=None, typed=True)
def spellings(cls: type["Coded"]) -> Mapping[str, "Coded"]:
    """``cls``'s members by spelling; raises when two members share one."""
    by_text: dict[str, Coded] = {}
    for member in cls:
        text = member.text
        if text in by_text:
            raise TypeError(f"{cls.__name__}.{member.name} and .{by_text[text].name} are both spelled {text!r}")
        by_text[text] = member
    return by_text


class Coded(enum.Enum):
    """Base of a closed category (see the module docstring). Members are ``enum.auto()``; a subclass may set
    ``SEPARATOR`` (what joins the words of a member's name in its spelling) and ``SPELLINGS`` (member name -> a
    spelling that is not the name's, ``"c++"``), both ``enum.nonmember``."""

    @property
    def text(self) -> str:
        """This member's boundary spelling."""
        own = type(self).__dict__
        separator = cast("str", own.get("SEPARATOR", SEPARATOR))
        overrides = cast("Mapping[str, str]", own.get("SPELLINGS", {}))
        return overrides.get(self.name) or self.name.lower().replace("_", separator)

    @classmethod
    def parse(cls, text: str) -> Self:
        """The member spelled ``text``; ValueError naming the spellings when none is."""
        member = spellings(cls).get(text)
        if member is None:
            raise ValueError(f"{text!r} is not a {cls.__name__}; one of {', '.join(spellings(cls))}")
        return cast("Self", member)  # spellings(cls) holds cls's members only

    @classmethod
    def texts(cls) -> tuple[str, ...]:
        """Every spelling, in definition order."""
        return tuple(spellings(cls))

    @classmethod
    def validate(cls, value: object) -> Self:
        """A member, or the member a spelling names: what a boundary model accepts."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.parse(value)
        raise ValueError(f"{value!r} is not a {cls.__name__} spelling")

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type[Any], handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls.validate,
            serialization=core_schema.plain_serializer_function_ser_schema(lambda member: member.text),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": "string", "enum": list(cls.texts())}
