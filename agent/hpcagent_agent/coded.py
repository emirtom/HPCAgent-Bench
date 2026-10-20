# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Closed categories as ``enum.auto()`` enums, spelled as text only at a boundary via :attr:`Coded.text` /
:meth:`Coded.parse` (a member is never compared to a string). A spelling is the lower-case name joined by
``SEPARATOR`` unless ``SPELLINGS`` overrides it; spellings of one class are distinct."""

import enum
import functools
from collections.abc import Mapping
from typing import Any, cast

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
    """Base of a closed category; a subclass may set ``SEPARATOR`` and ``SPELLINGS``, both ``enum.nonmember``."""

    @property
    def text(self) -> str:
        """This member's boundary spelling."""
        own = type(self).__dict__
        separator = cast("str", own.get("SEPARATOR", SEPARATOR))
        overrides = cast("Mapping[str, str]", own.get("SPELLINGS", {}))
        return overrides.get(self.name) or self.name.lower().replace("_", separator)

    @classmethod
    def parse[CodedT: Coded](cls: type[CodedT], text: str) -> CodedT:
        """The member spelled ``text``; ValueError naming the spellings when none is."""
        member = spellings(cls).get(text)
        if member is None:
            raise ValueError(f"{text!r} is not a {cls.__name__}; one of {', '.join(spellings(cls))}")
        return cast("CodedT", member)  # spellings(cls) holds cls's members only

    @classmethod
    def texts(cls) -> tuple[str, ...]:
        """Every spelling, in definition order."""
        return tuple(spellings(cls))

    @classmethod
    def validate[CodedT: Coded](cls: type[CodedT], value: object) -> CodedT:
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
