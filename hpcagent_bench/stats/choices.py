# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A closed figure option (an ``enum.auto()`` member) at the CLI or config edge: the only place it becomes text
(:func:`spelling`) or is read back from text (:func:`parse`, :func:`add_argument`)."""

import argparse
import enum

__all__ = ["add_argument", "parse", "spelling"]


def spelling(member: enum.Enum) -> str:
    """``member`` as text: its name in lower case, ``_`` written ``-``."""
    return member.name.lower().replace("_", "-")


def parse[Choice: enum.Enum](kind: type[Choice], text: str) -> Choice:
    """The member of ``kind`` that :func:`spelling` writes as ``text``; ``ValueError`` naming the choices otherwise."""
    for member in kind:
        if spelling(member) == text:
            return member
    raise ValueError(f"{text!r} is not one of {[spelling(member) for member in kind]}")


def add_argument[Choice: enum.Enum](
    parser: argparse.ArgumentParser, flag: str, kind: type[Choice], default: Choice, help_text: str
) -> None:
    """Add ``flag`` to ``parser``, read as a member of ``kind`` by its :func:`spelling`; the parsed value is the
    member itself."""

    def read(text: str) -> Choice:
        try:
            return parse(kind, text)
        except ValueError as unknown:
            raise argparse.ArgumentTypeError(str(unknown)) from None

    parser.add_argument(
        flag, type=read, choices=tuple(kind), default=default,
        metavar="{" + ",".join(spelling(member) for member in kind) + "}", help=help_text,
    )  # fmt: skip
