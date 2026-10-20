# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A closed figure option as a CLI flag: parsed to its :class:`~hpcagent_agent.coded.Coded` member."""

import argparse

from hpcagent_agent.coded import Coded

__all__ = ["add_argument"]


def add_argument[Choice: Coded](
    parser: argparse.ArgumentParser, flag: str, kind: type[Choice], default: Choice, help_text: str
) -> None:
    """Add ``flag`` to ``parser``, read as a member of ``kind``; the parsed value is the member itself."""

    def read(text: str) -> Choice:
        try:
            return kind.parse(text)
        except ValueError as unknown:
            raise argparse.ArgumentTypeError(str(unknown)) from None

    parser.add_argument(
        flag, type=read, choices=tuple(kind), default=default, metavar="{" + ",".join(kind.texts()) + "}",
        help=help_text,
    )  # fmt: skip
