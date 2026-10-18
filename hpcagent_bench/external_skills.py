# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Skill pages and packets kept outside the repository: a study's own treatment, run without committing it.

``skills.external`` (``HPCAGENT_BENCH_SKILLS_EXTERNAL``) lists folders. Each holds a :data:`CONFIG_NAME` naming its
packets and, for each, the page files it stages, as paths relative to the folder::

    packets:
      preconditioner-v4-single:
        name: Solver preconditioners v4, one page
        pages: [single/solver-preconditioners.md]
      preconditioner-v4-split:
        name: Solver preconditioners v4, five pages
        pages: [split/solver-preconditioners-verdict.md, split/solver-preconditioners-frozen.md]

A page is a markdown file with the shipped pages' frontmatter (``name``, ``description``, ``when``); its ``name``
is the page's identity, the name the task lists it under and the file it is staged as. An external packet stages
pages and nothing else: it carries no tools, env switches or methods, since a tool runs code beside the judge and
an external one could read what grading hides. A packet may compose shipped packets and pages (``packets``).

Names are one namespace with the shipped ones: an external page or packet that reuses a shipped name, or another
folder's, is refused, so a study never silently replaces a page another study measured.
"""

import dataclasses
import functools
import pathlib

import pydantic
import yaml

from hpcagent_bench import config
from hpcagent_bench.skill_frontmatter import split_frontmatter
from hpcagent_bench.vocabulary import PACKETS, PacketDef

__all__ = [
    "CONFIG_NAME",
    "ExternalPacket",
    "ExternalSkills",
    "ExternalSkillsFile",
    "load",
    "page_path",
    "register",
]

#: The file in each external folder that names its packets.
CONFIG_NAME = "skills.yaml"


class ExternalPacket(pydantic.BaseModel):
    """One packet of an external folder: its display name and the pages it stages. No ``tools``, ``env`` or
    ``method``: :mod:`pydantic` refuses any field this model does not declare."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str = pydantic.Field(min_length=1)
    #: Page files, relative to the folder, in the order the task lists them.
    pages: tuple[str, ...] = pydantic.Field(default=(), min_length=1)
    #: Shipped packets or pages this one stages too.
    packets: tuple[str, ...] = ()
    short: str = ""
    marker: str = ""


class ExternalSkillsFile(pydantic.BaseModel):
    """A folder's :data:`CONFIG_NAME`."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    packets: dict[str, ExternalPacket] = pydantic.Field(min_length=1)


@dataclasses.dataclass(frozen=True, slots=True)
class ExternalSkills:
    """Every external folder's pages and packets, merged."""

    #: Page name (its frontmatter ``name``) -> the file.
    pages: dict[str, pathlib.Path]
    #: Packet key -> its definition, ``skills`` naming pages by name.
    packets: dict[str, PacketDef]


def read_folder(folder: pathlib.Path, shipped_pages: frozenset[str]) -> ExternalSkills:
    """One folder's pages and packets. Raises ``ValueError`` naming the folder and what is wrong."""
    config_file = folder / CONFIG_NAME
    if not config_file.is_file():
        raise ValueError(f"skills.external folder {folder} has no {CONFIG_NAME}")
    declared = ExternalSkillsFile.model_validate(yaml.safe_load(config_file.read_text(encoding="utf-8")) or {})
    root = folder.resolve()
    pages: dict[str, pathlib.Path] = {}
    packets: dict[str, PacketDef] = {}
    for key, packet in declared.packets.items():
        names: list[str] = []
        for relative in packet.pages:
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or path.suffix != ".md" or not path.is_file():
                raise ValueError(f"{config_file}: packet {key!r} page {relative!r} is not a markdown file in {root}")
            meta = split_frontmatter(path.read_text(encoding="utf-8"))[0]
            if not (meta.name and meta.description and meta.when):
                raise ValueError(f"{path}: an external page carries name, description and when in its frontmatter")
            if meta.name in shipped_pages:
                raise ValueError(f"{path}: page name {meta.name!r} is a shipped skill page")
            if pages.get(meta.name, path) != path:
                raise ValueError(f"{config_file}: two files are page {meta.name!r}: {pages[meta.name]} and {path}")
            pages[meta.name] = path
            names.append(meta.name)
        packets[key] = PacketDef(
            name=packet.name,
            skills=tuple(names),
            packets=packet.packets,
            env=(),
            method="",
            tools=(),
            device="",
            frozen="",
            marker=packet.marker,
            short=packet.short,
        )
    return ExternalSkills(pages, packets)


@functools.lru_cache(maxsize=4, typed=True)
def load_folders(folders: tuple[str, ...], shipped_pages: frozenset[str]) -> ExternalSkills:
    """The pages and packets of ``folders``, merged; a name two folders both define is refused."""
    pages: dict[str, pathlib.Path] = {}
    packets: dict[str, PacketDef] = {}
    for folder in folders:
        found = read_folder(pathlib.Path(folder), shipped_pages)
        clash = sorted((pages.keys() & found.pages.keys()) | (packets.keys() & found.packets.keys()))
        if clash:
            raise ValueError(f"skills.external: {clash} defined in more than one folder")
        pages.update(found.pages)
        packets.update(found.packets)
    return ExternalSkills(pages, packets)


def shipped_page_names() -> frozenset[str]:
    """The shipped skill pages, by directory name."""
    shipped = pathlib.Path(__file__).resolve().parent / "skills"
    return frozenset(entry.name for entry in shipped.iterdir() if (entry / "SKILL.md").is_file())


def load() -> ExternalSkills:
    """The external pages and packets ``skills.external`` names; empty when it names none."""
    folders = tuple(str(folder) for folder in config.current().skills.external)
    if not folders:
        return ExternalSkills({}, {})
    return load_folders(folders, shipped_page_names())


def page_path(page: str) -> pathlib.Path | None:
    """The file of external page ``page``; None when no external folder defines it."""
    return load().pages.get(page)


def register() -> None:
    """Add every external packet to the packet vocabulary, after the shipped ones in key order, so a figure gives
    it a slot of its own. A key the vocabulary already holds is refused unless it is the same definition."""
    shipped_pages = shipped_page_names()
    for key, definition in sorted(load().packets.items()):
        if key in shipped_pages:
            raise ValueError(f"skills.external: packet {key!r} is a shipped skill page's name")
        if key in PACKETS.entries:
            if PACKETS.entries[key] != definition:
                raise ValueError(f"skills.external: packet {key!r} is a shipped packet")
            continue
        PACKETS.add(key, definition, order=PACKETS.next_order(), aliases=())
