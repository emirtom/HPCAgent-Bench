# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""External skill packets (``skills.external``): a study's pages, read from a folder outside the repository.

- a folder's ``skills.yaml`` names packets and the page files each stages; a page is known by its frontmatter name;
- an external packet resolves like a shipped one, its pages in the order the folder lists them;
- it carries pages only (no tools, env or method), every page lies inside its folder and carries the frontmatter
  the task index needs, and no external name replaces a shipped one or another folder's.
"""

import pathlib
from collections.abc import Iterator

import pydantic
import pytest

from hpcagent_bench import config, external_skills, packets
from hpcagent_bench import study_tags as tags
from hpcagent_bench.harness import prompts
from hpcagent_bench.vocabulary import PACKETS

PAGE = '---\nname: {name}\ndescription: "What {name} covers."\nwhen: "ALWAYS read {name} first"\n---\n\n# {name}\n'


def write_folder(folder: pathlib.Path, config_text: str, pages: dict[str, str]) -> pathlib.Path:
    """``folder`` with ``config_text`` as its skills.yaml and each ``relative path -> page name`` as a page."""
    for relative, name in pages.items():
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(PAGE.format(name=name), encoding="utf-8")
    (folder / external_skills.CONFIG_NAME).write_text(config_text, encoding="utf-8")
    return folder


SPLIT = """packets:
  precond-single:
    name: Preconditioners, one page
    pages: [single/precond.md]
  precond-split:
    name: Preconditioners, two pages
    pages: [split/precond-verdict.md, split/precond-verify.md]
"""
SPLIT_PAGES = {
    "single/precond.md": "precond",
    "split/precond-verdict.md": "precond-verdict",
    "split/precond-verify.md": "precond-verify",
}


@pytest.fixture(autouse=True)
def fresh_vocabulary() -> Iterator[None]:
    """Each test registers into the packet vocabulary on a clean slate and leaves it as it found it."""
    entries, orders = dict(PACKETS.entries), dict(PACKETS.orders)

    def clear() -> None:
        tags.registry.cache_clear()
        packets.tool_pages.cache_clear()
        packets.page_applies.cache_clear()
        external_skills.load_folders.cache_clear()

    clear()
    yield
    PACKETS.entries.clear()
    PACKETS.entries.update(entries)
    PACKETS.orders.clear()
    PACKETS.orders.update(orders)
    clear()


def test_an_external_packet_resolves_to_its_pages_in_folder_order(tmp_path: pathlib.Path) -> None:
    folder = write_folder(tmp_path / "ext", SPLIT, SPLIT_PAGES)
    with config.overridden("skills.external", (str(folder),)):
        assert packets.resolve("precond-split", "c").pages == ("precond-verdict", "precond-verify")
        assert packets.resolve("precond-single", "c").pages == ("precond",)
        assert packets.page_file("precond-verify") == (folder / "split/precond-verify.md").resolve()


def test_an_external_packet_takes_a_slot_after_the_shipped_ones(tmp_path: pathlib.Path) -> None:
    shipped_last = sum(slot is not None for slot in PACKETS.orders.values())
    folder = write_folder(tmp_path / "ext", SPLIT, SPLIT_PAGES)
    with config.overridden("skills.external", (str(folder),)):
        registry = tags.registry()
        assert registry.packets["precond-split"] == "Preconditioners, two pages"
        assert tags.slot("packets", "precond-single") == shipped_last
        assert tags.slot("packets", "precond-split") == shipped_last + 1


def test_the_task_index_lists_an_external_page_under_its_name(tmp_path: pathlib.Path) -> None:
    folder = write_folder(tmp_path / "ext", SPLIT, SPLIT_PAGES)
    with config.overridden("skills.external", (str(folder),)):
        by_file = {skill.file: skill for skill in prompts.load_skills()}
        assert by_file["precond-verdict"].when == "ALWAYS read precond-verdict first"


def test_no_external_folder_registers_nothing() -> None:
    before = set(PACKETS.entries)
    assert external_skills.load() == external_skills.ExternalSkills({}, {})
    tags.registry()
    assert set(PACKETS.entries) == before


@pytest.mark.parametrize("field", ["tools: [canonical_parallel_form]", "env: {X: '1'}", "method: autokernel"])
def test_an_external_packet_carries_pages_only(tmp_path: pathlib.Path, field: str) -> None:
    """A tool runs beside the judge; an external one could read what grading hides."""
    folder = write_folder(
        tmp_path / "ext", f"packets:\n  p:\n    name: P\n    pages: [a.md]\n    {field}\n", {"a.md": "page-a"}
    )
    with config.overridden("skills.external", (str(folder),)), pytest.raises(pydantic.ValidationError):
        external_skills.load()


@pytest.mark.parametrize("page", ["../outside.md", "a.txt", "missing.md"])
def test_a_page_is_a_markdown_file_inside_its_folder(tmp_path: pathlib.Path, page: str) -> None:
    (tmp_path / "outside.md").write_text(PAGE.format(name="outside"), encoding="utf-8")
    folder = write_folder(tmp_path / "ext", f"packets:\n  p:\n    name: P\n    pages: [{page}]\n", {"a.txt": "a"})
    with config.overridden("skills.external", (str(folder),)), pytest.raises(ValueError, match="not a markdown file"):
        external_skills.load()


def test_a_page_without_its_trigger_is_refused(tmp_path: pathlib.Path) -> None:
    folder = tmp_path / "ext"
    folder.mkdir()
    (folder / "a.md").write_text('---\nname: a\ndescription: "d"\n---\nbody\n', encoding="utf-8")
    (folder / external_skills.CONFIG_NAME).write_text("packets:\n  p:\n    name: P\n    pages: [a.md]\n")
    with config.overridden("skills.external", (str(folder),)), pytest.raises(ValueError, match="frontmatter"):
        external_skills.load()


def test_an_external_page_never_replaces_a_shipped_one(tmp_path: pathlib.Path) -> None:
    folder = write_folder(tmp_path / "ext", "packets:\n  p:\n    name: P\n    pages: [s.md]\n", {"s.md": "solver"})
    with config.overridden("skills.external", (str(folder),)), pytest.raises(ValueError, match="shipped skill page"):
        external_skills.load()


def test_an_external_packet_never_reuses_a_shipped_name(tmp_path: pathlib.Path) -> None:
    folder = write_folder(
        tmp_path / "ext", "packets:\n  cpf-tool-cpu:\n    name: X\n    pages: [a.md]\n", {"a.md": "a"}
    )
    with config.overridden("skills.external", (str(folder),)), pytest.raises(ValueError, match="shipped"):
        external_skills.register()


def test_two_folders_defining_one_packet_are_refused(tmp_path: pathlib.Path) -> None:
    first = write_folder(tmp_path / "one", SPLIT, SPLIT_PAGES)
    second = write_folder(
        tmp_path / "two", "packets:\n  precond-single:\n    name: X\n    pages: [b.md]\n", {"b.md": "b"}
    )
    with (
        config.overridden("skills.external", (str(first), str(second))),
        pytest.raises(ValueError, match="more than one folder"),
    ):
        external_skills.load()
