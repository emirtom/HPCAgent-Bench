# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A plugin's ``[install]`` command installs its package into the interpreter the judge runs in: ``{python}`` and
``{root}`` are filled in, it runs in ``root``, a plugin without one is left alone, and a failure names the plugin."""

import pathlib
import sys

import pytest
from hpcagent_agent import plugins as agent_plugins

from hpcagent_bench import plugins


def plugin_folder(repo: pathlib.Path, name: str, install: str) -> pathlib.Path:
    """``repo/plugin/<name>``, a plugin whose manifest carries ``install`` (TOML, empty for none)."""
    folder = repo / "plugin" / name
    folder.mkdir(parents=True)
    (folder / "plugin.toml").write_text(f'name = "{name}"\n{install}', encoding="utf-8")
    return folder


def test_the_install_command_runs_in_its_root_with_the_interpreter_and_root_filled_in(tmp_path: pathlib.Path) -> None:
    folder = plugin_folder(tmp_path, "probe", "")

    record = "import os, sys, pathlib; pathlib.Path(sys.argv[2]).write_text(sys.executable + '|' + sys.argv[1] + '|' + os.getcwd())"
    out = tmp_path / "installed.txt"
    folder.joinpath("plugin.toml").write_text(
        f'name = "probe"\n[install]\nroot = "../.."\ncommand = ["{{python}}", "-c", "{record}", "{{root}}", "{out}"]\n'
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'probe'\n")

    sut = plugins.install(agent_plugins.find([folder]), sys.executable)

    assert sut == ["probe"]
    python, root, cwd = out.read_text().split("|")
    assert python == sys.executable and root == cwd, "the command runs in the root it is given"
    assert pathlib.Path(root).name == tmp_path.name and pathlib.Path(root) != tmp_path, "a writable copy of the root"


def test_the_fingerprint_follows_the_plugin_sources(tmp_path: pathlib.Path) -> None:
    folder = plugin_folder(tmp_path, "probe", '\n[install]\nroot = "../.."\ncommand = ["{python}", "-V"]\n')
    found = agent_plugins.find([folder])
    before = plugins.fingerprint(found)

    (tmp_path / "module.py").write_text("changed = True\n")

    assert plugins.fingerprint(found) != before
    assert plugins.fingerprint(
        agent_plugins.find([plugin_folder(tmp_path / "other", "plain", "")])
    ) == plugins.fingerprint(())


def test_a_plugin_without_an_install_command_is_left_alone(tmp_path: pathlib.Path) -> None:
    folder = plugin_folder(tmp_path, "plain", "")

    assert plugins.install(agent_plugins.find([folder])) == []


def test_a_failing_install_command_names_its_plugin(tmp_path: pathlib.Path) -> None:
    folder = plugin_folder(tmp_path, "broken", '\n[install]\ncommand = ["{python}", "-c", "raise SystemExit(3)"]\n')

    with pytest.raises(RuntimeError, match="plugin 'broken'"):
        plugins.install(agent_plugins.find([folder]))


def test_an_install_block_without_a_command_is_refused_by_its_file(tmp_path: pathlib.Path) -> None:
    folder = plugin_folder(tmp_path, "empty", "\n[install]\ncommand = []\n")

    with pytest.raises(ValueError, match="plugin.toml"):
        agent_plugins.find([folder])
