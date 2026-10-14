# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-setup offload and python-device keys resolve through ``config.env_value``, so a fused
judge grading one setup under ``config.scoped_environment(fused.judge_overlay(setup))`` reads that
setup's contract and not whatever the job's process environment happens to carry."""

import pytest

from hpcagent_bench import config, languages
from hpcagent_bench.harness import service

#: Every per-setup key submit.sh pins for an offload or a triton-device setup.
SETUP_KEYS = (
    languages.OFFLOAD_MODEL_ENV,
    languages.OFFLOAD_MEMORY_ENV,
    languages.OFFLOAD_RESIDENCY_ENV,
    languages.PYTHON_DEVICE_ENV,
)


@pytest.fixture(name="bare_process_env")
def fixture_bare_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No setup key in the process environment: only a scope can supply one."""
    for name in SETUP_KEYS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("bare_process_env")
def test_a_scoped_offload_setup_reaches_the_offload_readers() -> None:
    model = languages.OFFLOAD_MODELS[0]
    scope = {
        languages.OFFLOAD_MODEL_ENV: model,
        languages.OFFLOAD_MEMORY_ENV: "unified",
        languages.OFFLOAD_RESIDENCY_ENV: "device",
    }
    with config.scoped_environment(scope):
        assert languages.offload_model() == model
        assert languages.offload_memory_mode() == "unified"
        assert languages.offload_device_residency()
    assert languages.offload_model() == ""
    assert languages.offload_memory_mode() == "explicit"
    assert not languages.offload_device_residency()


@pytest.mark.usefixtures("bare_process_env")
def test_a_scoped_python_device_setup_is_not_refused() -> None:
    with config.scoped_environment({languages.PYTHON_DEVICE_ENV: "1"}):
        assert languages.python_device_setup()
        assert service.python_residency_refusal(languages.PYTHON_DEVICE_LANGUAGE) is None
    assert not languages.python_device_setup()
    assert service.python_residency_refusal(languages.PYTHON_DEVICE_LANGUAGE) is not None


def test_a_scoped_none_removes_a_process_setup_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A setup that does not pin a key the job env carries must not inherit it."""
    monkeypatch.setenv(languages.PYTHON_DEVICE_ENV, "1")
    monkeypatch.setenv(languages.OFFLOAD_MODEL_ENV, languages.OFFLOAD_MODELS[0])
    with config.scoped_environment({languages.PYTHON_DEVICE_ENV: None, languages.OFFLOAD_MODEL_ENV: None}):
        assert not languages.python_device_setup()
        assert languages.offload_model() == ""
