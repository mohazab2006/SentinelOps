"""Smoke tests for the package layout and command-line entry point."""

from __future__ import annotations

import importlib

import pytest

import sentinelops
from sentinelops import cli

SUBPACKAGES = (
    "sim",
    "scenarios",
    "detect",
    "diagnose",
    "plan",
    "policy",
    "verify",
    "agent",
    "baselines",
    "eval",
    "api",
)


def test_version_is_exposed() -> None:
    assert sentinelops.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_is_importable_and_documented(name: str) -> None:
    module = importlib.import_module(f"sentinelops.{name}")
    assert module.__doc__, f"sentinelops.{name} should have a module docstring"


def test_cli_without_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    assert "usage: sentinelops" in capsys.readouterr().out


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"sentinelops {sentinelops.__version__}"
