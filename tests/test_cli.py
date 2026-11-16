"""Tests for the ``sentinelops run`` command."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from sentinelops import cli

RULES_INSTALLED = importlib.util.find_spec("sentinelops.baselines.rules_agent") is not None
SHORT = ["--horizon", "1601"]  # the shortest horizon every scenario family accepts


@pytest.mark.parametrize(
    ("text", "seeds"),
    [
        ("7", [7]),
        ("0-4", [0, 1, 2, 3, 4]),
        ("0,2,10-12", [0, 2, 10, 11, 12]),
        (" 3 , 1-2 ", [3, 1, 2]),
        ("1-3,2-4", [1, 2, 3, 4]),
    ],
)
def test_parse_seeds(text: str, seeds: list[int]) -> None:
    assert cli.parse_seeds(text) == seeds


@pytest.mark.parametrize("text", ["", "a", "3-1", "-2", "1-", "1,,2", "0-x"])
def test_parse_seeds_rejects_bad_lists(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_seeds(text)


def test_run_prints_one_row_per_seed_and_the_mean(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--family", "crash", "--agent", "noop", "--seeds", "0-1", *SHORT]) == 0
    out = capsys.readouterr().out
    assert "family=crash agent=noop horizon=1601 ticks" in out
    rows = [line.split() for line in out.splitlines() if line.split()[:1] in (["0"], ["1"])]
    assert [row[0] for row in rows] == ["0", "1"]
    assert all(row[1] == "crash" for row in rows)
    assert "mean failed share over 2 run(s):" in out
    assert "(sd " in out


def test_run_writes_event_logs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_dir = tmp_path / "events"
    args = ["run", "--family", "flood", "--agent", "noop", "--seeds", "4", *SHORT]
    assert cli.main([*args, "--events", str(out_dir)]) == 0
    log = out_dir / "flood-noop-seed4.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith('{"event":"run_started"')
    assert "mean failed share over 1 run(s):" in capsys.readouterr().out


def test_event_logs_are_identical_across_invocations(tmp_path: Path) -> None:
    args = ["run", "--family", "hardware", "--agent", "noop", "--seeds", "1", *SHORT]
    cli.main([*args, "--events", str(tmp_path / "a")])
    cli.main([*args, "--events", str(tmp_path / "b")])
    name = "hardware-noop-seed1.jsonl"
    assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "--family", "crash", "--agent", "oracle"], "invalid choice: 'oracle'"),
        (["run", "--family", "earthquake", "--agent", "noop"], "invalid choice: 'earthquake'"),
        (["run", "--family", "crash", "--agent", "noop", "--horizon", "900"], "horizon"),
        (["run", "--family", "crash", "--agent", "noop", "--seeds", "5-2"], "invalid seed range"),
        (["run", "--agent", "noop"], "--family"),
    ],
)
def test_run_rejects_bad_arguments(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.skipif(RULES_INSTALLED, reason="only meaningful before the baseline exists")
def test_unavailable_agent_is_a_clear_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "--family", "crash", "--agent", "rules"])
    assert exit_info.value.code == 2
    assert "agent 'rules' is not available" in capsys.readouterr().err
