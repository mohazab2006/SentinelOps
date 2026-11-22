"""Tests for the ``sentinelops run`` command."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import MappingProxyType

import pytest

from sentinelops import cli
from sentinelops.eval import harness

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


def test_unavailable_agent_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ghost = MappingProxyType({**harness.AGENTS, "rules": "sentinelops.ghost:Ghost"})
    monkeypatch.setattr(harness, "AGENTS", ghost)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "--family", "crash", "--agent", "rules"])
    assert exit_info.value.code == 2
    assert "agent 'rules' is not available" in capsys.readouterr().err


def test_unusable_events_path_is_a_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    taken = tmp_path / "events"
    taken.write_text("not a directory")
    argv = ["run", "--family", "crash", "--agent", "noop", *SHORT, "--events", str(taken)]
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2
    assert "cannot write event logs to" in capsys.readouterr().err
    assert taken.read_text() == "not a directory"  # nothing was simulated or overwritten


def test_report_detection_prints_detection_and_diagnosis(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = ["report-detection", "--tune-seeds", "100", "--train-seeds", "0"]
    assert cli.main([*args, "--healthy-seeds", "0", "--families", "crash", "hardware"]) == 0
    captured = capsys.readouterr()
    assert "tune seeds 100 | families: crash hardware" in captured.out
    assert "Detection (benign surges count as positives)" in captured.out
    assert "Sensitivity: benign surges count as healthy" in captured.out
    assert "Benign-surge windows alerted: 0.0% of 0" in captured.out
    assert "Diagnosis given an alert" in captured.out
    assert "labelling 2 tune runs" in captured.err  # progress goes to stderr


@pytest.mark.parametrize(
    ("args", "problem"),
    [
        (["--tune-seeds", "1000"], "not in the tune split"),  # the held-out test split
        (["--tune-seeds", "5"], "not in the tune split"),
        (["--train-seeds", "100"], "not in the train split"),
    ],
)
def test_report_detection_refuses_seeds_from_another_split(
    args: list[str], problem: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["report-detection", *args])
    assert exit_info.value.code == 2
    assert problem in capsys.readouterr().err


def test_serve_starts_uvicorn_with_the_console(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn

    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    assert cli.main(["serve", "--port", "8123", "--console", str(tmp_path)]) == 0
    (app, kwargs) = calls[0]
    assert kwargs == {"host": "127.0.0.1", "port": 8123}
    assert app.title == "SentinelOps console API"  # type: ignore[attr-defined]


def test_serve_says_when_there_is_no_built_console(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: None)
    assert cli.main(["serve", "--console", str(tmp_path / "missing")]) == 0
    assert "serving the API only" in capsys.readouterr().err
