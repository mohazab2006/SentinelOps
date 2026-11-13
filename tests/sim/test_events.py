"""Tests for the run event log."""

from __future__ import annotations

import json

import pytest

from sentinelops.sim.events import EventLog


def test_event_log_round_trips_to_sorted_json_lines() -> None:
    log = EventLog()
    log.append(0, "run_started", seed=3)
    log.append(5, "action", kind="restart", target="ott-1", weight=None)
    assert len(log) == 2
    assert [e.name for e in log] == ["run_started", "action"]
    lines = log.to_jsonl().splitlines()
    assert json.loads(lines[1]) == {
        "event": "action",
        "kind": "restart",
        "target": "ott-1",
        "tick": 5,
        "weight": None,
    }
    assert lines[1] == json.dumps(json.loads(lines[1]), sort_keys=True, separators=(",", ":"))
    assert [e.tick for e in log.named("action")] == [5]


def test_event_log_is_append_only_in_tick_order() -> None:
    log = EventLog()
    log.append(5, "a")
    with pytest.raises(ValueError):
        log.append(4, "b")
    with pytest.raises(TypeError):
        log.named("a")[0].data["x"] = 1  # type: ignore[index]
