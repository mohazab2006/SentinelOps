"""Tests for the console API: HTTP endpoints, the live WebSocket and LiveRun."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sentinelops.api import app as app_module
from sentinelops.api.app import (
    ALERT_PERSISTENCE,
    ALERT_THRESHOLD,
    VISIBLE_EVENTS,
    LiveRun,
    create_app,
)
from sentinelops.detect import IsolationDetector, healthy_windows
from sentinelops.detect.features import WARMUP_TICKS
from sentinelops.scenarios import FAMILIES, Scenario
from sentinelops.sim.faults import Fault, FaultKind
from sentinelops.sim.topology import default_topology

TOPO = default_topology()
SHORT = WARMUP_TICKS + 20  # a few scored ticks after the warm-up


def _short_scenario(family: str, seed: int, topology: object) -> Scenario:
    """Stand-in for ``generate``: real families need 1600+ ticks, too slow to stream here."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return Scenario(family, seed, SHORT, ())


@cache
def _detector() -> IsolationDetector:
    """A detector fitted on two healthy runs (built once)."""
    return IsolationDetector(seed=0).fit(healthy_windows(TOPO, [0, 1], horizon=300))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(app_module, "generate", _short_scenario)
    return TestClient(create_app(topology=TOPO, detector=_detector()))


def _messages(client: TestClient, url: str) -> list[dict[str, Any]]:
    """Every message of a live run, until the server closes the socket."""
    messages = []
    with client.websocket_connect(url) as ws:
        try:
            while True:
                messages.append(json.loads(ws.receive_text()))
        except WebSocketDisconnect:
            pass
    return messages


# ---------------------------------------------------------------------- HTTP


def test_health(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"


def test_topology_has_every_site_with_coordinates(client: TestClient) -> None:
    body = client.get("/api/topology").json()
    assert [s["id"] for s in body["sites"]] == sorted(TOPO.sites)
    lon = next(s for s in body["sites"] if s["id"] == "lon")
    assert lon["name"] == "London"
    assert lon["lat"] == pytest.approx(51.5074)
    assert lon["servers"] == ["lon-1", "lon-2"]
    assert len(body["regions"]) == len(TOPO.regions)


def test_options(client: TestClient) -> None:
    body = client.get("/api/options").json()
    assert body["families"] == list(FAMILIES)
    assert "noop" in body["agents"]
    assert body["alert_threshold"] == ALERT_THRESHOLD


# ----------------------------------------------------------------- WebSocket


def test_live_run_sends_start_every_tick_and_finish(client: TestClient) -> None:
    messages = _messages(client, "/ws/live?family=crash&seed=3&speed=1000")
    start, *ticks, finish = messages
    assert start == {
        "type": "run_started",
        "family": "crash",
        "seed": 3,
        "agent": "noop",
        "horizon": SHORT,
        "alert_threshold": ALERT_THRESHOLD,
    }
    assert [m["tick"] for m in ticks] == list(range(SHORT))
    assert all(m["type"] == "tick" for m in ticks)
    assert finish["type"] == "run_finished"
    assert finish["tick"] == SHORT - 1
    assert 0.0 <= finish["legit_failure_share"] <= 1.0
    # Scores start once the first full window exists.
    first_scored = ticks[WARMUP_TICKS - 1]["sites"]["lon"]
    assert ticks[WARMUP_TICKS - 2]["sites"]["lon"]["status"] == "warmup"
    assert first_scored["score"] is not None
    assert first_scored["status"] in {"ok", "watch"}
    assert set(ticks[0]["sites"]) == set(TOPO.sites)


@pytest.mark.parametrize(
    ("query", "problem"),
    [
        ("family=meteor", "unknown family"),
        ("agent=oracle", "unknown agent"),
        ("speed=0", "speed must be"),
        ("speed=5000", "speed must be"),
        ("seed=-1", "seed must be"),
    ],
)
def test_bad_live_parameters_get_an_error_and_close(
    client: TestClient, query: str, problem: str
) -> None:
    with client.websocket_connect(f"/ws/live?{query}") as ws:
        message = json.loads(ws.receive_text())
        assert message["type"] == "error"
        assert problem in message["message"]
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
    assert closed.value.code == 1008


def test_default_detector_is_fitted_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "generate", _short_scenario)
    with TestClient(create_app(topology=TOPO)) as client:
        messages = _messages(client, "/ws/live?speed=1000")
    assert messages[-1]["type"] == "run_finished"


def test_built_console_is_served_at_the_root(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<h1>console</h1>", encoding="utf-8")
    client = TestClient(create_app(topology=TOPO, detector=_detector(), static_dir=tmp_path))
    assert "console" in client.get("/").text
    assert client.get("/api/health").status_code == 200  # API routes still win


def test_console_leaving_mid_run_ends_it_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "generate", _short_scenario)
    # A given detector skips the startup fit; the context manager runs the lifespan.
    with TestClient(create_app(topology=TOPO, detector=_detector())) as client:
        with client.websocket_connect("/ws/live?speed=50") as ws:
            assert json.loads(ws.receive_text())["type"] == "run_started"
            assert json.loads(ws.receive_text())["tick"] == 0
        # The server notices on its next send and stops; the app keeps serving.
        assert client.get("/api/health").status_code == 200


def test_the_real_server_can_speak_websocket() -> None:
    # The test client needs no WebSocket library, but uvicorn does: without one,
    # every /ws/live request is answered 404 (caught running `sentinelops serve`).
    from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol

    assert AutoWebSocketsProtocol is not None


# ------------------------------------------------------------------- LiveRun


def _crash_run(agent: str = "noop") -> tuple[LiveRun, list[Any]]:
    """Crash one London server shortly after the warm-up; play the whole run."""
    crash = Fault(FaultKind.CRASH, "lon-1", onset=WARMUP_TICKS + 30, duration=None, severity=1.0)
    scenario = Scenario("crash", 0, WARMUP_TICKS + 150, (crash,))
    run = LiveRun(scenario, agent, TOPO, _detector())
    frames = []
    while not run.done:
        frames.append(run.step())
    return run, frames


def test_crashed_site_turns_red_and_stays_red() -> None:
    run, frames = _crash_run()
    lon = [f.sites["lon"].status for f in frames]
    onset = WARMUP_TICKS + 30
    assert "alert" not in lon[:onset]  # no red before the fault
    first_red = lon.index("alert")
    assert onset < first_red
    # Red needs ALERT_PERSISTENCE crossings in a row: the previous ones were "watch".
    assert lon[first_red - ALERT_PERSISTENCE + 1 : first_red] == ["watch"] * (ALERT_PERSISTENCE - 1)
    assert lon[-1] == "alert"
    with pytest.raises(RuntimeError):
        run.step()


def test_only_operator_visible_events_are_streamed() -> None:
    run, frames = _crash_run(agent="rules")
    streamed = [e for f in frames for e in f.events]
    assert {e["event"] for e in streamed} <= VISIBLE_EVENTS
    assert any(e["event"] == "action" for e in streamed)  # the rules agent reacted
    assert any(e.name == "fault_started" for e in run.env.events)  # ...but truth stayed home
    assert run.finished().tick == frames[-1].tick
