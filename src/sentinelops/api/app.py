"""The console API: the static network over HTTP, live runs over a WebSocket.

A browser opens ``/ws/live?family=crash&seed=3`` and the server plays that
scenario tick by tick, sending one :class:`~sentinelops.api.schemas.TickFrame`
per tick. Each frame holds every site's telemetry and the detector's verdict:
the window features of :mod:`sentinelops.detect.features` are scored by the
Isolation Forest and compared with the alert threshold tuned on the tune split
(``sentinelops report-detection``); a site turns red only after
:data:`ALERT_PERSISTENCE` alerted ticks in a row. The console only draws what it is sent; it
never computes anything about the simulation itself.

Run it with ``sentinelops serve``; the built web console is served from the
same address.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from sentinelops import __version__
from sentinelops.api.schemas import (
    ErrorMessage,
    Options,
    RegionInfo,
    RunFinished,
    RunStarted,
    SiteInfo,
    SiteState,
    Status,
    TickFrame,
    TopologyInfo,
)
from sentinelops.detect.features import WindowFeaturizer, to_matrix
from sentinelops.detect.isolation import IsolationDetector, healthy_windows
from sentinelops.eval.harness import AGENTS, make_agent
from sentinelops.scenarios import FAMILIES, Scenario, generate
from sentinelops.scenarios.families import DEFAULT_HORIZON
from sentinelops.sim.env import DnsEnv
from sentinelops.sim.telemetry import Observation, SiteTelemetry
from sentinelops.sim.topology import SiteId, Topology, default_topology

ALERT_THRESHOLD = 0.5523
"""Detector score at which a site alerts: the best-F1 threshold that
``sentinelops report-detection`` tunes on tune seeds 100-105 with the default
detector (seed 0, healthy seeds 0-4). Frozen before any test-split run."""

ALERT_PERSISTENCE = 10
"""Consecutive above-threshold ticks before a site turns to ``"alert"``.

One window over the threshold is often noise: measured on tune seeds 100-101
of every single-fault family, 1.24% of healthy site-ticks cross it. Requiring
10 in a row cuts that to 0.02%, misses no faulty site, and delays detection by
a median of 9 ticks (90th percentile 83). Single crossings show as ``"watch"``.
"""

HEALTHY_SEEDS = range(5)
"""Fault-free runs the default detector is fitted on."""

VISIBLE_EVENTS = frozenset({"action", "probe_started", "probe_result"})
"""Event-log entries an operator could see; fault events are ground truth."""

MAX_SPEED = 1000.0
"""Fastest playback, in ticks per second."""


def _fail_ratio(telemetry: SiteTelemetry) -> float | None:
    ok, failed = telemetry.ok_replies, telemetry.failed_replies
    if ok is None or failed is None:
        return None
    return failed / (ok + failed) if ok + failed > 0 else 0.0


class LiveRun:
    """Plays one scenario a tick at a time and describes each tick for the console."""

    def __init__(
        self,
        scenario: Scenario,
        agent: str,
        topology: Topology,
        detector: IsolationDetector,
        threshold: float = ALERT_THRESHOLD,
    ) -> None:
        """Set up the run; the first :meth:`step` simulates tick 0.

        Raises:
            ValueError: If ``agent`` is not a known agent.
        """
        self.scenario = scenario
        self.env = DnsEnv(topology, scenario.faults, scenario.horizon, seed=scenario.seed)
        self.agent = make_agent(agent)
        self.agent.reset(topology)
        self.featurizer = WindowFeaturizer(topology)
        self.detector = detector
        self.threshold = threshold
        self._obs: Observation | None = None
        self._streak: dict[SiteId, int] = dict.fromkeys(topology.sites, 0)
        self._events_seen = 0

    @property
    def done(self) -> bool:
        """Whether the last tick has been played."""
        return self._obs is not None and self.env.done

    def step(self) -> TickFrame:
        """Simulate the next tick and describe it.

        Raises:
            RuntimeError: If the run is already done.
        """
        if self.done:
            raise RuntimeError("the run is finished")
        obs = self.env.reset() if self._obs is None else self.env.step(self.agent.act(self._obs))
        self._obs = obs
        return TickFrame(tick=obs.tick, sites=self._sites(obs), events=self._new_events())

    def _scores(self, obs: Observation) -> Mapping[SiteId, float]:
        vectors = self.featurizer.update(obs)
        if not vectors:
            return {}
        scores = self.detector.score(to_matrix(list(vectors.values())))
        return dict(zip(vectors, scores.tolist(), strict=True))

    def _sites(self, obs: Observation) -> dict[str, SiteState]:
        scores = self._scores(obs)
        states: dict[str, SiteState] = {}
        for site in sorted(obs.sites):
            telemetry = obs.sites[site]
            score = scores.get(site)
            above = score is not None and score >= self.threshold
            self._streak[site] = self._streak[site] + 1 if above else 0
            states[site] = SiteState(
                qps=telemetry.qps,
                p95_latency_ms=telemetry.p95_latency_ms,
                loss=telemetry.loss,
                fail_ratio=_fail_ratio(telemetry),
                announced=telemetry.announced,
                weight=telemetry.weight,
                score=score,
                status=self._status(site, score),
            )
        return states

    def _status(self, site: SiteId, score: float | None) -> Status:
        if score is None:
            return "warmup"
        if self._streak[site] >= ALERT_PERSISTENCE:
            return "alert"
        return "watch" if self._streak[site] > 0 else "ok"

    def _new_events(self) -> list[dict[str, str | int | float | bool | None]]:
        new = list(itertools.islice(self.env.events, self._events_seen, None))
        self._events_seen += len(new)
        return [
            {"tick": event.tick, "event": event.name, **event.data}
            for event in new
            if event.name in VISIBLE_EVENTS
        ]

    def finished(self) -> RunFinished:
        """The closing message, once :attr:`done`."""
        return RunFinished(
            tick=self.env.tick, legit_failure_share=self.env.truth.legit_failure_share()
        )


def _check_run(family: str, agent: str, speed: float) -> str | None:
    """Why a live run cannot start with these parameters, or ``None`` if it can."""
    if family not in FAMILIES:
        return f"unknown family {family!r}; expected one of {list(FAMILIES)}"
    if agent not in AGENTS:
        return f"unknown agent {agent!r}; expected one of {list(AGENTS)}"
    if not 0 < speed <= MAX_SPEED:
        return f"speed must be in (0, {MAX_SPEED:g}] ticks per second, got {speed:g}"
    return None


def create_app(
    *,
    topology: Topology | None = None,
    detector: IsolationDetector | None = None,
    threshold: float = ALERT_THRESHOLD,
    static_dir: Path | None = None,
) -> FastAPI:
    """Build the API.

    Args:
        topology: The network; defaults to :func:`default_topology`.
        detector: A fitted detector. ``None`` fits the default one (seed 0, on
            :data:`HEALTHY_SEEDS`) when the server starts, which takes a
            couple of seconds.
        threshold: Alert threshold for detector scores.
        static_dir: A built web console to serve at ``/``, if any.
    """
    topo = topology or default_topology()
    state: dict[str, IsolationDetector] = {}
    if detector is not None:
        state["detector"] = detector

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if "detector" not in state:
            windows = await asyncio.to_thread(healthy_windows, topo, HEALTHY_SEEDS)
            state["detector"] = IsolationDetector(seed=0).fit(windows)
        yield

    app = FastAPI(title="SentinelOps console API", version=__version__, lifespan=lifespan)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        """Liveness check."""
        return {"status": "ok", "version": __version__}

    @app.get("/api/topology")
    def topology_info() -> TopologyInfo:
        """Sites and client regions, with coordinates for the globe."""
        return TopologyInfo(
            sites=[
                SiteInfo(id=s.id, name=s.name, lat=s.lat, lon=s.lon, servers=sorted(s.servers))
                for s in sorted(topo.sites.values(), key=lambda s: s.id)
            ],
            regions=[
                RegionInfo(id=r.id, name=r.name, lat=r.lat, lon=r.lon)
                for r in sorted(topo.regions.values(), key=lambda r: r.id)
            ],
        )

    @app.get("/api/options")
    def options() -> Options:
        """Scenario families and agents a live run can use."""
        return Options(
            families=list(FAMILIES),
            agents=list(AGENTS),
            horizon=DEFAULT_HORIZON,
            alert_threshold=threshold,
        )

    @app.websocket("/ws/live")
    async def live(
        websocket: WebSocket,
        family: str = "crash",
        seed: int = 0,
        agent: str = "noop",
        speed: float = 20.0,
    ) -> None:
        """Play a scenario: ``run_started``, one ``tick`` per tick, ``run_finished``."""
        await websocket.accept()
        problem = _check_run(family, agent, speed)
        if problem is None:
            try:
                scenario = generate(family, seed, topo)
            except ValueError as error:
                problem = str(error)
        if problem is not None:
            await websocket.send_text(ErrorMessage(message=problem).model_dump_json())
            await websocket.close(code=1008)
            return
        run = LiveRun(scenario, agent, topo, state["detector"], threshold)
        started = RunStarted(
            family=family,
            seed=seed,
            agent=agent,
            horizon=scenario.horizon,
            alert_threshold=threshold,
        )
        try:
            await websocket.send_text(started.model_dump_json())
            while not run.done:
                await websocket.send_text(run.step().model_dump_json())
                await asyncio.sleep(1 / speed)
            await websocket.send_text(run.finished().model_dump_json())
            await websocket.close()
        except WebSocketDisconnect:
            return  # the console navigated away; nothing to clean up

    if static_dir is not None and static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="console")
    return app
