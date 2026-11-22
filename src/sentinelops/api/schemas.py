"""Messages the console API sends, as typed models (their JSON is the wire format).

Every message carries only what an operator could know: telemetry, the
detector's verdict, and the agent's own actions and probes. Ground truth (which
fault was injected, and when) never leaves the server, so the console cannot
show more than the agent sees.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Status = Literal["warmup", "ok", "watch", "alert"]
"""A site's state on the globe (see :class:`SiteState`)."""


class _Message(BaseModel):
    model_config = ConfigDict(frozen=True)


class SiteInfo(_Message):
    """A site of the network, for placing it on the globe."""

    id: str
    name: str
    lat: float
    lon: float
    servers: list[str]


class RegionInfo(_Message):
    """A client region (where queries come from)."""

    id: str
    name: str
    lat: float
    lon: float


class TopologyInfo(_Message):
    """The static network: sites and client regions."""

    sites: list[SiteInfo]
    regions: list[RegionInfo]


class Options(_Message):
    """What a live run can be started with."""

    families: list[str]
    agents: list[str]
    horizon: int
    alert_threshold: float


class SiteState(_Message):
    """One site at one tick, as monitoring and the detector see it.

    Telemetry readings are ``None`` when missing. ``score`` is ``None`` during
    the detector's warm-up, before the first full window. ``status`` is
    ``"warmup"`` then; ``"watch"`` when this tick's score reaches the alert
    threshold; ``"alert"`` once it has done so for ``ALERT_PERSISTENCE``
    ticks in a row; otherwise ``"ok"``.
    """

    qps: float | None
    p95_latency_ms: float | None
    loss: float | None
    fail_ratio: float | None
    announced: bool
    weight: float
    score: float | None
    status: Status


class RunStarted(_Message):
    """First message of a live run."""

    type: Literal["run_started"] = "run_started"
    family: str
    seed: int
    agent: str
    horizon: int
    alert_threshold: float


class TickFrame(_Message):
    """One simulated tick: every site's state and the operator-visible events."""

    type: Literal["tick"] = "tick"
    tick: int
    sites: dict[str, SiteState]
    events: list[dict[str, str | int | float | bool | None]]


class RunFinished(_Message):
    """Last message of a live run."""

    type: Literal["run_finished"] = "run_finished"
    tick: int
    legit_failure_share: float


class ErrorMessage(_Message):
    """The run could not start (bad parameters)."""

    type: Literal["error"] = "error"
    message: str
