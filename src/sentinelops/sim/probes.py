"""Diagnostic probes: the checks an operator runs to find out what is wrong.

Telemetry shows *that* a site is in trouble, but often not *why*: a crashed
server, a broken host, a lossy network path and a flood can all look like
"failed replies are up". Probes tell them apart, at the cost of time:

* ``LOCAL_DNS`` (server) — query one server from inside its site. Separates a
  crashed process (``refused``) from a host that is off (``timeout``).
* ``REMOTE_DNS`` (site) — query the site from external vantage points, over
  the public network path, so it sees path loss.
* ``NEIGHBOUR_DNS`` (site) — query the site from a neighbouring site over the
  private backbone, which bypasses the public path. Healthy here but lossy
  remotely means the path is at fault, not the site.
* ``TRAFFIC_SAMPLE`` (site) — inspect a sample of arriving queries and report
  their pattern: normal, a broad surge of real users, an attack signature, or
  an ambiguous burst of normal-looking queries from a few known resolvers.
* ``HW_CHECK`` (server) — ask the host's out-of-band management controller
  whether the hardware is healthy.

Outcomes are noisy, as real probes are: a probe occasionally times out on its
own, a remote probe through a lossy path sometimes gets every reply back, and
a traffic sample can misread a flood as a known resolver's burst or the other
way round. That ambiguity is deliberate: the agent has to weigh evidence, and
escalate when it cannot tell a real client's accident from an attack, instead
of rate limiting real users.

This module is pure: :func:`probe_outcome` maps a snapshot of the true state
of the target and a random generator to an outcome. The engine builds the
snapshot, so the probe logic can be tested without running a simulation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import numpy as np

from sentinelops.sim.topology import Topology


class ProbeKind(StrEnum):
    """The diagnostic probes available to every agent."""

    LOCAL_DNS = "local_dns"
    """Target: server. Query the server from inside its site."""

    REMOTE_DNS = "remote_dns"
    """Target: site. Query the site from external vantage points over the public path."""

    NEIGHBOUR_DNS = "neighbour_dns"
    """Target: site. Query the site from a neighbouring site over the private backbone."""

    TRAFFIC_SAMPLE = "traffic_sample"
    """Target: site. Inspect the sources and query names of arriving traffic."""

    HW_CHECK = "hw_check"
    """Target: server. Read the host's hardware health from its management controller."""


class ProbeOutcome(StrEnum):
    """Every result a probe can report. :data:`OUTCOMES` lists which kind gives which."""

    OK = "ok"
    """DNS probes: answered correctly and quickly."""

    SLOW = "slow"
    """DNS probes: answered correctly, but slowly (overload or a degraded server)."""

    LOSSY = "lossy"
    """``REMOTE_DNS``: some vantage points got no reply."""

    TIMEOUT = "timeout"
    """DNS probes: no reply at all."""

    REFUSED = "refused"
    """``LOCAL_DNS``: the host answered but the DNS process is not running."""

    SERVFAIL = "servfail"
    """Site DNS probes: the site's load balancer answered but has no working server."""

    HEALTHY = "healthy"
    """``HW_CHECK``: the hardware reports no fault."""

    HARDWARE_FAULT = "hardware_fault"
    """``HW_CHECK``: the hardware reports a fault."""

    UNREACHABLE = "unreachable"
    """``HW_CHECK``: the management controller did not answer."""

    NORMAL_MIX = "normal_mix"
    """``TRAFFIC_SAMPLE``: the usual mix of clients and query names."""

    BROAD_SURGE = "many_sources_normal_qnames"
    """``TRAFFIC_SAMPLE``: many sources asking for normal names — real users."""

    ATTACK_PATTERN = "few_sources_random_qnames"
    """``TRAFFIC_SAMPLE``: few sources asking for random names — an attack signature."""

    RESOLVER_BURST = "few_sources_normal_qnames"
    """``TRAFFIC_SAMPLE``: few known resolvers asking for normal names — ambiguous.

    A misconfigured but legitimate client can look exactly like this, and so can
    an attack that mimics one. Neither volume nor this pattern settles it.
    """

    NO_TRAFFIC = "no_traffic"
    """``TRAFFIC_SAMPLE``: nothing is reaching the site."""


OUTCOMES: Mapping[ProbeKind, tuple[ProbeOutcome, ...]] = MappingProxyType(
    {
        ProbeKind.LOCAL_DNS: (
            ProbeOutcome.OK,
            ProbeOutcome.SLOW,
            ProbeOutcome.REFUSED,
            ProbeOutcome.TIMEOUT,
        ),
        ProbeKind.REMOTE_DNS: (
            ProbeOutcome.OK,
            ProbeOutcome.SLOW,
            ProbeOutcome.LOSSY,
            ProbeOutcome.SERVFAIL,
            ProbeOutcome.TIMEOUT,
        ),
        ProbeKind.NEIGHBOUR_DNS: (
            ProbeOutcome.OK,
            ProbeOutcome.SLOW,
            ProbeOutcome.SERVFAIL,
            ProbeOutcome.TIMEOUT,
        ),
        ProbeKind.TRAFFIC_SAMPLE: (
            ProbeOutcome.NORMAL_MIX,
            ProbeOutcome.BROAD_SURGE,
            ProbeOutcome.ATTACK_PATTERN,
            ProbeOutcome.RESOLVER_BURST,
            ProbeOutcome.NO_TRAFFIC,
        ),
        ProbeKind.HW_CHECK: (
            ProbeOutcome.HEALTHY,
            ProbeOutcome.HARDWARE_FAULT,
            ProbeOutcome.UNREACHABLE,
        ),
    }
)
"""The possible outcomes of each probe kind, e.g. for the states of a Bayesian network."""

PROBE_TICKS: Mapping[ProbeKind, int] = MappingProxyType(
    {
        ProbeKind.LOCAL_DNS: 2,
        ProbeKind.NEIGHBOUR_DNS: 3,
        ProbeKind.REMOTE_DNS: 5,
        ProbeKind.TRAFFIC_SAMPLE: 10,
        ProbeKind.HW_CHECK: 15,
    }
)
"""Ticks from request to result. Cheap DNS queries are quick; capturing traffic
and querying a management controller take longer."""

FLAKE_PROB = 0.02
"""Chance that a probe fails on its own (a DNS probe times out, a hardware check
gets no answer), whatever the true state."""

VANTAGE_POINTS = 10
"""External vantage points a ``REMOTE_DNS`` probe queries; each reply is lost
independently with the site's path loss."""

SLOW_LEVEL = 0.5
"""Slowness (see :class:`ServerState`) above which a reply counts as slow."""

SLOW_NOISE_SD = 0.1
"""Measurement noise on slowness, so states near :data:`SLOW_LEVEL` read either way."""

HW_MISS_PROB = 0.05
"""Chance that a hardware check misses a real hardware fault."""

HW_FALSE_ALARM_PROB = 0.01
"""Chance that a hardware check reports a fault on healthy hardware."""

DOMINANT_SHARE = 0.25
"""Share of sampled traffic from a flood or surge above which it shapes the sample."""

NO_TRAFFIC_QPS = 1.0
"""Queries per tick below which a traffic sample finds nothing to inspect."""

_SAMPLE_READINGS: Mapping[str, tuple[tuple[ProbeOutcome, float], ...]] = MappingProxyType(
    {
        "attack": (
            (ProbeOutcome.ATTACK_PATTERN, 0.65),
            (ProbeOutcome.RESOLVER_BURST, 0.30),
            (ProbeOutcome.BROAD_SURGE, 0.05),
        ),
        "surge": (
            (ProbeOutcome.BROAD_SURGE, 0.70),
            (ProbeOutcome.RESOLVER_BURST, 0.25),
            (ProbeOutcome.ATTACK_PATTERN, 0.05),
        ),
        "normal": (
            (ProbeOutcome.NORMAL_MIX, 0.94),
            (ProbeOutcome.BROAD_SURGE, 0.03),
            (ProbeOutcome.RESOLVER_BURST, 0.03),
        ),
    }
)
"""How a traffic sample reads each true traffic mix. Attacks and surges both
sometimes look like a known resolver's burst: the D-024 ambiguity."""


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    """One probe requested by an agent.

    Attributes:
        kind: Which probe.
        target: Server id for ``LOCAL_DNS`` and ``HW_CHECK``; site id otherwise.
    """

    kind: ProbeKind
    target: str

    def __post_init__(self) -> None:
        """Convert the kind to :class:`ProbeKind`, rejecting unknown kinds."""
        object.__setattr__(self, "kind", ProbeKind(self.kind))


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The result of a finished probe, reported in the observation of ``completed``.

    Attributes:
        request: The probe as requested.
        issued: Tick at which the probe was requested.
        completed: Tick at which the result arrived.
        outcome: What the probe found.
    """

    request: ProbeRequest
    issued: int
    completed: int
    outcome: ProbeOutcome


@dataclass(frozen=True, slots=True)
class ServerState:
    """True state of one server, as needed by the server probes.

    Attributes:
        managed: Whether its management controller can be reached (not during a
            site outage or a repair).
        host_up: Whether the host is powered and reachable.
        process_up: Whether the DNS process is running (not crashed or restarting).
        hardware_fault: Whether the host has a hardware fault.
        slowness: How slowly it answers, in ``[0, 1]``: the larger of its lost
            capacity and its site's queue fill.
    """

    managed: bool
    host_up: bool
    process_up: bool
    hardware_fault: bool
    slowness: float


@dataclass(frozen=True, slots=True)
class SiteState:
    """True state of one site, as needed by the site probes.

    Attributes:
        path_loss: Fraction of queries lost on the public path to the site.
        powered: Whether the site is up (no site outage or site-wide repair).
        serving: Whether at least one of its servers can answer.
        slowness: Queue fill in ``[0, 1]``: how long replies wait.
        reaching_qps: Queries reaching the site this tick, after path loss.
        attack_share: Share of those queries that are attack traffic.
        surge_share: Share of those queries that are extra legitimate traffic
            from a surge.
    """

    path_loss: float
    powered: bool
    serving: bool
    slowness: float
    reaching_qps: float
    attack_share: float
    surge_share: float


def validate_probe(request: ProbeRequest, topology: Topology) -> None:
    """Check that a probe's target exists and is the right type for its kind.

    Raises:
        ValueError: If the target is unknown or the wrong type of thing.
    """
    if request.kind in (ProbeKind.LOCAL_DNS, ProbeKind.HW_CHECK):
        valid, expected = request.target in topology.servers, "server"
    else:
        valid, expected = request.target in topology.sites, "site"
    if not valid:
        raise ValueError(
            f"{request.kind} probe targets {request.target!r}, which is not a {expected}"
        )


def _is_slow(slowness: float, rng: np.random.Generator) -> bool:
    return slowness + rng.normal(0.0, SLOW_NOISE_SD) > SLOW_LEVEL


def _pick(
    readings: tuple[tuple[ProbeOutcome, float], ...], rng: np.random.Generator
) -> ProbeOutcome:
    index = int(rng.choice(len(readings), p=[probability for _, probability in readings]))
    return readings[index][0]


def _server_outcome(kind: ProbeKind, state: ServerState, rng: np.random.Generator) -> ProbeOutcome:
    flaked = rng.random() < FLAKE_PROB
    if kind == ProbeKind.HW_CHECK:
        if flaked or not state.managed:
            return ProbeOutcome.UNREACHABLE
        misread = rng.random() < (HW_MISS_PROB if state.hardware_fault else HW_FALSE_ALARM_PROB)
        return (
            ProbeOutcome.HARDWARE_FAULT if state.hardware_fault != misread else ProbeOutcome.HEALTHY
        )
    # LOCAL_DNS
    if flaked or not state.host_up:
        return ProbeOutcome.TIMEOUT
    if not state.process_up:
        return ProbeOutcome.REFUSED
    return ProbeOutcome.SLOW if _is_slow(state.slowness, rng) else ProbeOutcome.OK


def _site_outcome(kind: ProbeKind, state: SiteState, rng: np.random.Generator) -> ProbeOutcome:
    if kind == ProbeKind.TRAFFIC_SAMPLE:
        if state.reaching_qps < NO_TRAFFIC_QPS:
            return ProbeOutcome.NO_TRAFFIC
        mix = "normal"
        if max(state.attack_share, state.surge_share) >= DOMINANT_SHARE:
            mix = "attack" if state.attack_share >= state.surge_share else "surge"
        return _pick(_SAMPLE_READINGS[mix], rng)
    # REMOTE_DNS and NEIGHBOUR_DNS
    if rng.random() < FLAKE_PROB or not state.powered:
        return ProbeOutcome.TIMEOUT
    lost = 0
    if kind == ProbeKind.REMOTE_DNS:  # only the public path is lossy
        lost = int(rng.binomial(VANTAGE_POINTS, min(1.0, max(0.0, state.path_loss))))
        if lost == VANTAGE_POINTS:
            return ProbeOutcome.TIMEOUT
    if not state.serving:
        return ProbeOutcome.SERVFAIL
    if lost > 0:
        return ProbeOutcome.LOSSY
    return ProbeOutcome.SLOW if _is_slow(state.slowness, rng) else ProbeOutcome.OK


def probe_outcome(
    kind: ProbeKind, state: ServerState | SiteState, rng: np.random.Generator
) -> ProbeOutcome:
    """Return what a probe of ``kind`` reports about a target in ``state``.

    Args:
        kind: The probe.
        state: True state of the target: a :class:`ServerState` for
            ``LOCAL_DNS`` and ``HW_CHECK``, a :class:`SiteState` otherwise.
        rng: Source of the probe's noise.

    Returns:
        One of ``OUTCOMES[kind]``.

    Raises:
        TypeError: If ``state`` is the wrong type for ``kind``.
    """
    server_probe = kind in (ProbeKind.LOCAL_DNS, ProbeKind.HW_CHECK)
    if server_probe and isinstance(state, ServerState):
        return _server_outcome(kind, state, rng)
    if not server_probe and isinstance(state, SiteState):
        return _site_outcome(kind, state, rng)
    raise TypeError(f"{kind} probe needs a {'ServerState' if server_probe else 'SiteState'}")
