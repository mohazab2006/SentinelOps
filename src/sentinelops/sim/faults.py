"""Fault model: what can break in the simulated network and what that does.

A :class:`Fault` is plain data, for example "server ``ott-1`` crashes at tick
420". The simulation engine does not need to know how each kind of fault works.
Every tick it calls :func:`compute_effects`, which turns all faults that are
active at that tick into one :class:`FaultEffects` summary: which servers are
down, how much capacity the others have left, how much traffic is lost on the
way to each site, and how much extra traffic each client region sends.

:func:`clears` says which repair action resolves which fault. Restarting fixes
software problems (a crash, a memory leak) but not broken hardware, which needs
a manual repair. Path loss, floods and benign surges are never "cleared": they
end on their own after their duration, and the agent works around them (by
withdrawing a site or rate limiting) instead.

Both functions are pure: the same inputs always give the same output, and
nothing is stored between calls. That keeps runs reproducible and easy to test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
from enum import StrEnum

from sentinelops.sim.topology import RegionId, ServerId, SiteId, Topology

EXHAUSTION_RAMP_TICKS = 600
"""Ticks over which a severity-1.0 memory leak uses up all of a server's capacity."""

EXHAUSTION_MIN_CAPACITY = 0.15
"""Lowest capacity factor an exhausted server falls to (it slows down, it does not stop)."""

FLOOD_MULTIPLIER = 8.0
"""A severity-1.0 flood adds this many times the region's normal demand as attack traffic."""

SURGE_MULTIPLIER = 4.0
"""A severity-1.0 benign surge adds this many times the region's normal demand as real traffic."""


class FaultKind(StrEnum):
    """The kinds of fault the simulator can inject.

    Each kind targets one type of thing: a server, a site or a client region.
    """

    CRASH = "crash"
    """Target: server. The server is down until a restart completes."""

    EXHAUSTION = "exhaustion"
    """Target: server. A memory leak slowly reduces capacity; a restart clears it."""

    HARDWARE = "hardware"
    """Target: server. The server is down; a restart does not help, a repair does."""

    PATH_LOSS = "path_loss"
    """Target: site. A fraction ``severity`` of queries arriving at the site is lost."""

    SITE_OUTAGE = "site_outage"
    """Target: site. Every server at the site is down and all arriving traffic is lost."""

    FLOOD = "flood"
    """Target: region. The region sends extra attack queries (a DDoS flood)."""

    BENIGN_SURGE = "benign_surge"
    """Target: region. The region sends extra legitimate queries. Not a fault to fix."""


_SERVER_KINDS = frozenset({FaultKind.CRASH, FaultKind.EXHAUSTION, FaultKind.HARDWARE})
_SITE_KINDS = frozenset({FaultKind.PATH_LOSS, FaultKind.SITE_OUTAGE})
_REGION_KINDS = frozenset({FaultKind.FLOOD, FaultKind.BENIGN_SURGE})


@dataclass(frozen=True, slots=True)
class Fault:
    """One injected fault.

    Attributes:
        kind: What goes wrong.
        target: The id of the server, site or region affected, depending on
            ``kind``. Whether it exists in a topology is checked separately by
            :func:`validate_fault`, because a fault does not know the topology.
        onset: First tick at which the fault is active.
        duration: How many ticks the fault lasts, or ``None`` if it lasts until
            an action clears it.
        severity: How bad the fault is, in ``(0, 1]``. Its meaning depends on
            the kind (for example the fraction of traffic lost for path loss).
    """

    kind: FaultKind
    target: str
    onset: int
    duration: int | None
    severity: float

    def __post_init__(self) -> None:
        """Reject an empty target, a negative onset, a bad duration or severity."""
        if not self.target:
            raise ValueError(f"{self.kind} fault needs a non-empty target")
        if self.onset < 0:
            raise ValueError(f"fault onset must be >= 0, got {self.onset}")
        if self.duration is not None and self.duration <= 0:
            raise ValueError(f"fault duration must be None or > 0, got {self.duration}")
        if not 0.0 < self.severity <= 1.0:
            raise ValueError(f"fault severity must be in (0, 1], got {self.severity}")

    def active_at(self, tick: int) -> bool:
        """Return whether the fault is active at ``tick``.

        A fault is active from ``onset`` up to, but not including,
        ``onset + duration``. A fault with no duration never ends by itself.
        """
        if tick < self.onset:
            return False
        return self.duration is None or tick < self.onset + self.duration


@dataclass(frozen=True, slots=True)
class FaultEffects:
    """The combined effect of all active faults at one tick.

    The defaults mean "no effect", so ``FaultEffects()`` describes a healthy
    network. A key missing from a mapping also means no effect for that item.

    Attributes:
        server_down: Servers that serve nothing this tick.
        server_capacity_factor: Multiplier in ``[0, 1]`` on a server's normal
            capacity. Missing key means 1.0.
        server_memory_pressure: Extra memory use in ``[0, 1]`` reported by a
            server. Missing key means 0.
        site_path_loss: Fraction of queries lost on the way to a site, in
            ``[0, 1]``. Missing key means 0.
        region_extra_legit_qps: Extra legitimate queries per tick from a region.
        region_extra_attack_qps: Extra attack queries per tick from a region.
    """

    server_down: frozenset[ServerId] = field(default_factory=frozenset)
    server_capacity_factor: Mapping[ServerId, float] = field(default_factory=dict)
    server_memory_pressure: Mapping[ServerId, float] = field(default_factory=dict)
    site_path_loss: Mapping[SiteId, float] = field(default_factory=dict)
    region_extra_legit_qps: Mapping[RegionId, float] = field(default_factory=dict)
    region_extra_attack_qps: Mapping[RegionId, float] = field(default_factory=dict)


class ClearingAction(StrEnum):
    """The agent actions that can resolve a fault outright."""

    RESTART = "restart"
    REPAIR = "repair"


def validate_fault(fault: Fault, topology: Topology) -> None:
    """Check that a fault's target exists and is the right type for its kind.

    Server faults need a server id, site faults a site id, and region faults a
    region id, each present in ``topology``.

    Raises:
        ValueError: If the target is missing or is the wrong type of thing.
    """
    if fault.kind in _SERVER_KINDS:
        valid, expected = fault.target in topology.servers, "server"
    elif fault.kind in _SITE_KINDS:
        valid, expected = fault.target in topology.sites, "site"
    else:
        valid, expected = fault.target in topology.regions, "region"
    if not valid:
        raise ValueError(
            f"{fault.kind} fault targets {fault.target!r}, which is not a {expected} "
            "in this topology"
        )


def _combine_loss(a: float, b: float) -> float:
    """Combine two independent loss fractions: a query must survive both."""
    return 1.0 - (1.0 - a) * (1.0 - b)


def compute_effects(
    faults: Sequence[Fault], tick: int, topology: Topology, cleared: Set[int]
) -> FaultEffects:
    """Combine every active, not-yet-cleared fault into one summary for ``tick``.

    Overlapping faults combine as follows: capacity factors multiply, memory
    pressure takes the largest value, losses combine as ``1 - (1 - a)(1 - b)``
    (independent drops), and extra query rates add up.

    Args:
        faults: All faults of the scenario.
        tick: The current tick.
        topology: The network, used to find a site's servers and a region's
            normal demand.
        cleared: Indices into ``faults`` of faults an action has already
            resolved; these have no effect.

    Returns:
        The combined effects. Inputs are not modified.

    Raises:
        KeyError: If a fault targets something not in ``topology`` (call
            :func:`validate_fault` first to get a clear error instead).
    """
    down: set[ServerId] = set()
    capacity: dict[ServerId, float] = {}
    memory: dict[ServerId, float] = {}
    loss: dict[SiteId, float] = {}
    extra_legit: dict[RegionId, float] = {}
    extra_attack: dict[RegionId, float] = {}

    for index, fault in enumerate(faults):
        if index in cleared or not fault.active_at(tick):
            continue
        kind = fault.kind
        if kind in (FaultKind.CRASH, FaultKind.HARDWARE):
            down.add(ServerId(fault.target))
        elif kind == FaultKind.EXHAUSTION:
            server = ServerId(fault.target)
            used = fault.severity * (tick - fault.onset) / EXHAUSTION_RAMP_TICKS
            factor = max(EXHAUSTION_MIN_CAPACITY, 1.0 - used)
            capacity[server] = capacity.get(server, 1.0) * factor
            memory[server] = max(memory.get(server, 0.0), min(1.0, used))
        elif kind == FaultKind.SITE_OUTAGE:
            site = SiteId(fault.target)
            down.update(topology.servers_at(site))
            loss[site] = 1.0
        elif kind == FaultKind.PATH_LOSS:
            site = SiteId(fault.target)
            loss[site] = _combine_loss(loss.get(site, 0.0), fault.severity)
        elif kind == FaultKind.FLOOD:
            region = RegionId(fault.target)
            qps = fault.severity * FLOOD_MULTIPLIER * topology.regions[region].base_qps
            extra_attack[region] = extra_attack.get(region, 0.0) + qps
        else:  # BENIGN_SURGE
            region = RegionId(fault.target)
            qps = fault.severity * SURGE_MULTIPLIER * topology.regions[region].base_qps
            extra_legit[region] = extra_legit.get(region, 0.0) + qps

    return FaultEffects(
        server_down=frozenset(down),
        server_capacity_factor=capacity,
        server_memory_pressure=memory,
        site_path_loss=loss,
        region_extra_legit_qps=extra_legit,
        region_extra_attack_qps=extra_attack,
    )


def clears(fault: Fault, action: ClearingAction, target: str, topology: Topology) -> bool:
    """Return whether performing ``action`` on ``target`` resolves ``fault``.

    * Crash and exhaustion: a restart of the same server.
    * Hardware: a repair of the same server (a restart does not help).
    * Site outage: a repair of the same site.
    * Path loss, flood, benign surge: never. They end after their duration;
      the agent mitigates them by withdrawing a site or rate limiting.

    Args:
        fault: The fault in question.
        action: The clearing action performed.
        target: The server or site id the action was performed on.
        topology: The network. Not needed by the current rules; kept so the
            rules can depend on the network without changing the signature.
    """
    if target != fault.target:
        return False
    if fault.kind in (FaultKind.CRASH, FaultKind.EXHAUSTION):
        return action == ClearingAction.RESTART
    if fault.kind in (FaultKind.HARDWARE, FaultKind.SITE_OUTAGE):
        return action == ClearingAction.REPAIR
    return False
