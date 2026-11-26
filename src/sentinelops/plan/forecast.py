r"""Recovery forecasting: what each candidate action would do under each diagnosis.

Before acting, the agent asks "if I do *a* and the real fault is *h*, how many
legitimate queries fail over the next few minutes?" It answers by rolling the
network forward from its current estimate, once per (action, hypothesis) pair,
with the same physics as the simulator: per tick, route each region's demand to
the announcing sites, then run every site's pipeline (path loss, rate limit,
bounded queue, serve up to capacity, time out). Rejections, overflow, timeouts,
losses and unrouted queries are failures. Actions take their expected time:
a restart about a minute, a repair about ten, a routing change a few seconds,
plus any wait for approval, and they only fix what they really fix (a restart
does not fix broken hardware). So a forecast captures three costs the agent
cannot see by looking at one site alone:

* the **delay** before an action helps (approval, restart, repair);
* the **disruption** the action causes while it runs (a restarting server
  serves nothing); and
* the **collateral** load it pushes elsewhere (a withdrawn site's traffic,
  floods included, lands on its neighbours).

A hypothesis from the diagnosis (:class:`~sentinelops.diagnose.inference.Hypothesis`)
only says *what* is wrong, so :func:`hypothesis_faults` turns it into concrete
faults sized from the latest telemetry: the measured loss for a path fault,
the traffic above normal for a flood or surge, the memory in use for a leak.
Faults that end by themselves are assumed to last the whole horizon, which is
the cautious choice when their end is unknown.

:func:`forecast_belief` fills the table of forecasts for every candidate action
and every plausible hypothesis; weighting a row by the belief gives the
expected failures of that action, and the "wait" row is the reference the
planner compares against (``score(a) = sum_h P(h) [F_wait(h) - F_a(h)] - penalty``).
Each forecast also keeps failures at the incident site apart from failures
elsewhere, so an ablation can plan as if only the affected site existed.

Everything here is deterministic: expected demand, expected durations, no
random draws. A rollout fast-forwards through stretches where nothing changes,
so a 15-minute forecast usually costs only a few hundred simulated ticks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from sentinelops.diagnose.inference import Belief, Hypothesis
from sentinelops.eval.dataset import HEALTHY, affected_sites
from sentinelops.sim.actions import (
    REPAIR_TICKS,
    RESTART_TICKS,
    ROUTING_CONVERGENCE_TICKS,
    Action,
    ActionKind,
    action_targets,
)
from sentinelops.sim.env import EnvConfig
from sentinelops.sim.faults import (
    EXHAUSTION_MIN_CAPACITY,
    EXHAUSTION_RAMP_TICKS,
    FLOOD_MULTIPLIER,
    SURGE_MULTIPLIER,
    ClearingAction,
    Fault,
    FaultKind,
    clears,
    compute_effects,
)
from sentinelops.sim.site import SiteQueue, Traffic, process_tick
from sentinelops.sim.telemetry import Observation
from sentinelops.sim.topology import RegionId, ServerId, SiteId, Topology
from sentinelops.sim.traffic import catchment_shares

FloatArray = npt.NDArray[np.float64]

DEFAULT_HORIZON = 900
"""Ticks (seconds) a forecast looks ahead: long enough for a repair to finish and pay off."""

DEFAULT_MIN_PROB = 0.01
"""Hypotheses less likely than this are not forecast (their weight goes to the others)."""

ASSUMED_EXHAUSTION_SEVERITY = 0.7
"""How fast an unseen memory leak is assumed to grow (middle of the scenarios' range)."""

MIN_FAULT_SIZE = 0.05
"""Smallest loss fraction or leak share assumed when a hypothesis is tested."""

ASSUMED_TRANSIENT_TICKS = 300
"""How much longer a fault that ends by itself (path loss, flood, surge) is assumed to last."""

DEFAULT_RESTORE_AFTER = 30
"""Fault-free ticks after which a withdrawn site is assumed to be restored."""

_SURGE_THRESHOLD = 0.01
"""Traffic above normal, as a share of normal, below which no flood or surge is assumed."""

_STEADY_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class ForecastConfig:
    """How far ahead to look and how long actions are expected to take.

    Attributes:
        horizon: Ticks to forecast after the estimate's tick.
        restart_ticks: Expected restart duration (the simulator draws 30-90).
        repair_ticks: Expected repair duration (the simulator draws 480-720).
        exhaustion_severity: Assumed growth rate of a suspected memory leak.
        transient_ticks: How much longer a fault that ends by itself is assumed
            to last; ``None`` assumes it outlasts the horizon.
        restore_after: Ticks without a fault affecting a withdrawn site after
            which the agent is assumed to restore it, as it does after
            verifying recovery; ``None`` never restores it.
    """

    horizon: int = DEFAULT_HORIZON
    restart_ticks: int = (RESTART_TICKS[0] + RESTART_TICKS[1]) // 2
    repair_ticks: int = (REPAIR_TICKS[0] + REPAIR_TICKS[1]) // 2
    exhaustion_severity: float = ASSUMED_EXHAUSTION_SEVERITY
    transient_ticks: int | None = ASSUMED_TRANSIENT_TICKS
    restore_after: int | None = DEFAULT_RESTORE_AFTER

    def __post_init__(self) -> None:
        """Reject a non-positive horizon or duration, or a severity outside ``(0, 1]``."""
        ticks = [self.horizon, self.restart_ticks, self.repair_ticks]
        if self.transient_ticks is not None:
            ticks.append(self.transient_ticks)
        if min(ticks) < 1:
            raise ValueError("horizon and durations must be at least 1 tick")
        if self.restore_after is not None and self.restore_after < 0:
            raise ValueError(f"restore_after must be non-negative, got {self.restore_after}")
        if not 0.0 < self.exhaustion_severity <= 1.0:
            raise ValueError("exhaustion_severity must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class RunningAction:
    """An action the agent already started that is still under way.

    Attributes:
        action: The action.
        end_tick: Tick at which it finishes (its effect applies from then).
    """

    action: Action
    end_tick: int


def _frozen(mapping: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True, slots=True)
class NetworkEstimate:
    """What the agent believes the network looks like now: the start of every forecast.

    Attributes:
        tick: The tick of the estimate; forecasts start at the next tick.
        demand: Expected legitimate queries per tick from each region.
        weights: Announcement weight of each site.
        limits: Rate limit of each site in queries per tick (missing: none).
        queue_depth: Queries waiting at each site.
        reaching_qps: Queries reaching each site per tick (after path loss).
        loss: Measured path loss of each site.
        leak: Estimated share of each server's capacity used up by a memory leak.
        running: Actions still under way.
        held: Sites that must stay as they are (e.g. the policy blocks
            restoring them): never restored automatically in a forecast.
    """

    tick: int
    demand: Mapping[RegionId, float]
    weights: Mapping[SiteId, float]
    limits: Mapping[SiteId, float] = field(default_factory=dict)
    queue_depth: Mapping[SiteId, float] = field(default_factory=dict)
    reaching_qps: Mapping[SiteId, float] = field(default_factory=dict)
    loss: Mapping[SiteId, float] = field(default_factory=dict)
    leak: Mapping[ServerId, float] = field(default_factory=dict)
    running: tuple[RunningAction, ...] = ()
    held: frozenset[SiteId] = frozenset()

    def __post_init__(self) -> None:
        """Store read-only copies of the mappings."""
        for name in (
            "demand",
            "weights",
            "limits",
            "queue_depth",
            "reaching_qps",
            "loss",
            "leak",
        ):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        object.__setattr__(self, "running", tuple(self.running))
        object.__setattr__(self, "held", frozenset(self.held))

    @classmethod
    def healthy(cls, topology: Topology, tick: int = 0) -> NetworkEstimate:
        """A quiet network: normal demand, every site fully announced, empty queues."""
        return cls(
            tick=tick,
            demand={r: region.base_qps for r, region in topology.regions.items()},
            weights=dict.fromkeys(topology.sites, 1.0),
        )

    @classmethod
    def from_observation(
        cls,
        obs: Observation,
        topology: Topology,
        *,
        limits: Mapping[SiteId, float] | None = None,
        running: Sequence[RunningAction] = (),
        config: EnvConfig | None = None,
        previous: NetworkEstimate | None = None,
    ) -> NetworkEstimate:
        """Estimate the network from one observation and what the agent itself has done.

        Demand is the regions' normal demand (the operator's traffic profile);
        weights are exact; queue depth, traffic and loss are the noisy readings.
        A missing reading keeps the value of ``previous`` (zero without one), so
        one dropped sample does not make a fault look gone. A leak is read off memory use: a
        server's memory is ``base + 0.1 * load + (0.9 - base) * leak``, with the
        load taken from its CPU.

        Args:
            obs: The latest observation.
            topology: The network.
            limits: Rate limits the agent has set and not cleared.
            running: Actions the agent started that have not finished.
            config: The server profile (idle CPU, base memory); default profile if ``None``.
            previous: An earlier estimate whose values stand in for missing readings.
        """
        profile = config or EnvConfig()
        before = previous or cls(obs.tick, {}, {})

        def site_reading(name: str, values: Mapping[SiteId, float | None]) -> dict[SiteId, float]:
            old: Mapping[SiteId, float] = getattr(before, name)
            return {s: old.get(s, 0.0) if v is None else v for s, v in values.items()}

        leak: dict[ServerId, float] = dict(before.leak)
        for server, telemetry in obs.servers.items():
            if telemetry.mem is None or telemetry.cpu is None:
                continue
            load = max(0.0, telemetry.cpu - profile.idle_cpu) / (1.0 - profile.idle_cpu)
            share = (telemetry.mem - profile.base_mem - 0.10 * load) / (0.90 - profile.base_mem)
            leak[server] = min(1.0, max(0.0, share))
        return cls(
            tick=obs.tick,
            demand={r: region.base_qps for r, region in topology.regions.items()},
            weights={s: t.weight for s, t in obs.sites.items()},
            limits=dict(limits or {}),
            queue_depth=site_reading(
                "queue_depth", {s: t.queue_depth for s, t in obs.sites.items()}
            ),
            reaching_qps=site_reading("reaching_qps", {s: t.qps for s, t in obs.sites.items()}),
            loss=site_reading("loss", {s: t.loss for s, t in obs.sites.items()}),
            leak=leak,
            running=tuple(running),
        )


@dataclass(frozen=True, slots=True)
class Rollout:
    """Legitimate queries over one forecast, by where they failed.

    Attributes:
        offered: Legitimate queries sent over the horizon.
        at_site: Failed at the incident site.
        elsewhere: Failed at any other site.
        unrouted: Had nowhere to go (no site announcing).
    """

    offered: float
    at_site: float
    elsewhere: float
    unrouted: float

    @property
    def failed(self) -> float:
        """Every failed legitimate query."""
        return self.at_site + self.elsewhere + self.unrouted

    def cost(self, *, local_only: bool = False) -> float:
        """Failures the planner counts: all of them, or only the incident site's."""
        return self.at_site if local_only else self.failed


def candidate_actions(
    site: str, estimate: NetworkEstimate, topology: Topology
) -> tuple[Action | None, ...]:
    """The actions worth forecasting for an incident at ``site``; ``None`` (wait) first.

    Restart or repair each server, repair the whole site, withdraw the site
    (if it is announcing), restore it fully (if it is not), and set or clear a
    rate limit at the site's healthy capacity. Actions the simulator would
    refuse are left out: anything on a busy server, and a routing change while
    another one of the site is still under way.

    Raises:
        KeyError: If ``site`` is not a site of ``topology``.
    """
    site_id = SiteId(site)
    servers = sorted(topology.sites[site_id].servers)
    busy = {s for r in estimate.running for s in action_targets(r.action, topology)}
    actions: list[Action | None] = [None]
    routing = (ActionKind.WITHDRAW, ActionKind.RESTORE)
    rerouting = any(r.action.kind in routing and r.action.target == site for r in estimate.running)
    weight = estimate.weights.get(site_id, 1.0)
    actions += [Action(ActionKind.RESTART, s) for s in servers if s not in busy]
    if weight > 0.0 and not rerouting:
        actions.append(Action(ActionKind.WITHDRAW, site))
    if weight < 1.0 and not rerouting:
        actions.append(Action(ActionKind.RESTORE, site, weight=1.0))
    if site_id in estimate.limits:
        actions.append(Action(ActionKind.CLEAR_RATE_LIMIT, site))
    else:
        capacity = sum(topology.servers[s].capacity_qps for s in servers)
        actions.append(Action(ActionKind.RATE_LIMIT, site, limit_qps=capacity))
    actions += [Action(ActionKind.REQUEST_REPAIR, s) for s in servers if s not in busy]
    if not busy.intersection(servers):
        actions.append(Action(ActionKind.REQUEST_REPAIR, site))
    return tuple(actions)


def hypothesis_faults(
    hypothesis: Hypothesis,
    site: str,
    estimate: NetworkEstimate,
    topology: Topology,
    config: ForecastConfig | None = None,
) -> tuple[Fault, ...]:
    """Turn a diagnosis hypothesis about ``site`` into concrete faults, sized from telemetry.

    * crash, hardware: the named server is down; site outage: the whole site;
    * exhaustion: a leak on the named server that has already used up the
      estimated share of its capacity and keeps growing;
    * path loss: the site's measured loss;
    * flood, benign surge: the traffic reaching the site above what normal
      demand would bring, sent from the regions that reach the site, in
      proportion to their normal traffic there;
    * healthy: nothing.

    Every fault is active from the estimate's tick. Path loss, floods and
    surges end after ``config.transient_ticks``; the others last until an
    action clears them.

    Raises:
        KeyError: If ``site`` is not a site of ``topology``.
        ValueError: If the kind is unknown or a server hypothesis names no server.
    """
    cfg = config or ForecastConfig()
    site_id = SiteId(site)
    topology.sites[site_id]  # unknown site -> KeyError
    now = estimate.tick
    kind = hypothesis.kind
    if kind in (FaultKind.CRASH, FaultKind.HARDWARE, FaultKind.EXHAUSTION):
        if hypothesis.server is None:
            raise ValueError(f"a {kind} hypothesis must name a server")
        if kind != FaultKind.EXHAUSTION:
            return (Fault(FaultKind(kind), hypothesis.server, now, None, 1.0),)
        used = max(MIN_FAULT_SIZE, estimate.leak.get(hypothesis.server, 0.0))
        age = round(used * EXHAUSTION_RAMP_TICKS / cfg.exhaustion_severity)
        onset = max(0, now - age)
        return (
            Fault(FaultKind.EXHAUSTION, hypothesis.server, onset, None, cfg.exhaustion_severity),
        )
    if kind == FaultKind.SITE_OUTAGE:
        return (Fault(FaultKind.SITE_OUTAGE, site, now, None, 1.0),)
    if kind == FaultKind.PATH_LOSS:
        severity = min(1.0, max(MIN_FAULT_SIZE, estimate.loss.get(site_id, 0.0)))
        return (Fault(FaultKind.PATH_LOSS, site, now, cfg.transient_ticks, severity),)
    if kind in (FaultKind.FLOOD, FaultKind.BENIGN_SURGE):
        return _extra_traffic(FaultKind(kind), site_id, estimate, topology, cfg.transient_ticks)
    if kind == HEALTHY:
        return ()
    raise ValueError(f"unknown hypothesis kind {kind!r}")


def _extra_traffic(
    kind: FaultKind,
    site: SiteId,
    estimate: NetworkEstimate,
    topology: Topology,
    duration: int | None,
) -> tuple[Fault, ...]:
    """Size a flood or surge so it brings the observed excess traffic to ``site``."""
    # Each region's normal traffic that lands on the site under the current weights.
    contribution = {
        region: qps * catchment_shares(topology.preferences[region], estimate.weights).get(site, 0)
        for region, qps in estimate.demand.items()
    }
    normal = sum(contribution.values())
    excess = estimate.reaching_qps.get(site, 0.0) - normal
    if normal <= 0.0 or excess <= _SURGE_THRESHOLD * normal:
        return ()
    multiplier = FLOOD_MULTIPLIER if kind == FaultKind.FLOOD else SURGE_MULTIPLIER
    faults = []
    for region in sorted(contribution):
        if contribution[region] <= 0.0:
            continue
        # A region sending x extra queries puts x * share of them on the site, and
        # its share is contribution / demand, so x = excess * demand / normal.
        extra = excess * estimate.demand[region] / normal
        base = topology.regions[region].base_qps
        severity = min(1.0, extra / (multiplier * base))
        faults.append(Fault(kind, region, estimate.tick, duration, severity))
    return tuple(faults)


@dataclass(slots=True)
class _Running:
    action: Action
    end_tick: int


def _duration(action: Action, config: ForecastConfig) -> int:
    if action.kind == ActionKind.RESTART:
        return config.restart_ticks
    if action.kind == ActionKind.REQUEST_REPAIR:
        return config.repair_ticks
    return ROUTING_CONVERGENCE_TICKS


class _World:
    """The mutable state of one rollout: the simulator's rules, without randomness."""

    def __init__(
        self,
        estimate: NetworkEstimate,
        faults: tuple[Fault, ...],
        topology: Topology,
        config: ForecastConfig,
    ) -> None:
        self.topology = topology
        self.faults = faults
        self.config = config
        self.weights: dict[SiteId, float] = {
            s: estimate.weights.get(s, 1.0) for s in topology.sites
        }
        self.limits: dict[SiteId, float | None] = {
            s: estimate.limits.get(s) for s in topology.sites
        }
        self.running = [_Running(r.action, r.end_tick) for r in estimate.running]
        self.busy: set[ServerId] = {
            s for r in self.running for s in action_targets(r.action, topology)
        }
        self.cleared: set[int] = set()
        self.weights_changed = True
        self.fault_sites = tuple(affected_sites(f, topology) for f in faults)
        self.held = estimate.held

    def finish_actions(self, tick: int) -> None:
        due = [r for r in self.running if r.end_tick <= tick]
        self.running = [r for r in self.running if r.end_tick > tick]
        for run in due:
            action = run.action
            self.busy.difference_update(action_targets(action, self.topology))
            if action.kind == ActionKind.WITHDRAW:
                self.weights[SiteId(action.target)] = 0.0
                self.weights_changed = True
            elif action.kind == ActionKind.RESTORE:
                assert action.weight is not None  # guaranteed by Action validation
                self.weights[SiteId(action.target)] = action.weight
                self.weights_changed = True
            elif action.kind == ActionKind.RESTART:
                self._clear(tick, ClearingAction.RESTART, (action.target,))
            elif action.kind == ActionKind.REQUEST_REPAIR:
                targets = (action.target, *action_targets(action, self.topology))
                self._clear(tick, ClearingAction.REPAIR, targets)

    def _clear(self, tick: int, how: ClearingAction, targets: Sequence[str]) -> None:
        for index, fault in enumerate(self.faults):
            if index in self.cleared or not fault.active_at(tick) or fault.onset >= tick:
                continue
            if any(clears(fault, how, target, self.topology) for target in targets):
                self.cleared.add(index)

    def start_action(self, tick: int, action: Action) -> None:
        servers = action_targets(action, self.topology)
        routing = action.kind in (ActionKind.WITHDRAW, ActionKind.RESTORE)
        if (routing and self.rerouting(SiteId(action.target))) or self.busy.intersection(servers):
            return  # the simulator refuses it, so it does nothing
        if action.kind == ActionKind.RATE_LIMIT:
            self.limits[SiteId(action.target)] = action.limit_qps
        elif action.kind == ActionKind.CLEAR_RATE_LIMIT:
            self.limits[SiteId(action.target)] = None
        else:
            self.running.append(_Running(action, tick + _duration(action, self.config)))
            self.busy.update(servers)

    def quiet(self, site: SiteId, tick: int) -> bool:
        """Whether no fault affecting ``site`` is active (and uncleared) at ``tick``."""
        return not any(
            i not in self.cleared and site in self.fault_sites[i] and f.active_at(tick)
            for i, f in enumerate(self.faults)
        )

    def rerouting(self, site: SiteId) -> bool:
        """Whether a withdrawal or restore of ``site`` is under way."""
        routing = (ActionKind.WITHDRAW, ActionKind.RESTORE)
        return any(r.action.kind in routing and r.action.target == site for r in self.running)

    def next_change(self, tick: int, starts: Sequence[int]) -> int | None:
        """First tick after ``tick`` at which anything changes, or ``None`` if never.

        Returns ``tick + 1`` while a memory leak is still growing: then every
        tick is different.
        """
        changes = [r.end_tick for r in self.running] + [s for s in starts if s > tick]
        for index, fault in enumerate(self.faults):
            if index in self.cleared:
                continue
            if fault.kind == FaultKind.EXHAUSTION and fault.active_at(tick):
                floor_age = (1.0 - EXHAUSTION_MIN_CAPACITY) * EXHAUSTION_RAMP_TICKS
                if tick - fault.onset < floor_age / fault.severity:
                    return tick + 1
            changes.append(fault.onset)
            if fault.duration is not None:
                changes.append(fault.onset + fault.duration)
        later = [c for c in changes if c > tick]
        return min(later) if later else None


def _shares(
    topology: Topology, weights: Mapping[SiteId, float]
) -> dict[RegionId, dict[SiteId, float]]:
    return {r: catchment_shares(topology.preferences[r], weights) for r in topology.regions}


def _restore_out_sites(
    world: _World,
    tick: int,
    quiet_since: dict[SiteId, int | None],
    out_since: dict[SiteId, int | None],
    restore_after: int | None,
) -> int | None:
    """Restore every site that has been out and fault-free long enough.

    Updates the per-site stretches in place and returns the first later tick at
    which a site is due to be restored, or ``None``.
    """
    due: int | None = None
    for s in world.topology.sites:
        if not world.quiet(s, tick):
            quiet_since[s] = None
        elif quiet_since[s] is None:
            quiet_since[s] = tick
        if world.weights[s] >= 1.0:
            out_since[s] = None
        elif out_since[s] is None:
            out_since[s] = tick
        quiet, out = quiet_since[s], out_since[s]
        if restore_after is None or quiet is None or out is None:
            continue
        if s in world.held or world.rerouting(s):
            continue
        at = max(quiet, out) + restore_after
        if tick >= at:
            world.start_action(tick, Action(ActionKind.RESTORE, s, weight=1.0))
        else:
            due = at if due is None else min(due, at)
    return due


def rollout(
    estimate: NetworkEstimate,
    faults: Sequence[Fault],
    topology: Topology,
    *,
    site: str,
    plan: Sequence[tuple[Action, int]] = (),
    config: ForecastConfig | None = None,
) -> Rollout:
    """Forecast legitimate failures from ``estimate`` with ``faults`` and a plan of actions.

    Runs ``config.horizon`` ticks after ``estimate.tick`` with the simulator's
    rules: actions finish, then new ones start, then faults take effect,
    demand is routed and every site serves its queue. Durations are the
    expected ones and demand is the estimate's, so the result is deterministic.
    Once a site has been out (withdrawn or partly restored) with no fault
    affecting it for ``config.restore_after`` ticks, it is restored to full
    weight, as the agent would after verifying recovery, unless the estimate
    holds it.

    Args:
        estimate: The starting state.
        faults: The faults assumed present (absolute ticks, like the simulator's).
        topology: The network.
        site: The incident site, whose failures are counted apart.
        plan: ``(action, tick)`` pairs: start ``action`` at ``tick``.
        config: Horizon and expected durations.

    Raises:
        KeyError: If ``site`` is not a site of ``topology``.
        ValueError: If a planned action starts before the first forecast tick.
    """
    cfg = config or ForecastConfig()
    incident = SiteId(site)
    topology.sites[incident]  # unknown site -> KeyError
    if any(at <= estimate.tick for _, at in plan):
        raise ValueError(f"planned actions must start after tick {estimate.tick}")
    world = _World(estimate, tuple(faults), topology, cfg)
    sites = tuple(topology.sites)
    queues = {
        s: SiteQueue(topology.sites[s].queue_limit, topology.sites[s].timeout_ticks) for s in sites
    }
    starts = sorted({tick for _, tick in plan})
    start, end = estimate.tick + 1, estimate.tick + 1 + cfg.horizon
    shares = _shares(topology, world.weights)
    offered = at_site = elsewhere = unrouted = 0.0
    previous: tuple[float, ...] | None = None
    clock = 0  # the queues' own clock: it only moves on simulated ticks
    # per site: first tick of its current fault-free stretch, and of its current time out
    quiet_since: dict[SiteId, int | None] = dict.fromkeys(sites)
    out_since: dict[SiteId, int | None] = dict.fromkeys(sites)
    same = 0  # ticks in a row identical to the one before
    settle = max(topology.sites[s].timeout_ticks for s in sites) + 1
    tick = start
    while tick < end:
        world.finish_actions(tick)
        for action, at in plan:
            if at == tick:
                world.start_action(tick, action)
        restore_at = _restore_out_sites(world, tick, quiet_since, out_since, cfg.restore_after)
        effects = compute_effects(world.faults, tick, topology, world.cleared)
        if world.weights_changed:
            shares = _shares(topology, world.weights)
            world.weights_changed = False

        legit = dict(estimate.demand)
        for region, extra in effects.region_extra_legit_qps.items():
            legit[region] = legit.get(region, 0.0) + extra
        arrivals = {s: [0.0, 0.0] for s in sites}
        lost_unrouted = 0.0
        for traffic, column in ((legit, 0), (effects.region_extra_attack_qps, 1)):
            for region, qps in traffic.items():
                if not shares[region] and column == 0:
                    lost_unrouted += qps
                for target, share in shares[region].items():
                    arrivals[target][column] += qps * share

        tick_at_site = tick_elsewhere = 0.0
        signature: list[float] = [lost_unrouted]
        for s in sites:
            capacity = sum(
                0.0
                if server in effects.server_down or server in world.busy
                else topology.servers[server].capacity_qps
                * effects.server_capacity_factor.get(server, 1.0)
                for server in topology.servers_at(s)
            )
            arriving = Traffic(arrivals[s][0], arrivals[s][1])
            if tick == start and estimate.queue_depth.get(s, 0.0) > 0.0:
                depth = estimate.queue_depth[s]
                mix = (
                    arriving.scaled(depth / arriving.total)
                    if arriving.total > 0
                    else Traffic(depth)
                )
                queues[s].offer(clock - 1, mix)
            result = process_tick(
                queues[s],
                clock,
                arriving,
                path_loss=effects.site_path_loss.get(s, 0.0),
                limit_qps=world.limits[s],
                capacity=capacity,
            )
            failed = result.failed.legit
            if s == incident:
                tick_at_site += failed
            else:
                tick_elsewhere += failed
            signature += [failed, result.served.total, result.queue_depth]
        tick_offered = sum(legit.values())
        clock += 1

        # Fast-forward: once ticks have repeated exactly for longer than any query
        # can wait (so every queued batch has turned over) and nothing changes
        # before ``nxt``, every tick until then repeats as well.
        repeats = 1
        nxt = world.next_change(tick, starts)
        if restore_at is not None:
            nxt = restore_at if nxt is None else min(nxt, restore_at)
        repeated = previous is not None and all(
            math.isclose(a, b, rel_tol=_STEADY_TOLERANCE, abs_tol=_STEADY_TOLERANCE)
            for a, b in zip(signature, previous, strict=True)
        )
        same = same + 1 if repeated else 0
        if same >= settle:
            repeats = min(end if nxt is None else nxt, end) - tick
        offered += repeats * tick_offered
        at_site += repeats * tick_at_site
        elsewhere += repeats * tick_elsewhere
        unrouted += repeats * lost_unrouted
        previous = tuple(signature)
        tick += repeats
    return Rollout(offered, at_site, elsewhere, unrouted)


@dataclass(frozen=True, slots=True)
class ForecastTable:
    """Forecasts for every candidate action under every plausible hypothesis.

    Attributes:
        site: The incident site.
        actions: The candidate actions; ``None`` is waiting.
        hypotheses: The hypotheses forecast (those at least ``min_prob`` likely).
        probs: Their probabilities, renormalised to sum to 1.
        rollouts: ``rollouts[i][j]`` is action ``i`` under hypothesis ``j``.
    """

    site: SiteId
    actions: tuple[Action | None, ...]
    hypotheses: tuple[Hypothesis, ...]
    probs: FloatArray
    rollouts: tuple[tuple[Rollout, ...], ...]

    def failures(self, *, local_only: bool = False) -> FloatArray:
        """Forecast failures, shape ``(actions, hypotheses)``."""
        return np.array(
            [[r.cost(local_only=local_only) for r in row] for row in self.rollouts], dtype=float
        )

    def expected_failures(self, *, local_only: bool = False) -> FloatArray:
        """Belief-weighted forecast failures of each action: ``sum_h P(h) F_a(h)``."""
        result: FloatArray = self.failures(local_only=local_only) @ self.probs
        return result

    def expected_reduction(self, *, local_only: bool = False) -> FloatArray:
        """Failures each action is expected to prevent compared with waiting.

        ``sum_h P(h) [F_wait(h) - F_a(h)]``; negative when the action makes things
        worse.

        Raises:
            ValueError: If waiting (``None``) is not one of the actions.
        """
        if None not in self.actions:
            raise ValueError("the table has no 'wait' row to compare against")
        expected = self.expected_failures(local_only=local_only)
        reduction: FloatArray = expected[self.actions.index(None)] - expected
        return reduction


def forecast_belief(
    estimate: NetworkEstimate,
    belief: Belief,
    topology: Topology,
    *,
    actions: Sequence[Action | None] | None = None,
    delays: Mapping[Action, int] | None = None,
    min_prob: float = DEFAULT_MIN_PROB,
    config: ForecastConfig | None = None,
    context_faults: Sequence[Fault] = (),
    context_plan: Sequence[tuple[Action, int]] = (),
) -> ForecastTable:
    """Forecast every candidate action under every plausible hypothesis of ``belief``.

    Args:
        estimate: The current state of the network.
        belief: The diagnosis of the incident site.
        topology: The network.
        actions: Actions to compare; :func:`candidate_actions` if ``None``.
        delays: Ticks an action waits before it starts, e.g. for approval;
            missing actions start at the next tick.
        min_prob: Hypotheses less likely than this are skipped. The most likely
            hypothesis is always kept.
        config: Horizon and expected durations.
        context_faults: Faults assumed present elsewhere in every forecast,
            e.g. the likeliest causes of other open incidents.
        context_plan: ``(action, tick)`` pairs already decided for other
            incidents, run alongside every candidate.

    Raises:
        ValueError: If ``min_prob`` is outside ``[0, 1]``, a delay is negative,
            or a context action starts before the first forecast tick.
    """
    if not 0.0 <= min_prob <= 1.0:
        raise ValueError(f"min_prob must be in [0, 1], got {min_prob}")
    waits = dict(delays or {})
    if any(d < 0 for d in waits.values()):
        raise ValueError("delays must be non-negative")
    cfg = config or ForecastConfig()
    site = belief.site
    choices = tuple(candidate_actions(site, estimate, topology) if actions is None else actions)
    best = int(np.argmax(belief.probs))
    kept = [j for j, p in enumerate(belief.probs.tolist()) if p >= min_prob or j == best]
    probs = belief.probs[kept] / belief.probs[kept].sum()
    first = estimate.tick + 1
    rows: list[list[Rollout]] = [[] for _ in choices]
    for j in kept:
        faults = (
            *hypothesis_faults(belief.hypotheses[j], site, estimate, topology, cfg),
            *context_faults,
        )
        for i, action in enumerate(choices):
            plan = tuple(context_plan)
            if action is not None:
                plan += ((action, first + waits.get(action, 0)),)
            rows[i].append(rollout(estimate, faults, topology, site=site, plan=plan, config=cfg))
    probs.flags.writeable = False
    return ForecastTable(
        site=site,
        actions=choices,
        hypotheses=tuple(belief.hypotheses[j] for j in kept),
        probs=probs,
        rollouts=tuple(tuple(row) for row in rows),
    )
