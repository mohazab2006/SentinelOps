"""The simulation engine: one tick at a time, with agents acting in between.

:class:`DnsEnv` combines the topology, a list of injected faults and a seed into
a reproducible run. Each tick it:

1. finishes actions whose time is up and starts newly requested actions and
   probes;
2. works out the combined effect of all active faults;
3. draws the tick's demand, adds flood and surge traffic, and routes it to the
   announcing sites;
4. runs every site's query pipeline (loss, rate limit, queue, serve, timeout);
5. records the ground truth and an event log, and reports probes whose
   time is up; and
6. returns a noisy :class:`~sentinelops.sim.telemetry.Observation` to the agent.

The environment takes a list of faults rather than a scenario, so the
simulator does not depend on how scenarios are generated.

Randomness is split into independent streams (demand, telemetry noise, action
durations, probe noise) derived from the seed. Demand and noise never depend on what the
agent does, so two agents run on the same seed face exactly the same incident.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from sentinelops.sim.actions import (
    REPAIR_TICKS,
    RESTART_TICKS,
    ROUTING_CONVERGENCE_TICKS,
    Action,
    ActionKind,
    ActionStatus,
    ActionUpdate,
    action_targets,
    validate_action,
)
from sentinelops.sim.calendar import Calendar
from sentinelops.sim.events import EventLog
from sentinelops.sim.faults import (
    ClearingAction,
    Fault,
    FaultEffects,
    FaultKind,
    clears,
    compute_effects,
    validate_fault,
)
from sentinelops.sim.probes import (
    PROBE_TICKS,
    ProbeKind,
    ProbeRequest,
    ProbeResult,
    ServerState,
    SiteState,
    probe_outcome,
    validate_probe,
)
from sentinelops.sim.site import SiteQueue, SiteTick, Traffic, process_tick
from sentinelops.sim.telemetry import (
    FloatArray,
    NoiseConfig,
    Observation,
    ServerTelemetry,
    SiteTelemetry,
    add_noise,
    reading,
)
from sentinelops.sim.topology import RegionId, ServerId, SiteId, Topology, great_circle_km
from sentinelops.sim.traffic import DemandTrace, catchment_shares, generate_demand, route

FAILURE_CAUSES = ("lost", "rejected", "overflow", "timeout", "unrouted")
"""Every way a query can fail, in the order they are counted."""


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Tunable realism parameters of the simulation.

    Attributes:
        noise: Monitoring noise and missing-reading rate.
        demand_noise_cv: Per-tick variation of legitimate demand.
        demand_drift_cv: Slow variation of legitimate demand.
        base_latency_ms: Fixed processing latency added to network round trips.
        km_per_ms_rtt: Round-trip distance covered per millisecond; light in
            fibre covers about 200 km per ms one way, so 100 km round trip.
        idle_cpu: CPU utilisation of a running server with no load.
        base_mem: Memory utilisation of a healthy, idle server.
        calendar: Moment of the week of tick 0; each observation carries the
            moment of its tick. Only policies read it: demand, faults and noise
            are the same whatever the start time.
    """

    noise: NoiseConfig = field(default_factory=NoiseConfig)
    demand_noise_cv: float = 0.05
    demand_drift_cv: float = 0.08
    base_latency_ms: float = 2.0
    km_per_ms_rtt: float = 100.0
    idle_cpu: float = 0.03
    base_mem: float = 0.30
    calendar: Calendar = field(default_factory=Calendar)


class RunTruth:
    """Ground truth of a run, for training labels and scoring. Never shown to agents.

    Per-tick arrays have one entry per simulated tick. Failures are counted in
    the tick they happen; a query that times out is counted when it times out.

    Attributes:
        faults: The injected faults.
        cleared: Indices of faults resolved by an action.
        legit_offered: Legitimate queries sent by clients per tick.
        legit_served: Legitimate queries answered per tick.
        attack_offered: Attack queries sent per tick.
        legit_failed_by_cause: Legitimate failures per tick, by cause.
        attack_failed: Attack queries that failed per tick, any cause.
    """

    def __init__(self, faults: tuple[Fault, ...], horizon: int) -> None:
        """Create empty per-tick arrays for ``horizon`` ticks."""
        self.faults = faults
        self.cleared: set[int] = set()
        self.legit_offered = np.zeros(horizon)
        self.legit_served = np.zeros(horizon)
        self.attack_offered = np.zeros(horizon)
        self.attack_failed = np.zeros(horizon)
        self.legit_failed_by_cause = {cause: np.zeros(horizon) for cause in FAILURE_CAUSES}

    @property
    def legit_failed(self) -> FloatArray:
        """Legitimate queries that failed per tick, all causes together."""
        total: FloatArray = np.sum(list(self.legit_failed_by_cause.values()), axis=0)
        return total

    def legit_failure_share(self, start: int = 0, end: int | None = None) -> float:
        """Share of legitimate queries that failed over ticks ``[start, end)``.

        This is the project's primary metric. Returns 0.0 if nothing was offered.
        """
        offered = float(self.legit_offered[start:end].sum())
        if offered == 0.0:
            return 0.0
        return float(self.legit_failed[start:end].sum()) / offered


@dataclass(slots=True)
class _Running:
    action_id: int
    action: Action
    end_tick: int


@dataclass(frozen=True, slots=True)
class _PendingProbe:
    probe_id: int
    request: ProbeRequest
    issued: int
    due: int


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Return the ``q``-quantile of ``values`` where each value has a weight.

    The result is the smallest value whose cumulative weight share reaches ``q``.

    Raises:
        ValueError: If there are no positive weights.
    """
    pairs = sorted((v, w) for v, w in zip(values, weights, strict=True) if w > 0)
    total = sum(w for _, w in pairs)
    if total <= 0:
        raise ValueError("weighted_quantile needs at least one positive weight")
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= q * total:
            return value
    return pairs[-1][0]


def split_load(total: float, capacities: Sequence[float]) -> list[float]:
    """Spread ``total`` queries over servers like a round-robin load balancer.

    Every working server gets an equal share; a server that cannot handle its
    share serves up to its capacity and the rest spills over to the others.
    Servers with zero capacity get nothing.
    """
    load = [0.0] * len(capacities)
    open_servers = [i for i, c in enumerate(capacities) if c > 0]
    remaining = total
    while open_servers and remaining > 1e-12:
        share = remaining / len(open_servers)
        still_open = []
        for i in open_servers:
            take = min(share, capacities[i] - load[i])
            load[i] += take
            remaining -= take
            if capacities[i] - load[i] > 1e-12:
                still_open.append(i)
        if len(still_open) == len(open_servers):
            break
        open_servers = still_open
    return load


class DnsEnv:
    """A reproducible run of the anycast DNS network with injected faults.

    Usage::

        env = DnsEnv(topology, faults, horizon=1800, seed=0)
        obs = env.reset()
        while not env.done:
            obs = env.step(agent.act(obs))
        share = env.truth.legit_failure_share()
    """

    def __init__(
        self,
        topology: Topology,
        faults: Sequence[Fault],
        horizon: int,
        seed: int,
        config: EnvConfig | None = None,
    ) -> None:
        """Set up a run; call :meth:`reset` before stepping.

        Raises:
            ValueError: If ``horizon`` < 1 or a fault does not fit the topology.
        """
        if horizon < 1:
            raise ValueError("horizon must be at least 1 tick")
        for fault in faults:
            validate_fault(fault, topology)
        self.topology = topology
        self.faults = tuple(faults)
        self.horizon = horizon
        self.seed = seed
        self.config = config or EnvConfig()
        self._sites = tuple(topology.sites)
        self._servers = tuple(topology.servers)
        self._rtt_ms = {
            (region.id, site.id): self.config.base_latency_ms
            + great_circle_km(region.lat, region.lon, site.lat, site.lon)
            / self.config.km_per_ms_rtt
            for region in topology.regions.values()
            for site in topology.sites.values()
        }
        self._started = False
        self._init_state()

    # ------------------------------------------------------------------ public API

    @property
    def tick(self) -> int:
        """The last simulated tick (-1 before :meth:`reset`)."""
        return self._tick

    @property
    def done(self) -> bool:
        """Whether every tick of the horizon has been simulated."""
        return self._tick >= self.horizon - 1

    def reset(self) -> Observation:
        """Start the run from scratch and return the observation of tick 0."""
        self._init_state()
        self._started = True
        self.events.append(
            0,
            "run_started",
            horizon=self.horizon,
            seed=self.seed,
            start=str(self.config.calendar.start),
        )
        return self._advance(())

    def step(self, actions: Sequence[Action | ProbeRequest]) -> Observation:
        """Apply ``actions``, start any probes among them and simulate the next tick.

        A probe's result arrives in the observation of the tick it finishes, after
        ``PROBE_TICKS[kind]`` ticks, and describes the network at that tick.

        Every request is checked before any takes effect, so a bad one leaves the
        run exactly as it was.

        Raises:
            RuntimeError: If called before :meth:`reset` or after the run is done.
            TypeError: If a request is neither an Action nor a ProbeRequest.
            ValueError: If an action's or probe's target does not fit the topology.
        """
        if not self._started:
            raise RuntimeError("call reset() before step()")
        if self.done:
            raise RuntimeError("the run is finished; call reset() to start again")
        requests = tuple(actions)
        for request in requests:
            if isinstance(request, ProbeRequest):
                validate_probe(request, self.topology)
            elif isinstance(request, Action):
                validate_action(request, self.topology)
            else:
                raise TypeError(f"expected an Action or ProbeRequest, got {request!r}")
        return self._advance(requests)

    # ------------------------------------------------------------------ internals

    def _init_state(self) -> None:
        # Spawned children depend only on their index, so adding a stream at the
        # end leaves the earlier ones (and every existing run) unchanged.
        streams = np.random.SeedSequence(self.seed).spawn(4)
        demand_seq, noise_seq, action_seq, probe_seq = streams
        self._demand: DemandTrace = generate_demand(
            self.topology,
            self.horizon,
            np.random.default_rng(demand_seq),
            noise_cv=self.config.demand_noise_cv,
            drift_cv=self.config.demand_drift_cv,
        )
        self._noise_rng = np.random.default_rng(noise_seq)
        self._action_rng = np.random.default_rng(action_seq)
        self._probe_rng = np.random.default_rng(probe_seq)
        self._tick = -1
        self._queues = {
            sid: SiteQueue(site.queue_limit, site.timeout_ticks)
            for sid, site in self.topology.sites.items()
        }
        self._weights: dict[SiteId, float] = dict.fromkeys(self._sites, 1.0)
        self._limits: dict[SiteId, float | None] = dict.fromkeys(self._sites)
        self._running: dict[int, _Running] = {}
        self._busy_servers: dict[ServerId, int] = {}
        self._next_action_id = 0
        self._pending_probes: list[_PendingProbe] = []
        self._next_probe_id = 0
        self._active_faults: set[int] = set()
        self.events = EventLog()
        self.truth = RunTruth(self.faults, self.horizon)

    def _advance(self, requests: tuple[Action | ProbeRequest, ...]) -> Observation:
        self._tick += 1
        tick = self._tick
        updates: list[ActionUpdate] = []
        self._finish_actions(tick, updates)
        for request in requests:
            if isinstance(request, ProbeRequest):
                self._start_probe(tick, request)
            else:
                self._start_action(tick, request, updates)
        effects = compute_effects(self.faults, tick, self.topology, self.truth.cleared)
        self._log_fault_transitions(tick)
        capacity = {s: self._server_capacity(s, effects) for s in self._servers}
        results, unrouted, mix = self._route_and_serve(tick, effects, capacity)
        self._record_truth(tick, results, unrouted)
        probes = self._finish_probes(tick, effects, capacity, results)
        return self._observe(tick, effects, capacity, results, mix, tuple(updates), probes)

    # .................................................................. actions

    def _update(
        self,
        updates: list[ActionUpdate],
        tick: int,
        action_id: int,
        action: Action,
        status: ActionStatus,
    ) -> None:
        update = ActionUpdate(action_id, action, status, tick)
        updates.append(update)
        self.events.append(
            tick,
            "action",
            action_id=action_id,
            kind=str(action.kind),
            target=action.target,
            status=str(status),
            limit_qps=action.limit_qps,
            weight=action.weight,
        )

    def _start_action(self, tick: int, action: Action, updates: list[ActionUpdate]) -> None:
        action_id = self._next_action_id
        self._next_action_id += 1
        servers = action_targets(action, self.topology)
        rerouting = {
            r.action.target
            for r in self._running.values()
            if r.action.kind in (ActionKind.WITHDRAW, ActionKind.RESTORE)
        }
        is_routing = action.kind in (ActionKind.WITHDRAW, ActionKind.RESTORE)
        if (is_routing and action.target in rerouting) or any(
            s in self._busy_servers for s in servers
        ):
            # One routing change per site and one operation per server at a time.
            self._update(updates, tick, action_id, action, ActionStatus.FAILED)
            return

        self._update(updates, tick, action_id, action, ActionStatus.RUNNING)
        if action.kind == ActionKind.RATE_LIMIT:
            self._limits[SiteId(action.target)] = action.limit_qps
            self._update(updates, tick, action_id, action, ActionStatus.SUCCEEDED)
            return
        if action.kind == ActionKind.CLEAR_RATE_LIMIT:
            self._limits[SiteId(action.target)] = None
            self._update(updates, tick, action_id, action, ActionStatus.SUCCEEDED)
            return

        if action.kind == ActionKind.RESTART:
            duration = int(self._action_rng.integers(RESTART_TICKS[0], RESTART_TICKS[1] + 1))
        elif action.kind == ActionKind.REQUEST_REPAIR:
            duration = int(self._action_rng.integers(REPAIR_TICKS[0], REPAIR_TICKS[1] + 1))
        else:  # WITHDRAW, RESTORE
            duration = ROUTING_CONVERGENCE_TICKS
        self._running[action_id] = _Running(action_id, action, tick + duration)
        for server in servers:
            self._busy_servers[server] = action_id

    def _finish_actions(self, tick: int, updates: list[ActionUpdate]) -> None:
        due = sorted(aid for aid, run in self._running.items() if run.end_tick <= tick)
        for action_id in due:
            action = self._running.pop(action_id).action
            for server in action_targets(action, self.topology):
                self._busy_servers.pop(server, None)
            status = ActionStatus.SUCCEEDED
            if action.kind == ActionKind.WITHDRAW:
                self._weights[SiteId(action.target)] = 0.0
            elif action.kind == ActionKind.RESTORE:
                assert action.weight is not None  # guaranteed by Action validation
                self._weights[SiteId(action.target)] = action.weight
            elif action.kind == ActionKind.RESTART:
                self._clear_faults(tick, ClearingAction.RESTART, (action.target,))
                still_down = compute_effects(
                    self.faults, tick, self.topology, self.truth.cleared
                ).server_down
                if ServerId(action.target) in still_down:
                    status = ActionStatus.FAILED
            elif action.kind == ActionKind.REQUEST_REPAIR:
                targets = (action.target, *action_targets(action, self.topology))
                self._clear_faults(tick, ClearingAction.REPAIR, targets)
            self._update(updates, tick, action_id, action, status)

    def _clear_faults(self, tick: int, how: ClearingAction, targets: Sequence[str]) -> None:
        for index, fault in enumerate(self.faults):
            if index in self.truth.cleared or not fault.active_at(tick) or fault.onset >= tick:
                # Only a fault that is already happening can be fixed. Actions finish
                # before faults take effect, so one starting this very tick is not yet.
                continue
            if any(clears(fault, how, target, self.topology) for target in targets):
                self.truth.cleared.add(index)
                self._active_faults.discard(index)
                self.events.append(
                    tick, "fault_cleared", fault=index, kind=str(fault.kind), by=str(how)
                )

    def _log_fault_transitions(self, tick: int) -> None:
        active = {
            i
            for i, fault in enumerate(self.faults)
            if i not in self.truth.cleared and fault.active_at(tick)
        }
        for index in sorted(active - self._active_faults):
            fault = self.faults[index]
            self.events.append(
                tick,
                "fault_started",
                fault=index,
                kind=str(fault.kind),
                target=fault.target,
                severity=fault.severity,
            )
        for index in sorted(self._active_faults - active):
            self.events.append(tick, "fault_ended", fault=index, kind=str(self.faults[index].kind))
        self._active_faults = active

    # .................................................................. probes

    def _start_probe(self, tick: int, request: ProbeRequest) -> None:
        probe_id = self._next_probe_id
        self._next_probe_id += 1
        due = tick + PROBE_TICKS[request.kind]
        self._pending_probes.append(_PendingProbe(probe_id, request, tick, due))
        self.events.append(
            tick, "probe_started", probe_id=probe_id, kind=str(request.kind), target=request.target
        )

    def _finish_probes(
        self,
        tick: int,
        effects: FaultEffects,
        capacity: Mapping[ServerId, float],
        results: Mapping[SiteId, SiteTick],
    ) -> tuple[ProbeResult, ...]:
        due = [p for p in self._pending_probes if p.due <= tick]
        if not due:
            return ()
        self._pending_probes = [p for p in self._pending_probes if p.due > tick]
        surge: Mapping[SiteId, float] = {}
        if any(p.request.kind == ProbeKind.TRAFFIC_SAMPLE for p in due):
            surge = route(self.topology, effects.region_extra_legit_qps, self._weights).site_qps
        finished = []
        for probe in due:  # in the order they were requested
            target = probe.request.target
            state: ServerState | SiteState
            if target in self.topology.servers:
                state = self._server_state(ServerId(target), tick, effects, capacity, results)
            else:
                state = self._site_state(SiteId(target), tick, effects, capacity, results, surge)
            outcome = probe_outcome(probe.request.kind, state, self._probe_rng)
            finished.append(ProbeResult(probe.request, probe.issued, tick, outcome))
            self.events.append(
                tick,
                "probe_result",
                probe_id=probe.probe_id,
                kind=str(probe.request.kind),
                target=target,
                outcome=str(outcome),
            )
        return tuple(finished)

    def _fault_active(self, kind: FaultKind, target: str, tick: int) -> bool:
        return any(
            fault.kind == kind and fault.target == target and fault.active_at(tick)
            for index, fault in enumerate(self.faults)
            if index not in self.truth.cleared
        )

    def _site_down(self, site: SiteId, tick: int) -> bool:
        """Whether the whole site is off: an outage, or a site-wide repair under way."""
        return self._fault_active(FaultKind.SITE_OUTAGE, site, tick) or any(
            r.action.kind == ActionKind.REQUEST_REPAIR and r.action.target == site
            for r in self._running.values()
        )

    def _queue_fill(self, site: SiteId, results: Mapping[SiteId, SiteTick]) -> float:
        limit = self.topology.sites[site].queue_limit
        return min(1.0, results[site].queue_depth / limit) if limit > 0 else 0.0

    def _server_state(
        self,
        server: ServerId,
        tick: int,
        effects: FaultEffects,
        capacity: Mapping[ServerId, float],
        results: Mapping[SiteId, SiteTick],
    ) -> ServerState:
        site = self.topology.site_of(server)
        lost_capacity = 1.0 - effects.server_capacity_factor.get(server, 1.0)
        return ServerState(
            managed=not (self._repairing(server) or self._site_down(site, tick)),
            host_up=not self._host_unreachable(server, tick),
            process_up=capacity[server] > 0.0,
            hardware_fault=self._fault_active(FaultKind.HARDWARE, server, tick),
            slowness=max(lost_capacity, self._queue_fill(site, results)),
        )

    def _site_state(
        self,
        site: SiteId,
        tick: int,
        effects: FaultEffects,
        capacity: Mapping[ServerId, float],
        results: Mapping[SiteId, SiteTick],
        surge_routed: Mapping[SiteId, float],
    ) -> SiteState:
        result = results[site]
        path_loss = effects.site_path_loss.get(site, 0.0)
        reaching = result.arrived.total - result.lost.total
        attack = result.arrived.attack - result.lost.attack
        surge = surge_routed.get(site, 0.0) * (1.0 - path_loss)
        return SiteState(
            path_loss=path_loss,
            powered=not self._site_down(site, tick),
            serving=self._site_capacity(site, capacity) > 0.0,
            slowness=self._queue_fill(site, results),
            reaching_qps=reaching,
            attack_share=attack / reaching if reaching > 0.0 else 0.0,
            surge_share=surge / reaching if reaching > 0.0 else 0.0,
        )

    # .................................................................. traffic

    def _server_capacity(self, server: ServerId, effects: FaultEffects) -> float:
        if server in effects.server_down or server in self._busy_servers:
            return 0.0
        factor = effects.server_capacity_factor.get(server, 1.0)
        return self.topology.servers[server].capacity_qps * factor

    def _site_capacity(self, site: SiteId, capacity: Mapping[ServerId, float]) -> float:
        return sum(capacity[s] for s in self.topology.servers_at(site))

    def _route_and_serve(
        self, tick: int, effects: FaultEffects, capacity: Mapping[ServerId, float]
    ) -> tuple[dict[SiteId, SiteTick], Traffic, dict[SiteId, dict[RegionId, float]]]:
        legit = dict(self._demand.at(tick))
        for region, extra in effects.region_extra_legit_qps.items():
            legit[region] = legit.get(region, 0.0) + extra
        attack = dict(effects.region_extra_attack_qps)

        legit_routed = route(self.topology, legit, self._weights)
        attack_routed = route(self.topology, attack, self._weights)
        unrouted = Traffic(legit_routed.unrouted_qps, attack_routed.unrouted_qps)

        # Per-site mix of client regions, used for the latency distribution.
        mix: dict[SiteId, dict[RegionId, float]] = {s: {} for s in self._sites}
        for region in self.topology.regions:
            offered = legit.get(region, 0.0) + attack.get(region, 0.0)
            shares = catchment_shares(self.topology.preferences[region], self._weights)
            for site, share in shares.items():
                mix[site][region] = offered * share

        results: dict[SiteId, SiteTick] = {}
        for site in self._sites:
            results[site] = process_tick(
                self._queues[site],
                tick,
                Traffic(legit_routed.site_qps[site], attack_routed.site_qps[site]),
                path_loss=effects.site_path_loss.get(site, 0.0),
                limit_qps=self._limits[site],
                capacity=self._site_capacity(site, capacity),
            )
        return results, unrouted, mix

    def _record_truth(
        self, tick: int, results: Mapping[SiteId, SiteTick], unrouted: Traffic
    ) -> None:
        truth = self.truth
        by_cause = truth.legit_failed_by_cause
        truth.legit_offered[tick] = sum(r.arrived.legit for r in results.values()) + unrouted.legit
        truth.attack_offered[tick] = (
            sum(r.arrived.attack for r in results.values()) + unrouted.attack
        )
        truth.legit_served[tick] = sum(r.served.legit for r in results.values())
        by_cause["lost"][tick] = sum(r.lost.legit for r in results.values())
        by_cause["rejected"][tick] = sum(r.rejected.legit for r in results.values())
        by_cause["overflow"][tick] = sum(r.overflow.legit for r in results.values())
        by_cause["timeout"][tick] = sum(r.timeout.legit for r in results.values())
        by_cause["unrouted"][tick] = unrouted.legit
        truth.attack_failed[tick] = sum(r.failed.attack for r in results.values()) + unrouted.attack
        self.events.append(
            tick,
            "tick",
            legit_offered=round(float(truth.legit_offered[tick]), 3),
            legit_failed=round(float(truth.legit_failed[tick]), 3),
            attack_offered=round(float(truth.attack_offered[tick]), 3),
        )

    # .................................................................. telemetry

    def _repairing(self, server: ServerId) -> bool:
        """Whether a repair of the server, or of its whole site, is under way."""
        busy = self._busy_servers.get(server)
        return busy is not None and self._running[busy].action.kind == ActionKind.REQUEST_REPAIR

    def _host_unreachable(self, server: ServerId, tick: int) -> bool:
        """Whether the server's host is off (hardware fault, site outage, repair)."""
        return (
            self._repairing(server)
            or self._fault_active(FaultKind.HARDWARE, server, tick)
            or self._fault_active(FaultKind.SITE_OUTAGE, self.topology.site_of(server), tick)
        )

    def _site_latency_ms(
        self, site: SiteId, result: SiteTick, capacity: float, mix: Mapping[RegionId, float]
    ) -> float:
        if result.served.total <= 0.0 or not mix:
            return float("nan")
        regions = list(mix)
        network = weighted_quantile(
            [self._rtt_ms[(r, site)] for r in regions], [mix[r] for r in regions], 0.95
        )
        timeout_ms = 1000.0 * self.topology.sites[site].timeout_ticks
        queueing = (
            timeout_ms if capacity <= 0 else min(timeout_ms, 1000.0 * result.queue_depth / capacity)
        )
        return network + queueing

    def _observe(
        self,
        tick: int,
        effects: FaultEffects,
        capacity: Mapping[ServerId, float],
        results: Mapping[SiteId, SiteTick],
        mix: Mapping[SiteId, Mapping[RegionId, float]],
        updates: tuple[ActionUpdate, ...],
        probes: tuple[ProbeResult, ...],
    ) -> Observation:
        cfg = self.config
        # Site amounts: qps, latency, queue, ok, failed. Site fractions: loss.
        site_amounts = np.empty((len(self._sites), 5))
        site_fractions = np.empty((len(self._sites), 1))
        for i, site in enumerate(self._sites):
            result = results[site]
            site_amounts[i] = (
                result.arrived.total - result.lost.total,
                self._site_latency_ms(site, result, self._site_capacity(site, capacity), mix[site]),
                result.queue_depth,
                result.served.total,
                result.failed.total,
            )
            site_fractions[i, 0] = effects.site_path_loss.get(site, 0.0)

        # Server amounts: served. Server fractions: cpu, mem.
        served_by: dict[ServerId, float] = {}
        for site in self._sites:
            members = self.topology.servers_at(site)
            loads = split_load(results[site].served.total, [capacity[s] for s in members])
            served_by.update(zip(members, loads, strict=True))
        server_amounts = np.empty((len(self._servers), 1))
        server_fractions = np.empty((len(self._servers), 2))
        for i, server in enumerate(self._servers):
            own = capacity[server]
            served = served_by[server]
            utilisation = min(1.0, served / own) if own > 0 else 0.0
            pressure = effects.server_memory_pressure.get(server, 0.0)
            if self._host_unreachable(server, tick):
                cpu = mem = float("nan")
            elif own <= 0.0:  # process down: crashed or restarting
                cpu, mem = cfg.idle_cpu, cfg.base_mem / 2
            else:
                # A leak shrinks capacity, so the same load shows as higher CPU.
                cpu = cfg.idle_cpu + (1.0 - cfg.idle_cpu) * utilisation
                mem = min(1.0, cfg.base_mem + 0.10 * utilisation + (0.90 - cfg.base_mem) * pressure)
            server_amounts[i, 0] = served
            server_fractions[i] = (cpu, mem)

        noisy_site_amounts, noisy_site_fractions = add_noise(
            self._noise_rng, site_amounts, site_fractions, cfg.noise
        )
        noisy_server_amounts, noisy_server_fractions = add_noise(
            self._noise_rng, server_amounts, server_fractions, cfg.noise
        )

        sites = {
            site: SiteTelemetry(
                qps=reading(noisy_site_amounts[i, 0]),
                p95_latency_ms=reading(noisy_site_amounts[i, 1]),
                loss=reading(noisy_site_fractions[i, 0]),
                queue_depth=reading(noisy_site_amounts[i, 2]),
                ok_replies=reading(noisy_site_amounts[i, 3]),
                failed_replies=reading(noisy_site_amounts[i, 4]),
                announced=self._weights[site] > 0.0,
                weight=self._weights[site],
            )
            for i, site in enumerate(self._sites)
        }
        servers = {
            server: ServerTelemetry(
                cpu=reading(noisy_server_fractions[i, 0]),
                mem=reading(noisy_server_fractions[i, 1]),
                served_qps=reading(noisy_server_amounts[i, 0]),
            )
            for i, server in enumerate(self._servers)
        }
        return Observation(
            tick=tick,
            sites=MappingProxyType(sites),
            servers=MappingProxyType(servers),
            action_updates=updates,
            probe_results=probes,
            calendar=self.config.calendar.at(tick),
        )
