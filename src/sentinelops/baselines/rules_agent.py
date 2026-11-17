"""Threshold-rule baseline: what a simple alerting runbook would do.

This is the agent SentinelOps is compared against. It sees exactly the same
telemetry and can take exactly the same actions, but its decisions are fixed
rules: if a condition holds for ``sustain_ticks`` consecutive ticks, apply the
matching runbook step. It never probes, never forecasts and never learns.

Runbook, in priority order, per site (first match wins each tick):

1. A server is silent (serves nothing while the site has traffic) -> restart
   it; after ``max_restarts`` restarts, withdraw the site and request a repair.
2. Too much packet loss -> withdraw the site (never the last announced one).
   If (nearly) everything is lost, the site is down rather than lossy, so
   also request a repair of the whole site; otherwise it would stay broken.
3. Traffic far above normal and replies failing -> rate limit the site.
4. Replies failing and a server's memory above ``mem_high`` -> restart the
   server using most memory (the memory-leak runbook). Failures without high
   memory, such as a neighbour absorbing a withdrawn site's traffic, are left
   alone: restarting a healthy server only takes capacity away.
5. Housekeeping: restore a withdrawn site, or clear a rate limit, once
   ``restore_after_ticks`` have passed.

Each site's "normal" traffic is measured during the warm-up (ticks 0-119),
which every scenario guarantees is healthy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sentinelops.sim.actions import Action, ActionKind, ActionStatus
from sentinelops.sim.telemetry import Observation, SiteTelemetry
from sentinelops.sim.topology import ServerId, SiteId, Topology

WARMUP_TICKS = 120
"""Ticks 0-119 are healthy; the agent only measures normal traffic there."""

SILENT_SERVED_QPS = 1.0
"""A server serving fewer queries per tick than this counts as silent."""

SILENT_MIN_TRAFFIC = 0.2
"""...but only while its site receives more than this fraction of its normal traffic."""

SITE_DOWN_LOSS = 0.95
"""A site losing at least this fraction of its queries is down, not just lossy."""

RATE_LIMIT_HEADROOM = 1.2
"""A rate limit lets through this multiple of the site's normal traffic."""

_DONE = frozenset({ActionStatus.SUCCEEDED, ActionStatus.FAILED})


@dataclass(frozen=True, slots=True)
class RuleThresholds:
    """The baseline's tunable thresholds.

    The defaults are sensible starting points; they are tuned later on the
    tuning seeds, the same way SentinelOps is tuned, so the comparison is fair.

    Attributes:
        fail_ratio: A site is failing when ``failed / (ok + failed)`` exceeds this.
        loss: A site has a path problem when its loss exceeds this.
        qps_factor: A site is overloaded when its traffic exceeds this multiple
            of its normal traffic (and replies are failing).
        sustain_ticks: A condition must hold this many consecutive ticks
            before the agent acts on it.
        restore_after_ticks: Cooldown before a withdrawn site is restored or a
            rate limit cleared.
        max_restarts: Restarts per server before the agent gives up on
            restarting and asks for a repair.
        mem_high: A failing site's server is restarted only if its memory
            use exceeds this fraction. Healthy servers stay below about 0.45,
            even when fully loaded.
    """

    fail_ratio: float = 0.05
    loss: float = 0.10
    qps_factor: float = 2.5
    sustain_ticks: int = 15
    restore_after_ticks: int = 300
    max_restarts: int = 2
    mem_high: float = 0.5

    def __post_init__(self) -> None:
        """Reject thresholds that would make the rules meaningless."""
        if not all(0.0 <= v < 1.0 for v in (self.fail_ratio, self.loss, self.mem_high)):
            raise ValueError("fail_ratio, loss and mem_high must be in [0, 1)")
        if self.qps_factor <= 1.0:
            raise ValueError("qps_factor must be greater than 1")
        if self.sustain_ticks < 1 or self.restore_after_ticks < 1 or self.max_restarts < 0:
            raise ValueError(
                "sustain_ticks and restore_after_ticks must be >= 1, max_restarts >= 0"
            )


def fail_ratio(site: SiteTelemetry) -> float:
    """Return ``failed / (ok + failed)`` for a site; missing or zero counts give 0."""
    ok = site.ok_replies or 0.0
    failed = site.failed_replies or 0.0
    total = ok + failed
    return failed / total if total > 0 else 0.0


class RuleBasedAgent:
    """The threshold-rule baseline agent (implements the ``Agent`` protocol).

    Besides the runbook (see the module docstring), the agent keeps track of
    its own actions so it never sends a duplicate while one is in flight, and
    after any of its actions at a site finishes it watches that site afresh for
    ``sustain_ticks`` before acting there again.

    Attributes:
        name: ``"rules"``.
        thresholds: The thresholds in use.
    """

    name = "rules"

    def __init__(self, thresholds: RuleThresholds | None = None) -> None:
        """Create the agent; ``None`` means the default :class:`RuleThresholds`."""
        self.thresholds = thresholds if thresholds is not None else RuleThresholds()
        self._topology: Topology | None = None
        self._clear_state()

    def _clear_state(self) -> None:
        self._qps_sum: dict[SiteId, float] = {}
        self._qps_count: dict[SiteId, int] = {}
        self._counters: dict[tuple[SiteId, str], int] = {}
        self._in_flight: dict[tuple[ActionKind, str], int] = {}
        self._restarts: dict[ServerId, int] = {}
        self._withdrawn_since: dict[SiteId, int] = {}
        self._limited_since: dict[SiteId, int] = {}

    def reset(self, topology: Topology) -> None:
        """Forget everything from previous runs and prepare for ``topology``."""
        self._topology = topology
        self._clear_state()

    # ------------------------------------------------------------------- acting

    def act(self, obs: Observation) -> Sequence[Action]:
        """Return the runbook actions for this tick (at most one step per site).

        Raises:
            RuntimeError: If :meth:`reset` was never called.
        """
        topology = self._topology
        if topology is None:
            raise RuntimeError("call reset(topology) before act()")
        self._absorb_updates(obs, topology)
        if obs.tick < WARMUP_TICKS:
            self._record_normal(obs)
            return []
        self._track_withdrawals(obs)

        announced = {s for s, t in obs.sites.items() if t.announced}
        actions: list[Action] = []
        for site in topology.sites:
            telemetry = obs.sites.get(site)
            if telemetry is None:
                continue
            step = self._decide(site, telemetry, obs, topology, announced)
            for action in step:
                key = (action.kind, action.target)
                self._in_flight[key] = self._in_flight.get(key, 0) + 1
                if action.kind == ActionKind.RESTART:
                    server = ServerId(action.target)
                    self._restarts[server] = self._restarts.get(server, 0) + 1
                if action.kind == ActionKind.WITHDRAW:
                    announced.discard(site)
            actions.extend(step)
        return actions

    def _decide(
        self,
        site: SiteId,
        telemetry: SiteTelemetry,
        obs: Observation,
        topology: Topology,
        announced: set[SiteId],
    ) -> list[Action]:
        """Update the site's sustain counters and return its runbook step (maybe none)."""
        th = self.thresholds
        normal = self._normal(site)
        servers = topology.servers_at(site)

        # Update every counter every tick, whichever rule ends up firing.
        silent: dict[ServerId, int] = {}
        for server in servers:
            reading = obs.servers.get(server)
            holds: bool | None = None
            if reading is not None and reading.served_qps is not None and telemetry.qps is not None:
                holds = (
                    reading.served_qps < SILENT_SERVED_QPS
                    and telemetry.qps > SILENT_MIN_TRAFFIC * normal
                )
            silent[server] = self._sustain(site, f"silent:{server}", holds)
        failing = fail_ratio(telemetry) > th.fail_ratio
        lossy = None if telemetry.loss is None else telemetry.loss > th.loss
        overloaded = None
        if telemetry.qps is not None:
            overloaded = normal > 0 and telemetry.qps > th.qps_factor * normal and failing
        down = None if telemetry.loss is None else telemetry.loss >= SITE_DOWN_LOSS
        loss_count = self._sustain(site, "loss", lossy)
        down_count = self._sustain(site, "down", down)
        overload_count = self._sustain(site, "overload", overloaded)
        fail_count = self._sustain(site, "fail", failing)

        # 1. Silent server. Servers under repair are offline on purpose: ignore them.
        for server in servers:
            if silent[server] < th.sustain_ticks or self._repairing(server):
                continue
            if self._busy(server):
                return []  # a restart is already running: wait, try nothing else
            self._counters[(site, f"silent:{server}")] = 0
            if self._restarts.get(server, 0) < th.max_restarts:
                return [Action(ActionKind.RESTART, server)]
            step = [Action(ActionKind.REQUEST_REPAIR, server)]
            if self._can_withdraw(site, telemetry, announced):
                step.insert(0, Action(ActionKind.WITHDRAW, site))
            return step

        # 2. Path loss; a site that loses everything is down and needs a repair.
        if loss_count >= th.sustain_ticks:
            can_withdraw = self._can_withdraw(site, telemetry, announced)
            idle = not (self._busy(site) or any(self._busy(s) for s in servers))
            if down_count >= th.sustain_ticks and idle:
                self._counters[(site, "loss")] = self._counters[(site, "down")] = 0
                step = [Action(ActionKind.REQUEST_REPAIR, site)]
                if can_withdraw:
                    step.insert(0, Action(ActionKind.WITHDRAW, site))
                return step
            if can_withdraw:
                self._counters[(site, "loss")] = 0
                return [Action(ActionKind.WITHDRAW, site)]

        # 3. Overload.
        if (
            overload_count >= th.sustain_ticks
            and site not in self._limited_since
            and not self._busy(site)
        ):
            self._counters[(site, "overload")] = 0
            return [Action(ActionKind.RATE_LIMIT, site, limit_qps=RATE_LIMIT_HEADROOM * normal)]

        # 4. Failing with a server short of memory: restart the server using most
        #    memory, but never while another operation is running at the site.
        site_busy = self._busy(site) or any(self._busy(s) for s in servers)
        if fail_count >= th.sustain_ticks and not site_busy:
            candidates = [
                (mem, server)
                for server in servers
                if (reading := obs.servers.get(server)) is not None
                and (mem := reading.mem) is not None
                and mem > th.mem_high
            ]
            if candidates:
                self._counters[(site, "fail")] = 0
                best = max(candidates, key=lambda c: c[0])[1]
                return [Action(ActionKind.RESTART, best)]

        # 5. Housekeeping after the cooldown.
        cooldown = th.restore_after_ticks
        if not self._busy(site):
            since = self._withdrawn_since.get(site)
            if since is not None and obs.tick - since >= cooldown:
                return [Action(ActionKind.RESTORE, site, weight=1.0)]
            since = self._limited_since.get(site)
            if since is not None and obs.tick - since >= cooldown:
                return [Action(ActionKind.CLEAR_RATE_LIMIT, site)]
        return []

    # ---------------------------------------------------------------- bookkeeping

    def _sustain(self, site: SiteId, condition: str, holds: bool | None) -> int:
        """Advance a consecutive-tick counter and return it.

        ``True`` adds one, ``False`` resets to zero, ``None`` (reading missing)
        leaves it unchanged, so a single dropped reading neither triggers nor
        cancels an alert.
        """
        key = (site, condition)
        if holds is True:
            self._counters[key] = self._counters.get(key, 0) + 1
        elif holds is False:
            self._counters[key] = 0
        return self._counters.get(key, 0)

    def _busy(self, target: str) -> bool:
        """Return whether any of our actions on ``target`` is still in flight."""
        return any(n > 0 for (_, t), n in self._in_flight.items() if t == target)

    def _repairing(self, server: ServerId) -> bool:
        return self._in_flight.get((ActionKind.REQUEST_REPAIR, server), 0) > 0

    def _can_withdraw(self, site: SiteId, telemetry: SiteTelemetry, announced: set[SiteId]) -> bool:
        """A site can be withdrawn if it is announced, idle, and not the last one up."""
        return telemetry.announced and not self._busy(site) and len(announced - {site}) > 0

    def _normal(self, site: SiteId) -> float:
        count = self._qps_count.get(site, 0)
        return self._qps_sum.get(site, 0.0) / count if count else 0.0

    def _record_normal(self, obs: Observation) -> None:
        for site, telemetry in obs.sites.items():
            if telemetry.qps is not None:
                self._qps_sum[site] = self._qps_sum.get(site, 0.0) + telemetry.qps
                self._qps_count[site] = self._qps_count.get(site, 0) + 1

    def _track_withdrawals(self, obs: Observation) -> None:
        for site, telemetry in obs.sites.items():
            if telemetry.announced:
                self._withdrawn_since.pop(site, None)
            else:
                self._withdrawn_since.setdefault(site, obs.tick)

    def _absorb_updates(self, obs: Observation, topology: Topology) -> None:
        """Follow our actions' lifecycle from the engine's status updates."""
        for update in obs.action_updates:
            if update.status not in _DONE:
                continue
            action = update.action
            key = (action.kind, action.target)
            self._in_flight[key] = max(0, self._in_flight.get(key, 0) - 1)
            ok = update.status == ActionStatus.SUCCEEDED
            if action.kind == ActionKind.RATE_LIMIT and ok:
                self._limited_since[SiteId(action.target)] = update.tick
            elif action.kind == ActionKind.CLEAR_RATE_LIMIT and ok:
                self._limited_since.pop(SiteId(action.target), None)
            elif action.kind == ActionKind.RESTART and ok:
                self._restarts[ServerId(action.target)] = 0
            elif action.kind == ActionKind.REQUEST_REPAIR and ok:
                repaired = (
                    topology.servers_at(SiteId(action.target))
                    if action.target in topology.sites
                    else (ServerId(action.target),)
                )
                for server in repaired:
                    self._restarts[server] = 0
            # Watch the site afresh before acting there again.
            site = (
                topology.site_of(ServerId(action.target))
                if action.target in topology.servers
                else SiteId(action.target)
            )
            for counter in [k for k in self._counters if k[0] == site]:
                self._counters[counter] = 0
