"""Tests for the threshold-rule baseline agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from sentinelops.agent.base import Agent, NoOpAgent
from sentinelops.baselines import RuleBasedAgent, RuleThresholds
from sentinelops.baselines.rules_agent import fail_ratio
from sentinelops.eval.harness import make_agent, simulate
from sentinelops.scenarios import generate
from sentinelops.sim.actions import Action, ActionKind, ActionStatus, ActionUpdate
from sentinelops.sim.env import DnsEnv
from sentinelops.sim.faults import Fault, FaultKind
from sentinelops.sim.telemetry import Observation, ServerTelemetry, SiteTelemetry
from sentinelops.sim.topology import ServerId, SiteId, default_topology

TOPO = default_topology()
LON, LON1, LON2 = SiteId("lon"), ServerId("lon-1"), ServerId("lon-2")
NORMAL_QPS = 1000.0
SUSTAIN = RuleThresholds().sustain_ticks
COOLDOWN = RuleThresholds().restore_after_ticks

# ------------------------------------------------------------------ fake network


@dataclass
class FakeNet:
    """A tiny stand-in for DnsEnv that reports action lifecycles like the real engine.

    Actions returned by ``act()`` at tick t start at tick t+1: RUNNING (plus SUCCEEDED
    for instant ones) appears in the observation of t+1; RESTART finishes after
    ``restart_ticks`` with ``restart_status``, WITHDRAW/RESTORE after 5 ticks,
    REQUEST_REPAIR after 500. Telemetry is healthy unless a test overrides it.
    """

    restart_status: ActionStatus = ActionStatus.FAILED
    restart_ticks: int = 30
    site: dict[SiteId, dict[str, float | bool | None]] = field(default_factory=dict)
    server: dict[ServerId, dict[str, float | None]] = field(default_factory=dict)
    announced: dict[SiteId, bool] = field(default_factory=dict)
    limited: set[SiteId] = field(default_factory=set)
    _pending: list[tuple[int, int, Action, ActionStatus]] = field(default_factory=list)
    _next_id: int = 0

    def step(self, tick: int, actions: list[Action]) -> Observation:
        updates: list[ActionUpdate] = []
        for action in actions:
            action_id, self._next_id = self._next_id, self._next_id + 1
            updates.append(ActionUpdate(action_id, action, ActionStatus.RUNNING, tick))
            kind = action.kind
            if kind in (ActionKind.RATE_LIMIT, ActionKind.CLEAR_RATE_LIMIT):
                site = SiteId(action.target)
                if kind == ActionKind.RATE_LIMIT:
                    self.limited.add(site)
                else:
                    self.limited.discard(site)
                updates.append(ActionUpdate(action_id, action, ActionStatus.SUCCEEDED, tick))
                continue
            if kind == ActionKind.RESTART:
                end, status = tick + self.restart_ticks, self.restart_status
            elif kind == ActionKind.REQUEST_REPAIR:
                end, status = tick + 500, ActionStatus.SUCCEEDED
            else:
                end, status = tick + 5, ActionStatus.SUCCEEDED
            self._pending.append((end, action_id, action, status))
        for item in [p for p in self._pending if p[0] <= tick]:
            self._pending.remove(item)
            _, action_id, action, status = item
            if action.kind == ActionKind.WITHDRAW:
                self.announced[SiteId(action.target)] = False
            elif action.kind == ActionKind.RESTORE:
                self.announced[SiteId(action.target)] = True
            updates.append(ActionUpdate(action_id, action, status, tick))
        return self._observe(tick, tuple(updates))

    def _observe(self, tick: int, updates: tuple[ActionUpdate, ...]) -> Observation:
        sites: dict[SiteId, SiteTelemetry] = {}
        for sid in TOPO.sites:
            up = self.announced.get(sid, True)
            values: dict[str, float | bool | None] = {
                "qps": NORMAL_QPS,
                "p95_latency_ms": 20.0,
                "loss": 0.0,
                "queue_depth": 0.0,
                "ok_replies": NORMAL_QPS,
                "failed_replies": 0.0,
            }
            values.update(self.site.get(sid, {}))
            if sid in self.limited:
                values["failed_replies"] = 0.0
            if not up:  # withdrawn: no traffic reaches it
                values.update(qps=0.0, loss=0.0, ok_replies=0.0, failed_replies=0.0)
            sites[sid] = SiteTelemetry(**values, announced=up, weight=1.0 if up else 0.0)  # type: ignore[arg-type]
        servers = {
            srv: ServerTelemetry(
                **{"cpu": 0.5, "mem": 0.5, "served_qps": NORMAL_QPS / 2, **self.server.get(srv, {})}
            )
            for srv in TOPO.servers
        }
        return Observation(tick=tick, sites=sites, servers=servers, action_updates=updates)


Hook = Callable[[int, FakeNet], None]


def run(
    agent: RuleBasedAgent, net: FakeNet, ticks: int, hook: Hook | None = None, *, reset: bool = True
) -> dict[int, list[Action]]:
    """Drive ``agent`` against ``net`` for ``ticks`` ticks; return actions by tick."""
    if reset:
        agent.reset(TOPO)
    log: dict[int, list[Action]] = {}
    pending: list[Action] = []
    for tick in range(ticks):
        if hook is not None:
            hook(tick, net)
        obs = net.step(tick, pending)
        pending = list(agent.act(obs))
        if pending:
            log[tick] = pending
    return log


def after_warmup(change: Callable[[FakeNet], None]) -> Hook:
    """A hook that applies ``change`` from tick 120 on."""

    def hook(tick: int, net: FakeNet) -> None:
        if tick == 120:
            change(net)

    return hook


def lon1_silent(net: FakeNet) -> None:
    net.server[LON1] = {"served_qps": 0.0, "cpu": 0.03}


def flat(log: dict[int, list[Action]]) -> list[Action]:
    return [action for tick in sorted(log) for action in log[tick]]


# -------------------------------------------------------------- acceptance 1-9


def test_healthy_network_gets_no_actions() -> None:
    assert run(RuleBasedAgent(), FakeNet(), 400) == {}


def test_silent_server_is_restarted_exactly_when_sustained() -> None:
    log = run(RuleBasedAgent(), FakeNet(), 200, after_warmup(lon1_silent))
    first = min(log)
    assert first == 120 + SUSTAIN - 1  # counted on ticks 120..134
    assert log[first] == [Action(ActionKind.RESTART, LON1)]


def test_failed_restarts_lead_to_withdraw_and_repair() -> None:
    log = run(RuleBasedAgent(), FakeNet(), 400, after_warmup(lon1_silent))
    assert flat(log)[:4] == [
        Action(ActionKind.RESTART, LON1),
        Action(ActionKind.RESTART, LON1),
        Action(ActionKind.WITHDRAW, LON),
        Action(ActionKind.REQUEST_REPAIR, LON1),
    ]
    withdraw_tick = next(t for t, acts in log.items() if acts[0].kind == ActionKind.WITHDRAW)
    assert len(log[withdraw_tick]) == 2  # one runbook step, two actions, same tick


def test_high_loss_withdraws_the_site() -> None:
    log = run(
        RuleBasedAgent(),
        FakeNet(),
        200,
        after_warmup(lambda n: n.site.update({LON: {"loss": 0.3}})),
    )
    assert log == {120 + SUSTAIN - 1: [Action(ActionKind.WITHDRAW, LON)]}


def test_last_announced_site_is_never_withdrawn() -> None:
    net = FakeNet(announced={s: s == LON for s in TOPO.sites})
    log = run(
        RuleBasedAgent(), net, 200, after_warmup(lambda n: n.site.update({LON: {"loss": 0.3}}))
    )
    assert all(a.kind != ActionKind.WITHDRAW for a in flat(log))


def site_down(net: FakeNet) -> None:
    net.site[LON] = {"qps": 0.0, "loss": 1.0, "ok_replies": 0.0, "failed_replies": NORMAL_QPS}


def test_site_that_loses_everything_is_withdrawn_and_repaired() -> None:
    log = run(RuleBasedAgent(), FakeNet(), 200, after_warmup(site_down))
    assert log == {
        120 + SUSTAIN - 1: [
            Action(ActionKind.WITHDRAW, LON),
            Action(ActionKind.REQUEST_REPAIR, LON),
        ]
    }


def test_repaired_site_is_restored_only_after_the_repair_finishes() -> None:
    fired = 120 + SUSTAIN - 1
    repair_done = fired + 1 + 500  # FakeNet repairs take 500 ticks

    def outage_until_repaired(tick: int, net: FakeNet) -> None:
        if tick == 120:
            site_down(net)
        elif tick == repair_done:
            net.site.pop(LON)

    log = run(RuleBasedAgent(), FakeNet(), 1000, outage_until_repaired)
    (restored_at,) = (t for t, acts in log.items() if acts[0].kind == ActionKind.RESTORE)
    assert restored_at == repair_done
    assert sorted(log) == [fired, restored_at]


def test_last_site_down_is_repaired_once_without_withdrawing() -> None:
    net = FakeNet(announced={s: s == LON for s in TOPO.sites})
    actions = flat(run(RuleBasedAgent(), net, 450, after_warmup(site_down)))
    at_lon = [a for a in actions if a.target == LON]
    assert at_lon == [Action(ActionKind.REQUEST_REPAIR, LON)]  # no repeat while it runs


def test_heavy_but_partial_loss_is_not_a_site_repair() -> None:
    lossy = {"loss": 0.9}
    log = run(RuleBasedAgent(), FakeNet(), 200, after_warmup(lambda n: n.site.update({LON: lossy})))
    assert flat(log) == [Action(ActionKind.WITHDRAW, LON)]


def test_site_outage_run_ends_with_the_site_repaired() -> None:
    # Regression: the site used to be withdrawn and restored over and over, never repaired.
    scenario = generate("site_outage", 0, TOPO)
    env = simulate(scenario, RuleBasedAgent(), TOPO)
    repairs = [
        event.data["target"]
        for event in env.events.named("action")
        if event.data["kind"] == ActionKind.REQUEST_REPAIR
        and event.data["status"] == ActionStatus.SUCCEEDED
    ]
    assert repairs == [scenario.faults[0].target]
    assert env.truth.cleared == {0}


def test_overload_with_failures_is_rate_limited_at_normal_plus_headroom() -> None:
    overload = {"qps": 3 * NORMAL_QPS, "ok_replies": 2000.0, "failed_replies": 1000.0}
    log = run(
        RuleBasedAgent(), FakeNet(), 200, after_warmup(lambda n: n.site.update({LON: overload}))
    )
    (tick, actions), *_ = log.items()
    assert tick == 120 + SUSTAIN - 1
    assert actions[0].kind == ActionKind.RATE_LIMIT and actions[0].target == LON
    assert actions[0].limit_qps == pytest.approx(1.2 * NORMAL_QPS)


def test_rate_limit_is_cleared_after_the_cooldown() -> None:
    overload = {"qps": 3 * NORMAL_QPS, "ok_replies": 2000.0, "failed_replies": 1000.0}
    log = run(
        RuleBasedAgent(), FakeNet(), 600, after_warmup(lambda n: n.site.update({LON: overload}))
    )
    limit_tick = min(log)
    clears = [t for t, acts in log.items() if acts[0].kind == ActionKind.CLEAR_RATE_LIMIT]
    assert clears[0] == limit_tick + 1 + COOLDOWN  # counted from the tick it SUCCEEDED


def test_withdrawn_site_is_restored_after_the_cooldown_not_before() -> None:
    log = run(
        RuleBasedAgent(),
        FakeNet(),
        600,
        after_warmup(lambda n: n.site.update({LON: {"loss": 0.3}})),
    )
    withdraw_tick = min(log)
    withdrawn_seen = withdraw_tick + 1 + 5  # started next tick, converged 5 ticks later
    restores = [t for t, acts in log.items() if acts[0].kind == ActionKind.RESTORE]
    assert restores[0] == withdrawn_seen + COOLDOWN
    assert log[restores[0]] == [Action(ActionKind.RESTORE, LON, weight=1.0)]


def test_no_duplicate_action_while_one_is_running() -> None:
    net = FakeNet(restart_ticks=250, restart_status=ActionStatus.SUCCEEDED)
    log = run(RuleBasedAgent(), net, 360, after_warmup(lon1_silent))
    assert flat(log) == [Action(ActionKind.RESTART, LON1)]


def test_reset_clears_all_state() -> None:
    agent = RuleBasedAgent()
    run(agent, FakeNet(), 400, after_warmup(lon1_silent))  # restarts used up
    assert run(agent, FakeNet(), 400) == {}
    log = run(agent, FakeNet(), 200, after_warmup(lon1_silent))
    assert flat(log)[0] == Action(ActionKind.RESTART, LON1)  # not withdraw: counts were reset


def test_missing_readings_never_crash_the_agent() -> None:
    def hook(tick: int, net: FakeNet) -> None:
        missing = tick % 3 == 0
        for sid in TOPO.sites:
            net.site[sid] = (
                dict.fromkeys(
                    ("qps", "p95_latency_ms", "loss", "queue_depth", "ok_replies", "failed_replies")
                )
                if missing
                else {}
            )
        for srv in TOPO.servers:
            net.server[srv] = dict.fromkeys(("cpu", "mem", "served_qps")) if missing else {}

    assert run(RuleBasedAgent(), FakeNet(), 400, hook) == {}


# ---------------------------------------------------------- acceptance 10: real engine


def _failure_share(agent: Agent) -> tuple[float, list[Action]]:
    env = DnsEnv(TOPO, [Fault(FaultKind.CRASH, "lon-1", 300, None, 0.8)], 900, seed=7)
    agent.reset(TOPO)
    obs = env.reset()
    issued: list[Action] = []
    while not env.done:
        actions = list(agent.act(obs))
        issued.extend(actions)
        obs = env.step(actions)
    return env.truth.legit_failure_share(), issued


def test_end_to_end_crash_run_is_fixed_by_a_restart() -> None:
    rules_share, issued = _failure_share(RuleBasedAgent())
    noop_share, _ = _failure_share(NoOpAgent())
    assert Action(ActionKind.RESTART, "lon-1") in issued
    assert rules_share < noop_share


# ------------------------------------------------------------------ other rules


def test_failing_site_restarts_the_server_with_most_memory() -> None:
    def leak(net: FakeNet) -> None:
        net.site[LON] = {"ok_replies": 900.0, "failed_replies": 100.0}
        net.server[LON2] = {"mem": 0.95}

    log = run(
        RuleBasedAgent(), FakeNet(restart_status=ActionStatus.SUCCEEDED), 200, after_warmup(leak)
    )
    assert log[min(log)] == [Action(ActionKind.RESTART, LON2)]


def test_no_memory_readings_means_no_memory_restart() -> None:
    def failing(net: FakeNet) -> None:
        net.site[LON] = {"ok_replies": 900.0, "failed_replies": 100.0}
        net.server[LON1] = {"mem": None}
        net.server[LON2] = {"mem": None}

    assert run(RuleBasedAgent(), FakeNet(), 200, after_warmup(failing)) == {}


def test_failing_site_with_normal_memory_is_not_restarted() -> None:
    # e.g. a neighbour absorbing a withdrawn site's traffic: restarting a healthy
    # server there would only take more capacity away.
    def overloaded(net: FakeNet) -> None:
        net.site[LON] = {"qps": 1.8 * NORMAL_QPS, "ok_replies": 1500.0, "failed_replies": 300.0}
        net.server[LON1] = {"mem": 0.42}
        net.server[LON2] = {"mem": 0.44}

    assert run(RuleBasedAgent(), FakeNet(), 400, after_warmup(overloaded)) == {}


def test_memory_threshold_is_tunable() -> None:
    def failing(net: FakeNet) -> None:
        net.site[LON] = {"ok_replies": 900.0, "failed_replies": 100.0}
        net.server[LON1] = {"mem": 0.42}
        net.server[LON2] = {"mem": 0.35}

    agent = RuleBasedAgent(RuleThresholds(mem_high=0.4))
    log = run(agent, FakeNet(restart_status=ActionStatus.SUCCEEDED), 200, after_warmup(failing))
    assert log[min(log)] == [Action(ActionKind.RESTART, LON1)]


def test_hardware_fault_never_triggers_restarts_at_healthy_sites() -> None:
    # Regression: after withdrawing the broken site, its neighbour is overloaded
    # and used to have its healthy servers restarted one after another.
    scenario = generate("hardware", 0, TOPO)
    (fault,) = scenario.faults
    env = simulate(scenario, RuleBasedAgent(), TOPO)
    restarted = {
        event.data["target"]
        for event in env.events.named("action")
        if event.data["kind"] == ActionKind.RESTART
    }
    assert restarted == {fault.target}


def test_last_site_with_exhausted_restarts_still_gets_a_repair() -> None:
    net = FakeNet(announced={s: s == LON for s in TOPO.sites})
    log = run(RuleBasedAgent(), net, 260, after_warmup(lon1_silent))
    actions = flat(log)
    assert Action(ActionKind.REQUEST_REPAIR, LON1) in actions
    assert Action(ActionKind.WITHDRAW, LON) not in actions


def test_a_single_missing_reading_does_not_reset_the_count() -> None:
    def hook(tick: int, net: FakeNet) -> None:
        if tick == 120:
            net.site[LON] = {"loss": 0.3}
        elif tick == 125:
            net.site[LON] = {"loss": None}
        elif tick == 126:
            net.site[LON] = {"loss": 0.3}

    log = run(RuleBasedAgent(), FakeNet(), 200, hook)
    assert min(log) == 120 + SUSTAIN  # one tick later than normal, not 15 later


def test_warmup_takes_no_action_even_when_something_is_wrong() -> None:
    def hook(tick: int, net: FakeNet) -> None:
        if tick == 0:
            lon1_silent(net)

    log = run(RuleBasedAgent(), FakeNet(), 120, hook)
    assert log == {}


def test_successful_repair_resets_the_restart_count() -> None:
    log = run(RuleBasedAgent(), FakeNet(), 1000, after_warmup(lon1_silent))
    actions = flat(log)
    repair_at = actions.index(Action(ActionKind.REQUEST_REPAIR, LON1))
    # Once the repair has succeeded and lon-1 is silent again, restarts are back on.
    assert Action(ActionKind.RESTART, LON1) in actions[repair_at + 1 :]


def test_repairing_a_whole_site_resets_its_servers_restart_counts() -> None:
    agent = RuleBasedAgent()
    agent.reset(TOPO)
    agent._restarts.update({LON1: 2, LON2: 2})
    repair = Action(ActionKind.REQUEST_REPAIR, LON)
    obs = FakeNet()._observe(5, (ActionUpdate(0, repair, ActionStatus.SUCCEEDED, 5),))
    agent.act(obs)
    assert agent._restarts == {LON1: 0, LON2: 0}


def test_site_missing_from_an_observation_is_skipped() -> None:
    agent = RuleBasedAgent()
    run(agent, FakeNet(), 130, after_warmup(lambda n: n.site.update({LON: {"loss": 0.3}})))
    full = FakeNet().step(130, [])
    partial = Observation(
        tick=130,
        sites={s: t for s, t in full.sites.items() if s != LON},
        servers=full.servers,
        action_updates=(),
    )
    assert list(agent.act(partial)) == []


# ------------------------------------------------------------------ plumbing


def test_agent_follows_the_protocol_and_is_registered_as_rules() -> None:
    agent = RuleBasedAgent()
    assert isinstance(agent, Agent)
    assert agent.name == "rules"
    assert agent.thresholds == RuleThresholds()
    assert isinstance(make_agent("rules"), RuleBasedAgent)


def test_custom_thresholds_are_used() -> None:
    agent = RuleBasedAgent(RuleThresholds(sustain_ticks=3))
    log = run(agent, FakeNet(), 200, after_warmup(lon1_silent))
    assert min(log) == 120 + 3 - 1


def test_act_before_reset_is_an_error() -> None:
    obs = FakeNet().step(0, [])
    with pytest.raises(RuntimeError, match="reset"):
        RuleBasedAgent().act(obs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fail_ratio": 1.0},
        {"loss": -0.1},
        {"qps_factor": 1.0},
        {"sustain_ticks": 0},
        {"restore_after_ticks": 0},
        {"max_restarts": -1},
        {"mem_high": 1.0},
        {"mem_high": -0.1},
    ],
)
def test_nonsense_thresholds_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RuleThresholds(**kwargs)  # type: ignore[arg-type]


def test_fail_ratio_handles_missing_and_zero_counts() -> None:
    def site(ok: float | None, failed: float | None) -> SiteTelemetry:
        return SiteTelemetry(100.0, 20.0, 0.0, 0.0, ok, failed, True, 1.0)

    assert fail_ratio(site(900.0, 100.0)) == pytest.approx(0.1)
    assert fail_ratio(site(None, None)) == 0.0
    assert fail_ratio(site(0.0, 0.0)) == 0.0
    assert fail_ratio(site(None, 50.0)) == 1.0
