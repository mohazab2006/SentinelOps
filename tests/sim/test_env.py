"""Tests for the simulation engine."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from sentinelops.agent.base import NoOpAgent
from sentinelops.sim.actions import (
    REPAIR_TICKS,
    RESTART_TICKS,
    ROUTING_CONVERGENCE_TICKS,
    Action,
    ActionKind,
    ActionStatus,
    ActionUpdate,
)
from sentinelops.sim.env import (
    FAILURE_CAUSES,
    DnsEnv,
    EnvConfig,
    split_load,
    weighted_quantile,
)
from sentinelops.sim.faults import Fault, FaultKind
from sentinelops.sim.site import Traffic
from sentinelops.sim.telemetry import NoiseConfig, Observation
from sentinelops.sim.topology import ServerId, SiteId, default_topology

TOPO = default_topology()
Plan = Callable[[Observation], Sequence[Action]]


def _run(
    faults: Sequence[Fault] = (),
    plan: Plan = lambda obs: (),
    *,
    horizon: int = 600,
    seed: int = 0,
    config: EnvConfig | None = None,
) -> tuple[DnsEnv, list[Observation]]:
    env = DnsEnv(TOPO, faults, horizon, seed, config)
    observations = [env.reset()]
    while not env.done:
        observations.append(env.step(plan(observations[-1])))
    return env, observations


def _at(tick: int, *actions: Action) -> Plan:
    """A plan that issues ``actions`` right after seeing observation ``tick - 1``."""
    return lambda obs: actions if obs.tick == tick - 1 else ()


def _updates(observations: list[Observation]) -> list[ActionUpdate]:
    return [u for obs in observations for u in obs.action_updates]


def _statuses(observations: list[Observation]) -> list[tuple[int, str, ActionStatus]]:
    return [(u.tick, u.action.target, u.status) for u in _updates(observations)]


# ------------------------------------------------------------------- lifecycle


def test_reset_returns_tick_zero_and_run_covers_the_horizon() -> None:
    env, observations = _run(horizon=50)
    assert [obs.tick for obs in observations] == list(range(50))
    assert env.done and env.tick == 49
    assert set(observations[0].sites) == set(TOPO.sites)
    assert set(observations[0].servers) == set(TOPO.servers)


def test_step_before_reset_or_after_done_raises() -> None:
    env = DnsEnv(TOPO, [], horizon=2, seed=0)
    assert env.tick == -1
    with pytest.raises(RuntimeError):
        env.step(())
    env.reset()
    env.step(())
    with pytest.raises(RuntimeError):
        env.step(())


def test_reset_starts_over() -> None:
    env, _ = _run(horizon=20)
    first = env.events.to_jsonl()
    env.reset()
    while not env.done:
        env.step(())
    assert env.events.to_jsonl() == first


def test_constructor_validates_inputs() -> None:
    with pytest.raises(ValueError):
        DnsEnv(TOPO, [], horizon=0, seed=0)
    with pytest.raises(ValueError):
        DnsEnv(TOPO, [Fault(FaultKind.CRASH, "ott", 0, None, 1.0)], horizon=10, seed=0)


def test_invalid_action_target_raises() -> None:
    env = DnsEnv(TOPO, [], horizon=10, seed=0)
    env.reset()
    with pytest.raises(ValueError):
        env.step([Action(ActionKind.RESTART, "ott")])


# ----------------------------------------------------------------- determinism


def test_same_seed_and_actions_give_identical_runs() -> None:
    faults = [Fault(FaultKind.FLOOD, "eu-west", 100, 150, 0.7)]
    plan = _at(120, Action(ActionKind.RATE_LIMIT, "lon", limit_qps=3000.0))
    env_a, obs_a = _run(faults, plan, horizon=300, seed=11)
    env_b, obs_b = _run(faults, plan, horizon=300, seed=11)
    assert env_a.events.to_jsonl() == env_b.events.to_jsonl()
    assert obs_a == obs_b
    np.testing.assert_array_equal(env_a.truth.legit_failed, env_b.truth.legit_failed)


def test_different_seeds_differ() -> None:
    env_a, _ = _run(horizon=50, seed=1)
    env_b, _ = _run(horizon=50, seed=2)
    assert env_a.events.to_jsonl() != env_b.events.to_jsonl()


def test_agent_actions_do_not_change_demand_or_noise() -> None:
    env_idle, obs_idle = _run(horizon=200, seed=5)
    env_busy, obs_busy = _run(
        plan=_at(50, Action(ActionKind.RATE_LIMIT, "sin", limit_qps=1.0)), horizon=200, seed=5
    )
    np.testing.assert_array_equal(env_idle.truth.legit_offered, env_busy.truth.legit_offered)
    # An untouched site sees the same noisy readings in both runs.
    assert [o.sites[SiteId("ott")] for o in obs_idle] == [o.sites[SiteId("ott")] for o in obs_busy]


# ------------------------------------------------------------------ accounting


def test_healthy_network_fails_almost_nothing() -> None:
    env, _ = _run(horizon=600)
    assert env.truth.legit_failure_share() < 1e-4
    assert env.truth.attack_offered.sum() == 0.0


@pytest.mark.parametrize(
    "faults",
    [
        [Fault(FaultKind.SITE_OUTAGE, "lon", 50, None, 1.0)],
        [Fault(FaultKind.FLOOD, "na-east", 50, 100, 1.0)],
        [
            Fault(FaultKind.PATH_LOSS, "par", 30, 60, 0.5),
            Fault(FaultKind.CRASH, "sin-1", 40, None, 1),
        ],
    ],
)
def test_every_legitimate_query_is_accounted_for(faults: list[Fault]) -> None:
    env, _ = _run(faults, horizon=300)
    truth = env.truth
    queued = sum((q.queued for q in env._queues.values()), Traffic())
    offered = truth.legit_offered.sum()
    accounted = truth.legit_served.sum() + truth.legit_failed.sum() + queued.legit
    assert accounted == pytest.approx(offered, rel=1e-9)
    assert set(truth.legit_failed_by_cause) == set(FAILURE_CAUSES)


def test_failure_share_window_and_empty_window() -> None:
    env, _ = _run([Fault(FaultKind.SITE_OUTAGE, "lon", 100, None, 1.0)], horizon=200)
    assert env.truth.legit_failure_share(0, 100) < 1e-4
    assert env.truth.legit_failure_share(100) > 0.05
    assert env.truth.legit_failure_share(5, 5) == 0.0


def test_withdrawing_every_site_leaves_traffic_unrouted() -> None:
    plan = _at(10, *(Action(ActionKind.WITHDRAW, s) for s in TOPO.sites))
    env, observations = _run(plan=plan, horizon=40)
    after = 10 + ROUTING_CONVERGENCE_TICKS
    assert env.truth.legit_failed_by_cause["unrouted"][after:].sum() > 0
    assert env.truth.legit_failure_share(after) == pytest.approx(1.0)
    assert not any(t.announced for t in observations[-1].sites.values())


# ---------------------------------------------------------------------- faults


def test_crash_causes_failures_and_restart_fixes_it() -> None:
    crash = [Fault(FaultKind.CRASH, "yyz-1", 100, None, 1.0)]
    idle, _ = _run(crash, horizon=600)
    env, observations = _run(crash, _at(110, Action(ActionKind.RESTART, "yyz-1")), horizon=600)
    statuses = _statuses(observations)
    assert statuses[0] == (110, "yyz-1", ActionStatus.RUNNING)
    done_tick, _, final = statuses[-1]
    assert final == ActionStatus.SUCCEEDED
    assert 110 + RESTART_TICKS[0] <= done_tick <= 110 + RESTART_TICKS[1]
    assert env.truth.cleared == {0}
    assert env.truth.legit_failure_share(done_tick + 10) < 1e-4
    assert env.truth.legit_failure_share() < idle.truth.legit_failure_share()
    assert [e.name for e in env.events if e.name.startswith("fault")] == [
        "fault_started",
        "fault_cleared",
    ]


def test_restart_does_not_fix_hardware_but_repair_does() -> None:
    hardware = [Fault(FaultKind.HARDWARE, "yyz-1", 20, None, 1.0)]
    plan = _at(30, Action(ActionKind.RESTART, "yyz-1"))
    _, observations = _run(hardware, plan, horizon=200)
    assert _statuses(observations)[-1][2] == ActionStatus.FAILED

    env, observations = _run(
        hardware, _at(30, Action(ActionKind.REQUEST_REPAIR, "yyz-1")), horizon=800
    )
    done_tick, _, final = _statuses(observations)[-1]
    assert final == ActionStatus.SUCCEEDED
    assert 30 + REPAIR_TICKS[0] <= done_tick <= 30 + REPAIR_TICKS[1]
    assert env.truth.cleared == {0}


def test_site_repair_clears_site_outage() -> None:
    outage = [Fault(FaultKind.SITE_OUTAGE, "par", 0, None, 1.0)]
    env, _ = _run(outage, _at(5, Action(ActionKind.REQUEST_REPAIR, "par")), horizon=800)
    assert env.truth.cleared == {0}
    assert env.truth.legit_failure_share(750) < 1e-4


def test_hardware_fault_hides_host_metrics_but_not_served_count() -> None:
    _, observations = _run([Fault(FaultKind.HARDWARE, "ott-2", 0, None, 1.0)], horizon=5)
    server = observations[-1].servers[ServerId("ott-2")]
    assert server.cpu is None and server.mem is None
    assert server.served_qps == 0.0


def test_crashed_process_reports_idle_host() -> None:
    quiet = EnvConfig(noise=NoiseConfig(0.0, 0.0, 0.0))
    _, observations = _run([Fault(FaultKind.CRASH, "ott-2", 0, None, 1.0)], horizon=5, config=quiet)
    server = observations[-1].servers[ServerId("ott-2")]
    assert server.served_qps == 0.0
    assert server.cpu == pytest.approx(quiet.idle_cpu)


def test_memory_leak_shows_in_memory_readings() -> None:
    quiet = EnvConfig(noise=NoiseConfig(0.0, 0.0, 0.0))
    _, observations = _run(
        [Fault(FaultKind.EXHAUSTION, "fra-1", 0, None, 1.0)], horizon=400, config=quiet
    )
    leaking = observations[-1].servers[ServerId("fra-1")]
    healthy = observations[-1].servers[ServerId("fra-2")]
    assert leaking.mem is not None and healthy.mem is not None
    assert leaking.mem > healthy.mem + 0.3
    assert leaking.cpu is not None and healthy.cpu is not None
    assert leaking.cpu > healthy.cpu


def test_fault_with_duration_ends_by_itself() -> None:
    env, _ = _run([Fault(FaultKind.PATH_LOSS, "par", 10, 20, 0.5)], horizon=60)
    names = [(e.tick, e.name) for e in env.events if e.name.startswith("fault")]
    assert names == [(10, "fault_started"), (30, "fault_ended")]
    assert env.truth.legit_failed_by_cause["lost"][10:30].sum() > 0
    assert env.truth.legit_failed_by_cause["lost"][30:].sum() == 0


# --------------------------------------------------------------------- actions


def test_withdraw_moves_traffic_after_convergence_and_restore_brings_it_back() -> None:
    plan = lambda obs: (  # noqa: E731
        [Action(ActionKind.WITHDRAW, "fra")]
        if obs.tick == 19
        else [Action(ActionKind.RESTORE, "fra", weight=0.5)]
        if obs.tick == 59
        else []
    )
    _, observations = _run(plan=plan, horizon=100, config=EnvConfig(noise=NoiseConfig(0, 0, 0)))
    switched = 20 + ROUTING_CONVERGENCE_TICKS
    fra = SiteId("fra")
    assert observations[switched - 1].sites[fra].announced
    assert not observations[switched].sites[fra].announced
    assert observations[switched + 1].sites[fra].qps == 0.0
    restored = observations[60 + ROUTING_CONVERGENCE_TICKS + 1].sites[fra]
    assert restored.announced and restored.weight == 0.5
    assert restored.qps is not None and restored.qps > 0


def test_rate_limit_is_immediate_and_cuts_a_flood() -> None:
    flood = [Fault(FaultKind.FLOOD, "eu-west", 50, 200, 0.8)]
    limit = Action(ActionKind.RATE_LIMIT, "lon", limit_qps=3200.0)
    idle, _ = _run(flood, horizon=300)
    env, observations = _run(flood, _at(55, limit), horizon=300)
    assert [s for _, _, s in _statuses(observations)] == [
        ActionStatus.RUNNING,
        ActionStatus.SUCCEEDED,
    ]
    assert env.truth.legit_failure_share() < idle.truth.legit_failure_share() / 3


def test_conflicting_actions_fail_but_compatible_ones_run() -> None:
    plan = _at(
        10,
        Action(ActionKind.RESTART, "lon-1"),
        Action(ActionKind.REQUEST_REPAIR, "lon"),  # lon-1 already busy
        Action(ActionKind.WITHDRAW, "lon"),
        Action(ActionKind.RESTORE, "lon", weight=1.0),  # lon already rerouting
        Action(ActionKind.RATE_LIMIT, "lon", limit_qps=10.0),
    )
    _, observations = _run(plan=plan, horizon=12)
    first = [(u.action.kind, u.status) for u in observations[10].action_updates]
    assert first == [
        (ActionKind.RESTART, ActionStatus.RUNNING),
        (ActionKind.REQUEST_REPAIR, ActionStatus.FAILED),
        (ActionKind.WITHDRAW, ActionStatus.RUNNING),
        (ActionKind.RESTORE, ActionStatus.FAILED),
        (ActionKind.RATE_LIMIT, ActionStatus.RUNNING),
        (ActionKind.RATE_LIMIT, ActionStatus.SUCCEEDED),
    ]
    assert len({u.action_id for u in observations[10].action_updates}) == 5


def test_repair_takes_its_servers_offline() -> None:
    _, observations = _run(plan=_at(5, Action(ActionKind.REQUEST_REPAIR, "sin")), horizon=20)
    for server in TOPO.servers_at(SiteId("sin")):
        reading = observations[-1].servers[server]
        assert reading.served_qps == 0.0 and reading.cpu is None


# --------------------------------------------------------------------- helpers


def test_weighted_quantile() -> None:
    assert weighted_quantile([10, 20, 30], [1, 1, 1], 0.5) == 20
    assert weighted_quantile([30, 10, 20], [0, 9, 1], 0.95) == 20
    assert weighted_quantile([5.0], [2.0], 1.0) == 5.0
    with pytest.raises(ValueError):
        weighted_quantile([1, 2], [0, 0], 0.5)


@pytest.mark.parametrize(
    ("total", "capacities", "expected"),
    [
        (100.0, [100.0, 100.0], [50.0, 50.0]),
        (100.0, [20.0, 100.0], [20.0, 80.0]),
        (100.0, [0.0, 100.0], [0.0, 100.0]),
        (500.0, [100.0, 100.0], [100.0, 100.0]),
        (10.0, [0.0, 0.0], [0.0, 0.0]),
        (90.0, [10.0, 50.0, 50.0], [10.0, 40.0, 40.0]),
    ],
)
def test_split_load_is_round_robin_with_spillover(
    total: float, capacities: list[float], expected: list[float]
) -> None:
    assert split_load(total, capacities) == pytest.approx(expected)


def test_latency_reflects_distance() -> None:
    _, observations = _run(horizon=3, config=EnvConfig(noise=NoiseConfig(0, 0, 0)))
    sites = observations[-1].sites
    near, far = sites[SiteId("lon")].p95_latency_ms, sites[SiteId("sin")].p95_latency_ms
    assert near is not None and far is not None and near < far


def test_clearing_a_rate_limit_lets_traffic_through_again() -> None:
    plan = lambda obs: (  # noqa: E731
        [Action(ActionKind.RATE_LIMIT, "par", limit_qps=10.0)]
        if obs.tick == 9
        else [Action(ActionKind.CLEAR_RATE_LIMIT, "par")]
        if obs.tick == 19
        else []
    )
    env, _ = _run(plan=plan, horizon=40)
    rejected = env.truth.legit_failed_by_cause["rejected"]
    assert rejected[10:20].min() > 0
    assert rejected[20:].sum() == 0


def test_restart_before_onset_does_not_prevent_a_later_crash() -> None:
    crash = [Fault(FaultKind.CRASH, "ott-1", 200, None, 1.0)]
    env, _ = _run(crash, _at(5, Action(ActionKind.RESTART, "ott-1")), horizon=260)
    assert env.truth.cleared == set()
    assert [e.name for e in env.events if e.name.startswith("fault")] == ["fault_started"]


def test_benign_surge_is_legitimate_and_rate_limiting_it_hurts_users() -> None:
    surge = [Fault(FaultKind.BENIGN_SURGE, "eu-central", 50, 100, 0.5)]
    healthy, _ = _run(horizon=200)
    idle, _ = _run(surge, horizon=200)
    limited, _ = _run(
        surge, _at(55, Action(ActionKind.RATE_LIMIT, "fra", limit_qps=3000.0)), horizon=200
    )
    extra = idle.truth.legit_offered[50:150] - healthy.truth.legit_offered[50:150]
    assert extra == pytest.approx(0.5 * 4.0 * 2200.0)
    assert idle.truth.attack_offered.sum() == 0.0
    assert limited.truth.legit_failure_share() > idle.truth.legit_failure_share()


def test_noop_agent_runs_a_full_episode() -> None:
    agent = NoOpAgent()
    agent.reset(TOPO)
    env = DnsEnv(TOPO, [], horizon=3, seed=0)
    obs = env.reset()
    while not env.done:
        obs = env.step(agent.act(obs))
    assert obs.tick == 2


def test_fixing_after_a_fault_ended_does_not_log_a_clear() -> None:
    blip = [Fault(FaultKind.CRASH, "ott-1", 10, 5, 1.0)]
    env, _ = _run(blip, _at(20, Action(ActionKind.RESTART, "ott-1")), horizon=120)
    assert env.truth.cleared == set()
    assert [e.name for e in env.events if e.name.startswith("fault")] == [
        "fault_started",
        "fault_ended",
    ]
