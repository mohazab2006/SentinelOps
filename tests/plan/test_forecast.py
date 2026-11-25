"""Tests for recovery forecasting."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from sentinelops.diagnose.inference import Belief, Hypothesis, hypotheses
from sentinelops.eval.splits import Split, split_scenarios
from sentinelops.plan.forecast import (
    ForecastConfig,
    ForecastTable,
    NetworkEstimate,
    Rollout,
    RunningAction,
    candidate_actions,
    forecast_belief,
    hypothesis_faults,
    rollout,
)
from sentinelops.scenarios.families import FAMILIES, Scenario, generate
from sentinelops.sim.actions import Action, ActionKind
from sentinelops.sim.env import DnsEnv, EnvConfig
from sentinelops.sim.faults import EXHAUSTION_RAMP_TICKS, Fault, FaultKind, compute_effects
from sentinelops.sim.telemetry import Observation, ServerTelemetry, SiteTelemetry
from sentinelops.sim.topology import RegionId, ServerId, SiteId, default_topology
from sentinelops.sim.traffic import route

TOPO = default_topology()
SITE = SiteId("lon")
SERVERS = sorted(TOPO.sites[SITE].servers)
QUIET = NetworkEstimate.healthy(TOPO)
EXACT = EnvConfig(demand_noise_cv=0.0, demand_drift_cv=0.0)  # demand = base_qps every tick
K = ActionKind


def _incident_site(fault: Fault) -> SiteId:
    if fault.target in TOPO.servers:
        return TOPO.site_of(ServerId(fault.target))
    if fault.target in TOPO.sites:
        return SiteId(fault.target)
    return TOPO.preferences[RegionId(fault.target)][0]


def _belief(truth: Hypothesis, site: str = SITE) -> Belief:
    hs = hypotheses(site, TOPO)
    probs = np.array([1.0 if h == truth else 0.0 for h in hs])
    return Belief(SiteId(site), hs, probs)


# ------------------------------------------------------------------ same physics as the simulator


def _simulate(scenario: Scenario, horizon: int, plan: tuple[tuple[Action, int], ...]) -> DnsEnv:
    env = DnsEnv(TOPO, scenario.faults, horizon, scenario.seed, EXACT)
    env.reset()
    while not env.done:
        env.step([a for a, at in plan if at == env.tick + 1])
    return env


def _forecast_matches(scenario: Scenario, plan: tuple[tuple[Action, int], ...]) -> None:
    horizon = 2400  # long enough for a repair to finish
    env = _simulate(scenario, horizon, plan)
    durations = [e.tick for e in env.events if e.name == "action"]
    config = ForecastConfig(horizon=horizon - 1, restore_after=None)
    if plan and plan[0][0].kind in (K.RESTART, K.REQUEST_REPAIR):
        taken = durations[-1] - durations[0]  # the duration the simulator drew
        field = "restart_ticks" if plan[0][0].kind == K.RESTART else "repair_ticks"
        config = ForecastConfig(horizon=horizon - 1, restore_after=None, **{field: taken})
    site = _incident_site(scenario.faults[0])
    result = rollout(QUIET, scenario.faults, TOPO, site=site, plan=plan, config=config)
    assert result.offered == pytest.approx(env.truth.legit_offered[1:].sum(), rel=1e-12)
    assert result.failed == pytest.approx(env.truth.legit_failed[1:].sum(), rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("family", FAMILIES)
def test_waiting_forecast_equals_the_simulator(family: str) -> None:
    """With the true faults and noise-free demand the forecast is the simulator, exactly."""
    _forecast_matches(generate(family, 3, TOPO), ())


@pytest.mark.parametrize(
    ("family", "kind"),
    [
        ("site_outage", K.WITHDRAW),
        ("flood", K.RATE_LIMIT),
        ("crash", K.RESTART),
        ("exhaustion", K.RESTART),
        ("hardware", K.REQUEST_REPAIR),
    ],
)
def test_action_forecast_equals_the_simulator(family: str, kind: ActionKind) -> None:
    scenario = generate(family, 3, TOPO)
    fault = scenario.faults[0]
    site = _incident_site(fault)
    target = fault.target if kind in (K.RESTART, K.REQUEST_REPAIR) else site
    limit = 4000.0 if kind == K.RATE_LIMIT else None
    _forecast_matches(scenario, ((Action(kind, target, limit_qps=limit), fault.onset + 40),))


def test_unrouted_traffic_is_counted_when_every_site_is_withdrawn() -> None:
    estimate = NetworkEstimate(0, QUIET.demand, dict.fromkeys(TOPO.sites, 0.0))
    config = ForecastConfig(horizon=50, restore_after=None)
    result = rollout(estimate, (), TOPO, site=SITE, config=config)
    assert result.unrouted == pytest.approx(result.offered)
    assert result.at_site == result.elsewhere == 0.0


# ------------------------------------------------------------------ what forecasts capture


def _cost(
    faults: tuple[Fault, ...],
    plan: tuple[tuple[Action, int], ...] = (),
    estimate: NetworkEstimate = QUIET,
    **config: int | None,
) -> Rollout:
    return rollout(estimate, faults, TOPO, site=SITE, plan=plan, config=ForecastConfig(**config))


CRASH = (Fault(FaultKind.CRASH, SERVERS[0], 0, None, 1.0),)
HARDWARE = (Fault(FaultKind.HARDWARE, SERVERS[0], 0, None, 1.0),)


def test_approval_delay_costs_failures() -> None:
    """Each tick a restart waits for approval is another tick of the crash."""
    restart = Action(K.RESTART, SERVERS[0])
    costs = [_cost(CRASH, ((restart, 1 + delay),)).failed for delay in (0, 60, 120, 600)]
    assert costs == sorted(costs)
    assert costs[0] < costs[-1] < _cost(CRASH).failed


def test_restart_does_not_fix_hardware_but_a_repair_does() -> None:
    restart = ((Action(K.RESTART, SERVERS[0]), 1),)
    repair = ((Action(K.REQUEST_REPAIR, SERVERS[0]), 1),)
    wait = _cost(HARDWARE).failed
    assert _cost(HARDWARE, restart).failed == pytest.approx(wait)
    assert _cost(HARDWARE, repair).failed < wait
    assert _cost(HARDWARE, repair, horizon=500).failed > 0  # still under repair at the end


def test_restarting_the_wrong_server_takes_the_whole_site_down() -> None:
    wrong = ((Action(K.RESTART, SERVERS[1]), 1),)
    assert _cost(CRASH, wrong).failed > _cost(CRASH).failed


def test_withdrawing_a_healthy_site_moves_failures_elsewhere() -> None:
    """The local-only ablation sees no harm in it; the full forecast sees the collateral."""
    busy = NetworkEstimate(0, {r: 1.5 * q for r, q in QUIET.demand.items()}, QUIET.weights)
    withdraw = ((Action(K.WITHDRAW, SITE), 1),)
    stay, moved = _cost((), estimate=busy), _cost((), withdraw, busy, restore_after=None)
    assert moved.cost(local_only=True) <= stay.cost(local_only=True)
    assert moved.elsewhere > stay.elsewhere
    assert moved.failed == moved.at_site + moved.elsewhere + moved.unrouted


def test_withdrawn_site_is_restored_once_the_fault_is_over() -> None:
    loss = (Fault(FaultKind.PATH_LOSS, SITE, 0, 100, 0.5),)
    busy = NetworkEstimate(0, {r: 1.5 * q for r, q in QUIET.demand.items()}, QUIET.weights)
    withdraw = ((Action(K.WITHDRAW, SITE), 1),)
    kept_out = _cost(loss, withdraw, busy, restore_after=None)
    restored = _cost(loss, withdraw, busy, restore_after=30)
    assert restored.elsewhere < kept_out.elsewhere
    # The countdown starts when the site goes out, not when the network went quiet.
    late = ((Action(K.WITHDRAW, SITE), 200),)
    out_for_30 = _cost((), late, busy, restore_after=30, horizon=300)
    out_for_0 = _cost((), late, busy, restore_after=0, horizon=300)
    stay_in = _cost((), estimate=busy, horizon=300)
    assert out_for_30.elsewhere > out_for_0.elsewhere > stay_in.elsewhere
    already_out = NetworkEstimate(0, busy.demand, {**busy.weights, SITE: 0.0})
    assert (
        _cost((), estimate=already_out).failed
        < _cost((), estimate=already_out, restore_after=None).failed
    )


def test_running_actions_finish_and_block_a_second_one() -> None:
    running = RunningAction(Action(K.RESTART, SERVERS[0]), end_tick=20)
    estimate = NetworkEstimate(0, QUIET.demand, QUIET.weights, running=(running,))
    helped = _cost(CRASH, estimate=estimate)
    assert helped.failed < _cost(CRASH).failed
    again = ((Action(K.RESTART, SERVERS[0]), 1),)
    assert _cost(CRASH, again, estimate).failed == pytest.approx(helped.failed)


def test_waiting_queries_time_out_at_a_dead_site() -> None:
    outage = (Fault(FaultKind.SITE_OUTAGE, SITE, 0, None, 1.0),)
    queued = NetworkEstimate(0, QUIET.demand, QUIET.weights, queue_depth={SITE: 1000.0})
    assert _cost(outage, estimate=queued).failed - _cost(outage).failed == pytest.approx(1000.0)


def test_queue_of_a_withdrawn_site_drains() -> None:
    weights = {**QUIET.weights, SITE: 0.0}
    queued = NetworkEstimate(0, QUIET.demand, weights, queue_depth={SITE: 500.0})
    assert _cost((), estimate=queued, restore_after=None).at_site == 0.0


def test_rate_limit_and_clearing_it() -> None:
    limited = NetworkEstimate(0, QUIET.demand, QUIET.weights, limits={SITE: 100.0})
    clear = ((Action(K.CLEAR_RATE_LIMIT, SITE), 1),)
    assert _cost((), clear, limited).failed < _cost((), estimate=limited).failed


# ------------------------------------------------------------------ hypotheses -> faults


def test_server_and_site_hypotheses_become_faults() -> None:
    for kind in (FaultKind.CRASH, FaultKind.HARDWARE):
        assert hypothesis_faults(Hypothesis(kind, SERVERS[1]), SITE, QUIET, TOPO) == (
            Fault(kind, SERVERS[1], 0, None, 1.0),
        )
    (outage,) = hypothesis_faults(Hypothesis("site_outage"), SITE, QUIET, TOPO)
    assert (outage.kind, outage.target, outage.duration) == (FaultKind.SITE_OUTAGE, SITE, None)
    assert hypothesis_faults(Hypothesis("healthy"), SITE, QUIET, TOPO) == ()


def test_path_loss_uses_the_measured_loss_and_ends_by_itself() -> None:
    estimate = NetworkEstimate(5, QUIET.demand, QUIET.weights, loss={SITE: 0.37})
    config = ForecastConfig(transient_ticks=200)
    (fault,) = hypothesis_faults(Hypothesis("path_loss"), SITE, estimate, TOPO, config)
    assert (fault.severity, fault.onset, fault.duration) == (0.37, 5, 200)
    (floor,) = hypothesis_faults(Hypothesis("path_loss"), SITE, QUIET, TOPO)
    assert floor.severity == 0.05


def test_leak_is_placed_where_its_measured_size_says() -> None:
    estimate = NetworkEstimate(1000, QUIET.demand, QUIET.weights, leak={SERVERS[0]: 0.35})
    config = ForecastConfig(exhaustion_severity=0.5)
    (fault,) = hypothesis_faults(Hypothesis("exhaustion", SERVERS[0]), SITE, estimate, TOPO, config)
    used = fault.severity * (estimate.tick - fault.onset) / EXHAUSTION_RAMP_TICKS
    assert used == pytest.approx(0.35, abs=0.002)
    (young,) = hypothesis_faults(Hypothesis("exhaustion", SERVERS[1]), SITE, QUIET, TOPO)
    assert young.onset == 0  # never before the run started


@pytest.mark.parametrize("kind", [FaultKind.FLOOD, FaultKind.BENIGN_SURGE])
def test_extra_traffic_brings_the_observed_excess(kind: FaultKind) -> None:
    normal = rollout(QUIET, (), TOPO, site=SITE, config=ForecastConfig(horizon=1))
    estimate = NetworkEstimate(0, QUIET.demand, QUIET.weights, reaching_qps={SITE: 9000.0})
    faults = hypothesis_faults(Hypothesis(kind), SITE, estimate, TOPO)
    assert faults and all(f.kind == kind and f.target in TOPO.regions for f in faults)
    # Re-route one tick with those faults: the site gets the observed traffic.
    effects = compute_effects(faults, 0, TOPO, set())
    extra = effects.region_extra_attack_qps or effects.region_extra_legit_qps
    demand = dict(QUIET.demand)
    reaching = route(TOPO, demand).site_qps[SITE] + route(TOPO, extra).site_qps[SITE]
    assert reaching == pytest.approx(9000.0)
    assert normal.offered > 0


def test_no_extra_traffic_without_an_excess_or_when_withdrawn() -> None:
    flood = Hypothesis("flood")
    assert hypothesis_faults(flood, SITE, QUIET, TOPO) == ()
    weights = {**QUIET.weights, SITE: 0.0}
    estimate = NetworkEstimate(0, QUIET.demand, weights, reaching_qps={SITE: 9000.0})
    assert hypothesis_faults(flood, SITE, estimate, TOPO) == ()
    huge = NetworkEstimate(0, QUIET.demand, QUIET.weights, reaching_qps={SITE: 1e9})
    assert all(f.severity == 1.0 for f in hypothesis_faults(flood, SITE, huge, TOPO))


def test_bad_hypotheses_are_rejected() -> None:
    with pytest.raises(ValueError, match="name a server"):
        hypothesis_faults(Hypothesis("crash"), SITE, QUIET, TOPO)
    with pytest.raises(KeyError):
        hypothesis_faults(Hypothesis("healthy"), "atlantis", QUIET, TOPO)
    with pytest.raises(ValueError, match="unknown hypothesis kind"):
        hypothesis_faults(Hypothesis("path-loss"), SITE, QUIET, TOPO)
    with pytest.raises(KeyError):
        rollout(QUIET, (), TOPO, site="atlantis")
    with pytest.raises(ValueError, match="after tick 0"):
        rollout(QUIET, (), TOPO, site=SITE, plan=((Action(K.WITHDRAW, SITE), 0),))


# ------------------------------------------------------------------ estimate and candidates


@cache
def _incident(family: str, seed: int) -> tuple[Scenario, DnsEnv, Observation]:
    """A tune run stepped to 30 ticks after its fault's onset."""
    scenario = next(s for s in split_scenarios(Split.TUNE, TOPO, (family,)) if s.seed == seed)
    env = DnsEnv(TOPO, scenario.faults, scenario.horizon, scenario.seed)
    obs = env.reset()
    while obs.tick < scenario.faults[0].onset + 30:
        obs = env.step(())
    return scenario, env, obs


def test_estimate_reads_the_observation() -> None:
    scenario, _, obs = _incident("exhaustion", 100)
    fault = scenario.faults[0]
    running = (RunningAction(Action(K.RESTART, "ott-1"), obs.tick + 5),)
    estimate = NetworkEstimate.from_observation(
        obs, TOPO, limits={SiteId("ott"): 10.0}, running=running
    )
    assert estimate.tick == obs.tick and estimate.running == running
    assert estimate.weights == {s: 1.0 for s in TOPO.sites}
    assert estimate.limits == {"ott": 10.0}
    assert dict(estimate.demand) == {r: region.base_qps for r, region in TOPO.regions.items()}
    used = fault.severity * (obs.tick - fault.onset) / EXHAUSTION_RAMP_TICKS
    assert estimate.leak[ServerId(fault.target)] == pytest.approx(used, abs=0.05)
    with pytest.raises(TypeError):
        estimate.weights[SITE] = 0.0  # type: ignore[index]


def test_estimate_skips_missing_readings() -> None:
    site = SiteTelemetry(None, None, None, None, None, None, announced=True, weight=1.0)
    obs = Observation(
        tick=3,
        sites={SITE: site},
        servers={SERVERS[0]: ServerTelemetry(None, 0.5, None)},
        action_updates=(),
    )
    estimate = NetworkEstimate.from_observation(obs, TOPO)
    assert estimate.leak == {}
    assert (estimate.queue_depth[SITE], estimate.loss[SITE], estimate.reaching_qps[SITE]) == (
        0,
        0,
        0,
    )
    earlier = NetworkEstimate(
        2,
        QUIET.demand,
        QUIET.weights,
        queue_depth={SITE: 7.0},
        reaching_qps={SITE: 900.0},
        loss={SITE: 0.4},
        leak={SERVERS[0]: 0.3},
    )
    kept = NetworkEstimate.from_observation(obs, TOPO, previous=earlier)
    assert (kept.queue_depth[SITE], kept.loss[SITE], kept.reaching_qps[SITE]) == (7, 0.4, 900)
    assert kept.leak == {SERVERS[0]: 0.3}


def test_candidate_actions() -> None:
    capacity = sum(TOPO.servers[s].capacity_qps for s in SERVERS)
    assert candidate_actions(SITE, QUIET, TOPO) == (
        None,
        *(Action(K.RESTART, s) for s in SERVERS),
        Action(K.WITHDRAW, SITE),
        Action(K.RATE_LIMIT, SITE, limit_qps=capacity),
        *(Action(K.REQUEST_REPAIR, s) for s in SERVERS),
        Action(K.REQUEST_REPAIR, SITE),
    )
    running = (RunningAction(Action(K.RESTART, SERVERS[0]), 10),)
    withdrawn = NetworkEstimate(0, QUIET.demand, {SITE: 0.0}, limits={SITE: 5.0}, running=running)
    actions = candidate_actions(SITE, withdrawn, TOPO)
    assert Action(K.RESTORE, SITE, weight=1.0) in actions
    assert Action(K.CLEAR_RATE_LIMIT, SITE) in actions
    assert not {Action(K.WITHDRAW, SITE), Action(K.RESTART, SERVERS[0])} & set(actions)
    assert Action(K.REQUEST_REPAIR, SITE) not in actions
    rerouting = (RunningAction(Action(K.RESTORE, SITE, weight=0.5), 10),)
    halfway = NetworkEstimate(0, QUIET.demand, {SITE: 0.25}, running=rerouting)
    routing = {K.WITHDRAW, K.RESTORE}
    assert not any(a and a.kind in routing for a in candidate_actions(SITE, halfway, TOPO))


def test_config_is_validated() -> None:
    for bad in (
        {"horizon": 0},
        {"restart_ticks": 0},
        {"transient_ticks": 0},
    ):
        with pytest.raises(ValueError, match="at least 1 tick"):
            ForecastConfig(**bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="restore_after"):
        ForecastConfig(restore_after=-1)
    assert ForecastConfig(transient_ticks=None, restore_after=0).restore_after == 0
    with pytest.raises(ValueError, match="exhaustion_severity"):
        ForecastConfig(exhaustion_severity=0.0)


# ------------------------------------------------------------------ the forecast table


def test_table_weights_rows_by_the_belief() -> None:
    hs = hypotheses(SITE, TOPO)
    probs = np.array([0.6 if h == Hypothesis("crash", SERVERS[0]) else 0.0 for h in hs])
    probs[hs.index(Hypothesis("healthy"))] = 0.395
    probs[hs.index(Hypothesis("flood"))] = 0.005  # below min_prob: skipped
    table = forecast_belief(
        QUIET, Belief(SITE, hs, probs), TOPO, config=ForecastConfig(horizon=300)
    )
    assert table.hypotheses == (Hypothesis("healthy"), Hypothesis("crash", SERVERS[0]))
    assert table.probs.sum() == pytest.approx(1.0)
    failures = table.failures()
    assert failures.shape == (len(table.actions), 2)
    assert table.expected_failures() == pytest.approx(failures @ table.probs)
    reduction = table.expected_reduction()
    assert reduction[0] == 0.0
    assert table.actions[int(np.argmax(reduction))] == Action(K.RESTART, SERVERS[0])
    assert table.expected_failures(local_only=True).shape == (len(table.actions),)


def test_most_likely_hypothesis_is_always_kept() -> None:
    hs = hypotheses(SITE, TOPO)
    uniform = Belief(SITE, hs, np.full(len(hs), 1 / len(hs)))
    table = forecast_belief(QUIET, uniform, TOPO, actions=[None], min_prob=1.0)
    assert table.hypotheses == (hs[0],)


def test_delays_change_the_forecast() -> None:
    restart = Action(K.RESTART, SERVERS[0])
    belief = _belief(Hypothesis("crash", SERVERS[0]))
    quick, slow = (
        forecast_belief(QUIET, belief, TOPO, actions=[None, restart], delays={restart: d})
        for d in (0, 300)
    )
    assert 0 < slow.expected_reduction()[1] < quick.expected_reduction()[1]


def test_table_argument_checks() -> None:
    belief = _belief(Hypothesis("healthy"))
    with pytest.raises(ValueError, match="min_prob"):
        forecast_belief(QUIET, belief, TOPO, min_prob=1.5)
    with pytest.raises(ValueError, match="non-negative"):
        forecast_belief(QUIET, belief, TOPO, delays={Action(K.WITHDRAW, SITE): -1})
    table = forecast_belief(QUIET, belief, TOPO, actions=[Action(K.WITHDRAW, SITE)])
    assert isinstance(table, ForecastTable)
    with pytest.raises(ValueError, match="wait"):
        table.expected_reduction()


@pytest.mark.parametrize(
    ("family", "seed", "expected"),
    [
        ("crash", 100, K.RESTART),
        ("hardware", 100, K.REQUEST_REPAIR),
        ("site_outage", 100, K.WITHDRAW),
        ("flood", 100, K.RATE_LIMIT),
        ("benign_surge", 100, None),
    ],
)
def test_forecast_picks_the_right_kind_of_repair_on_real_runs(
    family: str, seed: int, expected: ActionKind | None
) -> None:
    """Told the true cause, the forecast's best action is the textbook one, from real telemetry."""
    scenario, _, obs = _incident(family, seed)
    fault = scenario.faults[0]
    site = _incident_site(fault)
    server = ServerId(fault.target) if fault.target in TOPO.servers else None
    table = forecast_belief(
        NetworkEstimate.from_observation(obs, TOPO), _belief(Hypothesis(family, server), site), TOPO
    )
    best = table.actions[int(np.argmax(table.expected_reduction()))]
    assert (best.kind if best else None) == expected
    if server is not None and best is not None:
        assert best.target == server
