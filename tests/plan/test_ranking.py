from __future__ import annotations

import dataclasses
from functools import cache

import numpy as np
import pytest

from sentinelops.diagnose.inference import Belief, Hypothesis, hypotheses
from sentinelops.eval.dataset import HEALTHY
from sentinelops.eval.splits import Split, split_scenarios
from sentinelops.plan.forecast import ForecastConfig, NetworkEstimate
from sentinelops.plan.ranking import (
    APPROVAL_DELAY_TICKS,
    APPROVAL_UNITS,
    DEFAULT_DISRUPTION_COST,
    ActionRanking,
    Incident,
    RankingConfig,
    disruption,
    plan_incidents,
    rank_actions,
)
from sentinelops.policy.rules import Decision, Rule, RuleSet
from sentinelops.scenarios.families import Scenario, generate
from sentinelops.sim.actions import Action, ActionKind
from sentinelops.sim.env import DnsEnv
from sentinelops.sim.faults import Fault, FaultKind
from sentinelops.sim.topology import RegionId, ServerId, SiteId, default_topology

TOPO = default_topology()
QUIET = NetworkEstimate.healthy(TOPO)
K = ActionKind
SHORT = ForecastConfig(horizon=300)


def _belief(site: str, kind: str, server: str | None = None) -> Belief:
    hs = hypotheses(site, TOPO)
    truth = Hypothesis(kind, ServerId(server) if server else None)
    return Belief(SiteId(site), hs, np.array([1.0 if h == truth else 0.0 for h in hs]))


def _servers(site: str) -> list[ServerId]:
    return sorted(TOPO.sites[SiteId(site)].servers)


LON = _servers("lon")
CRASH_LON = _belief("lon", FaultKind.CRASH, LON[0])
RESTART_LON = Action(K.RESTART, LON[0])


# ------------------------------------------------------------------ disruption


def test_disruption_units() -> None:
    assert disruption(None, TOPO) == 0.0
    assert disruption(RESTART_LON, TOPO) == 1.0
    assert disruption(Action(K.WITHDRAW, "lon"), TOPO) == 2.0
    assert disruption(Action(K.RATE_LIMIT, "lon", limit_qps=10.0), TOPO) == 1.0
    assert disruption(Action(K.REQUEST_REPAIR, LON[0]), TOPO) == 3.0
    # a site repair takes both servers out: it costs as much as two server repairs
    assert disruption(Action(K.REQUEST_REPAIR, "lon"), TOPO) == 6.0
    free = RankingConfig(units={K.RESTART: 0.5})
    assert disruption(RESTART_LON, TOPO, free) == 0.5
    assert disruption(Action(K.WITHDRAW, "lon"), TOPO, free) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"disruption_cost": -1.0},
        {"approval_units": -0.1},
        {"approval_delay": -1},
        {"units": {K.RESTART: -1.0}},
    ],
)
def test_config_rejects_negative_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RankingConfig(**kwargs)  # type: ignore[arg-type]


def test_configs_hash_by_value() -> None:
    assert hash(RankingConfig()) == hash(RankingConfig(units=dict(RankingConfig().units)))
    assert {RankingConfig(), RankingConfig()} == {RankingConfig()}


def test_config_units_are_read_only() -> None:
    units = {K.RESTART: 1.0}
    config = RankingConfig(units=units)
    units[K.RESTART] = 99.0
    assert config.units[K.RESTART] == 1.0
    with pytest.raises(TypeError):
        config.units[K.RESTART] = 2.0  # type: ignore[index]


# ------------------------------------------------------------------ ranking one incident


def test_a_crash_ranks_the_restart_first() -> None:
    ranking = rank_actions(QUIET, CRASH_LON, TOPO, forecast=SHORT)
    best = ranking.best
    assert best.action == RESTART_LON and best.decision == Decision.ALLOW and best.delay == 0
    assert best.penalty == DEFAULT_DISRUPTION_COST * 1.0
    assert best.score == pytest.approx(best.reduction - best.penalty)
    assert best.score > 0
    scores = [r.score for r in ranking.ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranking.wait.action is None and ranking.wait.reduction == 0.0
    assert ranking.wait.score == 0.0
    assert ranking.projected_loss == ranking.wait.expected_failures > best.expected_failures
    assert ranking.blocked == () and ranking.site == "lon"


def test_scores_match_the_forecast_table() -> None:
    ranking = rank_actions(QUIET, CRASH_LON, TOPO, forecast=SHORT)
    table = ranking.table
    expected = table.expected_failures()
    for r in ranking.ranked:
        i = table.actions.index(r.action)
        assert r.expected_failures == pytest.approx(expected[i])
        assert r.reduction == pytest.approx(expected[0] - expected[i])


def test_nothing_wrong_means_waiting() -> None:
    """With no fault, no action prevents anything, so the disruption decides: wait."""
    ranking = rank_actions(QUIET, _belief("lon", HEALTHY), TOPO, forecast=SHORT)
    assert ranking.best.action is None
    assert all(r.score <= 0 for r in ranking.ranked)


def test_waiting_wins_a_tie() -> None:
    """Free actions that change nothing tie with waiting; waiting is listed first."""
    free = RankingConfig(disruption_cost=0.0)
    ranking = rank_actions(QUIET, _belief("lon", HEALTHY), TOPO, config=free, forecast=SHORT)
    assert ranking.best.action is None
    tied = [r.action for r in ranking.ranked if r.score == 0.0]
    assert tied[0] is None and len(tied) > 1


def test_a_high_disruption_cost_keeps_the_agent_waiting() -> None:
    costly = RankingConfig(disruption_cost=1e9)
    assert rank_actions(QUIET, CRASH_LON, TOPO, config=costly, forecast=SHORT).best.action is None


def test_local_only_counts_only_the_incident_site() -> None:
    local = RankingConfig(local_only=True)
    ranking = rank_actions(QUIET, CRASH_LON, TOPO, config=local, forecast=SHORT)
    expected = ranking.table.expected_failures(local_only=True)
    for r in ranking.ranked:
        assert r.expected_failures == pytest.approx(expected[ranking.table.actions.index(r.action)])


# ------------------------------------------------------------------ the policy


def _policy(decision: Decision, kind: ActionKind) -> RuleSet:
    rule = Rule(decision, kinds=frozenset({kind}), name=f"{decision} {kind}")
    return RuleSet([rule], default=Decision.ALLOW, topology=TOPO)


def test_blocked_actions_are_not_ranked() -> None:
    ranking = rank_actions(
        QUIET, CRASH_LON, TOPO, policy=_policy(Decision.BLOCK, K.RESTART), forecast=SHORT
    )
    assert set(ranking.blocked) == {Action(K.RESTART, s) for s in LON}
    assert all(r.action is None or r.action.kind != K.RESTART for r in ranking.ranked)
    assert ranking.best.action != RESTART_LON
    assert len(ranking.table.actions) == len(ranking.ranked)


def test_approval_costs_time_and_an_interruption() -> None:
    allowed = rank_actions(QUIET, CRASH_LON, TOPO, forecast=SHORT)
    gated = rank_actions(
        QUIET,
        CRASH_LON,
        TOPO,
        policy=_policy(Decision.REQUIRE_APPROVAL, K.RESTART),
        forecast=SHORT,
    )
    free = next(r for r in allowed.ranked if r.action == RESTART_LON)
    asked = next(r for r in gated.ranked if r.action == RESTART_LON)
    assert asked.decision == Decision.REQUIRE_APPROVAL and asked.delay == APPROVAL_DELAY_TICKS
    assert asked.reduction < free.reduction  # the crash lasts two more minutes
    assert asked.penalty == free.penalty + DEFAULT_DISRUPTION_COST * APPROVAL_UNITS


def test_an_approved_action_runs_at_once() -> None:
    policy = _policy(Decision.REQUIRE_APPROVAL, K.RESTART)
    allowed = rank_actions(QUIET, CRASH_LON, TOPO, forecast=SHORT)
    ok = rank_actions(QUIET, CRASH_LON, TOPO, policy=policy, approved={RESTART_LON}, forecast=SHORT)
    free = next(r for r in allowed.ranked if r.action == RESTART_LON)
    granted = next(r for r in ok.ranked if r.action == RESTART_LON)
    assert granted.decision == Decision.REQUIRE_APPROVAL and granted.delay == 0
    assert (granted.reduction, granted.penalty) == (free.reduction, free.penalty)


def test_the_policy_is_read_at_the_given_moment() -> None:
    from sentinelops.policy.rules import TimeWindow
    from sentinelops.sim.calendar import SimTime

    night = Rule(
        Decision.BLOCK, kinds=frozenset({K.RESTART}), window=TimeWindow(start_hour=22, end_hour=6)
    )
    policy = RuleSet([night], default=Decision.ALLOW)
    day = rank_actions(QUIET, CRASH_LON, TOPO, policy=policy, forecast=SHORT)
    late = rank_actions(
        QUIET, CRASH_LON, TOPO, policy=policy, when=SimTime(weekday=2, hour=23), forecast=SHORT
    )
    assert day.blocked == () and RESTART_LON in late.blocked


def test_a_site_the_policy_keeps_out_is_not_assumed_back() -> None:
    """fra is out on a maintenance hold: forecasts must not bring it back."""
    weights = {**QUIET.weights, SiteId("fra"): 0.0}
    estimate = NetworkEstimate(0, QUIET.demand, weights)
    hold = Rule(Decision.BLOCK, targets=frozenset({"fra"}), name="maintenance")
    policy = RuleSet([hold], default=Decision.ALLOW, topology=TOPO)
    free = rank_actions(estimate, CRASH_LON, TOPO, forecast=SHORT)
    held = rank_actions(estimate, CRASH_LON, TOPO, policy=policy, forecast=SHORT)
    # with fra kept out its traffic stays on the others, so waiting costs more
    assert held.projected_loss > free.projected_loss
    assert held.best.action == RESTART_LON


# ------------------------------------------------------------------ several incidents


def _summary(ranking: ActionRanking) -> list[tuple[Action | None, float]]:
    return [(r.action, r.score) for r in ranking.ranked]


def _incident(site: str, kind: str, server: str | None = None, deadline: int = 600) -> Incident:
    return Incident(_belief(site, kind, server), deadline)


def test_incidents_are_ordered_by_projected_loss() -> None:
    """A site outage loses more than a crash of one server: it goes first."""
    crash = _incident("lon", FaultKind.CRASH, LON[0])
    outage = _incident("fra", FaultKind.SITE_OUTAGE)
    plans = plan_incidents(QUIET, [crash, outage], TOPO, forecast=SHORT)
    assert [p.incident for p in plans] == [outage, crash]
    assert [p.priority for p in plans] == [0, 1]
    assert plans[0].projected_loss > plans[1].projected_loss
    assert not any(p.urgent for p in plans)


def test_an_incident_near_its_deadline_goes_first() -> None:
    crash = _incident("lon", FaultKind.CRASH, LON[0], deadline=100)
    outage = _incident("fra", FaultKind.SITE_OUTAGE, deadline=600)
    plans = plan_incidents(QUIET, [outage, crash], TOPO, forecast=SHORT)
    assert [p.incident for p in plans] == [crash, outage]
    assert [p.urgent for p in plans] == [True, False]
    # two urgent incidents: earliest deadline first, whatever their losses
    sooner = _incident("fra", FaultKind.SITE_OUTAGE, deadline=50)
    plans = plan_incidents(QUIET, [crash, sooner], TOPO, forecast=SHORT)
    assert [p.incident for p in plans] == [sooner, crash]


def test_equal_losses_are_ordered_by_deadline_then_site() -> None:
    a = _incident("lon", HEALTHY, deadline=500)
    b = _incident("fra", HEALTHY, deadline=500)
    c = _incident("yyz", HEALTHY, deadline=400)
    plans = plan_incidents(QUIET, [a, b, c], TOPO, forecast=SHORT)
    assert [p.incident.site for p in plans] == ["yyz", "fra", "lon"]


def test_reprioritising_follows_the_clock() -> None:
    crash = _incident("lon", FaultKind.CRASH, LON[0], deadline=400)
    outage = _incident("fra", FaultKind.SITE_OUTAGE, deadline=900)
    early = plan_incidents(QUIET, [crash, outage], TOPO, forecast=SHORT)
    later = plan_incidents(
        NetworkEstimate.healthy(TOPO, tick=300), [crash, outage], TOPO, forecast=SHORT
    )
    assert early[0].incident == outage and later[0].incident == crash


def test_independent_planning_ranks_each_incident_alone() -> None:
    crash = _incident("lon", FaultKind.CRASH, LON[0])
    outage = _incident("fra", FaultKind.SITE_OUTAGE)
    plans = plan_incidents(QUIET, [crash, outage], TOPO, forecast=SHORT, coordinate=False)
    for plan in plans:
        alone = rank_actions(QUIET, plan.incident.belief, TOPO, forecast=SHORT)
        assert _summary(plan.ranking) == _summary(alone)
        assert plan.choice == alone.best


def test_coordinated_planning_sees_the_other_incidents() -> None:
    """The second incident is forecast with the first one's fault and chosen action."""
    crash = _incident("lon", FaultKind.CRASH, LON[0])
    outage = _incident("fra", FaultKind.SITE_OUTAGE)
    first, second = plan_incidents(QUIET, [crash, outage], TOPO, forecast=SHORT)
    assert first.incident == outage and second.incident == crash
    alone = rank_actions(QUIET, crash.belief, TOPO, forecast=SHORT)
    # fra's outage pushes its traffic elsewhere, so the network fails more with it
    assert second.ranking.projected_loss > alone.projected_loss
    assert second.projected_loss == pytest.approx(alone.projected_loss)
    assert second.choice.action == RESTART_LON


def test_a_lone_incident_is_planned_once() -> None:
    crash = _incident("lon", FaultKind.CRASH, LON[0])
    (plan,) = plan_incidents(QUIET, [crash], TOPO, forecast=SHORT)
    assert _summary(plan.ranking) == _summary(
        rank_actions(QUIET, crash.belief, TOPO, forecast=SHORT)
    )
    assert plan_incidents(QUIET, [], TOPO) == ()


def test_plan_incidents_rejects_bad_input() -> None:
    crash = _incident("lon", FaultKind.CRASH, LON[0])
    with pytest.raises(ValueError, match="one incident per site"):
        plan_incidents(QUIET, [crash, _incident("lon", HEALTHY)], TOPO)
    with pytest.raises(ValueError, match="urgent_slack"):
        plan_incidents(QUIET, [crash], TOPO, urgent_slack=-1)


# ------------------------------------------------------------------ real runs


@cache
def _real(family: str, seed: int) -> tuple[Scenario, NetworkEstimate]:
    """A tune run stepped to 30 ticks after its fault's onset, and the agent's estimate."""
    scenario = next(s for s in split_scenarios(Split.TUNE, TOPO, (family,)) if s.seed == seed)
    env = DnsEnv(TOPO, scenario.faults, scenario.horizon, scenario.seed)
    obs = env.reset()
    estimate = None
    while obs.tick < scenario.faults[0].onset + 30:
        obs = env.step(())
        estimate = NetworkEstimate.from_observation(obs, TOPO, previous=estimate)
    assert estimate is not None
    return scenario, estimate


def _true_belief(fault: Fault) -> Belief:
    if fault.target in TOPO.servers:
        site = TOPO.site_of(ServerId(fault.target))
        return _belief(site, fault.kind, fault.target)
    if fault.target in TOPO.sites:
        return _belief(fault.target, fault.kind)
    return _belief(TOPO.preferences[RegionId(fault.target)][0], fault.kind)


@pytest.mark.parametrize(
    ("family", "kind"),
    [
        ("crash", K.RESTART),
        ("exhaustion", None),  # a young leak costs nothing yet; a restart costs a server
        ("hardware", K.REQUEST_REPAIR),
        ("flood", K.RATE_LIMIT),
        ("benign_surge", None),
    ],
)
def test_real_runs_pick_the_fitting_action(family: str, kind: ActionKind | None) -> None:
    scenario, estimate = _real(family, 100)
    best = rank_actions(estimate, _true_belief(scenario.faults[0]), TOPO).best
    assert (best.action.kind if best.action else None) == kind


def test_a_leak_is_restarted_only_where_it_costs_queries() -> None:
    """Most of its capacity leaked away, yyz-1 can't carry yyz's load; lon-1 can."""
    for site, expected in (("yyz", K.RESTART), ("lon", None)):
        server = _servers(site)[0]
        estimate = NetworkEstimate(0, QUIET.demand, QUIET.weights, leak={server: 0.85})
        best = rank_actions(estimate, _belief(site, FaultKind.EXHAUSTION, server), TOPO).best
        assert (best.action.kind if best.action else None) == expected
        if best.action:
            assert best.action.target == server


def _overlap() -> tuple[tuple[Fault, ...], int, NetworkEstimate, int]:
    """A hardware fault at sin and a crash at fra five ticks later (tune seeds 109, 159)."""
    hardware = generate("hardware", 109, TOPO).faults[0]
    crash = generate("crash", 159, TOPO).faults[0]
    onset = max(hardware.onset, crash.onset)
    faults = (
        dataclasses.replace(hardware, onset=onset),
        dataclasses.replace(crash, onset=onset + 5),
    )
    env = DnsEnv(TOPO, faults, onset + 1000, 109)
    obs = env.reset()
    estimate = None
    while obs.tick < onset + 30:
        obs = env.step(())
        estimate = NetworkEstimate.from_observation(obs, TOPO, previous=estimate)
    assert estimate is not None
    return faults, 109, estimate, obs.tick


def _real_failures(faults: tuple[Fault, ...], seed: int, plan: list[Action], tick: int) -> float:
    env = DnsEnv(TOPO, faults, tick + 901, seed)
    env.reset()
    while not env.done:
        env.step(plan if env.tick == tick else ())
    return float(env.truth.legit_failed[tick + 1 :].sum())


def test_coordination_avoids_two_withdrawals_on_a_real_overlap() -> None:
    """Alone, each incident would withdraw or repair its site; together, two sites out
    would overload the rest, so the coordinated plan restarts sin's server instead."""
    faults, seed, estimate, tick = _overlap()
    incidents = [Incident(_true_belief(f), tick + 600) for f in faults]
    joint = plan_incidents(estimate, incidents, TOPO)
    alone = plan_incidents(estimate, incidents, TOPO, coordinate=False)
    joint_actions = [p.choice.action for p in joint if p.choice.action]
    alone_actions = [p.choice.action for p in alone if p.choice.action]
    assert Action(K.WITHDRAW, "sin") in alone_actions
    assert Action(K.WITHDRAW, "sin") not in joint_actions
    assert _real_failures(faults, seed, joint_actions, tick) < 0.2 * _real_failures(
        faults, seed, alone_actions, tick
    )


def test_an_unsure_diagnosis_adds_no_context() -> None:
    from sentinelops.plan.ranking import _likeliest_faults

    hs = hypotheses("fra", TOPO)
    split = {FaultKind.SITE_OUTAGE: 0.4, FaultKind.PATH_LOSS: 0.3, FaultKind.FLOOD: 0.3}
    probs = np.array([split.get(FaultKind(h.kind), 0.0) if h.kind != HEALTHY else 0.0 for h in hs])
    unsure = Incident(Belief(SiteId("fra"), hs, probs), 600)
    sure = _incident("fra", FaultKind.SITE_OUTAGE)
    assert _likeliest_faults(unsure, QUIET, TOPO, SHORT) == ()
    assert _likeliest_faults(sure, QUIET, TOPO, SHORT) != ()
