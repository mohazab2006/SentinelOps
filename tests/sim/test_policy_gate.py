"""Tests for the environment's policy gate and approval delay."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from sentinelops.policy import (
    ApprovalStatus,
    Decision,
    MaintenanceHold,
    Preset,
    Rule,
    RuleSet,
    ScriptedApprover,
    TimeWindow,
    build_policy,
)
from sentinelops.sim.actions import Action, ActionKind, ActionStatus
from sentinelops.sim.calendar import Calendar, SimTime
from sentinelops.sim.env import DnsEnv, EnvConfig
from sentinelops.sim.events import Event
from sentinelops.sim.faults import Fault, FaultKind
from sentinelops.sim.telemetry import Observation
from sentinelops.sim.topology import default_topology

TOPO = default_topology()
CRASH = (Fault(FaultKind.CRASH, "yyz-1", 50, None, 1.0),)
RESTART = Action(ActionKind.RESTART, "yyz-1")
WITHDRAW = Action(ActionKind.WITHDRAW, "lon")
Plan = Callable[[Observation], Sequence[Action]]


def _env(
    policy: RuleSet | None = None,
    approver: ScriptedApprover | None = None,
    *,
    faults: Sequence[Fault] = CRASH,
    horizon: int = 600,
    start: SimTime | None = None,
) -> DnsEnv:
    config = EnvConfig(calendar=Calendar(start)) if start is not None else None
    return DnsEnv(TOPO, faults, horizon, seed=0, config=config, policy=policy, approver=approver)


def _run(env: DnsEnv, plan: Plan) -> list[Observation]:
    observations = [env.reset()]
    while not env.done:
        observations.append(env.step(plan(observations[-1])))
    return observations


def _at(tick: int, *actions: Action) -> Plan:
    """Issue ``actions`` on tick ``tick`` (right after seeing observation ``tick - 1``)."""
    return lambda obs: actions if obs.tick == tick - 1 else ()


def _history(observations: Sequence[Observation]) -> list[tuple[int, int, ActionStatus]]:
    return [(u.tick, u.action_id, u.status) for o in observations for u in o.action_updates]


def _action_events(env: DnsEnv) -> list[Event]:
    return list(env.events.named("action"))


def _always(decision: Decision, name: str = "") -> RuleSet:
    return RuleSet([Rule(decision, name=name)], default=Decision.BLOCK)


def test_an_approver_without_a_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs a policy"):
        _env(approver=ScriptedApprover())


def test_allowing_everything_is_identical_to_having_no_policy() -> None:
    plan = _at(100, RESTART)
    ungated = _env()
    _run(ungated, plan)
    allowed = _env(build_policy(Preset.FULL_AUTONOMY, topology=TOPO))
    _run(allowed, plan)
    assert allowed.events.to_jsonl() == ungated.events.to_jsonl()


def test_a_blocked_action_never_runs() -> None:
    env = _env(build_policy(Preset.OBSERVE_ONLY, topology=TOPO))
    observations = _run(env, _at(100, RESTART))
    assert _history(observations) == [(100, 0, ActionStatus.BLOCKED)]
    assert not env.truth.cleared
    (event,) = _action_events(env)
    assert event.data["rule"] == "default"
    assert "request_id" not in event.data


def test_an_approved_action_starts_after_the_approval_delay() -> None:
    env = _env(build_policy(Preset.CONSERVATIVE, topology=TOPO), ScriptedApprover(delay_ticks=120))
    observations = _run(env, _at(100, RESTART))
    history = _history(observations)
    assert history[:3] == [
        (100, 0, ActionStatus.PENDING_APPROVAL),
        (220, 0, ActionStatus.APPROVED),
        (220, 0, ActionStatus.RUNNING),
    ]
    assert history[3][2] == ActionStatus.SUCCEEDED
    assert env.truth.cleared == {0}
    assert env.approvals.status(0) == ApprovalStatus.USED
    pending = _action_events(env)[0]
    assert (pending.data["rule"], pending.data["request_id"]) == ("default", 0)
    assert env.approvals.pending() == ()


def test_a_zero_delay_approval_starts_on_the_tick_it_was_asked() -> None:
    env = _env(build_policy(Preset.CONSERVATIVE, topology=TOPO), ScriptedApprover(delay_ticks=0))
    observations = _run(env, _at(100, RESTART))
    assert _history(observations)[:3] == [
        (100, 0, ActionStatus.PENDING_APPROVAL),
        (100, 0, ActionStatus.APPROVED),
        (100, 0, ActionStatus.RUNNING),
    ]


def test_waiting_for_approval_costs_failed_queries() -> None:
    shares = []
    for delay in (0, 120, 300):
        env = _env(build_policy(Preset.CONSERVATIVE, topology=TOPO), ScriptedApprover(delay))
        _run(env, _at(100, RESTART))
        shares.append(env.truth.legit_failure_share())
    assert shares[0] < shares[1] < shares[2]


def test_a_rejected_action_never_runs() -> None:
    approver = ScriptedApprover(delay_ticks=60, reject=frozenset({ActionKind.RESTART}))
    env = _env(build_policy(Preset.CONSERVATIVE, topology=TOPO), approver)
    observations = _run(env, _at(100, RESTART))
    assert _history(observations) == [
        (100, 0, ActionStatus.PENDING_APPROVAL),
        (160, 0, ActionStatus.REJECTED),
    ]
    assert not env.truth.cleared
    assert env.approvals.status(0) == ApprovalStatus.REJECTED


def test_without_an_approver_a_request_waits_for_a_human() -> None:
    env = _env(build_policy(Preset.CONSERVATIVE, topology=TOPO))
    env.reset()
    for _ in range(99):
        env.step(())
    first = env.step([RESTART])  # tick 100
    assert [u.status for u in first.action_updates] == [ActionStatus.PENDING_APPROVAL]
    for _ in range(50):
        assert env.step(()).action_updates == ()
    (request,) = env.approvals.pending()
    assert (request.action, request.requested_tick, request.reason) == (RESTART, 100, "default")
    env.approvals.approve(request.request_id, env.tick)  # the operator clicks "approve"
    later = env.step(())
    assert [(u.action_id, u.status) for u in later.action_updates] == [
        (0, ActionStatus.APPROVED),
        (0, ActionStatus.RUNNING),
    ]


def test_a_human_rejection_is_reported_on_the_next_tick() -> None:
    env = _env(build_policy(Preset.CONSERVATIVE, topology=TOPO))
    env.reset()
    env.step([RESTART])
    env.approvals.reject(0, env.tick)
    assert [u.status for u in env.step(()).action_updates] == [ActionStatus.REJECTED]


def test_asking_again_while_waiting_reports_the_same_request() -> None:
    env = _env(build_policy(Preset.CONSERVATIVE, topology=TOPO), ScriptedApprover(delay_ticks=120))
    observations = _run(env, lambda obs: [RESTART] if obs.tick in (99, 109, 119) else [])
    history = _history(observations)
    assert history[:5] == [
        (100, 0, ActionStatus.PENDING_APPROVAL),
        (110, 0, ActionStatus.PENDING_APPROVAL),
        (120, 0, ActionStatus.PENDING_APPROVAL),
        (220, 0, ActionStatus.APPROVED),
        (220, 0, ActionStatus.RUNNING),
    ]
    assert {action_id for _, action_id, _ in history} == {0}
    assert _known(env, 0)
    assert not _known(env, 1)  # no second request was filed


def _known(env: DnsEnv, request_id: int) -> bool:
    try:
        env.approvals.status(request_id)
    except KeyError:
        return False
    return True


def test_after_an_approval_is_used_a_new_request_is_needed() -> None:
    env = _env(build_policy(Preset.CONSERVATIVE, topology=TOPO), ScriptedApprover(delay_ticks=10))
    observations = _run(env, lambda obs: [RESTART] if obs.tick in (99, 299) else [])
    pending = [(t, a) for t, a, s in _history(observations) if s == ActionStatus.PENDING_APPROVAL]
    assert pending == [(100, 0), (300, 1)]
    assert env.approvals.status(0) == env.approvals.status(1) == ApprovalStatus.USED


def test_an_action_blocked_while_it_waited_does_not_run() -> None:
    hold = MaintenanceHold("yyz", window=TimeWindow(start_hour=11, end_hour=12), reason="swap")
    policy = build_policy(Preset.CONSERVATIVE, holds=[hold], topology=TOPO)
    env = _env(policy, ScriptedApprover(delay_ticks=120), start=SimTime(0, 10, 58))
    observations = _run(env, _at(10, RESTART))  # 10:58:10, approved at 11:00:10
    assert _history(observations) == [
        (10, 0, ActionStatus.PENDING_APPROVAL),
        (130, 0, ActionStatus.APPROVED),
        (130, 0, ActionStatus.BLOCKED),
    ]
    assert _action_events(env)[-1].data["rule"] == "maintenance hold on yyz: swap"
    assert env.approvals.status(0) == ApprovalStatus.USED


def test_the_policy_is_read_at_the_simulated_time() -> None:
    lenient = build_policy(Preset.LENIENT, topology=TOPO)
    day = _env(lenient, ScriptedApprover(), faults=(), start=SimTime(0, 10))
    night = _env(lenient, ScriptedApprover(), faults=(), start=SimTime(0, 22))
    by_day = _history(_run(day, _at(100, WITHDRAW)))
    at_night = _history(_run(night, _at(100, WITHDRAW)))
    assert by_day[0] == (100, 0, ActionStatus.PENDING_APPROVAL)
    assert _action_events(day)[0].data["rule"] == "withdrawals in business hours need approval"
    assert at_night[0] == (100, 0, ActionStatus.RUNNING)


def test_an_unnamed_rule_is_reported_by_its_position() -> None:
    policy = RuleSet(
        [Rule(Decision.ALLOW, kinds=frozenset({ActionKind.WITHDRAW})), Rule(Decision.BLOCK)]
    )
    env = _env(policy)
    _run(env, _at(100, RESTART))
    assert _action_events(env)[0].data["rule"] == "rule 1"


def test_a_named_rule_is_reported_by_name() -> None:
    env = _env(_always(Decision.BLOCK, name="change freeze"))
    _run(env, _at(100, RESTART))
    assert _action_events(env)[0].data["rule"] == "change freeze"


def test_an_approved_action_goes_before_one_asked_for_on_the_same_tick() -> None:
    # The restart was asked for first, so on tick 100 it takes the server and the
    # repair asked for then finds it busy.
    policy = RuleSet(
        [Rule(Decision.ALLOW, kinds=frozenset({ActionKind.REQUEST_REPAIR}))],
        default=Decision.REQUIRE_APPROVAL,
        topology=TOPO,
    )
    repair = Action(ActionKind.REQUEST_REPAIR, "yyz-1")
    env = _env(policy, ScriptedApprover(delay_ticks=10))

    def plan(obs: Observation) -> Sequence[Action]:
        return {89: [RESTART], 99: [repair]}.get(obs.tick, [])

    history = _history(_run(env, plan))
    assert history[:4] == [
        (90, 0, ActionStatus.PENDING_APPROVAL),
        (100, 0, ActionStatus.APPROVED),
        (100, 0, ActionStatus.RUNNING),
        (100, 1, ActionStatus.FAILED),
    ]


def test_an_answer_names_the_rule_and_request_it_answers() -> None:
    approver = ScriptedApprover(delay_ticks=60, reject=frozenset({ActionKind.WITHDRAW}))
    env = _env(build_policy(Preset.CONSERVATIVE, topology=TOPO), approver)
    _run(env, _at(100, RESTART, WITHDRAW))
    answers = {
        e.data["status"]: (e.data["rule"], e.data["request_id"])
        for e in _action_events(env)
        if e.data["status"] in ("approved", "rejected")
    }
    assert answers == {"approved": ("default", 0), "rejected": ("default", 1)}


def test_gated_runs_are_deterministic() -> None:
    def gated() -> str:
        env = _env(build_policy(Preset.BALANCED, topology=TOPO), ScriptedApprover(delay_ticks=30))
        _run(env, lambda obs: [WITHDRAW, RESTART] if obs.tick == 99 else [])
        return env.events.to_jsonl()

    assert gated() == gated()


def test_reset_forgets_earlier_requests() -> None:
    env = _env(build_policy(Preset.CONSERVATIVE, topology=TOPO))
    env.reset()
    env.step([RESTART])
    assert len(env.approvals.pending()) == 1
    env.reset()
    assert env.approvals.pending() == ()
    assert env.step(()).action_updates == ()
