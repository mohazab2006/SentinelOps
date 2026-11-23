"""Tests for the operator policy rule engine."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from sentinelops.policy import Decision, Rule, RuleSet, TimeWindow
from sentinelops.sim.actions import Action, ActionKind
from sentinelops.sim.topology import default_topology

MON, FRI, SAT = 0, 4, 5
RESTART_LON1 = Action(ActionKind.RESTART, "lon-1")
WITHDRAW_YYZ = Action(ActionKind.WITHDRAW, "yyz")
BUSINESS_HOURS = TimeWindow(days=frozenset(range(5)), start_hour=9, end_hour=17)


# ------------------------------------------------------------------ defaults


def test_no_rules_falls_back_to_require_approval() -> None:
    assert RuleSet([]).evaluate(RESTART_LON1, MON, 10) == Decision.REQUIRE_APPROVAL


def test_custom_default_is_respected() -> None:
    policy = RuleSet([], default=Decision.ALLOW)
    assert policy.default == Decision.ALLOW
    assert policy.evaluate(RESTART_LON1, MON, 10) == Decision.ALLOW


def test_default_accepts_a_plain_string() -> None:
    assert RuleSet([], default="block").evaluate(RESTART_LON1, MON, 10) == Decision.BLOCK  # type: ignore[arg-type]


# ------------------------------------------------------------------ ordering


def test_first_matching_rule_wins() -> None:
    allow = Rule(Decision.ALLOW, kinds=frozenset({ActionKind.RESTART}))
    block = Rule(Decision.BLOCK, kinds=frozenset({ActionKind.RESTART}))
    assert RuleSet([allow, block]).evaluate(RESTART_LON1, MON, 10) == Decision.ALLOW
    assert RuleSet([block, allow]).evaluate(RESTART_LON1, MON, 10) == Decision.BLOCK


def test_rules_are_stored_as_a_read_only_tuple() -> None:
    rules = [Rule(Decision.ALLOW)]
    policy = RuleSet(rules)
    rules.append(Rule(Decision.BLOCK))  # changing the caller's list later has no effect
    assert policy.rules == (Rule(Decision.ALLOW),)
    assert isinstance(policy.rules, tuple)


# ------------------------------------------------------------------ kind filter


def test_kind_filter_only_affects_its_own_kinds() -> None:
    policy = RuleSet([Rule(Decision.BLOCK, kinds=frozenset({ActionKind.WITHDRAW}))])
    assert policy.evaluate(WITHDRAW_YYZ, MON, 10) == Decision.BLOCK
    assert policy.evaluate(RESTART_LON1, MON, 10) == Decision.REQUIRE_APPROVAL


def test_empty_rule_matches_every_action() -> None:
    policy = RuleSet([Rule(Decision.ALLOW)])
    for action in (RESTART_LON1, WITHDRAW_YYZ, Action(ActionKind.RATE_LIMIT, "ott", limit_qps=5)):
        assert policy.evaluate(action, SAT, 3) == Decision.ALLOW


# ------------------------------------------------------------------ targets


def test_site_target_covers_its_servers_with_a_topology() -> None:
    rule = Rule(Decision.BLOCK, targets=frozenset({"lon"}))
    policy = RuleSet([rule], topology=default_topology())
    assert policy.evaluate(RESTART_LON1, MON, 10) == Decision.BLOCK
    assert policy.evaluate(Action(ActionKind.WITHDRAW, "lon"), MON, 10) == Decision.BLOCK
    assert policy.evaluate(Action(ActionKind.RESTART, "yyz-1"), MON, 10) == (
        Decision.REQUIRE_APPROVAL
    )


def test_without_a_topology_targets_match_exactly() -> None:
    policy = RuleSet([Rule(Decision.BLOCK, targets=frozenset({"lon"}))])
    assert policy.evaluate(RESTART_LON1, MON, 10) == Decision.REQUIRE_APPROVAL
    assert policy.evaluate(Action(ActionKind.WITHDRAW, "lon"), MON, 10) == Decision.BLOCK


def test_server_target_matches_only_that_server() -> None:
    policy = RuleSet(
        [Rule(Decision.ALLOW, targets=frozenset({"lon-1"}))], topology=default_topology()
    )
    assert policy.evaluate(RESTART_LON1, MON, 10) == Decision.ALLOW
    assert policy.evaluate(Action(ActionKind.RESTART, "lon-2"), MON, 10) == (
        Decision.REQUIRE_APPROVAL
    )
    assert policy.evaluate(Action(ActionKind.WITHDRAW, "lon"), MON, 10) == (
        Decision.REQUIRE_APPROVAL
    )


# ------------------------------------------------------------------ time windows


def test_weekday_business_hours() -> None:
    assert BUSINESS_HOURS.contains(MON, 9)
    assert BUSINESS_HOURS.contains(FRI, 16)
    assert not BUSINESS_HOURS.contains(MON, 17)  # end hour is exclusive
    assert not BUSINESS_HOURS.contains(SAT, 10)


@pytest.mark.parametrize(
    ("hour", "inside"), [(22, True), (23, True), (0, True), (5, True), (6, False), (21, False)]
)
def test_window_wrapping_past_midnight(hour: int, inside: bool) -> None:
    assert TimeWindow(start_hour=22, end_hour=6).contains(MON, hour) is inside


def test_wrapping_window_checks_the_day_the_hour_falls_on() -> None:
    friday_nights = TimeWindow(days=frozenset({FRI}), start_hour=22, end_hour=6)
    assert friday_nights.contains(FRI, 23)
    assert friday_nights.contains(FRI, 3)
    assert not friday_nights.contains(SAT, 3)


def test_default_window_is_all_day_every_day() -> None:
    always = TimeWindow()
    assert all(always.contains(day, hour) for day in range(7) for hour in range(24))


def test_rule_window_limits_when_it_applies() -> None:
    policy = RuleSet(
        [Rule(Decision.BLOCK, kinds=frozenset({ActionKind.WITHDRAW}), window=BUSINESS_HOURS)]
    )
    assert policy.evaluate(WITHDRAW_YYZ, MON, 10) == Decision.BLOCK
    assert policy.evaluate(WITHDRAW_YYZ, SAT, 10) == Decision.REQUIRE_APPROVAL


def test_example_policy_from_the_task() -> None:
    policy = RuleSet(
        [
            Rule(Decision.ALLOW, kinds=frozenset({ActionKind.RESTART}), name="restarts any time"),
            Rule(
                Decision.BLOCK,
                kinds=frozenset({ActionKind.WITHDRAW}),
                targets=frozenset({"yyz"}),
                window=BUSINESS_HOURS,
                name="keep Toronto up in business hours",
            ),
        ],
        topology=default_topology(),
    )
    assert policy.evaluate(Action(ActionKind.RESTART, "yyz-1"), MON, 10) == Decision.ALLOW
    assert policy.evaluate(WITHDRAW_YYZ, MON, 10) == Decision.BLOCK
    assert policy.evaluate(WITHDRAW_YYZ, MON, 20) == Decision.REQUIRE_APPROVAL
    assert policy.evaluate(Action(ActionKind.WITHDRAW, "lon"), MON, 10) == (
        Decision.REQUIRE_APPROVAL
    )


# ------------------------------------------------------------------ explain


def test_explain_returns_the_deciding_rule_with_its_name() -> None:
    allow_restarts = Rule(
        Decision.ALLOW, kinds=frozenset({ActionKind.RESTART}), name="allow restarts any time"
    )
    policy = RuleSet([Rule(Decision.BLOCK, kinds=frozenset({ActionKind.WITHDRAW})), allow_restarts])
    decision, rule = policy.explain(RESTART_LON1, MON, 10)
    assert decision == Decision.ALLOW
    assert rule is allow_restarts
    assert rule.name == "allow restarts any time"


def test_explain_returns_none_when_the_default_applied() -> None:
    policy = RuleSet([Rule(Decision.BLOCK, kinds=frozenset({ActionKind.WITHDRAW}))])
    assert policy.explain(RESTART_LON1, MON, 10) == (Decision.REQUIRE_APPROVAL, None)


# ------------------------------------------------------------------ conversion + validation


def test_rule_converts_strings_to_enums() -> None:
    rule = Rule("allow", kinds=frozenset({"restart"}))  # type: ignore[arg-type]
    assert rule.decision is Decision.ALLOW
    assert rule.kinds == frozenset({ActionKind.RESTART})
    assert all(isinstance(kind, ActionKind) for kind in rule.kinds)


def test_plain_sets_are_stored_as_frozensets() -> None:
    rule = Rule(Decision.ALLOW, kinds={ActionKind.RESTART}, targets={"lon"})  # type: ignore[arg-type]
    window = TimeWindow(days={0, 1})  # type: ignore[arg-type]
    assert isinstance(rule.kinds, frozenset)
    assert isinstance(rule.targets, frozenset)
    assert isinstance(window.days, frozenset)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"days": frozenset({7})}, "days"),
        ({"days": frozenset({-1})}, "days"),
        ({"start_hour": 24}, "start_hour"),
        ({"start_hour": -1}, "start_hour"),
        ({"end_hour": 0}, "end_hour"),
        ({"end_hour": 25}, "end_hour"),
        ({"start_hour": 9, "end_hour": 9}, "both"),
    ],
)
def test_bad_time_windows_are_rejected(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TimeWindow(**kwargs)


def test_unknown_decision_string_is_rejected() -> None:
    with pytest.raises(ValueError):
        Rule("maybe")  # type: ignore[arg-type]


def test_unknown_kind_string_is_rejected() -> None:
    with pytest.raises(ValueError):
        Rule(Decision.ALLOW, kinds=frozenset({"reboot"}))  # type: ignore[arg-type]


@pytest.mark.parametrize(("weekday", "hour"), [(MON, 24), (-1, 10), (7, 10), (MON, -1)])
def test_evaluate_rejects_impossible_times(weekday: int, hour: int) -> None:
    policy = RuleSet([Rule(Decision.ALLOW)])
    with pytest.raises(ValueError):
        policy.evaluate(RESTART_LON1, weekday, hour)
    with pytest.raises(ValueError):
        policy.explain(RESTART_LON1, weekday, hour)


def test_rule_and_window_are_frozen() -> None:
    rule = Rule(Decision.ALLOW)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rule.decision = Decision.BLOCK  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        BUSINESS_HOURS.start_hour = 0  # type: ignore[misc]
