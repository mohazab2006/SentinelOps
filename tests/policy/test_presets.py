"""Tests for autonomy presets and maintenance holds."""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from sentinelops.policy import (
    BUSINESS_HOURS,
    Decision,
    MaintenanceHold,
    Preset,
    TimeWindow,
    build_policy,
    preset_default,
    preset_rules,
)
from sentinelops.sim.actions import Action, ActionKind
from sentinelops.sim.topology import default_topology

MON, SAT = 0, 5
A, B, R = Decision.ALLOW, Decision.BLOCK, Decision.REQUIRE_APPROVAL
K = ActionKind

# Expected decision per preset and kind, at (Monday 10:00, Saturday 03:00), straight from the table.
EXPECTED: dict[Preset, dict[ActionKind, tuple[Decision, Decision]]] = {
    Preset.OBSERVE_ONLY: {kind: (B, B) for kind in K},
    Preset.CONSERVATIVE: {kind: (R, R) for kind in K},
    Preset.BALANCED: {
        K.RESTART: (A, A),
        K.RATE_LIMIT: (A, A),
        K.CLEAR_RATE_LIMIT: (A, A),
        K.RESTORE: (A, A),
        K.WITHDRAW: (R, R),
        K.REQUEST_REPAIR: (R, R),
    },
    Preset.LENIENT: {
        K.RESTART: (A, A),
        K.RATE_LIMIT: (A, A),
        K.CLEAR_RATE_LIMIT: (A, A),
        K.RESTORE: (A, A),
        K.WITHDRAW: (R, A),  # needs approval only in business hours
        K.REQUEST_REPAIR: (R, R),
    },
    Preset.FULL_AUTONOMY: {kind: (A, A) for kind in K},
}

PERMISSIVENESS = {B: 0, R: 1, A: 2}


def action_for(kind: ActionKind, target: str = "lon") -> Action:
    """A valid action of ``kind`` (server kinds get the site's first server)."""
    if kind == K.RESTART:
        return Action(kind, f"{target}-1")
    if kind == K.RATE_LIMIT:
        return Action(kind, target, limit_qps=100.0)
    if kind == K.RESTORE:
        return Action(kind, target, weight=0.25)
    return Action(kind, target)


# ------------------------------------------------------------------ the preset table


@pytest.mark.parametrize("preset", list(Preset))
@pytest.mark.parametrize("kind", list(K))
@pytest.mark.parametrize(("when", "weekday", "hour"), [(0, MON, 10), (1, SAT, 3)])
def test_decision_table(
    preset: Preset, kind: ActionKind, when: int, weekday: int, hour: int
) -> None:
    policy = build_policy(preset, topology=default_topology())
    assert policy.evaluate(action_for(kind), weekday, hour) == EXPECTED[preset][kind][when]


def test_more_autonomy_is_never_stricter() -> None:
    policies = [build_policy(preset) for preset in Preset]  # listed least to most autonomous
    for kind, weekday, hour in itertools.product(K, range(7), range(24)):
        action = action_for(kind)
        levels = [PERMISSIVENESS[p.evaluate(action, weekday, hour)] for p in policies]
        assert levels == sorted(levels), (kind, weekday, hour, levels)


def test_preset_defaults_and_rule_names() -> None:
    assert [preset_default(p) for p in Preset] == [B, R, R, A, A]
    assert [rule.name for rule in preset_rules(Preset.BALANCED)] == ["allow low-risk actions"]
    assert [rule.name for rule in preset_rules(Preset.LENIENT)] == [
        "repairs call out engineers",
        "withdrawals in business hours need approval",
    ]
    assert preset_rules(Preset.LENIENT)[1].window == BUSINESS_HOURS


def test_explain_names_the_preset_rule_that_decided() -> None:
    policy = build_policy(Preset.LENIENT)
    decision, rule = policy.explain(Action(K.WITHDRAW, "yyz"), MON, 10)
    assert decision == R
    assert rule is not None
    assert rule.name == "withdrawals in business hours need approval"


# ------------------------------------------------------------------ maintenance holds


def test_hold_blocks_every_kind_even_under_full_autonomy() -> None:
    hold = MaintenanceHold("lon", window=TimeWindow(start_hour=22, end_hour=6))
    policy = build_policy(Preset.FULL_AUTONOMY, holds=[hold])
    for kind in (K.RATE_LIMIT, K.CLEAR_RATE_LIMIT, K.WITHDRAW, K.RESTORE, K.REQUEST_REPAIR):
        assert policy.evaluate(action_for(kind), MON, 23) == B
        assert policy.evaluate(action_for(kind), MON, 12) == A  # outside the window
    assert policy.evaluate(action_for(K.WITHDRAW, "yyz"), MON, 23) == A  # other sites unaffected


def test_hold_on_a_site_covers_its_servers_with_a_topology() -> None:
    policy = build_policy(
        Preset.FULL_AUTONOMY, holds=[MaintenanceHold("lon")], topology=default_topology()
    )
    assert policy.evaluate(Action(K.RESTART, "lon-1"), SAT, 3) == B
    assert policy.evaluate(Action(K.RESTART, "yyz-1"), SAT, 3) == A


def test_holds_come_before_the_preset_rules() -> None:
    holds = [MaintenanceHold("lon"), MaintenanceHold("ott")]
    policy = build_policy(Preset.BALANCED, holds=holds)
    assert policy.rules == (holds[0].rule(), holds[1].rule(), *preset_rules(Preset.BALANCED))


def test_explain_returns_the_hold_with_its_reason() -> None:
    policy = build_policy(Preset.LENIENT, holds=[MaintenanceHold("lon", reason="disk swap")])
    decision, rule = policy.explain(Action(K.REQUEST_REPAIR, "lon"), MON, 10)
    assert decision == B
    assert rule is not None
    assert rule.name == "maintenance hold on lon: disk swap"


def test_hold_rule_without_a_reason() -> None:
    rule = MaintenanceHold("lon-2").rule()
    assert rule.name == "maintenance hold on lon-2"
    assert (rule.decision, rule.kinds, rule.targets, rule.window) == (
        B,
        frozenset(),
        frozenset({"lon-2"}),
        None,
    )


def test_hold_is_frozen() -> None:
    hold = MaintenanceHold("lon")
    with pytest.raises(dataclasses.FrozenInstanceError):
        hold.target = "ott"  # type: ignore[misc]


def test_hold_on_an_unknown_target_with_a_topology_is_rejected() -> None:
    with pytest.raises(ValueError, match="tor"):
        build_policy(Preset.BALANCED, holds=[MaintenanceHold("tor")], topology=default_topology())


# ------------------------------------------------------------------ plain strings


def test_preset_names_work_as_plain_strings() -> None:
    policy = build_policy("lenient")
    assert policy.default == A
    assert policy.rules == preset_rules(Preset.LENIENT)
    assert preset_default("observe_only") == B
    assert preset_rules("balanced") == preset_rules(Preset.BALANCED)


@pytest.mark.parametrize("call", [build_policy, preset_rules, preset_default])
def test_unknown_preset_name_is_rejected(call: object) -> None:
    with pytest.raises(ValueError):
        call("yolo")  # type: ignore[operator]
