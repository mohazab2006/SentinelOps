"""Autonomy presets and maintenance holds: ready-made operator policies.

A preset is a :class:`~sentinelops.policy.rules.RuleSet` built from a fixed
list of named rules and a default decision. From least to most autonomous:

* **observe only** - no rules, default BLOCK: the agent never acts, it only
  reports what it would do.
* **conservative** - no rules, default REQUIRE_APPROVAL: the agent may act, but
  a human says yes every time.
* **balanced** - low-risk actions (restart, rate limit, clear rate limit,
  restore) are allowed; everything else needs approval.
* **lenient** - everything is allowed except repairs, which call out engineers,
  and withdrawals during business hours, when moving a site's traffic hurts
  most; those two need approval.
* **full autonomy** - no rules, default ALLOW.

Why those lines: a restart touches one server and is quick to undo; a
withdrawal moves a whole site's traffic onto other sites; a repair calls out
people and takes its target down for about ten minutes.

A **maintenance hold** ("engineers are working on London: don't touch it")
becomes one BLOCK rule on its target. Holds are placed *before* the preset's
rules, and the first matching rule decides, so a hold beats every preset, even
full autonomy: a human working on a site always outranks the agent. A
"custom" policy needs no code here: the operator builds their own ``RuleSet``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from sentinelops.policy.rules import Decision, Rule, RuleSet, TimeWindow
from sentinelops.sim.actions import ActionKind
from sentinelops.sim.topology import Topology


class Preset(StrEnum):
    """The ready-made policies, from least to most autonomous."""

    OBSERVE_ONLY = "observe_only"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    LENIENT = "lenient"
    FULL_AUTONOMY = "full_autonomy"


BUSINESS_HOURS = TimeWindow(days=frozenset(range(5)), start_hour=9, end_hour=17)
"""Monday to Friday, 9:00 to 17:00: when traffic is highest."""


@dataclass(frozen=True, slots=True)
class MaintenanceHold:
    """Engineers are working on a site or server: the agent must not touch it.

    Attributes:
        target: Site or server id. A site id also covers its servers when the
            policy has a topology.
        window: When the hold applies; ``None`` means until it is removed.
        reason: Short note shown to the operator, e.g. ``"disk swap"``.
    """

    target: str
    window: TimeWindow | None = None
    reason: str = ""

    def rule(self) -> Rule:
        """Return the BLOCK rule for this hold (any action kind on the target)."""
        name = f"maintenance hold on {self.target}"
        if self.reason:
            name += f": {self.reason}"
        return Rule(Decision.BLOCK, targets=frozenset({self.target}), window=self.window, name=name)


_PRESET_RULES: dict[Preset, tuple[Rule, ...]] = {
    Preset.OBSERVE_ONLY: (),
    Preset.CONSERVATIVE: (),
    Preset.BALANCED: (
        Rule(
            Decision.ALLOW,
            kinds=frozenset(
                {
                    ActionKind.RESTART,
                    ActionKind.RATE_LIMIT,
                    ActionKind.CLEAR_RATE_LIMIT,
                    ActionKind.RESTORE,
                }
            ),
            name="allow low-risk actions",
        ),
    ),
    Preset.LENIENT: (
        Rule(
            Decision.REQUIRE_APPROVAL,
            kinds=frozenset({ActionKind.REQUEST_REPAIR}),
            name="repairs call out engineers",
        ),
        Rule(
            Decision.REQUIRE_APPROVAL,
            kinds=frozenset({ActionKind.WITHDRAW}),
            window=BUSINESS_HOURS,
            name="withdrawals in business hours need approval",
        ),
    ),
    Preset.FULL_AUTONOMY: (),
}

_PRESET_DEFAULTS: dict[Preset, Decision] = {
    Preset.OBSERVE_ONLY: Decision.BLOCK,
    Preset.CONSERVATIVE: Decision.REQUIRE_APPROVAL,
    Preset.BALANCED: Decision.REQUIRE_APPROVAL,
    Preset.LENIENT: Decision.ALLOW,
    Preset.FULL_AUTONOMY: Decision.ALLOW,
}


def preset_rules(preset: Preset | str) -> tuple[Rule, ...]:
    """Return the preset's rules in priority order.

    Args:
        preset: A :class:`Preset` or its name, e.g. ``"balanced"``.

    Raises:
        ValueError: If ``preset`` is not a known preset name.
    """
    return _PRESET_RULES[Preset(preset)]


def preset_default(preset: Preset | str) -> Decision:
    """Return the decision the preset uses when none of its rules match.

    Raises:
        ValueError: If ``preset`` is not a known preset name.
    """
    return _PRESET_DEFAULTS[Preset(preset)]


def build_policy(
    preset: Preset | str,
    holds: Sequence[MaintenanceHold] = (),
    topology: Topology | None = None,
) -> RuleSet:
    """Build the full policy: maintenance holds first, then the preset's rules.

    Args:
        preset: A :class:`Preset` or its name.
        holds: Maintenance holds; they win over every preset rule.
        topology: Passed to the :class:`RuleSet`, so a hold on a site covers
            its servers and a misspelled target is rejected.

    Raises:
        ValueError: If ``preset`` is unknown, or a hold's target is not in
            ``topology``.
    """
    rules = [hold.rule() for hold in holds] + list(preset_rules(preset))
    return RuleSet(rules, default=preset_default(preset), topology=topology)
