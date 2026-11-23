"""Operator policy rules: may the agent take this action on its own?

For every action the agent proposes, a :class:`RuleSet` answers **allow**,
**block** or **require approval**, and can say which rule decided.

Rules form an ordered list and the **first rule that matches decides**, like a
firewall. That keeps a policy predictable: an operator reads it top to bottom
and :meth:`RuleSet.explain` shows exactly which line fired. A rule matches on
action kind, target and time window; an empty filter means "any". When no rule
matches, the default applies, which is ``REQUIRE_APPROVAL``: an action nobody
wrote a rule for goes to a human instead of happening silently.

Time is passed in as plain ``weekday`` and ``hour`` integers so that rules do
not depend on the simulation clock.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from sentinelops.sim.actions import Action, ActionKind
from sentinelops.sim.topology import ServerId, Topology


class Decision(StrEnum):
    """What the policy says about one proposed action."""

    ALLOW = "allow"
    """The agent may act on its own."""

    BLOCK = "block"
    """The action must not happen; the agent has to pick something else."""

    REQUIRE_APPROVAL = "require_approval"
    """A human operator must approve the action first."""


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A weekly time window, such as weekdays 9:00-17:00 or nights 22:00-06:00.

    If ``start_hour > end_hour`` the window wraps past midnight (22 -> 6 means
    22:00 until 06:00 the next morning). For a wrapping window the day check
    uses the day the hour falls on: with ``days={4}`` (Friday) and 22 -> 6,
    Friday 23:00 and Friday 03:00 are inside, Saturday 03:00 is not. This
    keeps the rule simple and easy to read.

    Attributes:
        days: Days the window applies on, 0 = Monday ... 6 = Sunday; empty means
            every day.
        start_hour: First hour inside the window, 0-23.
        end_hour: First hour after the window, 1-24.
    """

    days: frozenset[int] = frozenset()
    start_hour: int = 0
    end_hour: int = 24

    def __post_init__(self) -> None:
        """Store ``days`` as a frozenset and reject impossible days or hours.

        Raises:
            ValueError: If a day is outside 0-6, ``start_hour`` is outside
                0-23, ``end_hour`` is outside 1-24, or the two hours are equal.
        """
        object.__setattr__(self, "days", frozenset(self.days))
        if any(not 0 <= day <= 6 for day in self.days):
            raise ValueError(f"days must be in 0-6 (0 = Monday), got {sorted(self.days)}")
        if not 0 <= self.start_hour <= 23:
            raise ValueError(f"start_hour must be in 0-23, got {self.start_hour}")
        if not 1 <= self.end_hour <= 24:
            raise ValueError(f"end_hour must be in 1-24, got {self.end_hour}")
        if self.start_hour == self.end_hour:
            raise ValueError(f"start_hour and end_hour are both {self.start_hour}")

    def contains(self, weekday: int, hour: int) -> bool:
        """Return whether ``hour`` on ``weekday`` falls inside the window.

        Args:
            weekday: 0 = Monday ... 6 = Sunday.
            hour: Hour of the day, 0-23.
        """
        if self.days and weekday not in self.days:
            return False
        if self.start_hour < self.end_hour:
            return self.start_hour <= hour < self.end_hour
        return hour >= self.start_hour or hour < self.end_hour  # wraps past midnight


@dataclass(frozen=True, slots=True)
class Rule:
    """One line of an operator policy.

    Attributes:
        decision: What to answer when this rule matches.
        kinds: Action kinds the rule covers; empty means any kind.
        targets: Server or site ids the rule covers; empty means any target.
            A site id also covers that site's servers when the rule set has a
            topology.
        window: When the rule applies; ``None`` means at any time.
        name: Short human description shown in the console and handoff
            reports, e.g. ``"block withdrawals at night"``.
    """

    decision: Decision
    kinds: frozenset[ActionKind] = frozenset()
    targets: frozenset[str] = frozenset()
    window: TimeWindow | None = None
    name: str = ""

    def __post_init__(self) -> None:
        """Convert ``decision`` and ``kinds`` to their enums.

        Raises:
            ValueError: If the decision or a kind is an unknown string.
        """
        object.__setattr__(self, "decision", Decision(self.decision))
        object.__setattr__(self, "kinds", frozenset(ActionKind(kind) for kind in self.kinds))
        object.__setattr__(self, "targets", frozenset(self.targets))

    def matches(
        self, action: Action, weekday: int, hour: int, topology: Topology | None = None
    ) -> bool:
        """Return whether this rule applies to ``action`` at the given time.

        Args:
            action: The proposed action.
            weekday: 0 = Monday ... 6 = Sunday.
            hour: Hour of the day, 0-23.
            topology: Used to find a server's site; without it targets must
                match exactly.
        """
        if self.kinds and action.kind not in self.kinds:
            return False
        if self.targets and not self._covers(action.target, topology):
            return False
        return self.window is None or self.window.contains(weekday, hour)

    def _covers(self, target: str, topology: Topology | None) -> bool:
        if target in self.targets:
            return True
        server = None if topology is None else topology.servers.get(ServerId(target))
        return server is not None and server.site in self.targets


class RuleSet:
    """An ordered operator policy: the first matching rule decides.

    Args:
        rules: The rules, highest priority first.
        default: Decision when no rule matches.
        topology: Lets a site target also cover the site's servers; without
            it targets match exactly only.
    """

    def __init__(
        self,
        rules: Sequence[Rule],
        default: Decision = Decision.REQUIRE_APPROVAL,
        topology: Topology | None = None,
    ) -> None:
        """Store the rules read-only and convert ``default`` to a :class:`Decision`."""
        self._rules = tuple(rules)
        self._default = Decision(default)
        self._topology = topology

    @property
    def rules(self) -> tuple[Rule, ...]:
        """The rules in priority order (read-only)."""
        return self._rules

    @property
    def default(self) -> Decision:
        """The decision used when no rule matches."""
        return self._default

    def evaluate(self, action: Action, weekday: int, hour: int) -> Decision:
        """Decide whether ``action`` may happen at ``weekday``/``hour``.

        Raises:
            ValueError: If ``weekday`` is outside 0-6 or ``hour`` outside 0-23.
        """
        return self.explain(action, weekday, hour)[0]

    def explain(self, action: Action, weekday: int, hour: int) -> tuple[Decision, Rule | None]:
        """Decide like :meth:`evaluate` and also return the deciding rule.

        Returns:
            The decision and the first matching rule, or ``(default, None)``
            when no rule matched.

        Raises:
            ValueError: If ``weekday`` is outside 0-6 or ``hour`` outside 0-23.
        """
        if not 0 <= weekday <= 6:
            raise ValueError(f"weekday must be in 0-6 (0 = Monday), got {weekday}")
        if not 0 <= hour <= 23:
            raise ValueError(f"hour must be in 0-23, got {hour}")
        for rule in self._rules:
            if rule.matches(action, weekday, hour, self._topology):
                return rule.decision, rule
        return self._default, None
