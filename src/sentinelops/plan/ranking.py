r"""Action ranking and incident prioritisation: deciding what to do, and for which incident first.

The forecaster (:mod:`sentinelops.plan.forecast`) says how many legitimate
queries would fail under each candidate action and each diagnosis. This module
turns that table into a decision. An action's **score** is the failures it is
expected to prevent compared with waiting, minus a penalty for the disruption
it causes::

    score(a) = sum_h P(h) [F_wait(h) - F_a(h)] - disruption_cost * units(a)

Waiting scores 0 by definition, so the agent only acts when an action is
expected to save more queries than it disrupts. The disruption units stand for
what the forecast does not count: a routing change every resolver sees, a
server taken out of service, an engineer called out, an operator interrupted
for an approval. The policy is applied first: a BLOCKed action is never ranked,
and one that needs approval is forecast with the approval delay and pays for
the interruption.

When several sites have incidents at once, they compete: an action at one site
moves traffic onto the others. :func:`plan_incidents` puts the incidents in
priority order (any incident close to its deadline first, then the one that
would lose the most queries if left alone) and plans them in that order. Each
incident is planned with the likeliest causes of the others assumed present
and their actions assumed under way: the action chosen for each incident
planned before it, and for the rest the action each would pick on its own.
That way two incidents do not both take a site out and overload the same
neighbours. Called again every planning step, it reprioritises as losses and
deadlines change.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from sentinelops.diagnose.inference import Belief
from sentinelops.eval.dataset import HEALTHY
from sentinelops.plan.forecast import (
    DEFAULT_MIN_PROB,
    ForecastConfig,
    ForecastTable,
    NetworkEstimate,
    candidate_actions,
    forecast_belief,
    hypothesis_faults,
)
from sentinelops.policy.rules import Decision, RuleSet
from sentinelops.sim.actions import Action, ActionKind, action_targets
from sentinelops.sim.calendar import DEFAULT_START, SimTime
from sentinelops.sim.faults import Fault
from sentinelops.sim.topology import SiteId, Topology

DISRUPTION_UNITS: Mapping[ActionKind, float] = MappingProxyType(
    {
        ActionKind.RESTART: 1.0,
        ActionKind.RATE_LIMIT: 1.0,
        ActionKind.CLEAR_RATE_LIMIT: 0.5,
        ActionKind.WITHDRAW: 2.0,
        ActionKind.RESTORE: 1.0,
        ActionKind.REQUEST_REPAIR: 3.0,
    }
)
"""Disruption of each action kind, per server it takes out of service (routing and
rate-limit changes count once). Routing changes are seen by every resolver of the
site; a repair calls out an engineer."""

APPROVAL_UNITS = 2.0
"""Extra disruption of an action that needs approval: an operator is interrupted."""

DEFAULT_DISRUPTION_COST = 2000.0
"""Failed queries one unit of disruption is worth (tuned on the tune split)."""

APPROVAL_DELAY_TICKS = 120
"""Expected wait for an operator's answer (the scripted approver's delay)."""

DEFAULT_DEADLINE_TICKS = 600
"""Ticks after detection by which an incident should be resolved or escalated
(the orchestrator sets :attr:`Incident.deadline` from it)."""

URGENT_SLACK_TICKS = 120
"""An incident whose deadline is this close is planned before any other."""

CONTEXT_MIN_PROB = 0.5
"""How likely another incident's best hypothesis must be to count as context."""

MAX_PLANNING_PASSES = 3
"""Rounds of joint planning, stopping early once no incident changes its choice."""


@dataclass(frozen=True, slots=True)
class RankingConfig:
    """How actions are scored.

    Attributes:
        disruption_cost: Failed queries one unit of disruption is worth.
        units: Disruption units of each action kind (missing kinds: 0).
        approval_units: Extra units for an action that needs approval.
        approval_delay: Ticks an action waits for approval.
        local_only: Count only the incident site's failures (the ablation).
    """

    disruption_cost: float = DEFAULT_DISRUPTION_COST
    units: Mapping[ActionKind, float] = field(default_factory=lambda: DISRUPTION_UNITS)
    approval_units: float = APPROVAL_UNITS
    approval_delay: int = APPROVAL_DELAY_TICKS
    local_only: bool = False

    def __post_init__(self) -> None:
        """Reject negative costs, units or delays.

        Raises:
            ValueError: If any of them is negative.
        """
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))
        values = [self.disruption_cost, self.approval_units, self.approval_delay]
        if min(values) < 0 or any(u < 0 for u in self.units.values()):
            raise ValueError("costs, units and delays must be non-negative")

    def __hash__(self) -> int:
        """Hash by value, so a config can key a cache."""
        units = tuple(sorted(self.units.items()))
        return hash(
            (self.disruption_cost, units, self.approval_units, self.approval_delay, self.local_only)
        )


def disruption(
    action: Action | None, topology: Topology, config: RankingConfig | None = None
) -> float:
    """Disruption units of ``action`` (0 for waiting), before any approval.

    Restarts and repairs count once per server they take out of service, so
    repairing a whole site costs as much as repairing each of its servers.
    """
    if action is None:
        return 0.0
    cfg = config or RankingConfig()
    per = cfg.units.get(action.kind, 0.0)
    if action.kind in (ActionKind.RESTART, ActionKind.REQUEST_REPAIR):
        return per * len(action_targets(action, topology))
    return per


@dataclass(frozen=True, slots=True)
class RankedAction:
    """One candidate action with its forecast and score.

    Attributes:
        action: The action; ``None`` is waiting.
        decision: What the policy says (waiting is always allowed).
        delay: Ticks it waits before starting (approval).
        expected_failures: Belief-weighted forecast failures, ``sum_h P(h) F_a(h)``.
        reduction: Failures prevented compared with waiting (negative: worse).
        penalty: Disruption cost in failed-query equivalents.
    """

    action: Action | None
    decision: Decision
    delay: int
    expected_failures: float
    reduction: float
    penalty: float

    @property
    def score(self) -> float:
        """``reduction - penalty``: positive when acting beats waiting."""
        return self.reduction - self.penalty


@dataclass(frozen=True, slots=True)
class ActionRanking:
    """The candidate actions for one incident, best first.

    Attributes:
        site: The incident site.
        ranked: Every action the policy does not block, best first; waiting is
            always among them.
        blocked: Actions the policy blocks (not forecast).
        table: The forecasts behind the scores.
    """

    site: SiteId
    ranked: tuple[RankedAction, ...]
    blocked: tuple[Action, ...]
    table: ForecastTable

    @property
    def best(self) -> RankedAction:
        """The action to take (waiting when nothing beats it)."""
        return self.ranked[0]

    @property
    def wait(self) -> RankedAction:
        """The 'wait' entry, the reference every score is measured against."""
        return next(r for r in self.ranked if r.action is None)

    @property
    def projected_loss(self) -> float:
        """Failures expected over the horizon if the agent does nothing."""
        return self.wait.expected_failures


def _decide(action: Action, policy: RuleSet | None, when: SimTime) -> Decision:
    if policy is None:
        return Decision.ALLOW
    return policy.evaluate(action, when.weekday, when.hour)


def rank_actions(
    estimate: NetworkEstimate,
    belief: Belief,
    topology: Topology,
    *,
    policy: RuleSet | None = None,
    when: SimTime = DEFAULT_START,
    approved: Collection[Action] = (),
    config: RankingConfig | None = None,
    forecast: ForecastConfig | None = None,
    min_prob: float = DEFAULT_MIN_PROB,
    context_faults: Sequence[Fault] = (),
    context_plan: Sequence[tuple[Action, int]] = (),
) -> ActionRanking:
    """Score every candidate action for the incident at ``belief.site``.

    Ties are broken towards less disruption, then candidate order, so waiting
    wins any tie with an action. A withdrawn site the policy does not let the
    agent restore is held out in every forecast.

    Args:
        estimate: The current state of the network.
        belief: The diagnosis of the incident site.
        topology: The network.
        policy: The rules in force; ``None`` allows everything.
        when: The moment of the week the policy is evaluated at.
        approved: Actions an operator has already approved (no delay, no
            approval penalty).
        config: Scoring settings.
        forecast: Horizon and expected durations.
        min_prob: Hypotheses less likely than this are not forecast.
        context_faults: Faults assumed present elsewhere (other incidents).
        context_plan: Actions already chosen for other incidents.
    """
    cfg = config or RankingConfig()
    held = {
        SiteId(s)
        for s, w in estimate.weights.items()
        if w < 1.0
        and _decide(Action(ActionKind.RESTORE, s, weight=1.0), policy, when) == Decision.BLOCK
    }
    if held - estimate.held:
        estimate = dataclasses.replace(estimate, held=estimate.held | held)
    allowed: list[Action | None] = [None]
    blocked: list[Action] = []
    decisions: dict[Action, Decision] = {}
    for action in candidate_actions(belief.site, estimate, topology)[1:]:
        assert action is not None  # only the first candidate is "wait"
        decision = _decide(action, policy, when)
        if decision == Decision.BLOCK:
            blocked.append(action)
        else:
            allowed.append(action)
            decisions[action] = decision
    needs_approval = {
        a for a, d in decisions.items() if d == Decision.REQUIRE_APPROVAL and a not in approved
    }
    table = forecast_belief(
        estimate,
        belief,
        topology,
        actions=allowed,
        delays=dict.fromkeys(needs_approval, cfg.approval_delay),
        min_prob=min_prob,
        config=forecast,
        context_faults=context_faults,
        context_plan=context_plan,
    )
    expected = table.expected_failures(local_only=cfg.local_only)
    reduction = table.expected_reduction(local_only=cfg.local_only)
    entries = []
    for i, action in enumerate(allowed):
        units = disruption(action, topology, cfg)
        delay = 0
        if action in needs_approval:
            units += cfg.approval_units
            delay = cfg.approval_delay
        entries.append(
            RankedAction(
                action=action,
                decision=Decision.ALLOW if action is None else decisions[action],
                delay=delay,
                expected_failures=float(expected[i]),
                reduction=float(reduction[i]),
                penalty=cfg.disruption_cost * units,
            )
        )
    order = sorted(range(len(entries)), key=lambda i: (-entries[i].score, entries[i].penalty, i))
    return ActionRanking(
        site=belief.site,
        ranked=tuple(entries[i] for i in order),
        blocked=tuple(blocked),
        table=table,
    )


@dataclass(frozen=True, slots=True)
class Incident:
    """An open incident the agent is planning for.

    Attributes:
        belief: The current diagnosis (its site is the incident site).
        deadline: Tick by which it should be resolved or escalated.
    """

    belief: Belief
    deadline: int

    @property
    def site(self) -> SiteId:
        """The incident site."""
        return self.belief.site


@dataclass(frozen=True, slots=True)
class IncidentPlan:
    """One incident's place in the plan and the action chosen for it.

    Attributes:
        incident: The incident.
        priority: 0 for the incident planned first.
        urgent: Whether its deadline is close enough to put it first.
        projected_loss: Failures expected if it is left alone (planned on its own).
        ranking: Its actions, scored with the other incidents taken into account.
    """

    incident: Incident
    priority: int
    urgent: bool
    projected_loss: float
    ranking: ActionRanking

    @property
    def choice(self) -> RankedAction:
        """The action chosen for this incident."""
        return self.ranking.best


def _likeliest_faults(
    incident: Incident, estimate: NetworkEstimate, topology: Topology, forecast: ForecastConfig
) -> tuple[Fault, ...]:
    hypothesis, p = incident.belief.best()
    if hypothesis.kind == HEALTHY or p < CONTEXT_MIN_PROB:
        return ()  # too unsure to assume any one cause for the others
    return hypothesis_faults(hypothesis, incident.site, estimate, topology, forecast)


def plan_incidents(
    estimate: NetworkEstimate,
    incidents: Sequence[Incident],
    topology: Topology,
    *,
    policy: RuleSet | None = None,
    when: SimTime = DEFAULT_START,
    approved: Collection[Action] = (),
    config: RankingConfig | None = None,
    forecast: ForecastConfig | None = None,
    urgent_slack: int = URGENT_SLACK_TICKS,
    coordinate: bool = True,
) -> tuple[IncidentPlan, ...]:
    """Prioritise open incidents and choose an action for each, highest priority first.

    Priority: incidents whose deadline is within ``urgent_slack`` ticks come
    first (earliest deadline first); the rest follow by projected loss, the
    failures each would cause on its own if left alone (largest first), then by
    deadline and site.

    With ``coordinate``, each incident is then planned in that order assuming
    the likeliest cause of every other incident is present and every other
    incident's action is under way: the one chosen for it if it was planned
    already, otherwise the one it would pick on its own. Another incident's
    cause counts only if its diagnosis is at least ``CONTEXT_MIN_PROB`` sure.
    The round is repeated, up to ``MAX_PLANNING_PASSES`` times, until no
    incident changes its choice. Without it (the ablation), every incident is
    planned as if it were the only one.

    Args:
        estimate: The current state of the network.
        incidents: The open incidents, at most one per site.
        topology: The network.
        policy: The rules in force; ``None`` allows everything.
        when: The moment of the week the policy is evaluated at.
        approved: Actions an operator has already approved.
        config: Scoring settings.
        forecast: Horizon and expected durations.
        urgent_slack: Ticks before its deadline at which an incident goes first.
        coordinate: Plan incidents jointly (see above).

    Returns:
        One plan per incident, in priority order.

    Raises:
        ValueError: If two incidents share a site or ``urgent_slack`` is negative.
    """
    sites = [i.site for i in incidents]
    if len(set(sites)) != len(sites):
        raise ValueError("at most one incident per site")
    if urgent_slack < 0:
        raise ValueError(f"urgent_slack must be non-negative, got {urgent_slack}")
    cfg = config or RankingConfig()
    fcfg = forecast or ForecastConfig()

    def rank(
        incident: Incident,
        faults: Sequence[Fault] = (),
        plan: Sequence[tuple[Action, int]] = (),
    ) -> ActionRanking:
        return rank_actions(
            estimate,
            incident.belief,
            topology,
            policy=policy,
            when=when,
            approved=approved,
            config=cfg,
            forecast=fcfg,
            context_faults=faults,
            context_plan=plan,
        )

    alone = [rank(incident) for incident in incidents]
    urgent = [incident.deadline - estimate.tick <= urgent_slack for incident in incidents]
    order = sorted(
        range(len(incidents)),
        key=lambda i: (
            not urgent[i],
            incidents[i].deadline if urgent[i] else 0,
            -alone[i].projected_loss,
            incidents[i].deadline,
            incidents[i].site,
        ),
    )
    causes = [_likeliest_faults(incident, estimate, topology, fcfg) for incident in incidents]
    # what each incident is assumed to do: its own best action until it is planned jointly
    choices = [r.best for r in alone]
    rankings = list(alone)
    passes = MAX_PLANNING_PASSES if coordinate and len(incidents) > 1 else 0
    for _ in range(passes):
        changed = False
        for i in order:
            others = [j for j in range(len(incidents)) if j != i]
            faults = tuple(f for j in others for f in causes[j])
            plan = tuple(
                (c.action, estimate.tick + 1 + c.delay)
                for c in (choices[j] for j in others)
                if c.action is not None
            )
            rankings[i] = rank(incidents[i], faults, plan)
            changed |= rankings[i].best.action != choices[i].action
            choices[i] = rankings[i].best
        if not changed:
            break
    return tuple(
        IncidentPlan(incidents[i], priority, urgent[i], alone[i].projected_loss, rankings[i])
        for priority, i in enumerate(order)
    )
