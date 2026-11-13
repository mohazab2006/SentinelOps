"""Operator actions an agent can take on the network, and their lifecycle.

Every agent (SentinelOps or a baseline) chooses from the same small set of
actions. An action is plain data; the simulation engine starts it, lets it run
for a realistic time and reports each status change back to the agent through
:class:`ActionUpdate` records in its next observation.

Durations are part of the problem: a restart takes tens of seconds, a manual
repair about ten minutes, and routing changes need a few seconds to propagate.
An agent that picks the wrong action pays for it in time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sentinelops.sim.topology import ServerId, SiteId, Topology

RESTART_TICKS = (30, 90)
"""Inclusive range a server restart takes, drawn uniformly per restart."""

REPAIR_TICKS = (480, 720)
"""Inclusive range a manual repair takes (8 to 12 minutes), drawn uniformly per repair."""

ROUTING_CONVERGENCE_TICKS = 5
"""Ticks before a withdrawal or restore has propagated through routing."""


class ActionKind(StrEnum):
    """The operator actions available to every agent."""

    RESTART = "restart"
    """Target: server. Restart the DNS process. Clears crashes and memory leaks, not hardware."""

    RATE_LIMIT = "rate_limit"
    """Target: site. Drop queries above ``limit_qps``, mostly attack. Immediate."""

    CLEAR_RATE_LIMIT = "clear_rate_limit"
    """Target: site. Remove the site's rate limit. Takes effect at once."""

    WITHDRAW = "withdraw"
    """Target: site. Stop announcing the site; its traffic moves to the next preferred site."""

    RESTORE = "restore"
    """Target: site. Announce the site again with weight ``weight`` in ``(0, 1]``."""

    REQUEST_REPAIR = "request_repair"
    """Target: server or site. Engineers repair it; the target serves nothing meanwhile."""


@dataclass(frozen=True, slots=True)
class Action:
    """One action requested by an agent.

    Attributes:
        kind: What to do.
        target: Server or site id, depending on ``kind``.
        limit_qps: Queries per tick allowed through; only for ``RATE_LIMIT``.
        weight: Announcement weight in ``(0, 1]``; only for ``RESTORE``.
    """

    kind: ActionKind
    target: str
    limit_qps: float | None = None
    weight: float | None = None

    def __post_init__(self) -> None:
        """Convert the kind to :class:`ActionKind` and check the parameters fit it."""
        object.__setattr__(self, "kind", ActionKind(self.kind))
        if self.kind == ActionKind.RATE_LIMIT:
            if self.limit_qps is None or not self.limit_qps >= 0:
                raise ValueError("RATE_LIMIT needs limit_qps >= 0")
        elif self.limit_qps is not None:
            raise ValueError(f"{self.kind} does not take limit_qps")
        if self.kind == ActionKind.RESTORE:
            if self.weight is None or not 0.0 < self.weight <= 1.0:
                raise ValueError("RESTORE needs a weight in (0, 1]")
        elif self.weight is not None:
            raise ValueError(f"{self.kind} does not take a weight")


class ActionStatus(StrEnum):
    """Lifecycle states of an action.

    Without a policy layer every action is started as soon as it is requested
    (``RUNNING``). The approval states are used once operator policy is added.
    """

    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ActionUpdate:
    """A status change of one action, reported in the next observation.

    Attributes:
        action_id: Identifier the engine assigned to the action, unique per run.
        action: The action as requested.
        status: The new status.
        tick: Tick at which the status changed.
    """

    action_id: int
    action: Action
    status: ActionStatus
    tick: int


def action_targets(action: Action, topology: Topology) -> tuple[ServerId, ...]:
    """Return the servers an action takes out of service while it runs.

    A restart takes down its server; a repair takes down its server, or every
    server at its site. Routing and rate-limit actions affect no server.
    """
    if action.kind == ActionKind.RESTART:
        return (ServerId(action.target),)
    if action.kind == ActionKind.REQUEST_REPAIR:
        if action.target in topology.sites:
            return topology.servers_at(SiteId(action.target))
        return (ServerId(action.target),)
    return ()


def validate_action(action: Action, topology: Topology) -> None:
    """Check that an action's target exists and is the right type for its kind.

    Raises:
        ValueError: If the target is unknown or the wrong type of thing.
    """
    if action.kind == ActionKind.RESTART:
        valid, expected = action.target in topology.servers, "server"
    elif action.kind == ActionKind.REQUEST_REPAIR:
        valid = action.target in topology.servers or action.target in topology.sites
        expected = "server or site"
    else:
        valid, expected = action.target in topology.sites, "site"
    if not valid:
        raise ValueError(f"{action.kind} targets {action.target!r}, which is not a {expected}")
