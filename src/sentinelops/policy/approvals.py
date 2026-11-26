"""Approvals: what happens when the policy says REQUIRE_APPROVAL.

The agent files a request in an :class:`ApprovalBook`; a human operator (or, in
experiments, a :class:`ScriptedApprover`) approves or rejects it. A request
moves PENDING -> APPROVED or REJECTED, and APPROVED -> USED when the agent
actually runs the action.

An approval is **single-use** and covers **exactly one action**: the same kind,
target and parameters. Approving "restart yyz-1" does not approve "restart
yyz-2", nor a second restart of yyz-1 later, so one "yes" can never turn into a
standing permission. Asking again for an action that is still pending (or
approved but not yet used) returns the existing request instead of a duplicate.

Waiting for approval costs time while queries keep failing; the scripted
approver's fixed delay lets experiments measure that cost reproducibly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sentinelops.sim.actions import Action, ActionKind


class ApprovalStatus(StrEnum):
    """Where an approval request is in its life."""

    PENDING = "pending"
    """Waiting for an operator."""

    APPROVED = "approved"
    """The operator said yes; the agent has not run the action yet."""

    REJECTED = "rejected"
    """The operator said no."""

    USED = "used"
    """The agent ran the approved action; the approval is spent."""


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """One action the agent asked permission for.

    Attributes:
        request_id: ``0, 1, 2, ...`` in the order requests were filed.
        action: Exactly the action to approve (kind, target and parameters).
        requested_tick: Simulation tick the request was filed.
        reason: Why approval is needed, usually the name of the policy rule.
    """

    request_id: int
    action: Action
    requested_tick: int
    reason: str = ""


class ApprovalBook:
    """Every approval request with its status (one book per run)."""

    def __init__(self) -> None:
        """Start with no requests."""
        self._requests: dict[int, ApprovalRequest] = {}
        self._status: dict[int, ApprovalStatus] = {}
        self._decided: dict[int, int] = {}

    def request(self, action: Action, tick: int, reason: str = "") -> ApprovalRequest:
        """File a request for ``action``, or return the open one for it.

        A request for the identical action that is still PENDING, or APPROVED
        but not used yet, is returned instead of filing a duplicate.

        Args:
            action: The action the agent wants to take.
            tick: Current simulation tick.
            reason: Why approval is needed (shown to the operator).
        """
        open_states = (ApprovalStatus.PENDING, ApprovalStatus.APPROVED)
        for request_id, existing in self._requests.items():
            if existing.action == action and self._status[request_id] in open_states:
                return existing
        new = ApprovalRequest(len(self._requests), action, tick, reason)
        self._requests[new.request_id] = new
        self._status[new.request_id] = ApprovalStatus.PENDING
        return new

    def approve(self, request_id: int, tick: int) -> None:
        """Approve a pending request.

        Raises:
            KeyError: If there is no request with this id.
            ValueError: If the request is not PENDING, or ``tick`` is before
                the request was filed.
        """
        self._decide(request_id, tick, ApprovalStatus.APPROVED)

    def reject(self, request_id: int, tick: int) -> None:
        """Reject a pending request.

        Raises:
            KeyError: If there is no request with this id.
            ValueError: If the request is not PENDING, or ``tick`` is before
                the request was filed.
        """
        self._decide(request_id, tick, ApprovalStatus.REJECTED)

    def status(self, request_id: int) -> ApprovalStatus:
        """Return the request's current status.

        Raises:
            KeyError: If there is no request with this id.
        """
        return self._status[self._known(request_id)]

    def decided_tick(self, request_id: int) -> int | None:
        """Return the tick the request was approved or rejected, or ``None`` while pending.

        Raises:
            KeyError: If there is no request with this id.
        """
        return self._decided.get(self._known(request_id))

    def pending(self) -> tuple[ApprovalRequest, ...]:
        """Return the requests still waiting for an operator, oldest first."""
        return tuple(
            request
            for request_id, request in self._requests.items()
            if self._status[request_id] == ApprovalStatus.PENDING
        )

    def consume(self, action: Action) -> bool:
        """Use up the approval for exactly ``action``, if there is one.

        Returns:
            ``True`` if an APPROVED request for this exact action existed (the
            oldest such request becomes USED), otherwise ``False``.
        """
        for request_id, request in self._requests.items():
            if request.action == action and self._status[request_id] == ApprovalStatus.APPROVED:
                self._status[request_id] = ApprovalStatus.USED
                return True
        return False

    def _known(self, request_id: int) -> int:
        if request_id not in self._requests:
            raise KeyError(f"no approval request with id {request_id}")
        return request_id

    def _decide(self, request_id: int, tick: int, outcome: ApprovalStatus) -> None:
        request = self._requests[self._known(request_id)]
        if self._status[request_id] != ApprovalStatus.PENDING:
            raise ValueError(f"request {request_id} is {self._status[request_id]}, not pending")
        if tick < request.requested_tick:
            raise ValueError(
                f"tick {tick} is before request {request_id} was filed ({request.requested_tick})"
            )
        self._status[request_id] = outcome
        self._decided[request_id] = tick


class ScriptedApprover:
    """Stands in for the human operator in experiments.

    It answers every request after a fixed delay, so runs are reproducible
    and the cost of waiting for approval can be measured.

    Args:
        delay_ticks: How long a request waits before it is answered
            (default 120 ticks = 2 minutes); 0 answers at once.
        reject: Action kinds that are always rejected, e.g. to model "the
            operator never allows withdrawals".

    Raises:
        ValueError: If ``delay_ticks`` is negative.
    """

    def __init__(self, delay_ticks: int = 120, reject: frozenset[ActionKind] = frozenset()) -> None:
        """Store the delay and the kinds to reject."""
        if delay_ticks < 0:
            raise ValueError(f"delay_ticks must be >= 0, got {delay_ticks}")
        self.delay_ticks = delay_ticks
        self.reject = frozenset(ActionKind(kind) for kind in reject)

    def respond(self, book: ApprovalBook, tick: int) -> tuple[int, ...]:
        """Decide every pending request that has waited at least ``delay_ticks``.

        Args:
            book: The approval book to answer.
            tick: Current simulation tick.

        Returns:
            The ids of the requests decided now, oldest first.
        """
        decided = []
        for request in book.pending():
            if tick - request.requested_tick < self.delay_ticks:
                continue
            if request.action.kind in self.reject:
                book.reject(request.request_id, tick)
            else:
                book.approve(request.request_id, tick)
            decided.append(request.request_id)
        return tuple(decided)
