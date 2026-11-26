"""Tests for single-use approvals and the scripted approver."""

from __future__ import annotations

import dataclasses

import pytest

from sentinelops.policy import ApprovalBook, ApprovalRequest, ApprovalStatus, ScriptedApprover
from sentinelops.sim.actions import Action, ActionKind

RESTART_1 = Action(ActionKind.RESTART, "yyz-1")
RESTART_2 = Action(ActionKind.RESTART, "yyz-2")
WITHDRAW = Action(ActionKind.WITHDRAW, "yyz")
RATE_LIMIT = Action(ActionKind.RATE_LIMIT, "yyz", limit_qps=100.0)


# ------------------------------------------------------------------ lifecycle


def test_request_approve_consume_once() -> None:
    book = ApprovalBook()
    req = book.request(RESTART_1, tick=10, reason="conservative")
    assert (req.request_id, req.requested_tick, req.reason) == (0, 10, "conservative")
    assert book.status(0) == ApprovalStatus.PENDING
    assert book.decided_tick(0) is None
    book.approve(0, tick=40)
    assert book.status(0) == ApprovalStatus.APPROVED
    assert book.decided_tick(0) == 40
    assert book.consume(RESTART_1) is True
    assert book.status(0) == ApprovalStatus.USED
    assert book.consume(RESTART_1) is False  # single use


def test_approval_covers_only_that_exact_action() -> None:
    book = ApprovalBook()
    book.approve(book.request(RESTART_1, 0).request_id, 5)
    assert book.consume(RESTART_2) is False
    assert book.consume(RATE_LIMIT) is False
    assert book.status(0) == ApprovalStatus.APPROVED  # still waiting for its own action


def test_parameters_are_part_of_the_action() -> None:
    book = ApprovalBook()
    half = Action(ActionKind.RESTORE, "yyz", weight=0.5)
    book.approve(book.request(half, 0).request_id, 1)
    assert book.consume(Action(ActionKind.RESTORE, "yyz", weight=1.0)) is False
    assert book.consume(Action(ActionKind.RESTORE, "yyz", weight=0.5)) is True


def test_ids_count_up_in_filing_order() -> None:
    book = ApprovalBook()
    ids = [book.request(a, t).request_id for t, a in enumerate([RESTART_1, RESTART_2, WITHDRAW])]
    assert ids == [0, 1, 2]


def test_asking_again_while_open_returns_the_same_request() -> None:
    book = ApprovalBook()
    first = book.request(WITHDRAW, 10)
    assert book.request(WITHDRAW, 20) is first  # still pending
    book.approve(first.request_id, 30)
    assert book.request(WITHDRAW, 35) is first  # approved, not used yet
    assert book.consume(WITHDRAW)
    second = book.request(WITHDRAW, 50)  # used up: a new request
    assert second.request_id == 1
    assert second.requested_tick == 50


def test_asking_again_after_a_rejection_files_a_new_request() -> None:
    book = ApprovalBook()
    book.reject(book.request(WITHDRAW, 0).request_id, 5)
    assert book.request(WITHDRAW, 10).request_id == 1


def test_rejected_request_cannot_be_consumed() -> None:
    book = ApprovalBook()
    book.reject(book.request(WITHDRAW, 0).request_id, 7)
    assert book.status(0) == ApprovalStatus.REJECTED
    assert book.decided_tick(0) == 7
    assert book.consume(WITHDRAW) is False


def test_consume_with_no_requests_is_false() -> None:
    assert ApprovalBook().consume(RESTART_1) is False


# ------------------------------------------------------------------ errors


@pytest.mark.parametrize("call", ["approve", "reject", "status", "decided_tick"])
def test_unknown_id_is_a_key_error(call: str) -> None:
    book = ApprovalBook()
    book.request(RESTART_1, 0)
    args = (5, 1) if call in ("approve", "reject") else (5,)
    with pytest.raises(KeyError):
        getattr(book, call)(*args)


@pytest.mark.parametrize("second", ["approve", "reject"])
def test_deciding_twice_is_a_value_error(second: str) -> None:
    book = ApprovalBook()
    book.approve(book.request(RESTART_1, 0).request_id, 1)
    with pytest.raises(ValueError, match="not pending"):
        getattr(book, second)(0, 2)


def test_deciding_before_the_request_was_filed_is_a_value_error() -> None:
    book = ApprovalBook()
    book.request(RESTART_1, 100)
    with pytest.raises(ValueError, match="before"):
        book.approve(0, 99)
    with pytest.raises(ValueError, match="before"):
        book.reject(0, 99)
    assert book.status(0) == ApprovalStatus.PENDING


def test_negative_delay_is_rejected() -> None:
    with pytest.raises(ValueError, match="delay_ticks"):
        ScriptedApprover(delay_ticks=-1)


# ------------------------------------------------------------------ scripted approver


def test_scripted_approver_waits_exactly_the_delay() -> None:
    book = ApprovalBook()
    book.request(RESTART_1, 100)
    approver = ScriptedApprover(delay_ticks=120)
    assert approver.respond(book, 219) == ()
    assert book.status(0) == ApprovalStatus.PENDING
    assert approver.respond(book, 220) == (0,)
    assert book.status(0) == ApprovalStatus.APPROVED
    assert book.decided_tick(0) == 220
    assert approver.respond(book, 300) == ()  # nothing left to decide


def test_scripted_approver_rejects_listed_kinds_and_answers_oldest_first() -> None:
    book = ApprovalBook()
    book.request(RESTART_1, 0)
    book.request(WITHDRAW, 5)
    book.request(RESTART_2, 50)  # too recent at tick 130
    approver = ScriptedApprover(reject=frozenset({ActionKind.WITHDRAW}))
    assert approver.respond(book, 130) == (0, 1)
    assert book.status(0) == ApprovalStatus.APPROVED
    assert book.status(1) == ApprovalStatus.REJECTED
    assert book.status(2) == ApprovalStatus.PENDING
    assert approver.respond(book, 170) == (2,)


def test_zero_delay_answers_at_once() -> None:
    book = ApprovalBook()
    book.request(RESTART_1, 42)
    assert ScriptedApprover(delay_ticks=0).respond(book, 42) == (0,)
    assert book.consume(RESTART_1)


def test_default_delay_is_two_minutes() -> None:
    assert ScriptedApprover().delay_ticks == 120


def test_reject_kinds_accept_plain_strings() -> None:
    approver = ScriptedApprover(reject=frozenset({"withdraw"}))  # type: ignore[arg-type]
    assert approver.reject == frozenset({ActionKind.WITHDRAW})


# ------------------------------------------------------------------ read-only views


def test_request_is_frozen() -> None:
    req = ApprovalBook().request(RESTART_1, 0)
    assert isinstance(req, ApprovalRequest)
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.requested_tick = 5  # type: ignore[misc]


def test_pending_is_an_oldest_first_tuple_the_caller_cannot_change() -> None:
    book = ApprovalBook()
    book.request(RESTART_1, 0)
    book.request(RESTART_2, 1)
    book.request(WITHDRAW, 2)
    book.approve(1, 3)
    pending = book.pending()
    assert isinstance(pending, tuple)
    assert [r.request_id for r in pending] == [0, 2]
    as_list = list(pending)
    as_list.clear()
    assert [r.request_id for r in book.pending()] == [0, 2]
