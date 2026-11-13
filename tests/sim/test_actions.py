"""Tests for operator actions."""

from __future__ import annotations

import pytest

from sentinelops.sim.actions import (
    Action,
    ActionKind,
    action_targets,
    validate_action,
)
from sentinelops.sim.topology import default_topology

TOPO = default_topology()


def test_action_converts_string_kind() -> None:
    assert Action("restart", "ott-1").kind is ActionKind.RESTART  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Action("reboot", "ott-1")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "build",
    [
        lambda: Action(ActionKind.RATE_LIMIT, "ott"),
        lambda: Action(ActionKind.RATE_LIMIT, "ott", limit_qps=-1.0),
        lambda: Action(ActionKind.RESTART, "ott-1", limit_qps=10.0),
        lambda: Action(ActionKind.RESTORE, "ott"),
        lambda: Action(ActionKind.RESTORE, "ott", weight=0.0),
        lambda: Action(ActionKind.RESTORE, "ott", weight=1.5),
        lambda: Action(ActionKind.WITHDRAW, "ott", weight=0.5),
    ],
)
def test_action_parameters_must_fit_the_kind(build: object) -> None:
    with pytest.raises(ValueError):
        build()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("action", "ok"),
    [
        (Action(ActionKind.RESTART, "ott-1"), True),
        (Action(ActionKind.RESTART, "ott"), False),
        (Action(ActionKind.WITHDRAW, "ott"), True),
        (Action(ActionKind.WITHDRAW, "ott-1"), False),
        (Action(ActionKind.REQUEST_REPAIR, "ott"), True),
        (Action(ActionKind.REQUEST_REPAIR, "ott-2"), True),
        (Action(ActionKind.REQUEST_REPAIR, "mars"), False),
        (Action(ActionKind.RATE_LIMIT, "lon", limit_qps=5.0), True),
    ],
)
def test_validate_action(action: Action, ok: bool) -> None:
    if ok:
        validate_action(action, TOPO)
    else:
        with pytest.raises(ValueError):
            validate_action(action, TOPO)


def test_action_targets() -> None:
    assert action_targets(Action(ActionKind.RESTART, "ott-1"), TOPO) == ("ott-1",)
    assert action_targets(Action(ActionKind.REQUEST_REPAIR, "lon"), TOPO) == ("lon-1", "lon-2")
    assert action_targets(Action(ActionKind.REQUEST_REPAIR, "lon-2"), TOPO) == ("lon-2",)
    assert action_targets(Action(ActionKind.WITHDRAW, "lon"), TOPO) == ()
