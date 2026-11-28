"""Tests for gradual traffic restoration with rollback."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from sentinelops.sim.actions import ROUTING_CONVERGENCE_TICKS, Action, ActionKind
from sentinelops.sim.env import DnsEnv
from sentinelops.sim.faults import Fault, FaultKind
from sentinelops.sim.telemetry import SiteTelemetry
from sentinelops.sim.topology import SiteId, default_topology
from sentinelops.verify import RAMP, GradualRestore, RestorePhase, RestoreStep

TOPOLOGY = default_topology()
LON = SiteId("lon")
HOLD = 30


def reading(**changes: Any) -> SiteTelemetry:
    """A healthy site reading, with ``changes`` applied."""
    healthy: dict[str, Any] = {
        "qps": 100.0,
        "p95_latency_ms": 40.0,
        "loss": 0.0,
        "queue_depth": 0.0,
        "ok_replies": 100.0,
        "failed_replies": 0.0,
        "announced": True,
        "weight": 1.0,
    }
    return SiteTelemetry(**{**healthy, **changes})


HEALTHY = reading()
BAD = reading(p95_latency_ms=2700.0)


def restore_to(weight: float) -> Action:
    return Action(ActionKind.RESTORE, LON, weight=weight)


def run(
    restore: GradualRestore, readings: list[SiteTelemetry], first_tick: int
) -> list[RestoreStep]:
    """Feed ``readings`` on consecutive ticks starting at ``first_tick``."""
    return [restore.update(first_tick + i, r) for i, r in enumerate(readings)]


def ticks_per_step(hold: int = HOLD) -> int:
    return ROUTING_CONVERGENCE_TICKS + hold


# ------------------------------------------------------------------ the happy path


def test_start_returns_the_first_restore_action() -> None:
    step = GradualRestore(LON, TOPOLOGY).start(100)
    assert step == RestoreStep(RestorePhase.RAMPING, 0.25, restore_to(0.25))


def test_healthy_site_ramps_25_50_100_then_done() -> None:
    restore = GradualRestore(LON, TOPOLOGY)
    actions = [restore.start(0).action]
    steps = run(restore, [HEALTHY] * (3 * ticks_per_step() + 20), first_tick=1)
    actions += [s.action for s in steps if s.action is not None]
    assert actions == [restore_to(0.25), restore_to(0.5), restore_to(1.0)]  # exactly 3 RESTOREs
    assert all(
        s.action is None for s in steps[:ROUTING_CONVERGENCE_TICKS]
    )  # ticks 1-5 after a step
    assert steps[ticks_per_step() - 2].action is None
    assert steps[ticks_per_step() - 1].action == restore_to(0.5)  # 5 + 30 ticks after the start
    assert steps[2 * ticks_per_step() - 1].action == restore_to(1.0)
    assert steps[3 * ticks_per_step() - 2].phase == RestorePhase.RAMPING
    assert steps[3 * ticks_per_step() - 1] == RestoreStep(RestorePhase.DONE, 1.0, None)
    assert all(
        s == RestoreStep(RestorePhase.DONE, 1.0, None) for s in steps[3 * ticks_per_step() :]
    )


def test_custom_ramp_takes_two_steps() -> None:
    restore = GradualRestore(LON, TOPOLOGY, ramp=[0.5, 1.0])
    assert restore.ramp == (0.5, 1.0)
    assert restore.start(0).action == restore_to(0.5)
    steps = run(restore, [HEALTHY] * (2 * ticks_per_step()), first_tick=1)
    assert [s.action for s in steps if s.action is not None] == [restore_to(1.0)]
    assert steps[-1].phase == RestorePhase.DONE


def test_custom_hold_ticks_are_used() -> None:
    restore = GradualRestore(LON, TOPOLOGY, hold_ticks=3)
    restore.start(0)
    steps = run(restore, [HEALTHY] * ticks_per_step(3), first_tick=1)
    assert steps[-1].action == restore_to(0.5)


def test_default_ramp() -> None:
    assert RAMP == (0.25, 0.5, 1.0)


# ------------------------------------------------------------------ rollback


def test_bad_telemetry_at_50_percent_rolls_back_once() -> None:
    restore = GradualRestore(LON, TOPOLOGY)
    restore.start(0)
    first = run(restore, [HEALTHY] * ticks_per_step(), first_tick=1)
    assert first[-1].action == restore_to(0.5)
    after = run(
        restore, [HEALTHY] * ROUTING_CONVERGENCE_TICKS + [BAD] * 10 + [HEALTHY] * 50, first_tick=36
    )
    withdraws = [s for s in after if s.action is not None]
    assert withdraws == [
        RestoreStep(RestorePhase.ROLLED_BACK, 0.0, Action(ActionKind.WITHDRAW, LON))
    ]
    assert after.index(withdraws[0]) == ROUTING_CONVERGENCE_TICKS + 9  # tenth bad tick
    assert all(s == RestoreStep(RestorePhase.ROLLED_BACK, 0.0, None) for s in after[-50:])


def test_bad_telemetry_while_routing_converges_is_ignored() -> None:
    restore = GradualRestore(LON, TOPOLOGY)
    restore.start(0)
    steps = run(restore, [BAD] * ROUTING_CONVERGENCE_TICKS + [HEALTHY] * HOLD, first_tick=1)
    assert all(s.phase == RestorePhase.RAMPING for s in steps)
    assert steps[-1].action == restore_to(0.5)


def test_deadline_of_a_step_is_counted_after_convergence() -> None:
    restore = GradualRestore(LON, TOPOLOGY, hold_ticks=5, fail_after=3, deadline_ticks=10)
    restore.start(0)
    flapping = ([HEALTHY] * 4 + [BAD]) * 4  # never 5 good or 3 bad in a row
    steps = run(restore, flapping, first_tick=1)
    phases = [s.phase for s in steps]
    assert phases.index(RestorePhase.ROLLED_BACK) == ROUTING_CONVERGENCE_TICKS + 10 - 1  # tick 15


# ------------------------------------------------------------------ errors


@pytest.mark.parametrize(
    ("ramp", "message"),
    [
        ((), "empty"),
        ((0.5, 0.25, 1.0), "increasing"),
        ((0.5, 0.5, 1.0), "increasing"),
        ((0.25, 0.5), "end at 1.0"),
        ((0.0, 1.0), r"\(0, 1\]"),
        ((0.5, 1.5), r"\(0, 1\]"),
    ],
)
def test_bad_ramps_are_rejected(ramp: tuple[float, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GradualRestore(LON, TOPOLOGY, ramp=ramp)


def test_unknown_site_is_rejected() -> None:
    with pytest.raises(ValueError, match="tor"):
        GradualRestore(SiteId("tor"), TOPOLOGY)


def test_update_before_start_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="start"):
        GradualRestore(LON, TOPOLOGY).update(1, HEALTHY)


def test_starting_twice_is_an_error() -> None:
    restore = GradualRestore(LON, TOPOLOGY)
    restore.start(0)
    with pytest.raises(RuntimeError, match="already"):
        restore.start(1)


def test_restore_step_is_frozen() -> None:
    step = GradualRestore(LON, TOPOLOGY).start(0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        step.weight = 1.0  # type: ignore[misc]


# ------------------------------------------------------------------ real simulator runs


def drive(faults: tuple[Fault, ...]) -> tuple[RestoreStep, SiteTelemetry]:
    """Withdraw London at tick 100, restore it gradually from tick 200, follow its actions."""
    env = DnsEnv(TOPOLOGY, faults, horizon=500, seed=7)
    obs = env.reset()
    restore = GradualRestore(LON, TOPOLOGY)
    step: RestoreStep | None = None
    pending: list[Action] = []
    while not env.done:
        actions, pending = pending, []
        if obs.tick == 100:
            actions.append(Action(ActionKind.WITHDRAW, LON))
        if obs.tick == 200:
            step = restore.start(obs.tick)
        if step is not None and step.action is not None and obs.tick >= 200:
            actions.append(step.action)
            step = dataclasses.replace(step, action=None)  # send each action once
        obs = env.step(actions)
        if obs.tick > 200:
            step = restore.update(obs.tick, obs.sites[LON])
    assert step is not None
    return step, obs.sites[LON]


def test_real_run_restores_a_healthy_site_fully() -> None:
    step, london = drive(faults=())
    assert step.phase == RestorePhase.DONE
    assert (london.announced, london.weight) == (True, 1.0)


def test_real_run_rolls_back_a_site_that_is_still_broken() -> None:
    broken = tuple(
        Fault(kind=FaultKind.HARDWARE, target=server, onset=50, duration=None, severity=1.0)
        for server in TOPOLOGY.sites[LON].servers
    )
    step, london = drive(broken)
    assert step.phase == RestorePhase.ROLLED_BACK
    assert london.announced is False
