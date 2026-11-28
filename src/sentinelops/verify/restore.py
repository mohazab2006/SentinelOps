"""Gradual traffic restoration with rollback for a repaired, withdrawn site.

When a withdrawn site has been fixed, its traffic comes back **in steps**
(25%, then 50%, then 100%) rather than all at once. If the fix did not really
work, a site at 25% hurts a quarter of its users for a short time; at 100% it
would hurt all of them.

After each step the routing change needs ``ROUTING_CONVERGENCE_TICKS`` (5)
ticks to settle, so telemetry from those ticks still shows the old traffic
share and is ignored. Then a fresh :class:`~sentinelops.verify.checklist.SiteVerifier`
checks the site; when it passes, the next step starts.

If any step fails verification, the site is **rolled back** (withdrawn again)
instead of retried at a lower weight: a site that fails at any weight still has
its root cause, so the safe state is withdrawn and a human should look.

:class:`GradualRestore` never touches the simulator. It only says what the
agent should do each tick (:attr:`RestoreStep.action`), so it is easy to test
and the agent stays in charge.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from sentinelops.sim.actions import ROUTING_CONVERGENCE_TICKS, Action, ActionKind
from sentinelops.sim.telemetry import SiteTelemetry
from sentinelops.sim.topology import SiteId, Topology
from sentinelops.verify.checklist import HealthLimits, SiteVerifier, VerifyStatus

RAMP: tuple[float, ...] = (0.25, 0.5, 1.0)
"""Default announcement weights, one per step; the last is always full traffic."""


class RestorePhase(StrEnum):
    """Where a gradual restore stands."""

    RAMPING = "ramping"
    """Stepping the weight up and verifying each step."""

    DONE = "done"
    """Every step passed: the site carries its full traffic again."""

    ROLLED_BACK = "rolled_back"
    """A step failed verification and the site was withdrawn again."""


@dataclass(frozen=True, slots=True)
class RestoreStep:
    """What the restorer says after a call.

    Attributes:
        phase: The current phase.
        weight: The weight being verified now (1.0 when done, 0.0 after a rollback).
        action: What the agent should send this tick, or ``None``.
    """

    phase: RestorePhase
    weight: float
    action: Action | None


class GradualRestore:
    """Brings one site's traffic back step by step, rolling back on failure.

    Args:
        site: The withdrawn site to restore.
        topology: The network (for the site's queue limit when verifying).
        ramp: Weights to step through, strictly increasing in ``(0, 1]`` and
            ending at 1.0.
        hold_ticks: Healthy ticks in a row each step must pass (the
            verifier's ``required_ticks``).
        fail_after: Bad ticks in a row that fail a step.
        deadline_ticks: Ticks a step may take to pass before it fails.
        limits: Health limits; the checklist defaults if ``None``.

    Raises:
        ValueError: If the site is unknown, or ``ramp`` is empty, not strictly
            increasing, has a weight outside ``(0, 1]`` or does not end at 1.0.
    """

    def __init__(
        self,
        site: SiteId,
        topology: Topology,
        *,
        ramp: Sequence[float] = RAMP,
        hold_ticks: int = 30,
        fail_after: int = 10,
        deadline_ticks: int = 300,
        limits: HealthLimits | None = None,
    ) -> None:
        """Check the site and ramp; nothing happens until :meth:`start`."""
        if site not in topology.sites:
            raise ValueError(f"unknown site {site!r}")
        weights = tuple(float(w) for w in ramp)
        if not weights:
            raise ValueError("ramp must not be empty")
        if any(not 0.0 < w <= 1.0 for w in weights):
            raise ValueError(f"every ramp weight must be in (0, 1], got {weights}")
        if any(a >= b for a, b in itertools.pairwise(weights)):
            raise ValueError(f"ramp must be strictly increasing, got {weights}")
        if weights[-1] != 1.0:
            raise ValueError(f"ramp must end at 1.0, got {weights}")
        self._site = site
        self._topology = topology
        self._ramp = weights
        self._hold = hold_ticks
        self._fail_after = fail_after
        self._deadline = deadline_ticks
        self._limits = limits
        self._index = -1  # position in the ramp; -1 until start()
        self._step_tick = 0
        self._phase = RestorePhase.RAMPING
        self._verifier: SiteVerifier | None = None

    @property
    def ramp(self) -> tuple[float, ...]:
        """The weights this restore steps through."""
        return self._ramp

    def start(self, tick: int) -> RestoreStep:
        """Begin the restore with the first weight.

        Args:
            tick: Current simulation tick.

        Returns:
            RAMPING at ``ramp[0]`` with the ``RESTORE`` action to send.

        Raises:
            RuntimeError: If the restore was already started.
        """
        if self._index >= 0:
            raise RuntimeError("the restore was already started")
        return self._next_step(tick)

    def update(self, tick: int, telemetry: SiteTelemetry) -> RestoreStep:
        """Feed one tick of the site's telemetry.

        Args:
            tick: The tick the telemetry describes.
            telemetry: The site's readings for that tick.

        Returns:
            The current step. Its ``action`` is the next ``RESTORE`` when a
            step passed, ``WITHDRAW`` when one failed, otherwise ``None``.

        Raises:
            RuntimeError: If :meth:`start` was not called first.
        """
        if self._index < 0:
            raise RuntimeError("call start() before update()")
        if self._phase != RestorePhase.RAMPING:
            return self._current(None)
        if tick - self._step_tick <= ROUTING_CONVERGENCE_TICKS:
            return self._current(None)  # routing has not settled: telemetry shows the old share
        verifier = self._verifier_for_step()
        status = verifier.update(tick, telemetry)
        if status == VerifyStatus.FAILED:
            self._phase = RestorePhase.ROLLED_BACK
            return self._current(Action(ActionKind.WITHDRAW, self._site))
        if status == VerifyStatus.PASSED:
            if self._index == len(self._ramp) - 1:
                self._phase = RestorePhase.DONE
                return self._current(None)
            return self._next_step(tick)
        return self._current(None)

    def _next_step(self, tick: int) -> RestoreStep:
        self._index += 1
        self._step_tick = tick
        self._verifier = None  # a fresh verifier per step, created once routing settles
        weight = self._ramp[self._index]
        return self._current(Action(ActionKind.RESTORE, self._site, weight=weight))

    def _verifier_for_step(self) -> SiteVerifier:
        if self._verifier is None:
            self._verifier = SiteVerifier(
                self._site,
                self._topology,
                self._step_tick + ROUTING_CONVERGENCE_TICKS,
                required_ticks=self._hold,
                fail_after=self._fail_after,
                deadline_ticks=self._deadline,
                limits=self._limits,
            )
        return self._verifier

    def _current(self, action: Action | None) -> RestoreStep:
        weight = 0.0 if self._phase == RestorePhase.ROLLED_BACK else self._ramp[self._index]
        return RestoreStep(self._phase, weight, action)
