"""The interface every agent implements, plus a do-nothing reference agent.

SentinelOps, the rule-based baseline and any ablation all implement
:class:`Agent`, so the same evaluation harness can run each of them on the
same seeded scenarios and compare them fairly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from sentinelops.sim.actions import Action
from sentinelops.sim.telemetry import Observation
from sentinelops.sim.topology import Topology


@runtime_checkable
class Agent(Protocol):
    """Something that watches observations and decides on actions.

    Attributes:
        name: Short identifier used in results, e.g. ``"rules"``.
    """

    name: str

    def reset(self, topology: Topology) -> None:
        """Forget all state from previous runs and prepare for ``topology``."""
        ...

    def act(self, obs: Observation) -> Sequence[Action]:
        """Return the actions to take after seeing ``obs`` (may be empty)."""
        ...


class NoOpAgent:
    """An agent that never acts: the "do nothing" reference point.

    Comparing against it shows how much damage each incident does on its own.
    """

    name = "noop"

    def reset(self, topology: Topology) -> None:
        """Nothing to reset."""

    def act(self, obs: Observation) -> Sequence[Action]:
        """Never take an action."""
        return ()
