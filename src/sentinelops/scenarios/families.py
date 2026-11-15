"""Seeded scenario families: reproducible incidents to run agents against.

A scenario is a list of faults (what breaks, where, when, how badly) plus a run
length. :func:`generate` builds one from a *family* (the kind of incident, such
as ``"crash"`` or ``"flood"``) and a *seed*. All random choices come from one
NumPy generator seeded with ``[seed, family index]``, so:

* the same (family, seed) always gives exactly the same scenario, and
* seed 3 of ``"crash"`` and seed 3 of ``"flood"`` are unrelated incidents,
  because the family index is mixed into the seed.

Experiments use disjoint seed ranges for training, tuning and the held-out test
set, so agents are always evaluated on incidents they never saw.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from sentinelops.sim.faults import Fault, FaultKind, validate_fault
from sentinelops.sim.topology import Topology

FAMILIES: tuple[str, ...] = (
    "crash",
    "exhaustion",
    "hardware",
    "path_loss",
    "site_outage",
    "flood",
    "benign_surge",
    "mixed",
)
"""Scenario families. Every family except ``"mixed"`` is named after its fault kind."""

DEFAULT_HORIZON = 1800
"""Default run length in ticks (30 simulated minutes)."""

ONSET_RANGE = (300, 900)
"""Single-fault families: the fault starts at a tick drawn uniformly from this range."""

SEVERITY_RANGE = (0.4, 1.0)
"""Severity range for every kind except path loss."""

PATH_LOSS_SEVERITY_RANGE = (0.2, 0.6)
"""Path-loss severity range (the fraction of traffic lost)."""

DURATION_RANGES: dict[FaultKind, tuple[int, int]] = {
    FaultKind.PATH_LOSS: (300, 700),
    FaultKind.FLOOD: (180, 600),
    FaultKind.BENIGN_SURGE: (180, 600),
}
"""Duration ranges in ticks. Kinds not listed last until an action clears them."""

MIXED_FIRST_ONSET_RANGE = (300, 700)
"""Mixed family: onset range of the first fault."""

MIXED_GAP_RANGE = (30, 89)
"""Mixed family: ticks between one fault's onset and the next.

At most 89 so that even with three faults the last one starts (2 x 89 = 178
ticks after the first) before the shortest possible timed fault (180 ticks) has
ended. Every fault in a mixed scenario is therefore active at the same time.
"""

MIXED_FAULT_COUNTS = (2, 3)
"""Mixed family: number of faults, drawn uniformly from this inclusive range."""

_LATEST_SINGLE_END = ONSET_RANGE[1] + max(high for _, high in DURATION_RANGES.values())
_SERVER_KINDS = frozenset({FaultKind.CRASH, FaultKind.EXHAUSTION, FaultKind.HARDWARE})
_SITE_KINDS = frozenset({FaultKind.PATH_LOSS, FaultKind.SITE_OUTAGE})
_ALL_KINDS: tuple[FaultKind, ...] = tuple(FaultKind)


@dataclass(frozen=True, slots=True)
class Scenario:
    """One reproducible incident.

    Attributes:
        family: The family it was generated from.
        seed: The seed it was generated from.
        horizon: Total run length in ticks.
        faults: The faults to inject, sorted by onset.
    """

    family: str
    seed: int
    horizon: int
    faults: tuple[Fault, ...]


def _targets(kind: FaultKind, topology: Topology) -> list[str]:
    """Return the valid targets for ``kind``, sorted so dict order cannot matter."""
    if kind in _SERVER_KINDS:
        return sorted(topology.servers)
    if kind in _SITE_KINDS:
        return sorted(topology.sites)
    return sorted(topology.regions)


def _draw_fault(
    kind: FaultKind,
    onset: int,
    rng: np.random.Generator,
    topology: Topology,
    excluded: frozenset[str] = frozenset(),
) -> Fault:
    """Draw one fault of ``kind`` starting at ``onset``.

    Args:
        kind: The fault kind.
        onset: The fault's first active tick.
        rng: The scenario's random generator.
        topology: The network, for choosing a target.
        excluded: Targets that must not be chosen.

    Raises:
        ValueError: If no valid target is left.
    """
    candidates = [t for t in _targets(kind, topology) if t not in excluded]
    if not candidates:
        raise ValueError(f"no target left for a {kind} fault in this topology")
    target = candidates[int(rng.integers(len(candidates)))]
    low, high = PATH_LOSS_SEVERITY_RANGE if kind == FaultKind.PATH_LOSS else SEVERITY_RANGE
    severity = float(rng.uniform(low, high))
    duration: int | None = None
    if kind in DURATION_RANGES:
        d_low, d_high = DURATION_RANGES[kind]
        duration = int(rng.integers(d_low, d_high, endpoint=True))
    return Fault(kind, target, onset, duration, severity)


def _mixed_faults(rng: np.random.Generator, topology: Topology) -> list[Fault]:
    """Draw 2-3 faults of distinct kinds whose active windows all overlap.

    No two server faults are put on the same server.
    """
    count = int(rng.integers(MIXED_FAULT_COUNTS[0], MIXED_FAULT_COUNTS[1], endpoint=True))
    picks = rng.choice(len(_ALL_KINDS), size=count, replace=False)
    kinds = [_ALL_KINDS[int(i)] for i in picks]
    onset = int(rng.integers(*MIXED_FIRST_ONSET_RANGE, endpoint=True))
    faults: list[Fault] = []
    used_servers: set[str] = set()
    for position, kind in enumerate(kinds):
        if position > 0:
            onset += int(rng.integers(*MIXED_GAP_RANGE, endpoint=True))
        excluded = frozenset(used_servers) if kind in _SERVER_KINDS else frozenset()
        fault = _draw_fault(kind, onset, rng, topology, excluded)
        if kind in _SERVER_KINDS:
            used_servers.add(fault.target)
        faults.append(fault)
    return faults


def generate(
    family: str, seed: int, topology: Topology, horizon: int = DEFAULT_HORIZON
) -> Scenario:
    """Generate the scenario for ``(family, seed)``.

    Single-fault families draw one fault of that kind: onset uniform in
    ``ONSET_RANGE``, severity uniform in ``SEVERITY_RANGE`` (path loss:
    ``PATH_LOSS_SEVERITY_RANGE``), duration from ``DURATION_RANGES`` (or none),
    and a target chosen uniformly from the valid ones. The ``"mixed"`` family
    draws 2-3 faults of distinct kinds with overlapping windows.

    Args:
        family: One of :data:`FAMILIES`.
        seed: Any non-negative integer.
        topology: The network the scenario will run on.
        horizon: Run length in ticks; must leave room for the latest fault.

    Returns:
        The scenario, with faults sorted by onset. Deterministic in
        ``(family, seed, topology, horizon)``.

    Raises:
        ValueError: If the family is unknown, the horizon too short, or the
            topology has no valid target for a fault.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown scenario family {family!r}; valid families: {FAMILIES}")
    if horizon <= _LATEST_SINGLE_END:
        raise ValueError(f"horizon must be greater than {_LATEST_SINGLE_END}, got {horizon}")
    rng = np.random.default_rng([seed, FAMILIES.index(family)])
    if family == "mixed":
        faults = _mixed_faults(rng, topology)
    else:
        onset = int(rng.integers(*ONSET_RANGE, endpoint=True))
        faults = [_draw_fault(FaultKind(family), onset, rng, topology)]
    faults.sort(key=lambda fault: fault.onset)
    for fault in faults:
        validate_fault(fault, topology)
    return Scenario(family=family, seed=seed, horizon=horizon, faults=tuple(faults))


def generate_many(
    family: str, seeds: Iterable[int], topology: Topology, horizon: int = DEFAULT_HORIZON
) -> list[Scenario]:
    """Generate one scenario per seed, in the order the seeds are given."""
    return [generate(family, seed, topology, horizon) for seed in seeds]
