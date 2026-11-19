"""Train / tune / test splits by scenario seed.

Every learned part of SentinelOps is fitted on one set of incidents, tuned on a
second and judged on a third that it never saw. The split is by *run*: a seed
picks a whole scenario, so no window of a test incident can leak into
training. The same seed ranges are used for every scenario family:

* ``train`` - seeds 0-59: fit models (detector, Bayesian network).
* ``tune``  - seeds 100-139: choose thresholds and settings, compare variants.
* ``test``  - seeds 1000-1019: the 20 held-out incidents per family, used once,
  for the final results only.

The gaps between the ranges leave room to grow a split without overlapping
another. The test split is frozen: :func:`split_scenarios` only returns it when
asked with ``final=True``, so it cannot be used for tuning by accident.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from types import MappingProxyType

from sentinelops.scenarios import FAMILIES, Scenario, generate
from sentinelops.scenarios.families import DEFAULT_HORIZON
from sentinelops.sim.topology import Topology


class Split(StrEnum):
    """The three disjoint sets of incidents."""

    TRAIN = "train"
    """Fit models."""

    TUNE = "tune"
    """Choose thresholds and settings."""

    TEST = "test"
    """Held out for the final evaluation."""


SPLIT_SEEDS: Mapping[Split, range] = MappingProxyType(
    {
        Split.TRAIN: range(0, 60),
        Split.TUNE: range(100, 140),
        Split.TEST: range(1000, 1020),
    }
)
"""Scenario seeds of each split, the same for every family. Ranges never overlap."""


def split_of(seed: int) -> Split | None:
    """Return the split ``seed`` belongs to, or ``None`` if it is in no split."""
    for split, seeds in SPLIT_SEEDS.items():
        if seed in seeds:
            return split
    return None


def split_scenarios(
    split: Split | str,
    topology: Topology,
    families: Iterable[str] = FAMILIES,
    horizon: int = DEFAULT_HORIZON,
    *,
    final: bool = False,
) -> list[Scenario]:
    """Generate every scenario of a split, family by family, seeds ascending.

    Args:
        split: Which split.
        topology: The network the scenarios run on.
        families: Scenario families to include, in the order given.
        horizon: Run length in ticks.
        final: Must be ``True`` to get the test split, which is reserved for
            the final evaluation.

    Returns:
        One scenario per (family, seed), deterministic in the arguments.

    Raises:
        ValueError: If the split or a family is unknown, a family is repeated,
            or the test split is requested without ``final=True``.
    """
    split = Split(split)
    if split is Split.TEST and not final:
        raise ValueError(
            "the test split is held out for the final evaluation; pass final=True to use it"
        )
    chosen = tuple(families)
    unknown = [family for family in chosen if family not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown scenario families {unknown}; valid families: {FAMILIES}")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"families must not repeat, got {list(chosen)}")
    return [
        generate(family, seed, topology, horizon)
        for family in chosen
        for seed in SPLIT_SEEDS[split]
    ]
