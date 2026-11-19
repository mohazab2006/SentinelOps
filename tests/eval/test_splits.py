"""Tests for the train / tune / test seed splits."""

from __future__ import annotations

import itertools

import pytest

from sentinelops.eval.splits import SPLIT_SEEDS, Split, split_of, split_scenarios
from sentinelops.scenarios import FAMILIES, generate
from sentinelops.sim.topology import Topology, default_topology


@pytest.fixture(scope="module")
def topology() -> Topology:
    return default_topology()


def test_seed_ranges_match_the_plan() -> None:
    assert SPLIT_SEEDS[Split.TRAIN] == range(0, 60)
    assert SPLIT_SEEDS[Split.TUNE] == range(100, 140)
    assert SPLIT_SEEDS[Split.TEST] == range(1000, 1020)
    assert len(SPLIT_SEEDS[Split.TEST]) == 20  # 20 held-out incidents per family


def test_splits_never_share_a_seed() -> None:
    for a, b in itertools.combinations(SPLIT_SEEDS.values(), 2):
        assert not set(a) & set(b)


def test_seed_ranges_cannot_be_changed() -> None:
    with pytest.raises(TypeError):
        SPLIT_SEEDS[Split.TEST] = range(0, 20)  # type: ignore[index]


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (0, Split.TRAIN),
        (59, Split.TRAIN),
        (60, None),
        (99, None),
        (100, Split.TUNE),
        (139, Split.TUNE),
        (140, None),
        (1000, Split.TEST),
        (1019, Split.TEST),
        (1020, None),
        (-1, None),
    ],
)
def test_split_of(seed: int, expected: Split | None) -> None:
    assert split_of(seed) == expected


def test_scenarios_come_family_by_family_with_ascending_seeds(topology: Topology) -> None:
    scenarios = split_scenarios(Split.TUNE, topology, families=("flood", "crash"))
    assert [(s.family, s.seed) for s in scenarios] == [
        (family, seed) for family in ("flood", "crash") for seed in range(100, 140)
    ]


def test_scenarios_are_the_generated_ones(topology: Topology) -> None:
    scenarios = split_scenarios("train", topology, families=("mixed",), horizon=2000)
    assert scenarios == [generate("mixed", seed, topology, 2000) for seed in range(60)]


def test_every_family_by_default(topology: Topology) -> None:
    scenarios = split_scenarios(Split.TRAIN, topology)
    assert len(scenarios) == 60 * len(FAMILIES)
    assert {s.family for s in scenarios} == set(FAMILIES)


def test_test_split_is_held_out_unless_final(topology: Topology) -> None:
    with pytest.raises(ValueError, match="held out"):
        split_scenarios(Split.TEST, topology)
    with pytest.raises(ValueError, match="held out"):
        split_scenarios("test", topology, families=("crash",))
    scenarios = split_scenarios(Split.TEST, topology, families=("crash",), final=True)
    assert [s.seed for s in scenarios] == list(range(1000, 1020))


def test_final_does_not_change_other_splits(topology: Topology) -> None:
    assert split_scenarios(Split.TRAIN, topology, ("crash",), final=True) == split_scenarios(
        Split.TRAIN, topology, ("crash",)
    )


@pytest.mark.parametrize(
    ("split", "families", "message"),
    [
        ("validation", ("crash",), "validation"),
        ("train", ("crash", "meteor"), "unknown scenario families"),
        ("train", ("crash", "crash"), "must not repeat"),
    ],
)
def test_bad_arguments(
    topology: Topology, split: str, families: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        split_scenarios(split, topology, families)
