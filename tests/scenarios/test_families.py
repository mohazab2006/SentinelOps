"""Tests for the seeded scenario families."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from sentinelops.scenarios import FAMILIES, Scenario, generate, generate_many
from sentinelops.scenarios.families import (
    DEFAULT_HORIZON,
    DURATION_RANGES,
    MIXED_FIRST_ONSET_RANGE,
    MIXED_GAP_RANGE,
    ONSET_RANGE,
    PATH_LOSS_SEVERITY_RANGE,
    SEVERITY_RANGE,
    _draw_fault,
)
from sentinelops.sim.faults import Fault, FaultKind, validate_fault
from sentinelops.sim.topology import default_topology

TOPO = default_topology()
SEEDS = range(50)
SINGLE_FAMILIES = tuple(f for f in FAMILIES if f != "mixed")
SERVER_KINDS = {FaultKind.CRASH, FaultKind.EXHAUSTION, FaultKind.HARDWARE}


def _check_fault_ranges(fault: Fault) -> None:
    """Assert a fault's severity, duration and target follow the family rules."""
    low, high = PATH_LOSS_SEVERITY_RANGE if fault.kind == FaultKind.PATH_LOSS else SEVERITY_RANGE
    assert low <= fault.severity <= high
    if fault.kind in DURATION_RANGES:
        d_low, d_high = DURATION_RANGES[fault.kind]
        assert fault.duration is not None
        assert d_low <= fault.duration <= d_high
    else:
        assert fault.duration is None
    validate_fault(fault, TOPO)


# ---------------------------------------------------------------- determinism


@pytest.mark.parametrize("family", FAMILIES)
def test_same_family_and_seed_give_equal_scenarios(family: str) -> None:
    assert generate(family, 7, TOPO) == generate(family, 7, TOPO)


@pytest.mark.parametrize("family", FAMILIES)
def test_different_seeds_give_different_scenarios(family: str) -> None:
    scenarios = {generate(family, seed, TOPO).faults for seed in range(20)}
    assert len(scenarios) >= 15


def test_family_is_mixed_into_the_seed() -> None:
    # crash and hardware draw exactly the same way (onset, server, severity), so if
    # the family were not part of the seed their onsets would be identical.
    crash = [generate("crash", seed, TOPO).faults[0].onset for seed in range(20)]
    hardware = [generate("hardware", seed, TOPO).faults[0].onset for seed in range(20)]
    assert crash != hardware


def test_generate_many_matches_generate_one_by_one() -> None:
    seeds = [3, 1, 4, 1, 5]
    assert generate_many("flood", seeds, TOPO) == [generate("flood", s, TOPO) for s in seeds]


def test_scenario_records_family_seed_and_horizon() -> None:
    scenario = generate("path_loss", 11, TOPO, horizon=2000)
    assert isinstance(scenario, Scenario)
    assert (scenario.family, scenario.seed, scenario.horizon) == ("path_loss", 11, 2000)
    assert generate("path_loss", 11, TOPO).horizon == DEFAULT_HORIZON


# ------------------------------------------------------------ single families


@pytest.mark.parametrize("family", SINGLE_FAMILIES)
def test_single_family_scenarios_follow_the_rules(family: str) -> None:
    for seed in SEEDS:
        scenario = generate(family, seed, TOPO)
        assert len(scenario.faults) == 1
        fault = scenario.faults[0]
        assert fault.kind == FaultKind(family)
        assert ONSET_RANGE[0] <= fault.onset <= ONSET_RANGE[1]
        _check_fault_ranges(fault)


def test_targets_are_spread_over_the_valid_choices() -> None:
    servers = {generate("crash", seed, TOPO).faults[0].target for seed in SEEDS}
    sites = {generate("site_outage", seed, TOPO).faults[0].target for seed in SEEDS}
    assert len(servers) >= 8  # of 12 servers
    assert len(sites) >= 5  # of 6 sites


# ------------------------------------------------------------------ mixed family


def test_mixed_scenarios_follow_the_rules() -> None:
    counts = set()
    for seed in SEEDS:
        faults = generate("mixed", seed, TOPO).faults
        counts.add(len(faults))
        assert 2 <= len(faults) <= 3
        assert len({f.kind for f in faults}) == len(faults)

        onsets = [f.onset for f in faults]
        assert onsets == sorted(onsets)
        assert MIXED_FIRST_ONSET_RANGE[0] <= onsets[0] <= MIXED_FIRST_ONSET_RANGE[1]
        for earlier, later in itertools.pairwise(onsets):
            assert MIXED_GAP_RANGE[0] <= later - earlier <= MIXED_GAP_RANGE[1]

        # All windows overlap: every fault is active when the last one starts.
        assert all(f.active_at(onsets[-1]) for f in faults)

        server_targets = [f.target for f in faults if f.kind in SERVER_KINDS]
        assert len(server_targets) == len(set(server_targets))

        for fault in faults:
            _check_fault_ranges(fault)
    assert counts == {2, 3}


# ------------------------------------------------------------------- bad inputs


def test_unknown_family_is_rejected_with_the_valid_list() -> None:
    with pytest.raises(ValueError, match="valid families"):
        generate("earthquake", 0, TOPO)


def test_running_out_of_targets_is_a_clear_error() -> None:
    every_server = frozenset(TOPO.servers)
    with pytest.raises(ValueError, match="no target left"):
        _draw_fault(FaultKind.CRASH, 400, np.random.default_rng(0), TOPO, every_server)


def test_too_short_horizon_is_rejected() -> None:
    with pytest.raises(ValueError, match="horizon"):
        generate("crash", 0, TOPO, horizon=1600)
    assert generate("crash", 0, TOPO, horizon=1601).horizon == 1601
