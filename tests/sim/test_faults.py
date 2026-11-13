"""Tests for the fault model."""

from __future__ import annotations

import pytest

from sentinelops.sim.faults import (
    EXHAUSTION_MIN_CAPACITY,
    ClearingAction,
    Fault,
    FaultEffects,
    FaultKind,
    clears,
    compute_effects,
    validate_fault,
)
from sentinelops.sim.topology import (
    Region,
    RegionId,
    Server,
    ServerId,
    Site,
    SiteId,
    Topology,
    default_topology,
)

A1, A2, B1 = ServerId("a-1"), ServerId("a-2"), ServerId("b-1")
SITE_A, SITE_B = SiteId("a"), SiteId("b")
REGION = RegionId("r")


def _topology() -> Topology:
    """Two sites (a: 2 servers, b: 1 server) and one region with base demand 100 qps."""
    sites = {
        SITE_A: Site(SITE_A, "A", 0.0, 0.0, (A1, A2), queue_limit=100, timeout_ticks=2),
        SITE_B: Site(SITE_B, "B", 0.0, 10.0, (B1,), queue_limit=100, timeout_ticks=2),
    }
    servers = {
        A1: Server(A1, SITE_A, 10.0),
        A2: Server(A2, SITE_A, 10.0),
        B1: Server(B1, SITE_B, 10.0),
    }
    regions = {REGION: Region(REGION, "R", 0.0, 1.0, 100.0)}
    return Topology(sites, servers, regions, {REGION: (SITE_A, SITE_B)})


TOPO = _topology()


def _fault(
    kind: FaultKind,
    target: str,
    onset: int = 10,
    duration: int | None = None,
    severity: float = 1.0,
) -> Fault:
    return Fault(kind, target, onset, duration, severity)


def _effects(
    *faults: Fault, tick: int = 100, cleared: frozenset[int] = frozenset()
) -> FaultEffects:
    return compute_effects(faults, tick, TOPO, cleared)


# ------------------------------------------------------------------- Fault data


def test_active_at_boundaries_with_duration() -> None:
    fault = _fault(FaultKind.PATH_LOSS, SITE_A, onset=10, duration=5, severity=0.5)
    assert not fault.active_at(9)
    assert fault.active_at(10)
    assert fault.active_at(14)
    assert not fault.active_at(15)


def test_fault_without_duration_stays_active_forever() -> None:
    fault = _fault(FaultKind.CRASH, A1, onset=10)
    assert not fault.active_at(9)
    assert fault.active_at(10)
    assert fault.active_at(1_000_000)


@pytest.mark.parametrize("severity", [0.0, -0.1, 1.5, float("nan")])
def test_severity_outside_zero_to_one_is_rejected(severity: float) -> None:
    with pytest.raises(ValueError, match="severity"):
        _fault(FaultKind.CRASH, A1, severity=severity)


def test_severity_of_exactly_one_is_allowed() -> None:
    assert _fault(FaultKind.CRASH, A1, severity=1.0).severity == 1.0


def test_negative_onset_is_rejected() -> None:
    with pytest.raises(ValueError, match="onset"):
        _fault(FaultKind.CRASH, A1, onset=-1)


@pytest.mark.parametrize("duration", [0, -5])
def test_non_positive_duration_is_rejected(duration: int) -> None:
    with pytest.raises(ValueError, match="duration"):
        _fault(FaultKind.FLOOD, REGION, duration=duration)


def test_empty_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="target"):
        _fault(FaultKind.CRASH, "")


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="bogus"):
        Fault("bogus", "ott-1", 0, None, 1.0)  # type: ignore[arg-type]


def test_kind_given_as_string_becomes_the_enum() -> None:
    fault = Fault("crash", "ott-1", 0, None, 1.0)  # type: ignore[arg-type]
    assert fault.kind is FaultKind.CRASH


# --------------------------------------------------------------- FaultEffects


def test_fault_effects_mappings_are_read_only() -> None:
    effects = _effects(_fault(FaultKind.PATH_LOSS, SITE_A, severity=0.5))
    with pytest.raises(TypeError):
        effects.site_path_loss[SITE_A] = 0.0  # type: ignore[index]


def test_fault_effects_copies_the_mappings_it_is_given() -> None:
    loss = {SITE_A: 0.5}
    effects = FaultEffects(site_path_loss=loss)
    loss[SITE_A] = 0.9
    assert effects.site_path_loss[SITE_A] == 0.5


# -------------------------------------------------------------- validate_fault


def test_validate_rejects_server_fault_on_a_site() -> None:
    with pytest.raises(ValueError, match="not a server"):
        validate_fault(_fault(FaultKind.CRASH, SITE_A), TOPO)


def test_validate_rejects_region_fault_on_a_server() -> None:
    with pytest.raises(ValueError, match="not a region"):
        validate_fault(_fault(FaultKind.FLOOD, A1), TOPO)


def test_validate_rejects_site_fault_on_unknown_site() -> None:
    with pytest.raises(ValueError, match="not a site"):
        validate_fault(_fault(FaultKind.PATH_LOSS, "nowhere", severity=0.3), TOPO)


@pytest.mark.parametrize(
    ("kind", "target"),
    [
        (FaultKind.CRASH, A1),
        (FaultKind.EXHAUSTION, A2),
        (FaultKind.HARDWARE, B1),
        (FaultKind.PATH_LOSS, SITE_A),
        (FaultKind.SITE_OUTAGE, SITE_B),
        (FaultKind.FLOOD, REGION),
        (FaultKind.BENIGN_SURGE, REGION),
    ],
)
def test_validate_accepts_every_kind_on_its_own_target_type(kind: FaultKind, target: str) -> None:
    validate_fault(_fault(kind, target), TOPO)


def test_validate_works_on_the_default_topology() -> None:
    topo = default_topology()
    validate_fault(_fault(FaultKind.CRASH, "ott-1"), topo)
    validate_fault(_fault(FaultKind.SITE_OUTAGE, "sin"), topo)
    validate_fault(_fault(FaultKind.FLOOD, "eu-west"), topo)


# ------------------------------------------------------------- compute_effects


def test_crash_takes_only_its_server_down() -> None:
    effects = _effects(_fault(FaultKind.CRASH, A1))
    assert effects.server_down == frozenset({A1})


def test_hardware_fault_takes_its_server_down() -> None:
    assert _effects(_fault(FaultKind.HARDWARE, B1)).server_down == frozenset({B1})


def test_exhaustion_halfway_through_the_ramp() -> None:
    effects = _effects(_fault(FaultKind.EXHAUSTION, A1, onset=10), tick=310)
    assert effects.server_capacity_factor[A1] == pytest.approx(0.5)
    assert effects.server_memory_pressure[A1] == pytest.approx(0.5)
    assert A1 not in effects.server_down


def test_exhaustion_bottoms_out_at_minimum_capacity_and_full_memory() -> None:
    effects = _effects(_fault(FaultKind.EXHAUSTION, A1, onset=10), tick=10_010)
    assert effects.server_capacity_factor[A1] == pytest.approx(EXHAUSTION_MIN_CAPACITY)
    assert effects.server_memory_pressure[A1] == pytest.approx(1.0)


def test_exhaustion_ramp_scales_with_severity() -> None:
    effects = _effects(_fault(FaultKind.EXHAUSTION, A1, onset=10, severity=0.5), tick=310)
    assert effects.server_capacity_factor[A1] == pytest.approx(0.75)
    assert effects.server_memory_pressure[A1] == pytest.approx(0.25)


def test_site_outage_takes_all_site_servers_down_and_loses_all_traffic() -> None:
    effects = _effects(_fault(FaultKind.SITE_OUTAGE, SITE_A))
    assert effects.server_down == frozenset({A1, A2})
    assert effects.site_path_loss[SITE_A] == 1.0
    assert SITE_B not in effects.site_path_loss


def test_two_path_losses_on_one_site_combine_as_independent_drops() -> None:
    effects = _effects(
        _fault(FaultKind.PATH_LOSS, SITE_A, severity=0.2),
        _fault(FaultKind.PATH_LOSS, SITE_A, severity=0.5),
    )
    assert effects.site_path_loss[SITE_A] == pytest.approx(0.6)


def test_two_exhaustions_on_one_server_multiply_capacity_and_take_max_memory() -> None:
    effects = _effects(
        _fault(FaultKind.EXHAUSTION, A1, onset=10, severity=1.0),
        _fault(FaultKind.EXHAUSTION, A1, onset=10, severity=0.5),
        tick=310,
    )
    assert effects.server_capacity_factor[A1] == pytest.approx(0.5 * 0.75)
    assert effects.server_memory_pressure[A1] == pytest.approx(0.5)


def test_flood_adds_attack_traffic_from_its_region() -> None:
    effects = _effects(_fault(FaultKind.FLOOD, REGION, severity=0.5))
    assert effects.region_extra_attack_qps[REGION] == pytest.approx(400.0)
    assert REGION not in effects.region_extra_legit_qps


def test_benign_surge_adds_legitimate_traffic_from_its_region() -> None:
    effects = _effects(_fault(FaultKind.BENIGN_SURGE, REGION, severity=0.5))
    assert effects.region_extra_legit_qps[REGION] == pytest.approx(200.0)
    assert REGION not in effects.region_extra_attack_qps


def test_overlapping_floods_add_up() -> None:
    effects = _effects(
        _fault(FaultKind.FLOOD, REGION, severity=0.5),
        _fault(FaultKind.FLOOD, REGION, severity=0.25),
    )
    assert effects.region_extra_attack_qps[REGION] == pytest.approx(600.0)


def test_cleared_fault_has_no_effect() -> None:
    effects = _effects(
        _fault(FaultKind.CRASH, A1), _fault(FaultKind.CRASH, B1), cleared=frozenset({0})
    )
    assert effects.server_down == frozenset({B1})


def test_fault_before_onset_has_no_effect() -> None:
    assert _effects(_fault(FaultKind.CRASH, A1, onset=500), tick=499) == FaultEffects()


def test_fault_after_its_duration_has_no_effect() -> None:
    fault = _fault(FaultKind.PATH_LOSS, SITE_A, onset=10, duration=5, severity=0.5)
    assert _effects(fault, tick=15) == FaultEffects()


def test_no_faults_gives_empty_effects() -> None:
    effects = _effects()
    assert effects == FaultEffects()
    assert effects.server_down == frozenset()
    assert dict(effects.site_path_loss) == {}


def test_compute_effects_is_pure() -> None:
    faults = [
        _fault(FaultKind.EXHAUSTION, A1, onset=10),
        _fault(FaultKind.PATH_LOSS, SITE_B, severity=0.3),
        _fault(FaultKind.FLOOD, REGION, severity=0.7, duration=200),
    ]
    snapshot = list(faults)
    cleared: set[int] = set()
    first = compute_effects(faults, 150, TOPO, cleared)
    second = compute_effects(faults, 150, TOPO, cleared)
    assert first == second
    assert faults == snapshot
    assert cleared == set()


# ---------------------------------------------------------------------- clears


def test_restart_clears_crash_and_exhaustion_on_the_same_server_only() -> None:
    for kind in (FaultKind.CRASH, FaultKind.EXHAUSTION):
        fault = _fault(kind, A1)
        assert clears(fault, ClearingAction.RESTART, A1, TOPO)
        assert not clears(fault, ClearingAction.RESTART, A2, TOPO)


def test_restart_does_not_clear_hardware_fault() -> None:
    assert not clears(_fault(FaultKind.HARDWARE, A1), ClearingAction.RESTART, A1, TOPO)


def test_repair_clears_hardware_fault_on_the_same_server() -> None:
    fault = _fault(FaultKind.HARDWARE, A1)
    assert clears(fault, ClearingAction.REPAIR, A1, TOPO)
    assert not clears(fault, ClearingAction.REPAIR, A2, TOPO)


def test_repair_of_the_site_clears_a_site_outage() -> None:
    fault = _fault(FaultKind.SITE_OUTAGE, SITE_A)
    assert clears(fault, ClearingAction.REPAIR, SITE_A, TOPO)
    assert not clears(fault, ClearingAction.REPAIR, A1, TOPO)
    assert not clears(fault, ClearingAction.RESTART, SITE_A, TOPO)


@pytest.mark.parametrize(
    ("kind", "target"),
    [
        (FaultKind.PATH_LOSS, SITE_A),
        (FaultKind.FLOOD, REGION),
        (FaultKind.BENIGN_SURGE, REGION),
    ],
)
def test_nothing_clears_path_loss_flood_or_surge(kind: FaultKind, target: str) -> None:
    fault = _fault(kind, target, severity=0.5)
    for action in ClearingAction:
        assert not clears(fault, action, target, TOPO)
