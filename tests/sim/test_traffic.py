"""Tests for demand generation and anycast routing."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sentinelops.sim.topology import RegionId, SiteId, default_topology
from sentinelops.sim.traffic import DemandTrace, catchment_shares, generate_demand, route

TOPO = default_topology()
SITES = tuple(TOPO.sites)
REGIONS = tuple(TOPO.regions)


# ----------------------------------------------------------------------- demand


def _demand(seed: int, horizon: int = 600, **kwargs: float) -> DemandTrace:
    return generate_demand(TOPO, horizon, np.random.default_rng(seed), **kwargs)


def test_same_seed_gives_identical_demand() -> None:
    np.testing.assert_array_equal(_demand(7).qps, _demand(7).qps)
    assert not np.array_equal(_demand(7).qps, _demand(8).qps)


def test_demand_shape_order_and_positivity() -> None:
    trace = _demand(0, horizon=50)
    assert trace.regions == REGIONS
    assert trace.qps.shape == (50, len(REGIONS))
    assert trace.horizon == 50
    assert (trace.qps > 0).all()


def test_demand_is_read_only() -> None:
    trace = _demand(0, horizon=5)
    with pytest.raises(ValueError):
        trace.qps[0, 0] = 1.0


def test_trace_does_not_alias_caller_array() -> None:
    source = np.ones((2, len(REGIONS)))
    trace = DemandTrace(REGIONS, source)
    source[0, 0] = 99.0
    assert trace.qps[0, 0] == 1.0


def test_long_run_mean_matches_base_qps() -> None:
    trace = _demand(1, horizon=40_000)
    base = np.array([TOPO.regions[r].base_qps for r in REGIONS])
    np.testing.assert_allclose(trace.qps.mean(axis=0) / base, 1.0, atol=0.03)


def test_noise_only_has_requested_spread_and_no_memory() -> None:
    trace = _demand(2, horizon=20_000, noise_cv=0.1, drift_cv=0.0)
    ratio = trace.qps / np.array([TOPO.regions[r].base_qps for r in REGIONS])
    assert ratio.std(axis=0) == pytest.approx(np.full(len(REGIONS), 0.1), abs=0.01)
    lag1 = np.corrcoef(ratio[:-1, 0], ratio[1:, 0])[0, 1]
    assert abs(lag1) < 0.05


def test_drift_is_slow() -> None:
    trace = _demand(3, horizon=5_000, noise_cv=0.0, drift_cv=0.1, drift_ticks=120)
    series = trace.qps[:, 0]
    lag1 = np.corrcoef(series[:-1], series[1:])[0, 1]
    assert lag1 > 0.98


def test_zero_variation_reproduces_base_exactly() -> None:
    trace = _demand(4, horizon=10, noise_cv=0.0, drift_cv=0.0)
    for region, qps in trace.at(9).items():
        assert qps == pytest.approx(TOPO.regions[region].base_qps)


def test_at_rejects_ticks_outside_the_trace() -> None:
    trace = _demand(0, horizon=3)
    assert set(trace.at(2)) == set(REGIONS)
    for tick in (-1, 3):
        with pytest.raises(IndexError):
            trace.at(tick)


@pytest.mark.parametrize(
    "kwargs",
    [{"horizon": 0}, {"noise_cv": -0.1}, {"drift_cv": -0.1}, {"drift_ticks": 0.0}],
)
def test_generate_demand_rejects_bad_arguments(kwargs: dict[str, float]) -> None:
    params: dict[str, float] = {"horizon": 10, **kwargs}
    horizon = int(params.pop("horizon"))
    with pytest.raises(ValueError):
        generate_demand(TOPO, horizon, np.random.default_rng(0), **params)


def test_trace_rejects_mismatched_shape() -> None:
    with pytest.raises(ValueError):
        DemandTrace(REGIONS, np.ones((3, 2)))


# ---------------------------------------------------------------------- routing

A, B, C = SiteId("a"), SiteId("b"), SiteId("c")


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ({}, {A: 1.0}),
        ({A: 0.0}, {B: 1.0}),
        ({A: 0.0, B: 0.0}, {C: 1.0}),
        ({A: 0.25}, {A: 0.25, B: 0.75}),
        ({A: 0.0, B: 0.0, C: 0.0}, {}),
        # Every site partial: the residual 1/8 is spread proportionally.
        ({A: 0.5, B: 0.5, C: 0.5}, {A: 4 / 7, B: 2 / 7, C: 1 / 7}),
    ],
)
def test_catchment_shares(weights: dict[SiteId, float], expected: dict[SiteId, float]) -> None:
    shares = catchment_shares((A, B, C), weights)
    assert shares.keys() == expected.keys()
    for site, share in expected.items():
        assert shares[site] == pytest.approx(share)


def test_healthy_network_sends_each_region_to_its_nearest_site() -> None:
    region = RegionId("eu-central")
    result = route(TOPO, {region: 100.0})
    assert result.site_qps[SiteId("fra")] == 100.0
    assert sum(result.site_qps.values()) == 100.0
    assert result.unrouted_qps == 0.0
    assert set(result.site_qps) == set(SITES)


def test_withdrawal_moves_the_whole_catchment_to_the_next_site() -> None:
    region = RegionId("eu-central")
    second = TOPO.preferences[region][1]
    result = route(TOPO, {region: 100.0}, {SiteId("fra"): 0.0})
    assert result.site_qps[SiteId("fra")] == 0.0
    assert result.site_qps[second] == 100.0


def _utilisation(withdrawn: str | None = None) -> dict[SiteId, float]:
    topo = TOPO
    demand = {r.id: r.base_qps for r in topo.regions.values()}
    weights = {SiteId(withdrawn): 0.0} if withdrawn else {}
    arrivals = route(topo, demand, weights).site_qps
    return {
        s: arrivals[s] / sum(topo.servers[v].capacity_qps for v in topo.servers_at(s))
        for s in topo.sites
    }


def test_healthy_sites_have_headroom() -> None:
    for site, util in _utilisation().items():
        assert 0.45 <= util <= 0.70, site


def test_withdrawal_can_overload_a_neighbour_but_not_always() -> None:
    assert _utilisation(withdrawn="yyz")[SiteId("ott")] > 1.0
    assert max(_utilisation(withdrawn="par").values()) < 1.0


def test_all_sites_withdrawn_leaves_traffic_unrouted() -> None:
    result = route(TOPO, {RegionId("na-east"): 50.0}, dict.fromkeys(SITES, 0.0))
    assert result.unrouted_qps == 50.0
    assert sum(result.site_qps.values()) == 0.0


@given(
    demand=st.lists(st.floats(0, 1e5), min_size=len(REGIONS), max_size=len(REGIONS)),
    weights=st.lists(
        st.sampled_from([0.0, 0.1, 0.25, 0.5, 0.9, 1.0]), min_size=len(SITES), max_size=len(SITES)
    ),
)
def test_routing_conserves_traffic(demand: list[float], weights: list[float]) -> None:
    offered = dict(zip(REGIONS, demand, strict=True))
    result = route(TOPO, offered, dict(zip(SITES, weights, strict=True)))
    delivered = sum(result.site_qps.values()) + result.unrouted_qps
    assert delivered == pytest.approx(sum(demand), rel=1e-9, abs=1e-6)
    assert all(qps >= 0 for qps in result.site_qps.values())
    assert result.unrouted_qps == 0.0 or not any(weights)


@pytest.mark.parametrize(
    ("demand", "weights", "error"),
    [
        ({RegionId("mars"): 1.0}, {}, KeyError),
        ({}, {SiteId("zzz"): 1.0}, KeyError),
        ({}, {SiteId("ott"): 1.5}, ValueError),
        ({}, {SiteId("ott"): -0.1}, ValueError),
        ({RegionId("na-east"): -1.0}, {}, ValueError),
    ],
)
def test_route_rejects_bad_input(
    demand: dict[RegionId, float], weights: dict[SiteId, float], error: type[Exception]
) -> None:
    with pytest.raises(error):
        route(TOPO, demand, weights)
