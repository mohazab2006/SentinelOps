"""Tests for the static network description."""

from __future__ import annotations

import dataclasses
import math

import pytest

from sentinelops.sim.topology import (
    EARTH_RADIUS_KM,
    Region,
    RegionId,
    Server,
    ServerId,
    Site,
    SiteId,
    Topology,
    default_topology,
    great_circle_km,
    preferences_by_distance,
)

SITE_IDS = ("ott", "yyz", "lon", "fra", "par", "sin")


def _site(ident: str, *servers: str, lat: float = 0.0, lon: float = 0.0) -> Site:
    return Site(
        id=SiteId(ident),
        name=ident.upper(),
        lat=lat,
        lon=lon,
        servers=tuple(ServerId(s) for s in servers),
        queue_limit=100,
        timeout_ticks=2,
    )


def _tiny(**overrides: object) -> Topology:
    """A valid two-site, one-region topology, with optional field overrides."""
    a, b = SiteId("a"), SiteId("b")
    fields: dict[str, object] = {
        "sites": {a: _site("a", "a-1"), b: _site("b", "b-1", lon=10.0)},
        "servers": {
            ServerId("a-1"): Server(ServerId("a-1"), a, 10.0),
            ServerId("b-1"): Server(ServerId("b-1"), b, 10.0),
        },
        "regions": {RegionId("r"): Region(RegionId("r"), "R", 0.0, 1.0, 5.0)},
        "preferences": {RegionId("r"): (a, b)},
    }
    fields.update(overrides)
    return Topology(**fields)  # type: ignore[arg-type]


# ----------------------------------------------------------------- default network


def test_default_topology_has_the_six_documented_sites() -> None:
    topo = default_topology()
    assert tuple(topo.sites) == SITE_IDS
    assert topo.sites[SiteId("ott")].name == "Ottawa"
    for site in topo.sites.values():
        assert site.servers == (ServerId(f"{site.id}-1"), ServerId(f"{site.id}-2"))
    assert len(topo.servers) == 12


def test_server_lookups() -> None:
    topo = default_topology()
    assert topo.site_of(ServerId("lon-2")) == "lon"
    assert topo.servers_at(SiteId("sin")) == ("sin-1", "sin-2")
    with pytest.raises(KeyError):
        topo.site_of(ServerId("nope-1"))


def test_every_region_ranks_every_site_once() -> None:
    topo = default_topology()
    for order in topo.preferences.values():
        assert sorted(order) == sorted(SITE_IDS)


@pytest.mark.parametrize(
    ("region", "nearest"),
    [
        ("na-east", "ott"),
        ("na-central", "yyz"),
        ("eu-west", "lon"),
        ("eu-south", "par"),
        ("eu-central", "fra"),
        ("ap-east", "sin"),
    ],
)
def test_regions_prefer_their_nearest_site(region: str, nearest: str) -> None:
    assert default_topology().preferences[RegionId(region)][0] == nearest


def test_default_topology_is_cached_and_read_only() -> None:
    topo = default_topology()
    assert default_topology() is topo
    with pytest.raises(TypeError):
        topo.sites[SiteId("new")] = topo.sites[SiteId("ott")]  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        topo.sites = {}  # type: ignore[misc]


def test_queue_holds_about_one_timeout_of_work() -> None:
    topo = default_topology()
    for site in topo.sites.values():
        capacity = sum(topo.servers[s].capacity_qps for s in site.servers)
        assert site.queue_limit == round(capacity * site.timeout_ticks)


# -------------------------------------------------------------------- distances


def test_great_circle_matches_known_distance() -> None:
    # London to Paris is about 344 km.
    assert great_circle_km(51.5074, -0.1278, 48.8566, 2.3522) == pytest.approx(343.5, abs=1.0)


def test_great_circle_basic_properties() -> None:
    assert great_circle_km(10.0, 20.0, 10.0, 20.0) == 0.0
    assert great_circle_km(1.0, 2.0, 3.0, 4.0) == great_circle_km(3.0, 4.0, 1.0, 2.0)
    assert great_circle_km(0.0, 0.0, 0.0, 180.0) == pytest.approx(math.pi * EARTH_RADIUS_KM)


def test_preference_ties_break_by_site_id() -> None:
    sites = [_site("b", "b-1", lon=1.0), _site("a", "a-1", lon=-1.0)]
    region = Region(RegionId("r"), "R", 0.0, 0.0, 1.0)
    assert preferences_by_distance(sites, [region]) == {"r": ("a", "b")}


# ------------------------------------------------------------------- validation


def test_tiny_topology_is_valid() -> None:
    assert _tiny().site_of(ServerId("b-1")) == "b"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sites": {}}, "at least one site"),
        ({"regions": {RegionId("x"): Region(RegionId("r"), "R", 0, 0, 1)}}, "does not match"),
        (
            {"servers": {ServerId("a-1"): Server(ServerId("a-1"), SiteId("a"), 1.0)}},
            "unknown server",
        ),
        (
            {
                "servers": {
                    ServerId("a-1"): Server(ServerId("a-1"), SiteId("zz"), 1.0),
                    ServerId("b-1"): Server(ServerId("b-1"), SiteId("b"), 1.0),
                }
            },
            "unknown site",
        ),
        (
            {
                "servers": {
                    ServerId("a-1"): Server(ServerId("a-1"), SiteId("b"), 1.0),
                    ServerId("b-1"): Server(ServerId("b-1"), SiteId("b"), 1.0),
                }
            },
            "does not list",
        ),
        ({"preferences": {}}, "cover exactly"),
        ({"preferences": {RegionId("r"): (SiteId("a"),)}}, "every site once"),
    ],
)
def test_inconsistent_topology_is_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _tiny(**overrides)


@pytest.mark.parametrize(
    "build",
    [
        lambda: Server(ServerId("a-1"), SiteId("a"), 0.0),
        lambda: _site("a"),
        lambda: _site("a", "a-1", "a-1"),
        lambda: _site("a", "a-1", lat=91.0),
        lambda: dataclasses.replace(_site("a", "a-1"), queue_limit=-1),
        lambda: dataclasses.replace(_site("a", "a-1"), timeout_ticks=0),
        lambda: Region(RegionId("r"), "R", 0.0, 181.0, 1.0),
        lambda: Region(RegionId("r"), "R", 0.0, 0.0, -1.0),
    ],
)
def test_invalid_parts_are_rejected(build: object) -> None:
    with pytest.raises(ValueError):
        build()  # type: ignore[operator]
