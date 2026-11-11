"""Static description of the simulated anycast DNS network.

A :class:`Topology` lists the anycast sites, the DNS servers at each site, the
client regions that generate queries, and each region's routing preferences: the
order in which it would reach the sites if they were all announcing the service
address. Preferences follow great-circle distance. That is a simplification: real
BGP catchments are not purely geographic, but distance gives a stable, explainable
ordering that is enough to study how traffic shifts when a site is withdrawn.

Everything here is immutable. Dynamic state (which sites are announcing, server
health, queues) lives in the simulation engine.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import NewType

SiteId = NewType("SiteId", str)
ServerId = NewType("ServerId", str)
RegionId = NewType("RegionId", str)

EARTH_RADIUS_KM = 6371.0088
"""Mean Earth radius (IUGG), used for great-circle distances."""


def _check_coordinates(kind: str, ident: str, lat: float, lon: float) -> None:
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f"{kind} {ident!r} has invalid coordinates ({lat}, {lon})")


@dataclass(frozen=True, slots=True)
class Server:
    """One DNS server process at a site.

    Attributes:
        id: Server identifier, conventionally ``"<site>-<n>"``.
        site: Site that hosts the server.
        capacity_qps: Maximum queries answered per tick (per second) when healthy.
    """

    id: ServerId
    site: SiteId
    capacity_qps: float

    def __post_init__(self) -> None:
        """Reject a non-positive capacity."""
        if not self.capacity_qps > 0:
            raise ValueError(f"server {self.id!r} needs a positive capacity_qps")


@dataclass(frozen=True, slots=True)
class Site:
    """An anycast site: a location that announces the service address.

    Attributes:
        id: Short site identifier, e.g. ``"ott"``.
        name: Human-readable city name.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        servers: Servers behind the site's load balancer.
        queue_limit: Maximum number of queries waiting at the site; arrivals
            beyond it are dropped (overflow).
        timeout_ticks: A query that has waited longer than this many ticks
            fails (timeout).
    """

    id: SiteId
    name: str
    lat: float
    lon: float
    servers: tuple[ServerId, ...]
    queue_limit: int
    timeout_ticks: int

    def __post_init__(self) -> None:
        """Reject impossible coordinates, servers, queue limits or timeouts."""
        _check_coordinates("site", self.id, self.lat, self.lon)
        if not self.servers:
            raise ValueError(f"site {self.id!r} needs at least one server")
        if len(set(self.servers)) != len(self.servers):
            raise ValueError(f"site {self.id!r} lists a server twice")
        if self.queue_limit < 0:
            raise ValueError(f"site {self.id!r} needs a non-negative queue_limit")
        if self.timeout_ticks < 1:
            raise ValueError(f"site {self.id!r} needs timeout_ticks >= 1")


@dataclass(frozen=True, slots=True)
class Region:
    """A population of clients that sends DNS queries.

    Attributes:
        id: Region identifier, e.g. ``"na-east"``.
        name: Human-readable name of the region's main city.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        base_qps: Mean legitimate demand in queries per tick.
    """

    id: RegionId
    name: str
    lat: float
    lon: float
    base_qps: float

    def __post_init__(self) -> None:
        """Reject impossible coordinates or negative demand."""
        _check_coordinates("region", self.id, self.lat, self.lon)
        if self.base_qps < 0:
            raise ValueError(f"region {self.id!r} needs a non-negative base_qps")


@dataclass(frozen=True, slots=True)
class Topology:
    """The whole network: sites, servers, client regions and routing preferences.

    The constructor checks that the parts are consistent (ids match their keys,
    every server belongs to the site that lists it, every region ranks every
    site exactly once) and stores read-only views of the mappings.

    Attributes:
        sites: Sites by id.
        servers: Servers by id.
        regions: Client regions by id.
        preferences: For each region, every site ordered from most to least
            preferred. Traffic goes to the first announcing site.
    """

    sites: Mapping[SiteId, Site]
    servers: Mapping[ServerId, Server]
    regions: Mapping[RegionId, Region]
    preferences: Mapping[RegionId, tuple[SiteId, ...]]

    def __post_init__(self) -> None:
        """Freeze the mappings and check the parts fit together."""
        for name in ("sites", "servers", "regions", "preferences"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        self._validate()

    def _validate(self) -> None:
        if not self.sites:
            raise ValueError("topology needs at least one site")
        keyed: list[tuple[str, str]] = [
            *((k, v.id) for k, v in self.sites.items()),
            *((k, v.id) for k, v in self.servers.items()),
            *((k, v.id) for k, v in self.regions.items()),
        ]
        for key, ident in keyed:
            if key != ident:
                raise ValueError(f"key {key!r} does not match id {ident!r}")
        for server in self.servers.values():
            if server.site not in self.sites:
                raise ValueError(f"server {server.id!r} is at unknown site {server.site!r}")
            if server.id not in self.sites[server.site].servers:
                raise ValueError(f"site {server.site!r} does not list server {server.id!r}")
        for site in self.sites.values():
            for server_id in site.servers:
                if server_id not in self.servers:
                    raise ValueError(f"site {site.id!r} lists unknown server {server_id!r}")
        if set(self.preferences) != set(self.regions):
            raise ValueError("preferences must cover exactly the topology's regions")
        for region_id, order in self.preferences.items():
            if sorted(order) != sorted(self.sites):
                raise ValueError(f"preferences for {region_id!r} must rank every site once")

    def site_of(self, server: ServerId) -> SiteId:
        """Return the site hosting ``server``.

        Raises:
            KeyError: If the server does not exist.
        """
        return self.servers[server].site

    def servers_at(self, site: SiteId) -> tuple[ServerId, ...]:
        """Return the servers at ``site``, in the site's own order.

        Raises:
            KeyError: If the site does not exist.
        """
        return self.sites[site].servers


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance between two points, in kilometres.

    Uses the haversine formula, which stays accurate for short distances.
    """
    phi1, lam1, phi2, lam2 = map(math.radians, (lat1, lon1, lat2, lon2))
    h = (
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin((lam2 - lam1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def preferences_by_distance(
    sites: Iterable[Site], regions: Iterable[Region]
) -> dict[RegionId, tuple[SiteId, ...]]:
    """Rank all sites for each region, nearest first.

    Ties in distance are broken by site id so the order is always deterministic.
    """
    site_list = list(sites)
    return {
        region.id: tuple(
            site.id
            for site in sorted(
                site_list,
                key=lambda s: (great_circle_km(region.lat, region.lon, s.lat, s.lon), s.id),
            )
        )
        for region in regions
    }


# Default network. Capacities and demand are chosen so every site runs at
# 50-65% utilisation when healthy, while withdrawing some sites (e.g. yyz, whose
# catchment falls back on ott) overloads a neighbour. That makes "just withdraw
# the broken site" a genuine trade-off rather than a free fix.
_SERVERS_PER_SITE = 2
_DEFAULT_TIMEOUT_TICKS = 3

# id, city, lat, lon, per-server capacity (qps)
_SITES: tuple[tuple[str, str, float, float, float], ...] = (
    ("ott", "Ottawa", 45.4215, -75.6972, 2000.0),
    ("yyz", "Toronto", 43.6532, -79.3832, 3000.0),
    ("lon", "London", 51.5074, -0.1278, 2500.0),
    ("fra", "Frankfurt", 50.1109, 8.6821, 3000.0),
    ("par", "Paris", 48.8566, 2.3522, 2000.0),
    ("sin", "Singapore", 1.3521, 103.8198, 2500.0),
)

# id, city, lat, lon, mean legitimate demand (qps)
_REGIONS: tuple[tuple[str, str, float, float, float], ...] = (
    ("na-east", "New York", 40.7128, -74.0060, 2400.0),
    ("na-central", "Chicago", 41.8781, -87.6298, 1700.0),
    ("na-west", "San Francisco", 37.7749, -122.4194, 1500.0),
    ("sa-east", "Sao Paulo", -23.5505, -46.6333, 700.0),
    ("eu-west", "Dublin", 53.3498, -6.2603, 2600.0),
    ("eu-south", "Madrid", 40.4168, -3.7038, 2200.0),
    ("eu-central", "Berlin", 52.5200, 13.4050, 2200.0),
    ("me-central", "Dubai", 25.2048, 55.2708, 900.0),
    ("ap-south", "Mumbai", 19.0760, 72.8777, 1100.0),
    ("ap-east", "Tokyo", 35.6762, 139.6503, 1400.0),
    ("oc-east", "Sydney", -33.8688, 151.2093, 600.0),
)


@cache
def default_topology() -> Topology:
    """Return the six-site demo network used throughout the project.

    Each site has two equal servers; its queue holds up to the site's capacity
    times the timeout, so a full queue is roughly as many queries as the site
    can drain before they time out. The result is immutable and cached.
    """
    sites: dict[SiteId, Site] = {}
    servers: dict[ServerId, Server] = {}
    for raw_id, city, lat, lon, capacity in _SITES:
        site_id = SiteId(raw_id)
        ids = tuple(ServerId(f"{raw_id}-{n}") for n in range(1, _SERVERS_PER_SITE + 1))
        for server_id in ids:
            servers[server_id] = Server(server_id, site_id, capacity)
        site_capacity = capacity * _SERVERS_PER_SITE
        sites[site_id] = Site(
            id=site_id,
            name=city,
            lat=lat,
            lon=lon,
            servers=ids,
            queue_limit=round(site_capacity * _DEFAULT_TIMEOUT_TICKS),
            timeout_ticks=_DEFAULT_TIMEOUT_TICKS,
        )
    regions = {
        RegionId(raw_id): Region(RegionId(raw_id), city, lat, lon, qps)
        for raw_id, city, lat, lon, qps in _REGIONS
    }
    return Topology(
        sites=sites,
        servers=servers,
        regions=regions,
        preferences=preferences_by_distance(sites.values(), regions.values()),
    )
