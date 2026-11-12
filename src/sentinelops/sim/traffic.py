"""Legitimate query demand and anycast routing of traffic to sites.

Traffic is modelled as a fluid: each tick every client region produces a number
of queries (a float), and routing splits that number across sites. There are two
parts:

* :func:`generate_demand` draws a whole run's legitimate demand up front from a
  seeded generator, so demand never depends on what the agent does.
* :func:`route` sends one tick of per-region traffic to sites, honouring which
  sites are announcing and at what restore weight.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from sentinelops.sim.topology import RegionId, SiteId, Topology

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class DemandTrace:
    """Legitimate demand for every region over a whole run.

    Attributes:
        regions: Column order of ``qps``.
        qps: Read-only array of shape ``(horizon, len(regions))``; entry
            ``[t, i]`` is the legitimate queries per tick from ``regions[i]`` at
            tick ``t``.
    """

    regions: tuple[RegionId, ...]
    qps: FloatArray

    def __post_init__(self) -> None:
        """Check the shape and store a read-only copy of the array."""
        if self.qps.ndim != 2 or self.qps.shape[1] != len(self.regions):
            raise ValueError("qps must have shape (horizon, len(regions))")
        frozen = self.qps.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "qps", frozen)

    @property
    def horizon(self) -> int:
        """Number of ticks covered by the trace."""
        return int(self.qps.shape[0])

    def at(self, tick: int) -> Mapping[RegionId, float]:
        """Return each region's legitimate demand at ``tick``.

        Raises:
            IndexError: If ``tick`` is outside ``[0, horizon)``.
        """
        if not 0 <= tick < self.horizon:
            raise IndexError(f"tick {tick} outside trace of {self.horizon} ticks")
        row = self.qps[tick]
        return MappingProxyType({region: float(row[i]) for i, region in enumerate(self.regions)})


def generate_demand(
    topology: Topology,
    horizon: int,
    rng: np.random.Generator,
    *,
    noise_cv: float = 0.05,
    drift_cv: float = 0.08,
    drift_ticks: float = 120.0,
) -> DemandTrace:
    """Draw a run's legitimate demand for every region.

    Each region's demand is its ``base_qps`` times two independent,
    mean-one log-normal factors:

    * a slow drift, an AR(1) process in log space with stationary coefficient
      of variation ``drift_cv`` and correlation time ``drift_ticks``, so load
      wanders realistically instead of sitting flat; and
    * per-tick noise with coefficient of variation ``noise_cv``.

    Both are mean-one, so the long-run average equals ``base_qps``. The drift
    starts in its stationary distribution, so there is no warm-up transient.

    Args:
        topology: Network whose regions generate demand.
        horizon: Number of ticks to generate.
        rng: Random generator; the same state always yields the same trace.
        noise_cv: Coefficient of variation of the per-tick noise.
        drift_cv: Stationary coefficient of variation of the slow drift.
        drift_ticks: Correlation time of the drift, in ticks.

    Returns:
        The demand trace, with regions in topology order.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 tick")
    if noise_cv < 0 or drift_cv < 0:
        raise ValueError("coefficients of variation must be non-negative")
    if drift_ticks <= 0:
        raise ValueError("drift_ticks must be positive")

    regions = tuple(topology.regions)
    base = np.array([topology.regions[r].base_qps for r in regions], dtype=np.float64)

    # A log-normal factor exp(X) with X ~ N(-s^2/2, s^2) has mean 1 and
    # coefficient of variation sqrt(exp(s^2) - 1); invert that for s.
    drift_sigma = math.sqrt(math.log1p(drift_cv**2))
    noise_sigma = math.sqrt(math.log1p(noise_cv**2))

    phi = math.exp(-1.0 / drift_ticks)
    innovation = drift_sigma * math.sqrt(1.0 - phi**2)
    log_drift = np.empty((horizon, len(regions)), dtype=np.float64)
    log_drift[0] = rng.normal(0.0, drift_sigma, size=len(regions))
    shocks = rng.normal(0.0, innovation, size=(horizon - 1, len(regions)))
    for t in range(1, horizon):
        log_drift[t] = phi * log_drift[t - 1] + shocks[t - 1]

    log_noise = rng.normal(0.0, noise_sigma, size=(horizon, len(regions)))
    log_factor = log_drift - drift_sigma**2 / 2 + log_noise - noise_sigma**2 / 2
    return DemandTrace(regions=regions, qps=base * np.exp(log_factor))


@dataclass(frozen=True, slots=True)
class RouteResult:
    """Where one tick of traffic went.

    Attributes:
        site_qps: Queries arriving at each site this tick (every site present,
            zero if it received nothing).
        unrouted_qps: Queries with nowhere to go because no site was announcing.
    """

    site_qps: Mapping[SiteId, float]
    unrouted_qps: float


def catchment_shares(
    preferences: tuple[SiteId, ...], weights: Mapping[SiteId, float]
) -> dict[SiteId, float]:
    """Split one region's traffic across sites under anycast routing.

    The region walks its preference list. A site with announcement weight
    ``w`` takes fraction ``w`` of the traffic still unassigned and the rest
    spills to the next site, so weight 1 takes everything that reaches it,
    weight 0 (withdrawn) takes nothing, and a partially restored site takes a
    proportional slice. If some traffic is still unassigned after the last site
    (only possible when every site has weight below 1), it is spread over the
    sites that took traffic, in proportion to what they took.

    Args:
        preferences: The region's sites, most preferred first.
        weights: Announcement weight per site in ``[0, 1]``; missing sites
            count as fully announced (1.0).

    Returns:
        Fraction of the region's traffic per site, summing to 1, or an empty
        dict if no site is announcing.
    """
    shares: dict[SiteId, float] = {}
    remaining = 1.0
    for site in preferences:
        weight = weights.get(site, 1.0)
        if weight > 0.0:
            shares[site] = remaining * weight
            remaining -= shares[site]
            if remaining <= 0.0:
                break
    total = sum(shares.values())
    if total == 0.0:
        return {}
    return {site: share / total for site, share in shares.items()}


def route(
    topology: Topology,
    region_qps: Mapping[RegionId, float],
    weights: Mapping[SiteId, float] | None = None,
) -> RouteResult:
    """Send one tick of per-region traffic to sites.

    Traffic is conserved: the site arrivals plus ``unrouted_qps`` always add up
    to the total offered. Legitimate and attack traffic follow the same routes,
    so callers route each separately with the same weights.

    Args:
        topology: Network providing each region's preferences.
        region_qps: Queries per tick offered by each region; regions not listed
            send nothing.
        weights: Announcement weight per site in ``[0, 1]``: 0 means withdrawn,
            1 fully announced; missing sites count as 1.0. ``None`` means all
            sites are fully announced.

    Returns:
        Arrivals per site plus any traffic that had nowhere to go.

    Raises:
        KeyError: If a region or site is not part of the topology.
        ValueError: If a weight is outside ``[0, 1]`` or a rate is negative.
    """
    weights = weights or {}
    for site, weight in weights.items():
        if site not in topology.sites:
            raise KeyError(f"unknown site {site!r}")
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight for {site!r} must be in [0, 1], got {weight}")

    arrivals = dict.fromkeys(topology.sites, 0.0)
    unrouted = 0.0
    for region, qps in region_qps.items():
        if region not in topology.regions:
            raise KeyError(f"unknown region {region!r}")
        if qps < 0:
            raise ValueError(f"demand for {region!r} must be non-negative, got {qps}")
        shares = catchment_shares(topology.preferences[region], weights)
        if not shares:
            unrouted += qps
        for site, share in shares.items():
            arrivals[site] += qps * share
    return RouteResult(site_qps=MappingProxyType(arrivals), unrouted_qps=unrouted)
