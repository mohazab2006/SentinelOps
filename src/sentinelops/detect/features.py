"""Sliding-window features: summarise each site's recent telemetry as numbers.

A single noisy reading says little; the last 30 ticks say much more. For every
site and every tick (once the warm-up is over) this module computes seven
per-tick metrics and summarises each over the window with three statistics:

* **mean** - the level ("is latency high?"),
* **std** - the instability ("is the queue jumping around?"),
* **slope** - the trend per tick ("is memory climbing?").

That gives 7 x 3 = 21 features per site per tick: the input of the anomaly
detector.

Traffic is divided by each site's *normal* traffic, measured during the
healthy warm-up, so ``qps_rel = 1.0`` means "normal" at every site and one
detector can serve all of them. Missing readings (``None``) are forward-filled
with the last value seen for that site, so no feature is ever NaN.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sentinelops.sim.telemetry import Observation, SiteTelemetry
from sentinelops.sim.topology import SiteId, Topology

FloatArray = npt.NDArray[np.float64]

WINDOW_TICKS = 30
"""Length of the sliding window, in ticks."""

WARMUP_TICKS = 120
"""Ticks 0-119 are healthy; each site's normal traffic is measured over them."""

METRICS: tuple[str, ...] = (
    "qps_rel",
    "latency_ms",
    "loss",
    "queue_frac",
    "fail_ratio",
    "cpu_max",
    "mem_max",
)
"""Per-tick metrics of a site, in feature order."""

STATS: tuple[str, ...] = ("mean", "std", "slope")
"""Statistics computed over the window for each metric, in feature order."""

FEATURE_NAMES: tuple[str, ...] = tuple(f"{m}_{s}" for m in METRICS for s in STATS)
"""The 21 feature names, e.g. ``"qps_rel_mean"``, in the order of ``FeatureVector.values``."""

_QPS = METRICS.index("qps_rel")


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """The features of one site at one tick.

    Attributes:
        site: The site.
        tick: The last tick of the window.
        values: The 21 features in :data:`FEATURE_NAMES` order. Read-only and
            never NaN.
    """

    site: SiteId
    tick: int
    values: FloatArray


def _slope(window: FloatArray) -> FloatArray:
    """Least-squares slope per tick of each column against ``0..n-1``."""
    n = window.shape[0]
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    centred = window - window.mean(axis=0)
    result: FloatArray = (x @ centred) / float(x @ x)
    return result


class WindowFeaturizer:
    """Turns a run's observations, one tick at a time, into feature vectors.

    Feed every observation of a run in order (ticks 0, 1, 2, ...). From tick
    ``warmup_ticks - 1`` on, each call returns one feature vector per site.
    """

    def __init__(
        self,
        topology: Topology,
        window: int = WINDOW_TICKS,
        warmup_ticks: int = WARMUP_TICKS,
    ) -> None:
        """Create a featurizer for ``topology``.

        Args:
            topology: The network; gives each site's servers and queue limit.
            window: Number of ticks summarised by each feature vector.
            warmup_ticks: Length of the healthy warm-up used to measure normal
                traffic. Must be at least ``window`` so the first vector has a
                full window.

        Raises:
            ValueError: If ``window < 2`` or ``warmup_ticks < window``.
        """
        if window < 2:
            raise ValueError(f"window must be at least 2 ticks, got {window}")
        if warmup_ticks < window:
            raise ValueError(f"warmup_ticks ({warmup_ticks}) must be at least window ({window})")
        self.topology = topology
        self.window = window
        self.warmup_ticks = warmup_ticks
        self._sites: tuple[SiteId, ...] = tuple(sorted(topology.sites))
        self.reset()

    def reset(self) -> None:
        """Forget everything, ready for a new run."""
        self._next_tick = 0
        self._rows: dict[SiteId, deque[list[float]]] = {
            site: deque(maxlen=self.window) for site in self._sites
        }
        self._last: dict[SiteId, list[float]] = {site: [0.0] * len(METRICS) for site in self._sites}
        self._qps_sum: dict[SiteId, float] = dict.fromkeys(self._sites, 0.0)
        self._qps_count: dict[SiteId, int] = dict.fromkeys(self._sites, 0)
        self._normal: dict[SiteId, float] | None = None

    def update(self, obs: Observation) -> dict[SiteId, FeatureVector]:
        """Add one tick of telemetry and return the features it completes.

        Returns:
            ``{}`` until ``obs.tick`` reaches ``warmup_ticks - 1``; from then on
            one :class:`FeatureVector` per site, keyed by site id in sorted
            order.

        Raises:
            ValueError: If ticks are skipped or repeated.
        """
        if obs.tick != self._next_tick:
            raise ValueError(f"expected tick {self._next_tick}, got {obs.tick}")
        self._next_tick += 1

        for site in self._sites:
            telemetry = obs.sites.get(site)
            row = self._raw_row(site, telemetry, obs)
            self._rows[site].append(row)
            if obs.tick < self.warmup_ticks and telemetry is not None and telemetry.qps is not None:
                self._qps_sum[site] += telemetry.qps
                self._qps_count[site] += 1

        if obs.tick < self.warmup_ticks - 1:
            return {}
        if self._normal is None:
            self._normal = {site: self._measured_normal(site) for site in self._sites}
        return {site: self._vector(site, obs.tick) for site in self._sites}

    def _measured_normal(self, site: SiteId) -> float:
        count = self._qps_count[site]
        normal = self._qps_sum[site] / count if count else 0.0
        return normal if normal > 0 else 1.0

    def _raw_row(
        self, site: SiteId, telemetry: SiteTelemetry | None, obs: Observation
    ) -> list[float]:
        """Return this tick's metrics for ``site`` (qps still raw), forward-filling gaps."""
        values: list[float | None] = [None] * len(METRICS)
        if telemetry is not None:
            values[0] = telemetry.qps
            values[1] = telemetry.p95_latency_ms
            values[2] = telemetry.loss
            limit = self.topology.sites[site].queue_limit
            if telemetry.queue_depth is not None and limit > 0:
                values[3] = telemetry.queue_depth / limit
            ok, failed = telemetry.ok_replies, telemetry.failed_replies
            if ok is not None and failed is not None:
                total = ok + failed
                values[4] = failed / total if total > 0 else 0.0
        readings = [obs.servers.get(s) for s in self.topology.servers_at(site)]
        cpus = [r.cpu for r in readings if r is not None and r.cpu is not None]
        mems = [r.mem for r in readings if r is not None and r.mem is not None]
        values[5] = max(cpus) if cpus else None
        values[6] = max(mems) if mems else None

        last = self._last[site]
        row = [last[i] if value is None else float(value) for i, value in enumerate(values)]
        self._last[site] = row
        return row

    def _vector(self, site: SiteId, tick: int) -> FeatureVector:
        assert self._normal is not None  # set before the first vector
        window = np.array(self._rows[site], dtype=np.float64)
        window[:, _QPS] /= self._normal[site]
        stats = np.stack([window.mean(axis=0), window.std(axis=0), _slope(window)], axis=1)
        values = stats.reshape(-1)  # metric-major: m0 mean, m0 std, m0 slope, m1 mean, ...
        values.flags.writeable = False
        return FeatureVector(site=site, tick=tick, values=values)


def featurize_run(
    topology: Topology,
    observations: Iterable[Observation],
    window: int = WINDOW_TICKS,
    warmup_ticks: int = WARMUP_TICKS,
) -> list[FeatureVector]:
    """Featurize a whole run: every vector in tick order, sites sorted within a tick."""
    featurizer = WindowFeaturizer(topology, window, warmup_ticks)
    vectors: list[FeatureVector] = []
    for obs in observations:
        vectors.extend(featurizer.update(obs).values())
    return vectors


def to_matrix(vectors: Sequence[FeatureVector]) -> FloatArray:
    """Stack feature vectors into an ``(n, 21)`` array (``(0, 21)`` if empty)."""
    if not vectors:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.stack([v.values for v in vectors]).astype(np.float64, copy=False)
