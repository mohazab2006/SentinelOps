"""Tests for the sliding-window telemetry features."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pytest

from sentinelops.agent.base import NoOpAgent
from sentinelops.detect import (
    FEATURE_NAMES,
    FeatureVector,
    WindowFeaturizer,
    featurize_run,
    to_matrix,
)
from sentinelops.detect.features import METRICS, STATS, WARMUP_TICKS, WINDOW_TICKS
from sentinelops.sim.env import DnsEnv
from sentinelops.sim.telemetry import Observation, ServerTelemetry, SiteTelemetry
from sentinelops.sim.topology import ServerId, SiteId, default_topology

TOPO = default_topology()
SITES = tuple(sorted(TOPO.sites))
LON = SiteId("lon")
FIRST = WARMUP_TICKS - 1  # first tick that yields features

SiteFields = dict[str, float | None]
ServerFields = dict[str, float | None]


def _feature(vector: FeatureVector, name: str) -> float:
    return float(vector.values[FEATURE_NAMES.index(name)])


def obs_at(
    tick: int,
    site: Callable[[int, SiteId], SiteFields] | None = None,
    server: Callable[[int, ServerId], ServerFields] | None = None,
) -> Observation:
    """A healthy observation, with optional per-tick overrides for sites and servers."""
    sites = {}
    for sid in TOPO.sites:
        fields: SiteFields = {
            "qps": 1000.0,
            "p95_latency_ms": 20.0,
            "loss": 0.01,
            "queue_depth": 0.0,
            "ok_replies": 990.0,
            "failed_replies": 10.0,
        }
        if site is not None:
            fields.update(site(tick, sid))
        sites[sid] = SiteTelemetry(**fields, announced=True, weight=1.0)  # type: ignore[arg-type]
    servers = {}
    for srv in TOPO.servers:
        sfields: ServerFields = {"cpu": 0.5, "mem": 0.4, "served_qps": 500.0}
        if server is not None:
            sfields.update(server(tick, srv))
        servers[srv] = ServerTelemetry(**sfields)  # type: ignore[arg-type]
    return Observation(tick=tick, sites=sites, servers=servers, action_updates=())


def run(
    ticks: int, **overrides: Callable[[int, str], dict[str, float | None]]
) -> Iterator[Observation]:
    return (obs_at(t, **overrides) for t in range(ticks))  # type: ignore[arg-type]


def last_vectors(
    ticks: int, **overrides: Callable[[int, str], dict[str, float | None]]
) -> dict[SiteId, FeatureVector]:
    featurizer = WindowFeaturizer(TOPO)
    out: dict[SiteId, FeatureVector] = {}
    for obs in run(ticks, **overrides):
        out = featurizer.update(obs)
    return out


# ------------------------------------------------------------------ 1. names


def test_feature_names_are_21_unique_in_metric_then_stat_order() -> None:
    assert len(FEATURE_NAMES) == 21 == len(set(FEATURE_NAMES))
    assert FEATURE_NAMES[:4] == ("qps_rel_mean", "qps_rel_std", "qps_rel_slope", "latency_ms_mean")
    assert FEATURE_NAMES[-1] == "mem_max_slope"
    assert list(FEATURE_NAMES) == [f"{m}_{s}" for m in METRICS for s in STATS]


# ----------------------------------------------------------- 2. when output starts


def test_nothing_before_warmup_then_one_vector_per_site() -> None:
    featurizer = WindowFeaturizer(TOPO)
    for tick in range(FIRST):
        assert featurizer.update(obs_at(tick)) == {}
    out = featurizer.update(obs_at(FIRST))
    assert tuple(out) == SITES
    assert all(v.site == s and v.tick == FIRST for s, v in out.items())
    assert tuple(featurizer.update(obs_at(FIRST + 1))) == SITES


# --------------------------------------------------------- 3-5. the statistics


def test_constant_telemetry_gives_zero_std_and_slope() -> None:
    vector = last_vectors(200)[LON]
    for metric in METRICS:
        assert _feature(vector, f"{metric}_std") == pytest.approx(0.0, abs=1e-12)
        assert _feature(vector, f"{metric}_slope") == pytest.approx(0.0, abs=1e-12)
    assert _feature(vector, "qps_rel_mean") == pytest.approx(1.0)
    assert _feature(vector, "latency_ms_mean") == pytest.approx(20.0)
    assert _feature(vector, "fail_ratio_mean") == pytest.approx(0.01)
    assert _feature(vector, "cpu_max_mean") == pytest.approx(0.5)


def test_metric_rising_by_one_per_tick_has_slope_one() -> None:
    vector = last_vectors(200, site=lambda t, s: {"p95_latency_ms": float(t)})[LON]
    assert _feature(vector, "latency_ms_slope") == pytest.approx(1.0)
    assert _feature(vector, "latency_ms_mean") == pytest.approx(199 - (WINDOW_TICKS - 1) / 2)
    # population std of 30 consecutive integers
    assert _feature(vector, "latency_ms_std") == pytest.approx(np.std(np.arange(30.0)))


def test_qps_is_relative_to_the_warmup_normal() -> None:
    def doubled_after_warmup(tick: int, site: str) -> SiteFields:
        base = 500.0 if site == LON else 1000.0  # sites have different normals
        return {"qps": base * (2.0 if tick >= WARMUP_TICKS else 1.0)}

    out = last_vectors(200, site=doubled_after_warmup)
    for vector in out.values():
        assert _feature(vector, "qps_rel_mean") == pytest.approx(2.0)


def test_normal_is_frozen_after_the_warmup() -> None:
    featurizer = WindowFeaturizer(TOPO)
    for obs in run(WARMUP_TICKS):
        featurizer.update(obs)
    for tick in range(WARMUP_TICKS, WARMUP_TICKS + 40):
        out = featurizer.update(obs_at(tick, site=lambda t, s: {"qps": 3000.0}))
    assert _feature(out[LON], "qps_rel_mean") == pytest.approx(3.0)


def test_site_without_any_warmup_qps_uses_normal_one() -> None:
    vector = last_vectors(200, site=lambda t, s: {"qps": None if t < WARMUP_TICKS else 7.0})[LON]
    assert _feature(vector, "qps_rel_mean") == pytest.approx(7.0)


# ----------------------------------------------------------- 6. missing readings


def test_missing_readings_are_forward_filled() -> None:
    def latency(tick: int, site: str) -> SiteFields:
        if tick >= 180:
            return {"p95_latency_ms": None}  # gap: repeats 50.0
        return {"p95_latency_ms": 50.0 if tick >= 150 else 20.0}

    vector = last_vectors(200, site=latency)[LON]
    assert _feature(vector, "latency_ms_mean") == pytest.approx(50.0)
    assert _feature(vector, "latency_ms_std") == pytest.approx(0.0, abs=1e-12)


def test_never_seen_reading_is_zero_and_nothing_is_nan() -> None:
    def all_missing(tick: int, site: str) -> SiteFields:
        return dict.fromkeys(
            ("qps", "p95_latency_ms", "loss", "queue_depth", "ok_replies", "failed_replies")
        )

    def no_servers(tick: int, server: str) -> ServerFields:
        return dict.fromkeys(("cpu", "mem", "served_qps"))

    vector = last_vectors(200, site=all_missing, server=no_servers)[LON]
    assert np.all(np.isfinite(vector.values))
    assert _feature(vector, "latency_ms_mean") == 0.0
    assert _feature(vector, "qps_rel_mean") == 0.0  # never seen -> 0, divided by normal 1.0
    assert _feature(vector, "cpu_max_mean") == 0.0


def test_site_missing_from_the_observation_is_forward_filled() -> None:
    featurizer = WindowFeaturizer(TOPO)
    for obs in run(150):
        featurizer.update(obs)
    full = obs_at(150)
    partial = Observation(
        tick=150,
        sites={s: t for s, t in full.sites.items() if s != LON},
        servers=full.servers,
        action_updates=(),
    )
    vector = featurizer.update(partial)[LON]
    assert _feature(vector, "latency_ms_mean") == pytest.approx(20.0)


# ---------------------------------------------------------- 7. derived metrics


def test_fail_ratio_and_queue_fraction() -> None:
    limit = TOPO.sites[LON].queue_limit

    def site(tick: int, sid: str) -> SiteFields:
        return {"ok_replies": 750.0, "failed_replies": 250.0, "queue_depth": limit / 4}

    vector = last_vectors(200, site=site)[LON]
    assert _feature(vector, "fail_ratio_mean") == pytest.approx(0.25)
    assert _feature(vector, "queue_frac_mean") == pytest.approx(0.25)


def test_fail_ratio_is_zero_when_there_are_no_replies() -> None:
    vector = last_vectors(200, site=lambda t, s: {"ok_replies": 0.0, "failed_replies": 0.0})[LON]
    assert _feature(vector, "fail_ratio_mean") == 0.0


def test_cpu_and_mem_max_ignore_missing_servers() -> None:
    def server(tick: int, srv: str) -> ServerFields:
        if srv == "lon-1":
            return {"cpu": None, "mem": None}
        return {"cpu": 0.7, "mem": 0.6} if srv == "lon-2" else {}

    vector = last_vectors(200, server=server)[LON]
    assert _feature(vector, "cpu_max_mean") == pytest.approx(0.7)
    assert _feature(vector, "mem_max_mean") == pytest.approx(0.6)


def test_cpu_max_takes_the_busier_server() -> None:
    vector = last_vectors(200, server=lambda t, s: {"cpu": 0.9} if s == "lon-1" else {})[LON]
    assert _feature(vector, "cpu_max_mean") == pytest.approx(0.9)


# ------------------------------------------------------------ 8-9. bad usage


@pytest.mark.parametrize("ticks", [[0, 2], [0, 1, 1], [1]])
def test_ticks_out_of_order_are_rejected(ticks: list[int]) -> None:
    featurizer = WindowFeaturizer(TOPO)
    with pytest.raises(ValueError, match="expected tick"):
        for tick in ticks:
            featurizer.update(obs_at(tick))


def test_reset_starts_a_fresh_run() -> None:
    featurizer = WindowFeaturizer(TOPO)
    for obs in run(200, site=lambda t, s: {"qps": 5000.0}):
        featurizer.update(obs)
    featurizer.reset()
    out: dict[SiteId, FeatureVector] = {}
    for obs in run(WARMUP_TICKS):
        out = featurizer.update(obs)
    assert _feature(out[LON], "qps_rel_mean") == pytest.approx(1.0)


@pytest.mark.parametrize(("window", "warmup"), [(1, 120), (0, 120), (40, 30)])
def test_bad_settings_are_rejected(window: int, warmup: int) -> None:
    with pytest.raises(ValueError):
        WindowFeaturizer(TOPO, window=window, warmup_ticks=warmup)


def test_smaller_window_and_warmup_are_supported() -> None:
    vectors = featurize_run(TOPO, run(10), window=3, warmup_ticks=5)
    assert len(vectors) == (10 - 4) * len(SITES)


# --------------------------------------------------- 10. real run + helpers


def test_real_noop_run_gives_finite_read_only_features() -> None:
    env = DnsEnv(default_topology(), (), 300, seed=1)
    agent = NoOpAgent()
    agent.reset(TOPO)

    def observations() -> Iterator[Observation]:
        obs = env.reset()
        yield obs
        while not env.done:
            obs = env.step(agent.act(obs))
            yield obs

    vectors = featurize_run(TOPO, observations())
    assert len(vectors) == (300 - FIRST) * len(SITES)
    assert [v.tick for v in vectors[: len(SITES)]] == [FIRST] * len(SITES)
    assert [v.site for v in vectors[: len(SITES)]] == list(SITES)
    matrix = to_matrix(vectors)
    assert matrix.shape == (len(vectors), 21)
    assert np.all(np.isfinite(matrix))
    assert not vectors[0].values.flags.writeable
    with pytest.raises(ValueError):
        vectors[0].values[0] = 1.0
    # healthy traffic hovers around its normal
    assert 0.8 < matrix[:, FEATURE_NAMES.index("qps_rel_mean")].mean() < 1.2


def test_to_matrix_of_nothing_is_empty_with_21_columns() -> None:
    assert to_matrix([]).shape == (0, 21)
