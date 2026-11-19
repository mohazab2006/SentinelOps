"""Tests for the Isolation Forest anomaly detector."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from sentinelops.agent.base import NoOpAgent
from sentinelops.detect import (
    FEATURE_NAMES,
    IsolationDetector,
    featurize_run,
    healthy_windows,
    to_matrix,
)
from sentinelops.detect.features import WARMUP_TICKS
from sentinelops.sim.env import DnsEnv
from sentinelops.sim.faults import Fault, FaultKind
from sentinelops.sim.telemetry import Observation
from sentinelops.sim.topology import SiteId, default_topology

TOPO = default_topology()
N_SITES = len(TOPO.sites)
N_FEATURES = len(FEATURE_NAMES)


@pytest.fixture(scope="module")
def healthy() -> np.ndarray:
    """Healthy training windows from seeds 0-2 (built once for the whole file)."""
    return healthy_windows(TOPO, seeds=range(3), horizon=400)


@pytest.fixture(scope="module")
def detector(healthy: np.ndarray) -> IsolationDetector:
    return IsolationDetector(seed=0).fit(healthy)


def _run(faults: Sequence[Fault], horizon: int, seed: int) -> list[Observation]:
    env = DnsEnv(TOPO, faults, horizon, seed)
    agent = NoOpAgent()
    agent.reset(TOPO)
    observations = [env.reset()]
    while not env.done:
        observations.append(env.step(agent.act(observations[-1])))
    return observations


def _site_rows(observations: list[Observation], site: str, from_tick: int) -> np.ndarray:
    vectors = featurize_run(TOPO, observations)
    return to_matrix([v for v in vectors if v.site == SiteId(site) and v.tick >= from_tick])


# ------------------------------------------------------- 1-2. it detects things


def test_crashed_site_scores_higher_than_the_same_site_when_healthy(
    detector: IsolationDetector,
) -> None:
    onset = 300
    crash = _run([Fault(FaultKind.CRASH, "lon-1", onset, None, 0.8)], 500, seed=10)
    calm = _run([], 500, seed=10)
    faulty = detector.score(_site_rows(crash, "lon", onset + 30)).mean()
    normal = detector.score(_site_rows(calm, "lon", onset + 30)).mean()
    assert faulty > normal


def test_extreme_outlier_scores_above_every_training_row(
    healthy: np.ndarray, detector: IsolationDetector
) -> None:
    outlier = healthy.mean(axis=0) + 10 * healthy.std(axis=0) + 10.0
    assert detector.score(outlier[np.newaxis, :])[0] > detector.score(healthy).max()


# ----------------------------------------------------------- 3. determinism


def test_same_seed_same_scores_different_seed_different_scores(healthy: np.ndarray) -> None:
    sample = healthy[:200]
    a = IsolationDetector(n_estimators=50, seed=1).fit(healthy).score(sample)
    b = IsolationDetector(n_estimators=50, seed=1).fit(healthy).score(sample)
    c = IsolationDetector(n_estimators=50, seed=2).fit(healthy).score(sample)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ------------------------------------------------------ 4-5, 7. usage errors


def test_score_before_fit_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        IsolationDetector().score(np.zeros((1, N_FEATURES)))


def test_score_with_wrong_column_count_is_an_error(detector: IsolationDetector) -> None:
    with pytest.raises(ValueError, match="columns"):
        detector.score(np.zeros((3, N_FEATURES - 1)))


def test_score_rejects_nan(detector: IsolationDetector) -> None:
    rows = np.zeros((2, N_FEATURES))
    rows[1, 4] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        detector.score(rows)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(N_FEATURES),  # 1-D
        np.zeros((1, N_FEATURES)),  # fewer than 2 rows
        np.array([[0.0, np.nan], [1.0, 2.0]]),  # NaN
        np.array([[0.0, np.inf], [1.0, 2.0]]),  # inf
    ],
)
def test_fit_rejects_bad_training_data(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        IsolationDetector().fit(bad)


def test_fitted_flag_and_fit_returns_self(healthy: np.ndarray) -> None:
    detector = IsolationDetector(n_estimators=10)
    assert not detector.fitted
    assert detector.fit(healthy) is detector
    assert detector.fitted


def test_scores_have_one_value_per_row(detector: IsolationDetector, healthy: np.ndarray) -> None:
    scores = detector.score(healthy[:7])
    assert scores.shape == (7,)
    assert np.all(np.isfinite(scores))


def test_tiny_training_set_caps_max_samples() -> None:
    rows = np.random.default_rng(0).normal(size=(5, 3))
    detector = IsolationDetector(n_estimators=5, max_samples=256).fit(rows)
    assert detector.score(rows).shape == (5,)


@pytest.mark.parametrize(("n_estimators", "max_samples"), [(0, 256), (200, 0)])
def test_bad_settings_are_rejected(n_estimators: int, max_samples: int) -> None:
    with pytest.raises(ValueError):
        IsolationDetector(n_estimators=n_estimators, max_samples=max_samples)


# --------------------------------------------------------- 6. healthy_windows


def test_healthy_windows_shape_and_values(healthy: np.ndarray) -> None:
    assert healthy.shape == (3 * (400 - (WARMUP_TICKS - 1)) * N_SITES, N_FEATURES)
    assert np.all(np.isfinite(healthy))


@pytest.mark.parametrize("horizon", [WARMUP_TICKS, 50])
def test_healthy_windows_needs_a_horizon_past_the_warmup(horizon: int) -> None:
    with pytest.raises(ValueError, match="horizon"):
        healthy_windows(TOPO, seeds=[0], horizon=horizon)


def test_healthy_windows_with_no_seeds_is_empty() -> None:
    assert healthy_windows(TOPO, seeds=[], horizon=300).shape == (0, N_FEATURES)
