"""Isolation Forest anomaly detector: learn "healthy", flag anything unlike it.

An Isolation Forest builds many random trees. Each tree keeps splitting the
data on a random feature at a random value until every point is on its own.
Unusual points sit far from the crowd, so they are isolated after only a few
splits; normal points need many. The average number of splits over all trees
becomes the anomaly score.

The detector is trained only on windows from **healthy** runs, which we can
generate in any quantity. It therefore does not just recognise the faults it
has seen: anything that does not look healthy scores high. No feature scaling
is needed, because each split is drawn within a single feature's own range.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import IsolationForest

from sentinelops.agent.base import NoOpAgent
from sentinelops.detect.features import WARMUP_TICKS, featurize_run, to_matrix
from sentinelops.sim.env import DnsEnv
from sentinelops.sim.telemetry import Observation
from sentinelops.sim.topology import Topology

FloatArray = npt.NDArray[np.float64]


def _as_matrix(x: FloatArray, what: str) -> FloatArray:
    """Return ``x`` as a 2-D float array of finite numbers, or raise ``ValueError``."""
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{what} must be a 2-D array (rows x features), got {matrix.ndim}-D")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{what} contains NaN or infinite values")
    return matrix


class IsolationDetector:
    """Scores feature windows by how unlike healthy behaviour they are.

    Usage: ``IsolationDetector().fit(healthy).score(windows)``.
    """

    def __init__(self, n_estimators: int = 200, max_samples: int = 256, seed: int = 0) -> None:
        """Configure the detector.

        Args:
            n_estimators: Number of random trees. More trees give steadier scores.
            max_samples: Rows drawn to build each tree (capped at the number of
                training rows). Small samples isolate outliers well and are fast.
            seed: Seed for the forest's randomness; same seed and data give the
                same scores.

        Raises:
            ValueError: If ``n_estimators`` or ``max_samples`` is not positive.
        """
        if n_estimators < 1 or max_samples < 1:
            raise ValueError("n_estimators and max_samples must be at least 1")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed
        self._forest: IsolationForest | None = None
        self._n_features = 0

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._forest is not None

    def fit(self, healthy: FloatArray) -> IsolationDetector:
        """Learn what healthy windows look like.

        Args:
            healthy: Feature windows from healthy runs, shape ``(n, features)``.

        Returns:
            ``self``, so calls can be chained.

        Raises:
            ValueError: If ``healthy`` is not 2-D, has fewer than 2 rows, or
                contains NaN or infinite values.
        """
        data = _as_matrix(healthy, "training data")
        if data.shape[0] < 2:
            raise ValueError(f"need at least 2 training rows, got {data.shape[0]}")
        forest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, data.shape[0]),
            random_state=self.seed,
        )
        forest.fit(data)
        self._forest = forest
        self._n_features = data.shape[1]
        return self

    def score(self, x: FloatArray) -> FloatArray:
        """Return an anomaly score per row: **higher means more anomalous**.

        This is the negative of scikit-learn's ``score_samples``, whose sign is
        the opposite (lower = more anomalous there).

        Args:
            x: Feature windows, shape ``(n, features)`` with the same number of
                features as the training data.

        Returns:
            Scores of shape ``(n,)``.

        Raises:
            RuntimeError: If the detector has not been fitted.
            ValueError: If ``x`` is not 2-D, has NaN/inf, or the wrong number of
                columns.
        """
        if self._forest is None:
            raise RuntimeError("call fit() before score()")
        data = _as_matrix(x, "input")
        if data.shape[1] != self._n_features:
            raise ValueError(f"expected {self._n_features} feature columns, got {data.shape[1]}")
        scores: FloatArray = -np.asarray(self._forest.score_samples(data), dtype=np.float64)
        return scores


def _noop_observations(topology: Topology, horizon: int, seed: int) -> list[Observation]:
    """Run a fault-free simulation with an agent that never acts; return every observation."""
    env = DnsEnv(topology, (), horizon, seed)
    agent = NoOpAgent()
    agent.reset(topology)
    observations = [env.reset()]
    while not env.done:
        observations.append(env.step(agent.act(observations[-1])))
    return observations


def healthy_windows(topology: Topology, seeds: Iterable[int], horizon: int = 600) -> FloatArray:
    """Generate training data: feature windows from healthy, fault-free runs.

    Each seed gives one run with no faults and an agent that never acts.

    Args:
        topology: The network to simulate.
        seeds: One run per seed.
        horizon: Ticks per run; must be greater than the warm-up.

    Returns:
        All runs' feature vectors stacked, shape
        ``(n_seeds * (horizon - WARMUP_TICKS + 1) * n_sites, 21)``.

    Raises:
        ValueError: If ``horizon`` is not greater than ``WARMUP_TICKS``.
    """
    if horizon <= WARMUP_TICKS:
        raise ValueError(f"horizon must be greater than {WARMUP_TICKS}, got {horizon}")
    vectors = []
    for seed in seeds:
        vectors.extend(featurize_run(topology, _noop_observations(topology, horizon, seed)))
    return to_matrix(vectors)
