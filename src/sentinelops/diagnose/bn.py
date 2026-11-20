r"""Bayesian network for fault diagnosis: its structure and fitted probability tables.

The detector says *that* a site looks wrong; the network says *what* is most
likely wrong with it. Its structure is fixed by hand, from how faults cause
evidence::

                          Fault at the site
            (healthy, crash, exhaustion, hardware, path_loss,
             site_outage, flood, benign_surge)
             /        |         \                 \
       qps_rel    latency  ...  mem_max        local_dns  ...  hw_check
       (symptoms: low / normal / high)         (probe outcomes)

The fault is the only parent of every evidence node, so given the fault the
symptoms and probe outcomes are independent of each other. The symptoms are
six of the detector's window features, picked so that each measures a
different thing (traffic, latency, loss, queueing, failures, memory trend);
near-duplicate features would count the same evidence twice.

Each symptom is cut into three levels. "Normal" is the range that healthy
windows fall in (their 1st to 99th percentile), so the cut points are learned
from data rather than guessed; below it is "low", above it "high".

Server probes (``local_dns``, ``hw_check``) see the state of the server they
probe: a crash on one server leaves its sibling healthy. Their tables
therefore give the outcome given the *probed target's* state, which is how the
training rows are labelled; the inference step maps a site hypothesis such as
"crash on server 1" to each target's state.

Every table is fitted by maximum likelihood with Laplace smoothing::

    P(value | fault) = (count(value, fault) + alpha) / (count(fault) + alpha * n_values)

Smoothing keeps every probability above zero, so one surprising reading (a
probe that flaked) lowers a hypothesis instead of ruling it out for good.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from sentinelops.detect.features import FEATURE_NAMES
from sentinelops.eval.dataset import HEALTHY, Dataset
from sentinelops.sim.faults import FaultKind
from sentinelops.sim.probes import OUTCOMES, ProbeKind, ProbeOutcome

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
StrArray = npt.NDArray[np.str_]

FAULT_STATES: tuple[str, ...] = (HEALTHY, *(str(kind) for kind in FaultKind))
"""States of the fault node: ``"healthy"`` and every fault kind, in this order."""

SYMPTOMS: tuple[str, ...] = (
    "qps_rel_mean",
    "latency_ms_mean",
    "loss_mean",
    "queue_frac_mean",
    "fail_ratio_mean",
    "mem_max_slope",
)
"""Window features used as symptom nodes, in this order."""

LEVELS: tuple[str, ...] = ("low", "normal", "high")
"""States of every symptom node, in this order."""

DEFAULT_ALPHA = 1.0
"""Laplace smoothing pseudo-count added to every cell of every table."""

NORMAL_RANGE = (0.01, 0.99)
"""Quantiles of healthy windows that bound a symptom's "normal" level."""

_COLUMNS = tuple(FEATURE_NAMES.index(name) for name in SYMPTOMS)


def _frozen(array: FloatArray) -> FloatArray:
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class DiagnosisNetwork:
    """The fitted network: a prior over faults and one table per evidence node.

    Arrays are read-only. Rows follow :data:`FAULT_STATES`; symptom columns
    follow :data:`LEVELS` and probe columns follow ``OUTCOMES[kind]``. Every row
    sums to 1.

    Attributes:
        prior: ``P(fault)``, shape ``(n_states,)``, from how often each fault
            affects a site in the training windows.
        bounds: Shape ``(n_symptoms, 2)``: the low and high end of each
            symptom's normal level.
        symptoms: ``P(level | fault)``, shape ``(n_symptoms, n_states, 3)``.
        probes: ``P(outcome | fault at the probed target)`` per probe kind,
            each of shape ``(n_states, len(OUTCOMES[kind]))``.
    """

    prior: FloatArray
    bounds: FloatArray
    symptoms: FloatArray
    probes: Mapping[ProbeKind, FloatArray]

    def levels(self, features: FloatArray) -> IntArray:
        """Discretise feature windows into symptom levels.

        Args:
            features: Shape ``(n, 21)``, columns in ``FEATURE_NAMES`` order.

        Returns:
            Shape ``(n, n_symptoms)``: 0 (low), 1 (normal) or 2 (high).

        Raises:
            ValueError: If ``features`` does not have 21 columns.
        """
        return discretise(features, self.bounds)

    def p_symptom(self, symptom: str, level: str, fault: str) -> float:
        """Return ``P(symptom = level | fault)``.

        Raises:
            ValueError: If a name is not a known symptom, level or fault state.
        """
        return float(
            self.symptoms[
                _position(symptom, SYMPTOMS, "symptom"),
                _position(fault, FAULT_STATES, "fault state"),
                _position(level, LEVELS, "level"),
            ]
        )

    def p_outcome(self, probe: ProbeKind, outcome: ProbeOutcome, fault: str) -> float:
        """Return ``P(outcome | fault at the probed target)`` for one probe kind.

        Raises:
            ValueError: If ``outcome`` is not a possible result of ``probe`` or
                ``fault`` is not a fault state.
        """
        table = self.probes[ProbeKind(probe)]
        return float(
            table[
                _position(fault, FAULT_STATES, "fault state"),
                _position(ProbeOutcome(outcome), OUTCOMES[ProbeKind(probe)], f"{probe} outcome"),
            ]
        )


def _position(value: str, names: tuple[str, ...], what: str) -> int:
    try:
        return names.index(value)
    except ValueError:
        raise ValueError(f"unknown {what} {value!r}; expected one of {list(names)}") from None


def _indices(values: StrArray, names: tuple[str, ...], what: str) -> IntArray:
    """Map each value to its position in ``names`` (vectorised)."""
    unique, inverse = np.unique(values, return_inverse=True)
    unknown = sorted(set(unique.tolist()) - set(names))
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; expected one of {list(names)}")
    lookup = np.array([names.index(value) for value in unique.tolist()], dtype=np.int64)
    result: IntArray = lookup[inverse.reshape(-1)]
    return result


def discretise(features: FloatArray, bounds: FloatArray) -> IntArray:
    """Cut each symptom column at its bounds: 0 below, 1 within, 2 above.

    Args:
        features: Shape ``(n, 21)``, columns in ``FEATURE_NAMES`` order.
        bounds: Shape ``(n_symptoms, 2)``, low and high end of "normal".

    Returns:
        Shape ``(n, n_symptoms)`` of levels.

    Raises:
        ValueError: If ``features`` does not have 21 columns.
    """
    data = np.asarray(features, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"features must have shape (n, {len(FEATURE_NAMES)}), got {data.shape}")
    values = data[:, _COLUMNS]
    levels: IntArray = (values >= bounds[:, 0]).astype(np.int64) + (values > bounds[:, 1])
    return levels


def normal_bounds(
    healthy: FloatArray, normal_range: tuple[float, float] = NORMAL_RANGE
) -> FloatArray:
    """Return each symptom's normal range: quantiles of healthy windows.

    Args:
        healthy: Feature windows of healthy sites, shape ``(n, 21)``.
        normal_range: Lower and upper quantile, ``0 <= low < high <= 1``.

    Returns:
        Shape ``(n_symptoms, 2)``.

    Raises:
        ValueError: If there are no rows or ``normal_range`` is invalid.
    """
    low, high = normal_range
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"normal_range must satisfy 0 <= low < high <= 1, got {normal_range}")
    if len(healthy) == 0:
        raise ValueError("need at least one healthy window to set the normal range")
    bounds: FloatArray = np.quantile(healthy[:, _COLUMNS], [low, high], axis=0).T
    return bounds


def conditional_table(
    parents: IntArray, values: IntArray, n_parents: int, n_values: int, alpha: float
) -> FloatArray:
    """Fit ``P(value | parent)`` by counting, with Laplace smoothing.

    Args:
        parents: Parent state of each row, in ``range(n_parents)``.
        values: Child value of each row, in ``range(n_values)``.
        n_parents: Number of parent states (table rows).
        n_values: Number of child values (table columns).
        alpha: Pseudo-count added to every cell, ``> 0``.

    Returns:
        Shape ``(n_parents, n_values)``; each row sums to 1. A parent state with
        no rows gets the uniform distribution.
    """
    counts = np.zeros((n_parents, n_values))
    np.add.at(counts, (parents, values), 1.0)
    table: FloatArray = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * n_values)
    return table


def fit_network(
    dataset: Dataset,
    *,
    alpha: float = DEFAULT_ALPHA,
    normal_range: tuple[float, float] = NORMAL_RANGE,
) -> DiagnosisNetwork:
    """Fit every table of the network from labelled training runs.

    Args:
        dataset: Labelled windows and probe results (training split only).
        alpha: Laplace pseudo-count, ``> 0``.
        normal_range: Quantiles of healthy windows bounding "normal".

    Returns:
        The fitted network.

    Raises:
        ValueError: If ``alpha <= 0``, there are no healthy windows, or a label
            or probe outcome is not one the network knows.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    windows, rows = dataset.windows, dataset.probes
    n_states = len(FAULT_STATES)
    faults = _indices(windows.kind, FAULT_STATES, "fault label")
    bounds = normal_bounds(windows.features[faults == FAULT_STATES.index(HEALTHY)], normal_range)
    levels = discretise(windows.features, bounds)

    prior = conditional_table(np.zeros_like(faults), faults, 1, n_states, alpha)[0]
    symptoms = np.stack(
        [
            conditional_table(faults, levels[:, i], n_states, len(LEVELS), alpha)
            for i in range(len(SYMPTOMS))
        ]
    )
    probe_faults = _indices(rows.kind, FAULT_STATES, "fault label")
    probes = {}
    for kind, outcomes in OUTCOMES.items():
        mine = rows.probe == str(kind)
        values = _indices(rows.outcome[mine], tuple(str(o) for o in outcomes), f"{kind} outcome")
        probes[kind] = _frozen(
            conditional_table(probe_faults[mine], values, n_states, len(outcomes), alpha)
        )
    return DiagnosisNetwork(
        prior=_frozen(prior),
        bounds=_frozen(bounds),
        symptoms=_frozen(symptoms),
        probes=MappingProxyType(probes),
    )
