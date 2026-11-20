"""Tests for the diagnosis Bayesian network: structure and fitted tables."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from sentinelops.detect.features import FEATURE_NAMES
from sentinelops.diagnose import FAULT_STATES, LEVELS, SYMPTOMS, DiagnosisNetwork, fit_network
from sentinelops.diagnose.bn import conditional_table, discretise, normal_bounds
from sentinelops.eval.dataset import HEALTHY, Dataset, ProbeTable, WindowTable, build_dataset
from sentinelops.eval.splits import Split, split_scenarios
from sentinelops.sim.faults import FaultKind
from sentinelops.sim.probes import OUTCOMES, ProbeKind, ProbeOutcome
from sentinelops.sim.topology import default_topology

N_STATES = len(FAULT_STATES)
FAIL = SYMPTOMS.index("fail_ratio_mean")
QPS = SYMPTOMS.index("qps_rel_mean")


def _features(**symptoms: Sequence[float] | np.ndarray) -> np.ndarray:
    """Feature rows with the given symptom columns set and every other column 0."""
    n = len(next(iter(symptoms.values())))
    rows = np.zeros((n, len(FEATURE_NAMES)))
    for name, values in symptoms.items():
        rows[:, FEATURE_NAMES.index(name)] = values
    return rows


def _windows(kinds: Sequence[str], features: np.ndarray) -> WindowTable:
    n = len(kinds)
    return WindowTable(
        family=np.full(n, "test"),
        seed=np.zeros(n, dtype=np.int64),
        site=np.full(n, "lon"),
        tick=np.arange(n, dtype=np.int64),
        features=features,
        fault=np.array([k != HEALTHY for k in kinds]),
        kind=np.array(kinds),
        n_faults=np.array([int(k != HEALTHY) for k in kinds], dtype=np.int64),
    )


def _probes(rows: Sequence[tuple[str, str, str]]) -> ProbeTable:
    """Probe rows from ``(probe kind, outcome, fault label)`` triples."""
    n = len(rows)
    text = np.full(n, "lon-1")
    return ProbeTable(
        family=np.full(n, "test"),
        seed=np.zeros(n, dtype=np.int64),
        probe=np.array([r[0] for r in rows], dtype=np.str_),
        target=text,
        site=np.full(n, "lon"),
        tick=np.arange(n, dtype=np.int64),
        outcome=np.array([r[1] for r in rows], dtype=np.str_),
        fault=np.array([r[2] != HEALTHY for r in rows], dtype=np.bool_),
        kind=np.array([r[2] for r in rows], dtype=np.str_),
        n_faults=np.zeros(n, dtype=np.int64),
    )


@pytest.fixture(scope="module")
def tiny() -> Dataset:
    """4 healthy windows and 2 crash windows; 4 local probes and 1 hardware check.

    Healthy fail ratio is 0 and qps 0.9-1.2; both crash windows fail 0.5 at normal qps.
    """
    windows = _windows(
        [HEALTHY] * 4 + ["crash"] * 2,
        _features(
            qps_rel_mean=[0.9, 1.0, 1.1, 1.2, 1.0, 1.0],
            fail_ratio_mean=[0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
        ),
    )
    probes = _probes(
        [
            ("local_dns", "ok", HEALTHY),
            ("local_dns", "ok", HEALTHY),
            ("local_dns", "ok", HEALTHY),
            ("local_dns", "refused", "crash"),
            ("hw_check", "healthy", HEALTHY),
        ]
    )
    return Dataset(windows, probes)


@pytest.fixture(scope="module")
def network(tiny: Dataset) -> DiagnosisNetwork:
    return fit_network(tiny, normal_range=(0.0, 1.0))


# ------------------------------------------------------------------ structure


def test_fault_states_are_healthy_then_every_fault_kind() -> None:
    assert (HEALTHY, *(str(kind) for kind in FaultKind)) == FAULT_STATES
    assert len(FAULT_STATES) == 8


def test_symptoms_are_distinct_detector_features() -> None:
    assert set(SYMPTOMS) <= set(FEATURE_NAMES)
    assert len(set(SYMPTOMS)) == len(SYMPTOMS) == 6
    assert LEVELS == ("low", "normal", "high")


# -------------------------------------------------------------- discretising


def test_discretise_cuts_each_symptom_at_its_bounds() -> None:
    bounds = np.tile([0.0, 1.0], (len(SYMPTOMS), 1))
    rows = _features(qps_rel_mean=[-0.1, 0.0, 0.5, 1.0, 1.1])
    levels = discretise(rows, bounds)
    assert levels.shape == (5, len(SYMPTOMS))
    assert levels[:, QPS].tolist() == [0, 1, 1, 1, 2]  # bounds themselves are normal
    assert levels[:, FAIL].tolist() == [1] * 5


@pytest.mark.parametrize("shape", [(3, 20), (21,)])
def test_discretise_rejects_wrong_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        discretise(np.zeros(shape), np.zeros((len(SYMPTOMS), 2)))


def test_normal_bounds_are_healthy_quantiles() -> None:
    healthy = _features(qps_rel_mean=np.linspace(0.0, 100.0, 101))
    bounds = normal_bounds(healthy, (0.01, 0.99))
    assert bounds.shape == (len(SYMPTOMS), 2)
    assert bounds[QPS] == pytest.approx([1.0, 99.0])
    # A feature that is always 0 when healthy: 0 is normal, anything above is high.
    assert bounds[FAIL].tolist() == [0.0, 0.0]
    assert discretise(_features(fail_ratio_mean=[0.0, 0.01]), bounds)[:, FAIL].tolist() == [1, 2]


@pytest.mark.parametrize("normal_range", [(0.5, 0.5), (0.9, 0.1), (-0.1, 0.9), (0.1, 1.1)])
def test_normal_bounds_rejects_a_bad_range(normal_range: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="normal_range"):
        normal_bounds(_features(qps_rel_mean=[1.0, 2.0]), normal_range)


def test_normal_bounds_needs_healthy_rows() -> None:
    with pytest.raises(ValueError, match="healthy"):
        normal_bounds(np.zeros((0, len(FEATURE_NAMES))))


# ------------------------------------------------------- conditional tables


def test_conditional_table_by_hand() -> None:
    # parent 0: values (0, 1); parent 1: value 1; parent 2: no rows.
    table = conditional_table(np.array([0, 0, 1]), np.array([0, 1, 1]), 3, 2, alpha=1.0)
    assert table[0] == pytest.approx([2 / 4, 2 / 4])
    assert table[1] == pytest.approx([1 / 3, 2 / 3])
    assert table[2] == pytest.approx([1 / 2, 1 / 2])  # unseen parent: uniform


def test_conditional_table_tends_to_counting_as_alpha_shrinks() -> None:
    table = conditional_table(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 1]), 1, 2, alpha=1e-9)
    assert table[0] == pytest.approx([0.75, 0.25])


# --------------------------------------------------------- fitting, by hand


def test_prior_counts_window_labels_with_smoothing(network: DiagnosisNetwork) -> None:
    # 6 windows, 8 states, alpha 1 -> denominator 14.
    expected = np.full(N_STATES, 1 / 14)
    expected[FAULT_STATES.index(HEALTHY)] = 5 / 14
    expected[FAULT_STATES.index("crash")] = 3 / 14
    assert network.prior == pytest.approx(expected)


def test_bounds_come_from_healthy_windows_only(network: DiagnosisNetwork) -> None:
    assert network.bounds[QPS].tolist() == [0.9, 1.2]
    assert network.bounds[FAIL].tolist() == [0.0, 0.0]  # crash windows (0.5) not included


def test_symptom_tables_by_hand(network: DiagnosisNetwork) -> None:
    # Crash: 2 windows, both fail-ratio "high" -> (2+1)/(2+3); low and normal (0+1)/5.
    assert network.p_symptom("fail_ratio_mean", "high", "crash") == pytest.approx(3 / 5)
    assert network.p_symptom("fail_ratio_mean", "low", "crash") == pytest.approx(1 / 5)
    # Healthy: 4 windows, all normal -> (4+1)/(4+3).
    assert network.p_symptom("fail_ratio_mean", "normal", HEALTHY) == pytest.approx(5 / 7)
    # Flood: never seen -> uniform.
    assert network.p_symptom("qps_rel_mean", "high", "flood") == pytest.approx(1 / 3)


def test_probe_tables_by_hand(network: DiagnosisNetwork) -> None:
    local = ProbeKind.LOCAL_DNS
    # Crash: one "refused" out of 4 local outcomes -> (1+1)/(1+4).
    assert network.p_outcome(local, ProbeOutcome.REFUSED, "crash") == pytest.approx(2 / 5)
    assert network.p_outcome(local, ProbeOutcome.OK, "crash") == pytest.approx(1 / 5)
    # Healthy: three "ok" -> (3+1)/(3+4).
    assert network.p_outcome(local, ProbeOutcome.OK, HEALTHY) == pytest.approx(4 / 7)
    # Hardware check: one healthy row, 3 outcomes -> (1+1)/(1+3).
    hw = ProbeKind.HW_CHECK
    assert network.p_outcome(hw, ProbeOutcome.HEALTHY, HEALTHY) == pytest.approx(2 / 4)
    # A probe kind with no rows at all: uniform over its outcomes.
    remote = network.probes[ProbeKind.REMOTE_DNS]
    assert remote == pytest.approx(np.full((N_STATES, 5), 1 / 5))


def test_every_table_is_a_distribution_with_no_zeros(network: DiagnosisNetwork) -> None:
    tables = [network.prior[np.newaxis], *network.symptoms, *network.probes.values()]
    for table in tables:
        assert np.all(table > 0)
        assert table.sum(axis=-1) == pytest.approx(np.ones(table.shape[0]))
    assert network.symptoms.shape == (len(SYMPTOMS), N_STATES, len(LEVELS))
    for kind, outcomes in OUTCOMES.items():
        assert network.probes[kind].shape == (N_STATES, len(outcomes))


def test_levels_uses_the_fitted_bounds(network: DiagnosisNetwork) -> None:
    rows = _features(qps_rel_mean=[0.5, 1.0, 2.0], fail_ratio_mean=[0.0, 0.0, 0.3])
    levels = network.levels(rows)
    assert levels[:, QPS].tolist() == [0, 1, 2]
    assert levels[:, FAIL].tolist() == [1, 1, 2]


def test_tables_are_read_only(network: DiagnosisNetwork) -> None:
    local = network.probes[ProbeKind.LOCAL_DNS]
    for array in (network.prior, network.bounds, network.symptoms, local):
        with pytest.raises(ValueError):
            array[(0,) * array.ndim] = 0.5
    with pytest.raises(TypeError):
        network.probes[ProbeKind.LOCAL_DNS] = network.prior  # type: ignore[index]


# ------------------------------------------------------------------ errors


def test_lookups_reject_unknown_names(network: DiagnosisNetwork) -> None:
    with pytest.raises(ValueError, match="symptom"):
        network.p_symptom("cpu_max_mean", "high", HEALTHY)
    with pytest.raises(ValueError, match="level"):
        network.p_symptom("loss_mean", "extreme", HEALTHY)
    with pytest.raises(ValueError, match="fault state"):
        network.p_symptom("loss_mean", "high", "meteor")
    with pytest.raises(ValueError, match="local_dns outcome"):
        network.p_outcome(ProbeKind.LOCAL_DNS, ProbeOutcome.LOSSY, HEALTHY)


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_fit_rejects_non_positive_alpha(tiny: Dataset, alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        fit_network(tiny, alpha=alpha)


def test_fit_needs_healthy_windows(tiny: Dataset) -> None:
    crash_only = _windows(["crash"], _features(qps_rel_mean=[1.0]))
    with pytest.raises(ValueError, match="healthy"):
        fit_network(Dataset(crash_only, tiny.probes))


def test_fit_rejects_an_unknown_label(tiny: Dataset) -> None:
    windows = _windows([HEALTHY, "meteor"], _features(qps_rel_mean=[1.0, 1.0]))
    with pytest.raises(ValueError, match="meteor"):
        fit_network(Dataset(windows, tiny.probes))


def test_fit_rejects_an_outcome_the_probe_cannot_give(tiny: Dataset) -> None:
    probes = _probes([("local_dns", "lossy", HEALTHY)])
    with pytest.raises(ValueError, match="local_dns outcome"):
        fit_network(Dataset(tiny.windows, probes))


# ------------------------------------------------------- on simulated runs


@pytest.fixture(scope="module")
def simulated() -> Dataset:
    topology = default_topology()
    scenarios = split_scenarios(Split.TRAIN, topology, families=("crash", "flood"))
    picked = [s for s in scenarios if s.seed < 2]
    return build_dataset(picked, topology)


def test_fitted_on_real_runs_the_tables_point_the_right_way(simulated: Dataset) -> None:
    net = fit_network(simulated)
    local, sample = ProbeKind.LOCAL_DNS, ProbeKind.TRAFFIC_SAMPLE
    assert net.p_outcome(local, ProbeOutcome.REFUSED, "crash") > 0.8
    assert net.p_outcome(local, ProbeOutcome.REFUSED, HEALTHY) < 0.05
    assert net.p_outcome(sample, ProbeOutcome.ATTACK_PATTERN, "flood") > 0.4
    assert net.p_outcome(sample, ProbeOutcome.NORMAL_MIX, HEALTHY) > 0.8
    assert net.p_symptom("qps_rel_mean", "high", "flood") > net.p_symptom(
        "qps_rel_mean", "high", HEALTHY
    )
    assert int(np.argmax(net.prior)) == FAULT_STATES.index(HEALTHY)


def test_fitting_is_deterministic(simulated: Dataset) -> None:
    a, b = fit_network(simulated), fit_network(simulated)
    assert np.array_equal(a.symptoms, b.symptoms)
    assert np.array_equal(a.bounds, b.bounds)
    assert all(np.array_equal(a.probes[k], b.probes[k]) for k in OUTCOMES)
