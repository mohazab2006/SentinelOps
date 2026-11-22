"""Tests for the detection and diagnosis report."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from sentinelops.diagnose import DiagnosisNetwork, fit_network
from sentinelops.eval.dataset import Dataset, ProbeTable, build_dataset, label_run
from sentinelops.eval.report import (
    EVIDENCE,
    detection_report,
    diagnosis_report,
    format_detection,
    format_diagnosis,
    latest_probes,
    probe_groups,
)
from sentinelops.eval.splits import Split, split_scenarios
from sentinelops.scenarios import generate
from sentinelops.sim.probes import ProbeKind, ProbeOutcome
from sentinelops.sim.topology import default_topology

TOPO = default_topology()

# ------------------------------------------------------------------ detection

SCORES = [0.1, 0.2, 0.9, 0.8, 0.85, 0.3]
FAULTY = [False, False, True, True, True, True]
KINDS = ["healthy", "healthy", "crash", "flood", "benign_surge", "crash"]


def test_detection_counts_surges_as_positives_by_hand() -> None:
    report = detection_report(SCORES, FAULTY, KINDS)
    # Every fault scores >= 0.3 and every healthy window < 0.3: a perfect cut.
    assert report.headline.threshold == pytest.approx(0.3)
    assert report.headline.f1 == pytest.approx(1.0)
    assert report.surge_alert_rate == 1.0
    assert dict(report.recall_by_kind) == {"crash": 1.0, "flood": 1.0, "benign_surge": 1.0}
    assert (report.n_windows, report.n_surge) == (6, 1)


def test_sensitivity_counts_surges_as_false_alarms_by_hand() -> None:
    m = detection_report(SCORES, FAULTY, KINDS).surge_as_healthy
    assert m is not None
    # Positives 0.9, 0.8, 0.3; negatives 0.1, 0.2 and the surge at 0.85.
    # Threshold 0.3: tp 3, fp 1 -> precision 3/4, recall 1, F1 6/7 (the best).
    assert m.threshold == pytest.approx(0.3)
    assert (m.tp, m.fp, m.fn, m.tn) == (3, 1, 0, 2)
    assert m.f1 == pytest.approx(6 / 7)
    assert m.fpr == pytest.approx(1 / 3)


def test_recall_by_kind_at_the_headline_threshold() -> None:
    # A crash window far below the cut is missed; the best F1 keeps the cut above it.
    scores = [0.1, 0.2, 0.9, 0.8, 0.85, 0.05]  # cut at 0.8: F1 6/7 beats 0.05: F1 4/5
    report = detection_report(scores, FAULTY, KINDS)
    assert report.headline.threshold == pytest.approx(0.8)
    assert report.recall_by_kind["crash"] == pytest.approx(0.5)
    assert report.recall_by_kind["flood"] == 1.0


def test_no_surge_windows() -> None:
    report = detection_report([0.1, 0.9], [False, True], ["healthy", "crash"])
    assert (report.surge_alert_rate, report.n_surge) == (0.0, 0)
    assert "benign_surge" not in report.recall_by_kind


def test_surges_only_leave_no_sensitivity_line() -> None:
    report = detection_report([0.1, 0.9], [False, True], ["healthy", "benign_surge"])
    assert report.surge_as_healthy is None
    assert "n/a: no fault window other than benign surges" in format_detection(report)


def test_detection_inputs_must_match() -> None:
    with pytest.raises(ValueError, match="differ in shape"):
        detection_report(SCORES, FAULTY, KINDS[:-1])


def test_format_detection_shows_both_lines() -> None:
    text = format_detection(detection_report(SCORES, FAULTY, KINDS))
    assert "benign surges count as positives" in text
    assert "benign surges count as healthy" in text
    assert "Benign-surge windows alerted: 100.0% of 1" in text
    assert "crash" in text


# --------------------------------------------------------------- probe lookup


def _probes(rows: list[tuple[str, int, str, str, str, int, str]]) -> ProbeTable:
    """Probe table from (family, seed, site, probe, target, tick, outcome) rows."""
    family, seed, site, probe, target, tick, outcome = (list(c) for c in zip(*rows, strict=True))
    n = len(rows)
    return ProbeTable(
        family=np.array(family),
        seed=np.array(seed, dtype=np.int64),
        probe=np.array(probe),
        target=np.array(target),
        site=np.array(site),
        tick=np.array(tick, dtype=np.int64),
        outcome=np.array(outcome),
        fault=np.zeros(n, dtype=np.bool_),
        kind=np.full(n, "healthy"),
        n_faults=np.zeros(n, dtype=np.int64),
    )


LOCAL, HW = str(ProbeKind.LOCAL_DNS), str(ProbeKind.HW_CHECK)
PROBES = _probes(
    [
        ("crash", 0, "lon", LOCAL, "lon-1", 50, "ok"),
        ("crash", 0, "lon", LOCAL, "lon-1", 30, "timeout"),  # older, listed later
        ("crash", 0, "lon", LOCAL, "lon-2", 40, "refused"),
        ("crash", 0, "lon", HW, "lon-1", 90, "hardware_fault"),  # after the window
        ("crash", 0, "par", LOCAL, "par-1", 10, "ok"),  # another site
        ("crash", 1, "lon", LOCAL, "lon-1", 10, "refused"),  # another run
    ]
)


def test_probe_groups_split_runs_and_sites_in_tick_order() -> None:
    groups = probe_groups(PROBES)
    assert set(groups) == {("crash", 0, "lon"), ("crash", 0, "par"), ("crash", 1, "lon")}
    assert PROBES.tick[groups[("crash", 0, "lon")]].tolist() == [30, 40, 50, 90]


def test_latest_probes_keeps_the_newest_result_per_target_up_to_the_tick() -> None:
    rows = probe_groups(PROBES)[("crash", 0, "lon")]
    latest = latest_probes(PROBES, rows, tick=60)
    got = {(r.request.kind, r.request.target): r.outcome for r in latest}
    assert got == {
        (ProbeKind.LOCAL_DNS, "lon-1"): ProbeOutcome.OK,  # 50 replaces 30
        (ProbeKind.LOCAL_DNS, "lon-2"): ProbeOutcome.REFUSED,
    }
    assert latest_probes(PROBES, rows, tick=29) == []
    assert len(latest_probes(PROBES, rows, tick=90)) == 3  # the hardware check is done


# ------------------------------------------------------------------ diagnosis


@cache
def _network_and_runs() -> tuple[DiagnosisNetwork, Dataset]:
    families = ("crash", "hardware", "flood", "benign_surge")
    scenarios = [s for s in split_scenarios(Split.TRAIN, TOPO, families) if s.seed < 2]
    network = fit_network(build_dataset(scenarios, TOPO))
    runs = Dataset.concat([label_run(generate(f, 100, TOPO), TOPO) for f in ("crash", "hardware")])
    return network, runs


def test_diagnosis_report_on_simulated_runs() -> None:
    network, runs = _network_and_runs()
    report = diagnosis_report(network, runs, TOPO)
    assert report.n_windows == int((runs.windows.kind != "healthy").sum())
    assert list(report.accuracy) == list(EVIDENCE)
    # Probes separate crash from hardware (refused vs timeout, hw_check); symptoms barely can.
    assert report.accuracy["probes"] > 0.8
    assert report.accuracy["both"] > report.accuracy["symptoms"]
    assert set(report.by_kind) == {"crash", "hardware"}
    text = format_diagnosis(report)
    assert f"({report.n_windows} faulty windows)" in text
    assert "both" in text


def test_diagnosis_needs_a_faulty_window() -> None:
    network, _ = _network_and_runs()
    # One window per site, at the end of the warm-up: faults start at tick 300 or later.
    early = label_run(generate("crash", 100, TOPO), TOPO, stride=10_000, probe_every=None)
    assert (early.windows.kind == "healthy").all()
    with pytest.raises(ValueError, match="no faulty window"):
        diagnosis_report(network, early, TOPO)
