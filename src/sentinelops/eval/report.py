"""How well detection and diagnosis work on labelled runs: the numbers the report quotes.

Two questions, each answered on the *tune* split (the test split stays held
out for the final evaluation):

* **Detection** - does the Isolation Forest score, cut at a tuned threshold,
  raise an alert on the windows where a fault is active?
* **Diagnosis** - once something is wrong, does the Bayesian network name the
  right fault kind?

Benign traffic surges are labelled as faults in the dataset and are kept as
positives for detection: from traffic volume alone a surge looks like a flood,
so a detector that ignored surges would ignore floods too. Detection says
"investigate"; diagnosis rules the surge out, and the agent is penalised only
if it *acts* on one. Because that choice changes the headline numbers, the
report also gives a sensitivity line with surges counted as false alarms.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from sentinelops.detect.thresholds import DetectionMetrics, format_report, tune_threshold
from sentinelops.diagnose.bn import FAULT_STATES, DiagnosisNetwork
from sentinelops.diagnose.inference import posterior
from sentinelops.eval.dataset import HEALTHY, Dataset, ProbeTable
from sentinelops.sim.faults import FaultKind
from sentinelops.sim.probes import ProbeKind, ProbeOutcome, ProbeRequest, ProbeResult
from sentinelops.sim.topology import Topology

IntArray = npt.NDArray[np.int64]

SURGE = str(FaultKind.BENIGN_SURGE)

SENSITIVITY = "Sensitivity: benign surges count as healthy"

EVIDENCE = ("symptoms", "probes", "both")
"""Evidence the diagnosis is scored with, in report order."""


@dataclass(frozen=True, slots=True)
class DetectionReport:
    """Detection quality at the tuned threshold.

    Attributes:
        headline: Best-F1 threshold with surges as positives (the dataset labels).
        surge_as_healthy: Best-F1 threshold with surge windows counted as
            healthy; ``None`` if no other fault window is left.
        surge_alert_rate: Share of surge windows alerted at the headline threshold.
        recall_by_kind: Share of each fault kind's windows alerted at the
            headline threshold, in ``FAULT_STATES`` order (kinds present only).
        n_windows: Windows scored.
        n_surge: Of which surge windows.
    """

    headline: DetectionMetrics
    surge_as_healthy: DetectionMetrics | None
    surge_alert_rate: float
    recall_by_kind: Mapping[str, float]
    n_windows: int
    n_surge: int


@dataclass(frozen=True, slots=True)
class DiagnosisReport:
    """Diagnosis accuracy on faulty windows, given that an alert was raised.

    Attributes:
        accuracy: Share of windows whose most probable fault kind is the true
            one, for each entry of :data:`EVIDENCE`.
        by_kind: Accuracy with both kinds of evidence, per true fault kind.
        n_windows: Faulty windows diagnosed.
    """

    accuracy: Mapping[str, float]
    by_kind: Mapping[str, float]
    n_windows: int


def detection_report(
    scores: npt.ArrayLike, faulty: npt.ArrayLike, kinds: npt.ArrayLike
) -> DetectionReport:
    """Tune the alert threshold both ways and measure it.

    Args:
        scores: Anomaly score of each window.
        faulty: Whether a fault affects the window's site (dataset label).
        kinds: Kind label of each window (``"healthy"`` or a fault kind).

    Raises:
        ValueError: If the inputs differ in length or lack faulty or healthy
            windows (see :func:`~sentinelops.detect.thresholds.tune_threshold`).
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(faulty, dtype=np.bool_)
    k = np.asarray(kinds, dtype=np.str_)
    if not s.shape == y.shape == k.shape:
        raise ValueError(f"scores, labels and kinds differ in shape: {s.shape} {y.shape} {k.shape}")
    surge = k == SURGE
    real = y & ~surge
    headline = tune_threshold(s, y)
    alerts = s >= headline.threshold
    present = set(k[y].tolist())
    recall = {kind: float(alerts[k == kind].mean()) for kind in FAULT_STATES if kind in present}
    return DetectionReport(
        headline=headline,
        surge_as_healthy=tune_threshold(s, real) if real.any() else None,
        surge_alert_rate=float(alerts[surge].mean()) if surge.any() else 0.0,
        recall_by_kind=MappingProxyType(recall),
        n_windows=int(s.size),
        n_surge=int(surge.sum()),
    )


def probe_groups(probes: ProbeTable) -> dict[tuple[str, int, str], IntArray]:
    """Row indices of each run's probe results at each site, in completion order.

    Keyed by ``(family, seed, site)``; build once, then look up windows with
    :func:`latest_probes`.
    """
    order = np.lexsort((probes.tick, probes.site, probes.seed, probes.family))  # stable
    keys = zip(
        probes.family[order].tolist(),
        probes.seed[order].tolist(),
        probes.site[order].tolist(),
        strict=True,
    )
    groups: dict[tuple[str, int, str], list[int]] = {}
    for key, row in zip(keys, order.tolist(), strict=True):
        groups.setdefault(key, []).append(row)
    return {key: np.array(rows, dtype=np.int64) for key, rows in groups.items()}


def latest_probes(probes: ProbeTable, rows: IntArray, tick: int) -> list[ProbeResult]:
    """The latest result of every probe among ``rows`` that finished by ``tick``.

    One result per (probe kind, target): a newer result replaces an older one,
    as an operator trusts the most recent check.

    Args:
        probes: The probe table.
        rows: One group from :func:`probe_groups` (completion order).
        tick: The window's last tick.
    """
    done = rows[: int(np.searchsorted(probes.tick[rows], tick, side="right"))]
    latest: dict[tuple[str, str], int] = {}
    for i in done.tolist():
        latest[(str(probes.probe[i]), str(probes.target[i]))] = i
    return [
        ProbeResult(
            ProbeRequest(ProbeKind(kind), target),
            issued=int(probes.tick[i]),
            completed=int(probes.tick[i]),
            outcome=ProbeOutcome(str(probes.outcome[i])),
        )
        for (kind, target), i in latest.items()
    ]


def diagnosis_report(
    network: DiagnosisNetwork, dataset: Dataset, topology: Topology
) -> DiagnosisReport:
    """Diagnose every faulty window of ``dataset`` and score the fault kind.

    Each window is diagnosed given an alert (``exclude_healthy``), from its
    symptom levels, from the latest probe results at its site, and from both.
    The predicted kind is the one with the highest posterior summed over
    servers (:meth:`~sentinelops.diagnose.inference.Belief.kinds`).

    Raises:
        ValueError: If ``dataset`` has no faulty window.
    """
    w = dataset.windows
    faulty = np.flatnonzero(w.kind != HEALTHY)
    if faulty.size == 0:
        raise ValueError("the dataset has no faulty window to diagnose")
    levels = network.levels(w.features[faulty])
    groups = probe_groups(dataset.probes)
    none = np.empty(0, dtype=np.int64)
    right = {evidence: np.zeros(faulty.size, dtype=np.bool_) for evidence in EVIDENCE}
    for row, index in enumerate(faulty.tolist()):
        site, truth = str(w.site[index]), str(w.kind[index])
        rows = groups.get((str(w.family[index]), int(w.seed[index]), site), none)
        probes = latest_probes(dataset.probes, rows, int(w.tick[index]))
        evidence = {
            "symptoms": {"levels": levels[row]},
            "probes": {"probes": probes},
            "both": {"levels": levels[row], "probes": probes},
        }
        for name, given in evidence.items():
            kinds = posterior(network, site, topology, exclude_healthy=True, **given).kinds()
            right[name][row] = max(kinds, key=kinds.__getitem__) == truth
    truths = w.kind[faulty]
    present = set(truths.tolist())
    return DiagnosisReport(
        accuracy=MappingProxyType({name: float(right[name].mean()) for name in EVIDENCE}),
        by_kind=MappingProxyType(
            {
                kind: float(right["both"][truths == kind].mean())
                for kind in FAULT_STATES
                if kind in present
            }
        ),
        n_windows=int(faulty.size),
    )


def format_detection(report: DetectionReport) -> str:
    """Text block for :func:`detection_report` (the caller prints it)."""
    lines = [
        format_report(report.headline, "Detection (benign surges count as positives)"),
        "",
        (
            format_report(report.surge_as_healthy, SENSITIVITY)
            if report.surge_as_healthy is not None
            else f"{SENSITIVITY}\n  n/a: no fault window other than benign surges"
        ),
        "",
        f"Benign-surge windows alerted: {report.surge_alert_rate:.1%} of {report.n_surge}",
        "Recall by fault kind:",
        *(f"  {kind:<13}{value:.3f}" for kind, value in report.recall_by_kind.items()),
    ]
    return "\n".join(lines)


def format_diagnosis(report: DiagnosisReport) -> str:
    """Text block for :func:`diagnosis_report` (the caller prints it)."""
    lines = [
        f"Diagnosis given an alert ({report.n_windows} faulty windows)",
        *(f"  {name:<13}{value:.3f}" for name, value in report.accuracy.items()),
        "Accuracy by fault kind (symptoms + probes):",
        *(f"  {kind:<13}{value:.3f}" for kind, value in report.by_kind.items()),
    ]
    return "\n".join(lines)
