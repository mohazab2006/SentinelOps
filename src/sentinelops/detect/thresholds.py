"""Alert thresholds: turn anomaly scores into alerts, and measure how good they are.

The detector gives every window a score; an *alert* is raised when the score is
at least a threshold. Choosing the threshold is a trade-off:

* **precision** - of the alerts raised, the share that were real faults
  (low precision = false alarms, alert fatigue);
* **recall** - of the real faults, the share that raised an alert
  (low recall = missed incidents);
* **F1** - the harmonic mean of the two, high only when both are high;
* **FPR** (false-positive rate) - the share of healthy windows alerted on.

Accuracy is deliberately not used: faults are rare, so "never alert" would
already look very accurate while catching nothing.

The threshold is chosen on the tuning runs only (which include harmless
traffic surges, so spikes that need no fix are penalised) and then frozen
before the held-out test runs are scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    """How well one threshold separates fault windows from healthy ones.

    Attributes:
        threshold: Alert when ``score >= threshold``.
        tp: Fault windows alerted on (true positives).
        fp: Healthy windows alerted on (false positives).
        fn: Fault windows missed (false negatives).
        tn: Healthy windows correctly left alone (true negatives).
        precision: ``tp / (tp + fp)``; 0.0 if there were no alerts.
        recall: ``tp / (tp + fn)``; 0.0 if there were no fault windows.
        f1: Harmonic mean of precision and recall; 0.0 if both are 0.
        fpr: ``fp / (fp + tn)``; 0.0 if there were no healthy windows.
    """

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    fpr: float


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _metrics(threshold: float, tp: int, fp: int, n_pos: int, n_neg: int) -> DetectionMetrics:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, n_pos)
    total = precision + recall
    f1 = 2 * precision * recall / total if total > 0 else 0.0
    return DetectionMetrics(
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        fn=n_pos - tp,
        tn=n_neg - fp,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=_ratio(fp, n_neg),
    )


def _prepare(
    scores: npt.ArrayLike, labels: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Return scores and labels as matching 1-D arrays, or raise ``ValueError``."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if s.ndim != 1 or y.ndim != 1:
        raise ValueError("scores and labels must be 1-D")
    if s.shape != y.shape:
        raise ValueError(f"scores ({s.size}) and labels ({y.size}) differ in length")
    if s.size == 0:
        raise ValueError("scores and labels are empty")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores contain NaN or infinite values")
    return s, y


def evaluate(scores: npt.ArrayLike, labels: npt.ArrayLike, threshold: float) -> DetectionMetrics:
    """Measure one threshold.

    Args:
        scores: Anomaly scores, one per window (higher = more anomalous).
        labels: ``True`` for windows during a fault, ``False`` for healthy ones.
        threshold: Alert when ``score >= threshold``.

    Raises:
        ValueError: If the inputs are empty, not 1-D, differ in length, or
            contain NaN/inf scores.
    """
    s, y = _prepare(scores, labels)
    alerts = s >= threshold
    tp = int(np.sum(alerts & y))
    fp = int(np.sum(alerts & ~y))
    return _metrics(threshold, tp, fp, int(y.sum()), int((~y).sum()))


def pr_curve(scores: npt.ArrayLike, labels: npt.ArrayLike) -> list[DetectionMetrics]:
    """Measure every useful threshold: each distinct score, highest first.

    Going down the list the threshold falls, so more windows alert: recall
    never decreases while precision usually does.

    Raises:
        ValueError: As for :func:`evaluate`.
    """
    s, y = _prepare(scores, labels)
    candidates = np.unique(s)[::-1]  # distinct scores, descending
    positives = np.sort(s[y])
    negatives = np.sort(s[~y])
    # Windows with score >= t, counted by binary search in the sorted scores.
    tps = positives.size - np.searchsorted(positives, candidates, side="left")
    fps = negatives.size - np.searchsorted(negatives, candidates, side="left")
    return [
        _metrics(t, int(tp), int(fp), positives.size, negatives.size)
        for t, tp, fp in zip(candidates, tps, fps, strict=True)
    ]


def tune_threshold(
    scores: npt.ArrayLike, labels: npt.ArrayLike, *, max_fpr: float | None = None
) -> DetectionMetrics:
    """Choose the alert threshold on tuning data.

    Args:
        scores: Anomaly scores of the tuning windows.
        labels: Their true labels (``True`` = fault window).
        max_fpr: ``None`` to maximise F1. Otherwise maximise recall among
            thresholds whose false-positive rate is at most ``max_fpr``.

    Returns:
        The metrics of the chosen threshold. Ties keep the **higher** threshold,
        so the detector raises fewer alerts.

    Raises:
        ValueError: If the inputs are invalid (see :func:`evaluate`), there is
            no fault window or no healthy window, ``max_fpr`` is outside
            ``[0, 1]``, or no threshold meets ``max_fpr``.
    """
    s, y = _prepare(scores, labels)
    if y.all() or not y.any():
        raise ValueError("labels need at least one fault window and one healthy window")
    if max_fpr is not None and not 0.0 <= max_fpr <= 1.0:
        raise ValueError(f"max_fpr must be in [0, 1], got {max_fpr}")
    curve = pr_curve(s, y)  # thresholds descending, so the first best is the highest
    if max_fpr is None:
        best_f1 = max(m.f1 for m in curve)
        return next(m for m in curve if m.f1 >= best_f1 - _TIE_TOLERANCE)
    allowed = [m for m in curve if m.fpr <= max_fpr]
    if not allowed:
        raise ValueError(f"no threshold keeps the false-positive rate at or below {max_fpr}")
    best_recall = max(m.recall for m in allowed)
    return next(m for m in allowed if m.recall >= best_recall - _TIE_TOLERANCE)


def format_report(metrics: DetectionMetrics, title: str = "Detection") -> str:
    """Return a small aligned text table of ``metrics`` (the caller prints it)."""
    m = metrics
    return "\n".join(
        [
            title,
            f"  threshold  {m.threshold:.4f}",
            f"  precision  {m.precision:.3f}",
            f"  recall     {m.recall:.3f}",
            f"  F1         {m.f1:.3f}",
            f"  FPR        {m.fpr:.3f}",
            f"  TP {m.tp}  FP {m.fp}  FN {m.fn}  TN {m.tn}",
        ]
    )
