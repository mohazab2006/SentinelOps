"""Tests for alert threshold tuning and detection metrics (all numbers hand-computed)."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from sentinelops.detect import (
    DetectionMetrics,
    evaluate,
    format_report,
    pr_curve,
    tune_threshold,
)

T, F = True, False


# ------------------------------------------------------------------ evaluate


def test_evaluate_counts_and_rates_by_hand() -> None:
    # alerts at >= 0.35: 0.4 (healthy), 0.35 (fault), 0.8 (fault); 0.1 stays quiet
    m = evaluate([0.1, 0.4, 0.35, 0.8], [F, F, T, T], 0.35)
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 1, 0, 1)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == 1.0
    assert m.f1 == pytest.approx(0.8)  # 2 * (2/3 * 1) / (2/3 + 1)
    assert m.fpr == 0.5
    assert m.threshold == 0.35


def test_threshold_above_every_score_raises_no_alerts() -> None:
    m = evaluate([0.1, 0.4, 0.35, 0.8], [F, F, T, T], 0.9)
    assert (m.tp, m.fp, m.fn, m.tn) == (0, 0, 2, 2)
    assert (m.precision, m.recall, m.f1, m.fpr) == (0.0, 0.0, 0.0, 0.0)


def test_evaluate_with_only_one_class_uses_zero_for_empty_rates() -> None:
    m = evaluate([0.2, 0.7], [F, F], 0.5)
    assert (m.recall, m.fpr) == (0.0, 0.5)
    m = evaluate([0.2, 0.7], [T, T], 0.5)
    assert (m.recall, m.fpr) == (0.5, 0.0)


# ------------------------------------------------------------------ pr_curve


def test_pr_curve_has_one_entry_per_distinct_score_descending() -> None:
    curve = pr_curve([0.5, 0.2, 0.5, 0.9, 0.1], [T, F, F, T, F])
    thresholds = [m.threshold for m in curve]
    assert thresholds == [0.9, 0.5, 0.2, 0.1]
    assert all(a > b for a, b in itertools.pairwise(thresholds))
    recalls = [m.recall for m in curve]
    assert all(a <= b for a, b in itertools.pairwise(recalls))


def test_pr_curve_agrees_with_evaluate_at_every_threshold() -> None:
    rng = np.random.default_rng(3)
    scores = rng.integers(0, 20, 300) / 20  # many ties on purpose
    labels = rng.random(300) < 0.3
    for m in pr_curve(scores, labels):
        assert m == evaluate(scores, labels, m.threshold)


# ------------------------------------------------------------- tune_threshold


def test_perfectly_separable_data_reaches_f1_one_at_lowest_fault_score() -> None:
    m = tune_threshold([0.1, 0.2, 0.3, 0.6, 0.7, 0.9], [F, F, F, T, T, T])
    assert m.f1 == 1.0
    assert m.threshold == 0.6


def test_equal_f1_keeps_the_higher_threshold() -> None:
    # threshold 0.9: P=1, R=1/2 -> F1 2/3;  threshold 0.6: P=1/2, R=1 -> F1 2/3
    scores, labels = [0.9, 0.8, 0.7, 0.6], [T, F, F, T]
    assert evaluate(scores, labels, 0.9).f1 == pytest.approx(evaluate(scores, labels, 0.6).f1)
    assert tune_threshold(scores, labels).threshold == 0.9


def test_zero_false_positives_gives_the_best_recall_without_false_alarms() -> None:
    scores, labels = [0.9, 0.8, 0.7, 0.6, 0.1], [T, T, F, T, F]
    m = tune_threshold(scores, labels, max_fpr=0.0)
    assert (m.threshold, m.fp) == (0.8, 0)
    assert m.recall == pytest.approx(2 / 3)


def test_fpr_cap_ties_on_recall_keep_the_higher_threshold() -> None:
    # allowed (FPR <= 0.5): 0.8 (R 1/2, FPR 0) and 0.7 (R 1/2, FPR 1/2); 0.3 and 0.2 have
    # FPR 1. Both allowed ones have recall 1/2, so the higher threshold (fewer alerts) wins.
    m = tune_threshold([0.8, 0.7, 0.3, 0.2], [T, F, F, T], max_fpr=0.5)
    assert m.threshold == 0.8


def test_fpr_cap_that_nothing_meets_is_an_error() -> None:
    with pytest.raises(ValueError, match="false-positive"):
        tune_threshold([0.9, 0.5], [F, T], max_fpr=0.0)


@pytest.mark.parametrize(
    ("scores", "labels", "kwargs", "message"),
    [
        ([0.1, 0.2], [T], {}, "length"),
        ([], [], {}, "empty"),
        ([0.1, 0.2], [T, T], {}, "healthy"),
        ([0.1, 0.2], [F, F], {}, "fault"),
        ([0.1, 0.2], [T, F], {"max_fpr": 1.5}, "max_fpr"),
        ([0.1, 0.2], [T, F], {"max_fpr": -0.1}, "max_fpr"),
        ([0.1, float("nan")], [T, F], {}, "NaN"),
        ([[0.1, 0.2]], [[T, F]], {}, "1-D"),
    ],
)
def test_bad_inputs_are_rejected(
    scores: list[float], labels: list[bool], kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tune_threshold(scores, labels, **kwargs)


def test_tuned_f1_beats_or_matches_every_candidate() -> None:
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(5, 60))
        scores = rng.random(n).round(2)
        labels = rng.random(n) < 0.4
        labels[0], labels[1] = True, False  # both classes present
        best = tune_threshold(scores, labels)
        assert all(best.f1 >= evaluate(scores, labels, t).f1 - 1e-12 for t in np.unique(scores))


def test_ten_thousand_scores_are_fast() -> None:
    rng = np.random.default_rng(1)
    labels = rng.random(10_000) < 0.1
    scores = rng.normal(size=10_000) + 1.5 * labels
    start = time.perf_counter()
    m = tune_threshold(scores, labels)
    assert time.perf_counter() - start < 2.0
    assert 0.0 < m.f1 <= 1.0


# ---------------------------------------------------------------- format_report


def test_format_report_shows_threshold_and_f1() -> None:
    m = evaluate([0.1, 0.4, 0.35, 0.8], [F, F, T, T], 0.35)
    report = format_report(m, title="Tuning")
    assert report.splitlines()[0] == "Tuning"
    assert "0.3500" in report
    assert "0.800" in report
    assert "TP 2  FP 1  FN 0  TN 1" in report


def test_metrics_are_plain_python_numbers() -> None:
    m = tune_threshold(np.array([0.2, 0.9]), np.array([F, T]))
    assert isinstance(m, DetectionMetrics)
    assert type(m.threshold) is float and type(m.tp) is int
