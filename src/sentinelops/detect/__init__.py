"""Anomaly detection over windows of site telemetry (Isolation Forest)."""

from sentinelops.detect.features import (
    FEATURE_NAMES,
    FeatureVector,
    WindowFeaturizer,
    featurize_run,
    to_matrix,
)
from sentinelops.detect.isolation import IsolationDetector, healthy_windows
from sentinelops.detect.thresholds import (
    DetectionMetrics,
    evaluate,
    format_report,
    pr_curve,
    tune_threshold,
)

__all__ = [
    "FEATURE_NAMES",
    "DetectionMetrics",
    "FeatureVector",
    "IsolationDetector",
    "WindowFeaturizer",
    "evaluate",
    "featurize_run",
    "format_report",
    "healthy_windows",
    "pr_curve",
    "to_matrix",
    "tune_threshold",
]
