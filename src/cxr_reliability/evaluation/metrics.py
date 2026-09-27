"""Core classification and selective-prediction metrics.

Responsibility:
    AUROC, AUPRC, sensitivity at fixed specificity, risk-coverage curves and patient-level
    bootstrap confidence intervals.

Input:
    Labels, scores, patient ids.

Output:
    Metric values with confidence intervals.

Dependencies:
    scikit-learn, numpy

Implementation phase: P2
"""

from __future__ import annotations


def auroc_auprc(labels, scores) -> dict[str, float]:
    raise NotImplementedError("Phase 2")


def sensitivity_at_specificity(labels, scores, specificity: float) -> float:
    raise NotImplementedError("Phase 2")


def risk_coverage_curve(confidences, correct):
    raise NotImplementedError("Phase 7")


def patient_bootstrap_ci(metric_fn, labels, scores, patient_ids, n_resamples: int, seed: int):
    raise NotImplementedError("Phase 7")
