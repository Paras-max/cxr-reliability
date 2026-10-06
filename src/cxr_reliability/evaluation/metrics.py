"""Core classification and selective-prediction metrics.

Responsibility:
    AUROC, AUPRC, sensitivity at fixed specificity, risk-coverage curves and patient-level
    bootstrap confidence intervals.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def auroc_auprc(labels, scores) -> dict[str, float]:
    y_true = np.asarray(labels)
    y_score = np.asarray(scores)
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


def sensitivity_at_specificity(labels, scores, specificity: float = 0.95) -> float:
    y_true = np.asarray(labels)
    y_score = np.asarray(scores)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # specificity = 1 - fpr -> target fpr = 1 - specificity
    target_fpr = 1.0 - specificity
    idx = np.searchsorted(fpr, target_fpr)
    if idx >= len(tpr):
        idx = len(tpr) - 1
    return float(tpr[idx])


def risk_coverage_curve(confidences, correct):
    """Compute risk vs coverage curve ordered by descending confidence."""
    confs = np.asarray(confidences)
    corr = np.asarray(correct, dtype=bool)

    order = np.argsort(-confs)
    ordered_corr = corr[order]

    n = len(ordered_corr)
    coverages = np.arange(1, n + 1) / n
    cum_acc = np.cumsum(ordered_corr) / np.arange(1, n + 1)
    cum_risk = 1.0 - cum_acc

    return coverages, cum_risk


def patient_bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    labels: np.ndarray,
    scores: np.ndarray,
    patient_ids: np.ndarray,
    n_resamples: int = 100,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, list[np.ndarray]]:
    """
    Patient-level bootstrap confidence interval:
    Resamples whole patient IDs with replacement so images from the same
    patient are grouped together.
    Returns (lower_bound, upper_bound, sampled_patient_clusters).
    """
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    patient_to_indices = {}
    for idx, pid in enumerate(patient_ids):
        patient_to_indices.setdefault(pid, []).append(idx)

    boot_metrics = []
    resampled_patient_clusters = []

    for _ in range(n_resamples):
        sampled_pids = rng.choice(unique_patients, size=n_patients, replace=True)
        resampled_patient_clusters.append(sampled_pids)
        sampled_indices = []
        for pid in sampled_pids:
            sampled_indices.extend(patient_to_indices[pid])

        b_labels = labels[sampled_indices]
        b_scores = scores[sampled_indices]

        # Compute metric if both classes present
        if len(np.unique(b_labels)) > 1:
            try:
                boot_metrics.append(metric_fn(b_labels, b_scores))
            except Exception:
                pass

    if not boot_metrics:
        return 0.0, 0.0, resampled_patient_clusters

    lower = float(np.percentile(boot_metrics, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_metrics, 100 * (1 - alpha / 2)))
    return lower, upper, resampled_patient_clusters
