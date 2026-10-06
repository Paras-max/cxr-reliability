"""Tests for evaluation metrics (Phase 2/7)."""

from __future__ import annotations

import numpy as np
import pytest

from cxr_reliability.evaluation.action_breakdown import action_breakdown
from cxr_reliability.evaluation.calibration_metrics import expected_calibration_error
from cxr_reliability.evaluation.metrics import patient_bootstrap_ci
from cxr_reliability.evaluation.repair_recovery import laplacian_recovery_pct


def test_ece_zero_for_perfectly_calibrated_toy_case():
    """ECE is 0 for a constructed perfectly calibrated set."""
    # Bin 1 (0.0): 10 negative samples with p=0.0
    # Bin 2 (0.5): 10 samples (5 pos, 5 neg) with p=0.5
    # Bin 3 (1.0): 10 positive samples with p=1.0
    probs = np.array([0.0] * 10 + [0.5] * 10 + [1.0] * 10)
    labels = np.array([0] * 10 + [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] + [1] * 10)

    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert pytest.approx(ece, abs=1e-6) == 0.0


def test_recovery_pct_formula():
    """recovery_pct matches the PRD formula on hand-computed values."""
    # (150 - 50) / (200 - 50) * 100 = 100 / 150 * 100 = 66.666666...
    rec = laplacian_recovery_pct(var_original=200.0, var_corrupted=50.0, var_repaired=150.0)
    assert pytest.approx(rec, rel=1e-4) == 66.6667


def test_action_breakdown_counts_sum_to_total():
    """Bucket counts sum to the number of inputs."""
    actions = ["accept", "repair", "escalate", "reject", "accept", "escalate"]
    breakdown = action_breakdown(actions)
    total_counted = sum(breakdown.values())
    assert total_counted == len(actions)
    assert breakdown["accept"] == 2
    assert breakdown["escalate"] == 2
    assert breakdown["repair"] == 1
    assert breakdown["reject"] == 1


def test_bootstrap_resamples_by_patient():
    """Bootstrap resamples whole patients, never individual images."""
    patient_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.1, 0.3, 0.8, 0.9, 0.7, 0.8, 0.2, 0.1])

    def dummy_metric(y, p):
        return float(np.mean(p))

    lower, upper, patient_clusters = patient_bootstrap_ci(
        dummy_metric, labels, scores, patient_ids, n_resamples=10, seed=42
    )

    assert len(patient_clusters) == 10
    # Resampled clusters contain unique patient IDs
    for cluster in patient_clusters:
        assert len(cluster) == 5
        # All resampled IDs must be valid patient IDs
        assert set(cluster).issubset({1, 2, 3, 4, 5})
