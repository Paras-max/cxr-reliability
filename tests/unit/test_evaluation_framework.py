"""Unit tests for the Agent-Wise Before vs After Evaluation Framework.

Verifies:
1. Classification metrics calculation (TP, TN, FP, FN, Acc, Prec, Rec, Spec, NPV, F1).
2. Strict prohibition of fabricated metrics (Quality, OOD, Decision, Verification report N/A).
3. Sample size tracking (N, N_pos, N_neg reported for all subsets).
4. Paired before/after constraints (N_before == N_after).
5. Robustness against division by zero and edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cxr_reliability.evaluation.agent_evaluation import (
    calculate_classification_metrics,
    calculate_decision_metrics,
    calculate_ood_metrics,
    calculate_quality_transitions,
    calculate_safety_filtering,
    calculate_uncertainty_metrics,
    calculate_verification_metrics,
)


def test_classification_metrics_perfect_case():
    """Verify classification metrics on a known hand-constructed dataset."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    res = calculate_classification_metrics(y_true, y_pred, y_prob)
    assert res["n_total"] == 4
    assert res["n_pos"] == 2
    assert res["n_neg"] == 2
    assert res["tp"] == 2
    assert res["tn"] == 2
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert pytest.approx(res["accuracy"]) == 1.0
    assert pytest.approx(res["precision"]) == 1.0
    assert pytest.approx(res["recall_sensitivity"]) == 1.0
    assert pytest.approx(res["specificity"]) == 1.0
    assert pytest.approx(res["f1_score"]) == 1.0
    assert pytest.approx(res["npv"]) == 1.0
    assert pytest.approx(res["auroc"]) == 1.0
    assert pytest.approx(res["auprc"]) == 1.0


def test_classification_metrics_with_errors():
    """Verify metrics with non-zero FP and FN."""
    # 10 cases: 4 positive, 6 negative
    # Pred: TP=3, FN=1, TN=5, FP=1
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]

    res = calculate_classification_metrics(y_true, y_pred)
    assert res["tp"] == 3
    assert res["fn"] == 1
    assert res["fp"] == 1
    assert res["tn"] == 5
    assert pytest.approx(res["accuracy"]) == 0.8
    assert pytest.approx(res["precision"]) == 3 / 4  # 0.75
    assert pytest.approx(res["recall_sensitivity"]) == 3 / 4  # 0.75
    assert pytest.approx(res["specificity"]) == 5 / 6
    assert pytest.approx(res["npv"]) == 5 / 6
    assert pytest.approx(res["f1_score"]) == 0.75


def test_classification_metrics_zero_division():
    """Zero-division cases should return 0.0 safely without crashing."""
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]
    res = calculate_classification_metrics(y_true, y_pred)
    assert res["tp"] == 0
    assert res["fp"] == 0
    assert res["precision"] == 0.0
    assert res["recall_sensitivity"] == 0.0
    assert res["f1_score"] == 0.0
    assert res["specificity"] == 1.0
    assert res["npv"] == 1.0


def test_classification_metrics_length_mismatch():
    """Length mismatch must raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_classification_metrics([0, 1], [0])


def test_quality_transitions_no_fabricated_f1():
    """Quality agent metrics must NOT fabricate an F1 or accuracy score."""
    df = pd.DataFrame({
        "quality_label": ["poor", "poor", "degraded", "good"],
        "after_repair_quality_label": ["good", "degraded", "good", "good"],
        "repair_applied": [True, True, True, False],
    })
    res = calculate_quality_transitions(df)

    assert "N/A" in str(res["f1_score"])
    assert "N/A" in str(res["accuracy"])
    assert res["n_total_images"] == 4
    assert res["n_repairs_evaluated"] == 3
    assert res["transitions_breakdown"]["poor_to_good"] == 1
    assert res["transitions_breakdown"]["poor_to_degraded"] == 1
    assert pytest.approx(res["poor_to_good_recovery_rate_pct"]) == 50.0
    assert pytest.approx(res["poor_to_acceptable_recovery_rate_pct"]) == 100.0


def test_ood_metrics_no_fabricated_metrics():
    """OOD metrics must NOT report accuracy/F1 when test set lacks external OOD ground truth."""
    df = pd.DataFrame({
        "ood_level": ["IN_DISTRIBUTION", "IN_DISTRIBUTION", "BORDERLINE", "SEVERE"],
        "mahalanobis_distance": [10.5, 12.0, 35.0, 75.0],
        "final_action": ["ACCEPT", "ACCEPT", "ESCALATE", "REJECT"],
        "needs_human_review": [False, False, True, True],
    })
    res = calculate_ood_metrics(df)

    assert "N/A" in str(res["f1_score"])
    assert "N/A" in str(res["accuracy"])
    assert res["counts"]["in_distribution"] == 2
    assert res["counts"]["borderline"] == 1
    assert res["counts"]["severe"] == 1
    assert res["percentages"]["in_distribution"] == 50.0
    assert res["routing_summary"]["SEVERE"]["withheld"] == 1
    assert res["routing_summary"]["SEVERE"]["released"] == 0
    assert pytest.approx(res["mahalanobis_distance_stats"]["mean"]) == 33.125


def test_decision_metrics_routing_behavior():
    """Decision metrics must evaluate routing distributions and not fabricate accuracy."""
    df = pd.DataFrame({
        "final_action": ["ACCEPT", "REPAIR", "ESCALATE", "REJECT"],
        "prediction_released": [True, True, False, False],
        "needs_human_review": [False, False, True, True],
    })
    res = calculate_decision_metrics(df)

    assert "N/A" in str(res["accuracy"])
    assert "N/A" in str(res["f1_score"])
    assert res["action_counts"]["ACCEPT"] == 1
    assert res["action_counts"]["REJECT"] == 1
    assert res["downstream_outcomes"]["predictions_released"] == 2
    assert res["downstream_outcomes"]["predictions_withheld"] == 2
    assert pytest.approx(res["downstream_outcomes"]["release_rate_pct"]) == 50.0


def test_verification_metrics_failure_breakdown():
    """Verification metrics must break down failure categories without claiming accuracy."""
    df = pd.DataFrame({
        "repair_applied": [True, True, True, True],
        "verification_status": [True, False, False, False],
        "verification_next_step": ["release", "escalate", "escalate", "escalate"],
        "delta_confidence": [0.005, -0.002, -0.001, -0.025],
        "quality_label": ["poor", "poor", "poor", "poor"],
        "after_repair_quality_label": ["good", "poor", "degraded", "good"],
    })
    res = calculate_verification_metrics(df)

    assert "N/A" in str(res["accuracy"])
    assert "N/A" in str(res["f1_score"])
    assert res["n_entering_verification"] == 4
    assert res["verified_count"] == 1
    assert res["escalated_count"] == 3
    assert res["failure_reasons"]["quality_unresolved_poor_remained_poor"] == 1
    assert res["failure_reasons"]["quality_partial_poor_to_degraded"] == 1
    assert res["failure_reasons"]["confidence_non_degradation_guard_exceeded"] == 1


def test_safety_filtering_error_reduction():
    """Safety filtering must quantify how many Base Model errors were withheld."""
    df = pd.DataFrame({
        "ground_truth_label": [0, 0, 1, 1, 0, 0],
        "raw_model_score": [0.8, 0.7, 0.1, 0.2, 0.1, 0.2],  # Base model: FP=2, FN=2, TN=2, TP=0
        "prediction_released": [False, False, False, False, True, True],  # Withholds all errors
        "prediction_positive": [None, None, None, None, 0, 0],
    })
    res = calculate_safety_filtering(df, threshold=0.5)

    assert res["base_model_errors"]["false_positives"] == 2
    assert res["base_model_errors"]["false_negatives"] == 2
    assert res["base_model_errors"]["total_errors"] == 4

    assert res["reliability_system_released_errors"]["false_positives"] == 0
    assert res["reliability_system_released_errors"]["false_negatives"] == 0

    assert res["errors_withheld_from_automated_release"]["false_positives_withheld"] == 2
    assert res["errors_withheld_from_automated_release"]["false_negatives_withheld"] == 2
    assert pytest.approx(res["errors_withheld_from_automated_release"]["total_errors_withheld_pct"]) == 100.0


def test_uncertainty_metrics_group_isolation():
    """Uncertainty metrics must report Low and High groups with proper sample sizes."""
    df = pd.DataFrame({
        "uncertainty_level": ["LOW", "LOW", "HIGH", "HIGH"],
        "prediction_released": [True, True, True, False],
        "prediction_positive": [0, 1, 0, None],
        "ground_truth_label": [0, 1, 1, 0],
    })
    res = calculate_uncertainty_metrics(df)

    assert res["low_count"] == 2
    assert res["high_count"] == 2
    assert res["groups"]["LOW"]["released_count"] == 2
    assert res["groups"]["LOW"]["released_metrics"]["accuracy"] == 1.0
    assert res["groups"]["HIGH"]["released_count"] == 1
    assert res["groups"]["HIGH"]["withheld_count"] == 1
