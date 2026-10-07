"""
Unit Tests for Before vs After Project Improvement Evaluation and Dynamic Calculations.

Verifies:
- Calculations use actual disk artifacts without hardcoding.
- Confusion matrix and diagnostic metric mathematics.
- Before/after population identification and coverage.
- Quality transition calculations and recovery rates.
- OOD routing containment.
- Uncertainty risk stratification.
- Decision routing reconciliation equations.
- Paired repair diagnostic 4-way transitions and stability.
- Error containment and scientific nomenclature.
- Agent-wise effectiveness scorecard rules.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cxr_reliability.dashboard.before_after_data import (
    DEFAULT_OPERATING_THRESHOLD,
    build_agent_wise_scorecard,
    calculate_binary_classification_metrics,
    compute_baseline_vs_released_comparison,
    compute_decision_agent_analysis,
    compute_error_containment,
    compute_ood_agent_analysis,
    compute_paired_repair_diagnostic_validation,
    compute_quality_agent_analysis,
    compute_uncertainty_agent_analysis,
    compute_verification_agent_analysis,
    discover_evaluation_artifacts,
    generate_all_evaluation_artifacts,
    get_project_root,
)


@pytest.fixture
def project_root() -> Path:
    return get_project_root()


@pytest.fixture
def artifacts(project_root: Path) -> dict:
    return discover_evaluation_artifacts(project_root)


@pytest.fixture
def full_test_df(artifacts: dict) -> pd.DataFrame:
    path = artifacts["full_test_csv"]
    assert path is not None and path.is_file(), "Missing evaluation_full_test.csv"
    return pd.read_csv(path)


@pytest.fixture
def paired_df(artifacts: dict) -> pd.DataFrame:
    path = artifacts["paired_repair_csv"]
    assert path is not None and path.is_file(), "Missing final_paired_repair_diagnostic.csv"
    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# 1. Artifact Discovery & Non-Hardcoding Verification
# -----------------------------------------------------------------------------
def test_artifact_discovery(artifacts: dict):
    """Ensure all required evaluation artifacts are discovered on disk."""
    assert artifacts["full_test_csv"] is not None
    assert artifacts["paired_repair_csv"] is not None
    assert artifacts["full_test_csv"].is_file()
    assert artifacts["paired_repair_csv"].is_file()


def test_no_hardcoded_results_in_code(project_root: Path):
    """
    Verify that source files do NOT hardcode fixed results like accuracy = 0.9094,
    released = 7349, poor_to_good = 7766.
    """
    app_file = project_root / "src" / "cxr_reliability" / "dashboard" / "before_after_app.py"
    data_file = project_root / "src" / "cxr_reliability" / "dashboard" / "before_after_data.py"

    assert app_file.is_file()
    assert data_file.is_file()

    content_app = app_file.read_text(encoding="utf-8")
    content_data = data_file.read_text(encoding="utf-8")

    # Patterns representing fixed hardcoded project metric assignments
    forbidden_patterns = [
        r"accuracy\s*=\s*0\.9094",
        r"released\s*=\s*7349\b",
        r"poor_to_good\s*=\s*7766\b",
        r"withheld\s*=\s*9375\b",
        r"total_test_images\s*=\s*16724\b",
    ]

    for pat in forbidden_patterns:
        match_app = re.search(pat, content_app, re.IGNORECASE)
        match_data = re.search(pat, content_data, re.IGNORECASE)
        assert match_app is None, f"Found hardcoded metric assignment in before_after_app.py matching {pat}"
        assert match_data is None, f"Found hardcoded metric assignment in before_after_data.py matching {pat}"


# -----------------------------------------------------------------------------
# 2. Binary Classification Math Tests
# -----------------------------------------------------------------------------
def test_metric_calculation_math():
    """Verify diagnostic metric calculation formulas against known ground truth array."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])  # 4 pos, 6 neg
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])  # TP=2, FN=2, FP=1, TN=5

    res = calculate_binary_classification_metrics(y_true, y_pred)
    assert res["n_total"] == 10
    assert res["n_pos"] == 4
    assert res["n_neg"] == 6
    assert res["tp"] == 2
    assert res["tn"] == 5
    assert res["fp"] == 1
    assert res["fn"] == 2
    assert pytest.approx(res["accuracy"], 0.001) == 0.7
    assert pytest.approx(res["precision"], 0.001) == 2 / 3
    assert pytest.approx(res["recall"], 0.001) == 2 / 4
    assert pytest.approx(res["specificity"], 0.001) == 5 / 6
    assert pytest.approx(res["npv"], 0.001) == 5 / 7
    assert pytest.approx(res["f1"], 0.001) == 2 * (2 / 3) * (0.5) / ((2 / 3) + 0.5)


# -----------------------------------------------------------------------------
# 3. Baseline vs Released Comparison Tests
# -----------------------------------------------------------------------------
def test_baseline_vs_released_cohort_and_metrics(full_test_df: pd.DataFrame):
    """Verify Before (standalone) vs After (released) calculations on full test cohort."""
    comp = compute_baseline_vs_released_comparison(full_test_df, threshold=DEFAULT_OPERATING_THRESHOLD)

    before = comp["before"]
    after = comp["after"]

    # Full test cohort size
    assert before["n_total"] == len(full_test_df)
    assert before["n_pos"] == int((full_test_df["ground_truth_label"] == 1).sum())
    assert before["n_neg"] == int((full_test_df["ground_truth_label"] == 0).sum())
    assert before["coverage"] == 1.0

    # Released subset
    df_rel = full_test_df[full_test_df["prediction_released"] == True]
    assert after["n_total"] == len(df_rel)
    assert after["coverage"] == pytest.approx(len(df_rel) / len(full_test_df), 1e-4)
    assert after["withheld_rate"] == pytest.approx(1.0 - (len(df_rel) / len(full_test_df)), 1e-4)

    # Verification of confusion matrix sum
    assert before["tp"] + before["tn"] + before["fp"] + before["fn"] == before["n_total"]
    assert after["tp"] + after["tn"] + after["fp"] + after["fn"] == after["n_total"]


# -----------------------------------------------------------------------------
# 4. Error Containment Tests
# -----------------------------------------------------------------------------
def test_error_containment_calculations(full_test_df: pd.DataFrame):
    """Verify error containment arithmetic and preservation of false positives/negatives."""
    ec = compute_error_containment(full_test_df, threshold=DEFAULT_OPERATING_THRESHOLD)

    base_err = ec["baseline_errors"]
    rel_err = ec["released_errors"]
    withheld_err = ec["withheld_errors"]

    # Baseline total errors
    assert base_err["total_errors"] == base_err["false_positives"] + base_err["false_negatives"]
    # Released total errors
    assert rel_err["total_errors"] == rel_err["false_positives"] + rel_err["false_negatives"]

    # Withheld calculations
    assert withheld_err["false_positives_withheld"] == base_err["false_positives"] - rel_err["false_positives"]
    assert withheld_err["false_negatives_withheld"] == base_err["false_negatives"] - rel_err["false_negatives"]
    assert withheld_err["total_errors_withheld"] == base_err["total_errors"] - rel_err["total_errors"]

    # Positive containment rate
    assert withheld_err["total_errors_withheld"] > 0
    assert 0.0 < withheld_err["total_errors_withheld_pct"] <= 100.0


# -----------------------------------------------------------------------------
# 5. Quality Agent Transition Tests
# -----------------------------------------------------------------------------
def test_quality_transition_calculations(full_test_df: pd.DataFrame):
    """Verify quality label counts and repair transition rates."""
    qa = compute_quality_agent_analysis(full_test_df)

    assert qa["n_total"] == len(full_test_df)
    assert qa["initial_counts"]["good"] + qa["initial_counts"]["degraded"] + qa["initial_counts"]["poor"] == qa["n_total"]

    n_repaired = int((full_test_df["repair_applied"] == True).sum())
    assert qa["n_repairs_evaluated"] == n_repaired

    tr = qa["transitions"]
    assert qa["quality_improved_count"] == tr["poor_to_good"] + tr["poor_to_degraded"]
    assert qa["quality_unchanged_count"] == tr["poor_to_poor"]
    assert qa["quality_worsened_count"] == 0
    assert qa["improvement_rate_pct"] > 50.0  # Quality recovery demonstrated


# -----------------------------------------------------------------------------
# 6. OOD Routing & Containment Tests
# -----------------------------------------------------------------------------
def test_ood_routing_calculations(full_test_df: pd.DataFrame):
    """Verify OOD levels and rejection enforcement for severe cases."""
    ood = compute_ood_agent_analysis(full_test_df)

    counts = ood["counts"]
    assert counts["in_distribution"] + counts["borderline"] + counts["severe"] == len(full_test_df)

    routing = ood["routing_summary"]
    # 100% of severe OOD cases must be rejected
    severe_routing = routing["SEVERE"]
    assert severe_routing["released"] == 0
    assert severe_routing["actions"].get("REJECT", 0) == counts["severe"]

    # 100% of borderline cases must be escalated/withheld
    border_routing = routing["BORDERLINE"]
    assert border_routing["released"] == 0


# -----------------------------------------------------------------------------
# 7. Uncertainty Stratification Tests
# -----------------------------------------------------------------------------
def test_uncertainty_stratification(full_test_df: pd.DataFrame):
    """Verify that uncertainty correctly stratifies prediction accuracy."""
    unc = compute_uncertainty_agent_analysis(full_test_df, threshold=DEFAULT_OPERATING_THRESHOLD)

    assert unc["low_count"] + unc["high_count"] == len(full_test_df)
    groups = unc["groups"]

    low_acc = groups["LOW"]["released_metrics"]["accuracy"]
    high_acc = groups["HIGH"]["released_metrics"]["accuracy"]

    # Low uncertainty released cohort must achieve higher accuracy than high uncertainty
    assert low_acc > high_acc


# -----------------------------------------------------------------------------
# 8. Decision Agent Reconciliation Tests
# -----------------------------------------------------------------------------
def test_decision_reconciliation(full_test_df: pd.DataFrame):
    """Verify complete mathematical reconciliation of decision routing."""
    dec = compute_decision_agent_analysis(full_test_df)

    act = dec["action_counts"]
    outcomes = dec["downstream_outcomes"]
    n_total = len(full_test_df)

    # Total = Accept + Escalate + Reject + Repair
    assert (act["ACCEPT"] + act["ESCALATE"] + act["REJECT"] + act["REPAIR"]) == n_total

    # Total = Released + Withheld
    assert (outcomes["predictions_released"] + outcomes["predictions_withheld"]) == n_total

    # Released == Accept
    assert outcomes["predictions_released"] == act["ACCEPT"]

    # Reconciliation flag is True
    assert dec["reconciliation"]["is_valid"] is True


# -----------------------------------------------------------------------------
# 9. Paired Diagnostic Validation Tests
# -----------------------------------------------------------------------------
def test_paired_diagnostic_validation(paired_df: pd.DataFrame):
    """Verify paired diagnostic evaluation transitions, stability, and zero flips."""
    res = compute_paired_repair_diagnostic_validation(paired_df)

    repaired_subset = paired_df[paired_df["repair_applied"] == True]
    assert res["n_paired"] == len(repaired_subset)

    fwt = res["four_way_transitions"]
    assert fwt["Correct->Correct"] + fwt["Incorrect->Incorrect"] + fwt["Correct->Incorrect"] + fwt["Incorrect->Correct"] == res["n_paired"]

    # Prediction stability must be 100%
    assert res["prediction_flips"] == 0
    assert res["prediction_stability_pct"] == 100.0

    # No diagnostic regressions
    assert fwt["Correct->Incorrect"] == 0


# -----------------------------------------------------------------------------
# 10. Agent Scorecard Status Tests
# -----------------------------------------------------------------------------
def test_agent_wise_scorecard_rules(full_test_df: pd.DataFrame, paired_df: pd.DataFrame):
    """Verify agent scorecard generation and status assignment rules."""
    baseline_vs_rel = compute_baseline_vs_released_comparison(full_test_df)
    qa = compute_quality_agent_analysis(full_test_df)
    ood = compute_ood_agent_analysis(full_test_df)
    unc = compute_uncertainty_agent_analysis(full_test_df)
    dec = compute_decision_agent_analysis(full_test_df)
    ver = compute_verification_agent_analysis(full_test_df)
    paired = compute_paired_repair_diagnostic_validation(paired_df)

    scorecard = build_agent_wise_scorecard(
        full_test_comparison=baseline_vs_rel,
        quality_analysis=qa,
        ood_analysis=ood,
        unc_analysis=unc,
        decision_analysis=dec,
        paired_validation=paired,
        verification_analysis=ver,
    )

    assert len(scorecard) == 7
    valid_statuses = {
        "✓ DEMONSTRATED IMPROVEMENT",
        "≈ PRESERVED / STABLE",
        "⚠ INSUFFICIENT EVIDENCE",
        "✗ WORSENED",
        "✓ DEMONSTRATED (Quality) / ≈ PRESERVED (Diagnostic)",
    }
    for _, row in scorecard.iterrows():
        assert row["Status"] in valid_statuses
        assert len(row["Evidence"]) > 0
        assert len(row["Rule"]) > 0


# -----------------------------------------------------------------------------
# 11. Artifact Generation Functionality
# -----------------------------------------------------------------------------
def test_generate_all_evaluation_artifacts(project_root: Path):
    """Verify complete end-to-end artifact generation and output file validity."""
    res = generate_all_evaluation_artifacts(project_root)

    for fpath in res["exported_files"]:
        p = Path(fpath)
        assert p.is_file(), f"Expected file {p} does not exist"
        assert p.stat().st_size > 0, f"File {p} is empty"

    # Verify JSON content
    summary_json_path = project_root / "outputs" / "before_after_project_summary.json"
    with open(summary_json_path) as f:
        data = json.load(f)
    assert "baseline_vs_released" in data
    assert "error_containment" in data
    assert "agent_wise_scorecard" in data
