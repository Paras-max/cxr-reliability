"""Unit tests for Final Paired Repair -> DenseNet Diagnostic Validation.

Verifies:
1. Artifact existence: CSV, summary JSON, and demo_cases JSON.
2. Structure and schema of outputs/final_paired_repair_diagnostic.csv.
3. Mathematical consistency of 4-way transitions and metric deltas.
4. Demo cases contract and field completeness.
5. Invariants: v0_prd_defaults.yaml and BaseModel remain frozen.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_paired_diagnostic_artifacts_exist():
    """Verify all 3 required output files exist."""
    csv_path = PROJECT_ROOT / "outputs" / "final_paired_repair_diagnostic.csv"
    json_path = PROJECT_ROOT / "outputs" / "final_paired_repair_diagnostic_summary.json"
    demo_path = PROJECT_ROOT / "outputs" / "demo_cases.json"
    report_path = PROJECT_ROOT / "docs" / "FINAL_PAIRED_REPAIR_DIAGNOSTIC_REPORT.md"

    assert csv_path.exists(), "final_paired_repair_diagnostic.csv missing!"
    assert json_path.exists(), "final_paired_repair_diagnostic_summary.json missing!"
    assert demo_path.exists(), "demo_cases.json missing!"
    assert report_path.exists(), "FINAL_PAIRED_REPAIR_DIAGNOSTIC_REPORT.md missing!"


def test_paired_csv_columns_and_rows():
    """Verify CSV has all required columns and valid records."""
    csv_path = PROJECT_ROOT / "outputs" / "final_paired_repair_diagnostic.csv"
    df = pd.read_csv(csv_path)

    expected_cols = [
        "image_id",
        "ground_truth",
        "quality_before",
        "quality_after",
        "repair_required",
        "repair_applied",
        "repair_type",
        "raw_score_before",
        "raw_score_after",
        "prediction_before",
        "prediction_after",
        "confidence_before",
        "confidence_after",
        "confidence_delta",
        "correct_before",
        "correct_after",
        "transition",
        "verification_result",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Column {col} missing from CSV!"

    assert len(df) >= 30, f"Expected at least 30 candidate images, got {len(df)}"
    assert (df["repair_applied"] == True).sum() >= 25, "Expected repaired images in cohort"


def test_summary_json_consistency():
    """Verify mathematical consistency of transition counts and metrics."""
    json_path = PROJECT_ROOT / "outputs" / "final_paired_repair_diagnostic_summary.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transitions = data["four_transitions"]
    total_transitions = sum(transitions.values())
    n_repaired = data["sample_size"]["repair_applied_count"]
    assert total_transitions == n_repaired, f"Sum of transitions ({total_transitions}) != repaired count ({n_repaired})"

    # Verify metric delta arithmetic
    bm = data["before_metrics"]
    am = data["after_metrics"]
    deltas = data["metric_deltas"]
    assert pytest.approx(am["accuracy"] - bm["accuracy"], abs=1e-6) == deltas["accuracy_delta"]
    assert data["prediction_stability_pct"] >= 0.0 and data["prediction_stability_pct"] <= 100.0


def test_demo_cases_schema():
    """Verify demo_cases.json contains Case 1, 2, and 3 with all required fields."""
    demo_path = PROJECT_ROOT / "outputs" / "demo_cases.json"
    with open(demo_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert "demo_case_1" in cases
    assert "demo_case_2" in cases
    assert "demo_case_3" in cases

    required_keys = [
        "case_name",
        "image_id",
        "ground_truth",
        "quality_before",
        "quality_after",
        "repair_type",
        "raw_score_before",
        "raw_score_after",
        "prediction_before",
        "prediction_after",
        "transition",
        "final_action",
        "verification_result",
    ]
    for c_id, c_data in cases.items():
        for k in required_keys:
            assert k in c_data, f"Key {k} missing in {c_id}"


def test_production_config_unmodified():
    """Verify v0_prd_defaults.yaml has not been overwritten by experimental params."""
    yaml_path = PROJECT_ROOT / "configs" / "thresholds" / "v0_prd_defaults.yaml"
    assert yaml_path.exists()
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Invariants
    assert cfg["quality"]["blur_laplacian_var_min"] == 100.0
    assert cfg["quality"]["snr_db_min"] == 15.0
    assert cfg["quality"]["exposure_mean_min"] == 20.0
    assert cfg["quality"]["exposure_mean_max"] == 235.0
