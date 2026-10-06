"""tests/unit/test_calibration_integration.py

Comprehensive tests for Base Model probability calibration integration.

Verifies:
1. BaseModelResult schema additions: raw_pneumonia_score, calibrated_probability.
2. Platt calibrator loading from outputs/calibration/platt_calibrator.joblib.
3. Fallback when calibration artifacts are missing (graceful uncalibrated mode).
4. Raw score preservation during forward pass alongside calibrated probability.
5. Calibrated probability calculation using Platt scaling.
6. Uncertainty Agent strictly consuming raw_pneumonia_score (NOT calibrated_probability).
7. Proving no uncertainty collapse (raw score 0.50 yields HIGH uncertainty even with low calibrated probability).
8. PredictionSummary preserves raw_model_score, calibrated_probability, and threshold 0.522161.
9. Dynamically derived calibrated threshold from loaded Platt calibrator.
10. Dashboard component render_base_model_section displays raw score, calibrated prob, and threshold.
11. End-to-end pipeline integration with loaded calibration artifacts.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from cxr_reliability.agents.uncertainty import UncertaintyAgent
from cxr_reliability.calibration.probability_calibration import ProbabilityCalibrator
from cxr_reliability.contracts.base_model import BaseModelResult
from cxr_reliability.contracts.common import AgentName
from cxr_reliability.contracts.pipeline import (
    PredictionSummary,
)
from cxr_reliability.contracts.uncertainty import UncertaintyLevel
from cxr_reliability.dashboard.components import render_base_model_section
from cxr_reliability.models.base_model import (
    RAW_OPERATING_THRESHOLD,
    BaseModelAgent,
    ModelForward,
)
from cxr_reliability.pipeline.orchestrator import ReliabilityPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "outputs" / "calibration"


def make_test_base_model_result(
    prob: float = 0.50,
    raw_score: float | None = None,
    calibrated_prob: float | None = None,
    model_id: str = "densenet121-res224-nih",
) -> BaseModelResult:
    score = raw_score if raw_score is not None else prob
    return BaseModelResult(
        agent=AgentName.BASE_MODEL,
        version="0.4.0",
        model_id=model_id,
        target_pathology="Pneumonia",
        pneumonia_logit=score,
        pneumonia_probability=score,
        raw_pneumonia_score=score,
        calibrated_probability=calibrated_prob,
        score=score,
        label="Pneumonia" if score >= RAW_OPERATING_THRESHOLD else "Non-Pneumonia",
        reasoning=f"Base model test score: {score:.4f}",
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. BaseModelResult Schema & Validation
# ═══════════════════════════════════════════════════════════════════════

def test_base_model_result_fields():
    """Verify BaseModelResult supports both raw score and calibrated probability."""
    res = make_test_base_model_result(
        raw_score=0.45,
        calibrated_prob=0.0125,
    )
    assert res.raw_pneumonia_score == 0.45
    assert res.calibrated_probability == 0.0125
    assert res.pneumonia_probability == 0.45


def test_base_model_result_sync_validator():
    """Verify legacy construction without raw_pneumonia_score syncs properly."""
    res = BaseModelResult(
        agent=AgentName.BASE_MODEL,
        version="0.4.0",
        model_id="densenet121-res224-nih",
        label="Non-Pneumonia",
        reasoning="Legacy test",
        pneumonia_logit=0.60,
        pneumonia_probability=0.60,
    )
    assert res.raw_pneumonia_score == 0.60
    assert res.calibrated_probability is None


# ═══════════════════════════════════════════════════════════════════════
# 2. Platt Calibrator Loading & Fallback
# ═══════════════════════════════════════════════════════════════════════

def test_platt_calibrator_loading_from_outputs():
    """Verify Platt calibrator loads successfully from project calibration directory."""
    if not (CALIBRATION_DIR / "platt_calibrator.joblib").exists():
        pytest.skip("outputs/calibration/platt_calibrator.joblib not present")

    agent = BaseModelAgent(device="cpu", calibration_dir=CALIBRATION_DIR)
    assert agent.is_calibrated is True
    assert agent.calibrator is not None
    assert getattr(agent.calibrator, "_platt", None) is not None


def test_calibrator_missing_fallback(tmp_path):
    """Verify missing calibration artifacts result in safe uncalibrated mode without error."""
    agent = BaseModelAgent(device="cpu", calibration_dir=tmp_path / "nonexistent")
    assert agent.is_calibrated is False
    assert agent.calibrator is None


# ═══════════════════════════════════════════════════════════════════════
# 3. Raw Score Preservation & Calibrated Generation
# ═══════════════════════════════════════════════════════════════════════

def test_raw_score_preservation_and_calibrated_generation():
    """Verify forward inference preserves raw score and computes calibrated probability."""
    mock_cal = MagicMock()
    mock_cal.transform.side_effect = lambda s, method="platt": 0.0163 if abs(s - 0.5) < 1e-4 else 0.05

    agent = BaseModelAgent(device="cpu", calibrator=mock_cal)

    # Mock FeatureExtractor context manager to avoid needing a full PyTorch hook
    with patch("cxr_reliability.models.base_model.FeatureExtractor") as mock_extractor_cls:
        mock_extractor = MagicMock()
        mock_extractor.pooled_features.return_value = torch.zeros((1, 1024))
        mock_extractor_cls.return_value.__enter__.return_value = mock_extractor
        mock_extractor_cls.return_value.__exit__.return_value = False

        agent._model = MagicMock()
        mock_output = torch.zeros((1, 18))
        mock_output[0, 8] = 0.50
        agent._model.return_value = mock_output

        dummy_tensor = torch.zeros((1, 1, 224, 224))
        fwd = agent.run(dummy_tensor)

    assert fwd.result.raw_pneumonia_score == pytest.approx(0.50)
    assert fwd.result.pneumonia_logit == pytest.approx(0.50)
    assert fwd.result.calibrated_probability == pytest.approx(0.0163)
    mock_cal.transform.assert_called_once_with(0.50, method="platt")


# ═══════════════════════════════════════════════════════════════════════
# 4. Uncertainty Agent Strictly Consumes Raw Score
# ═══════════════════════════════════════════════════════════════════════

def test_uncertainty_agent_strictly_uses_raw_score():
    """
    CRITICAL TEST:
    Verify Uncertainty Agent evaluates uncertainty using raw_pneumonia_score.
    If raw score is 0.50 (maximum ambiguity), uncertainty MUST be HIGH.
    If it improperly consumed calibrated_probability (0.016), confidence would be 0.984
    and incorrectly classify as LOW uncertainty!
    """
    agent = UncertaintyAgent()

    # Ambiguous raw model score (0.50) with low calibrated probability (0.016)
    model_res = make_test_base_model_result(
        raw_score=0.50,
        calibrated_prob=0.01628,
    )

    unc_res = agent.run(model_res)

    # Raw score 0.50 -> conf = 0.50, norm_entropy = 1.0 -> HIGH uncertainty
    assert unc_res.raw_model_score == pytest.approx(0.50)
    assert unc_res.confidence == pytest.approx(0.50)
    assert unc_res.normalized_entropy == pytest.approx(1.0)
    assert unc_res.uncertainty_level == UncertaintyLevel.HIGH


def test_uncertainty_agent_handles_model_forward_with_raw_score():
    """Verify ModelForward input extracts raw_pneumonia_score."""
    agent = UncertaintyAgent()
    model_res = make_test_base_model_result(
        raw_score=0.52,
        calibrated_prob=0.018,
    )
    fwd = ModelForward(
        result=model_res,
        raw_probs=torch.full((18,), 0.52),
        features=torch.zeros(1024),
    )
    unc_res = agent.run(fwd)
    assert unc_res.raw_model_score == pytest.approx(0.52)
    assert unc_res.confidence == pytest.approx(0.52)


# ═══════════════════════════════════════════════════════════════════════
# 5. PredictionSummary & Operating Threshold 0.522161
# ═══════════════════════════════════════════════════════════════════════

def test_prediction_summary_threshold_and_scores():
    """Verify PredictionSummary correctly preserves raw score, calibrated prob, and threshold."""
    pred = PredictionSummary(
        raw_model_score=0.60,
        calibrated_probability=0.025,
        pneumonia_probability=0.025,
        positive=True,
        decision_threshold=0.522161,
        raw_decision_threshold=0.522161,
        calibrated_decision_threshold=0.0174,
        source_model_id="densenet121-res224-nih",
    )
    assert pred.raw_model_score == pytest.approx(0.60)
    assert pred.calibrated_probability == pytest.approx(0.025)
    assert pred.decision_threshold == pytest.approx(0.522161)
    assert pred.raw_decision_threshold == pytest.approx(0.522161)
    assert pred.calibrated_decision_threshold == pytest.approx(0.0174)
    assert pred.positive is True


def test_operating_threshold_classification():
    """Verify raw score comparison against validated threshold 0.522161."""
    assert RAW_OPERATING_THRESHOLD == pytest.approx(0.522161)

    # Sub-threshold: 0.51 -> Negative
    sub_res = make_test_base_model_result(raw_score=0.51)
    # Above threshold: 0.53 -> Positive
    above_res = make_test_base_model_result(raw_score=0.53)

    pipeline = ReliabilityPipeline.__new__(ReliabilityPipeline)
    pipeline.base_model = MagicMock(calibrator=None)

    pred_sub = pipeline._build_prediction_summary(sub_res)
    pred_above = pipeline._build_prediction_summary(above_res)

    assert pred_sub.positive is False
    assert pred_above.positive is True
    assert pred_sub.decision_threshold == pytest.approx(0.522161)


# ═══════════════════════════════════════════════════════════════════════
# 6. Dashboard Component Rendering
# ═══════════════════════════════════════════════════════════════════════

def test_dashboard_render_base_model_section():
    """Verify render_base_model_section displays raw score, calibrated prob, and threshold."""
    base_res = make_test_base_model_result(
        raw_score=0.55,
        calibrated_prob=0.0195,
    )
    pred = PredictionSummary(
        raw_model_score=0.55,
        calibrated_probability=0.0195,
        pneumonia_probability=0.0195,
        positive=True,
        decision_threshold=0.522161,
        raw_decision_threshold=0.522161,
        calibrated_decision_threshold=0.0174,
        source_model_id="densenet121-res224-nih",
    )

    with patch("streamlit.metric") as mock_metric, patch("streamlit.columns") as mock_cols, patch("streamlit.subheader"), patch("streamlit.caption"), patch("streamlit.write"):
        mock_cols.return_value = [MagicMock(), MagicMock(), MagicMock()]
        render_base_model_section(base_res, pred)

        calls = [c.args for c in mock_metric.call_args_list]
        metric_labels = [c[0] for c in calls]
        assert "Raw Model Sigmoid Score" in metric_labels
        assert "Calibrated Probability" in metric_labels
        assert "Raw-Score Decision Threshold" in metric_labels


# ═══════════════════════════════════════════════════════════════════════
# 7. End-to-End Pipeline Integration with Calibration
# ═══════════════════════════════════════════════════════════════════════

def test_pipeline_integration_build_prediction_summary_with_real_calibrator():
    """Verify pipeline integrates with loaded Platt calibrator dynamically."""
    if not (CALIBRATION_DIR / "platt_calibrator.joblib").exists():
        pytest.skip("outputs/calibration/platt_calibrator.joblib not present")

    calibrator = ProbabilityCalibrator.load(CALIBRATION_DIR)
    mock_base_model = MagicMock()
    mock_base_model.calibrator = calibrator

    pipeline = ReliabilityPipeline.__new__(ReliabilityPipeline)
    pipeline.base_model = mock_base_model

    base_res = make_test_base_model_result(raw_score=0.522161)

    summary = pipeline._build_prediction_summary(base_res)
    assert summary.raw_model_score == pytest.approx(0.522161)
    assert summary.calibrated_probability is not None
    # Platt transform of 0.522161 is ~0.0174
    assert 0.015 < summary.calibrated_probability < 0.020
    assert summary.calibrated_decision_threshold == pytest.approx(summary.calibrated_probability, rel=1e-3)
    assert summary.positive is True
