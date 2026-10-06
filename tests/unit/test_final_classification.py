"""Unit tests for the new FinalClassification field and additive dashboard display.

Verifies all 12 test requirements specified for the controlled, additive modification:
    TEST 1: Accepted positive prediction -> PNEUMONIA
    TEST 2: Accepted negative prediction -> NO_PNEUMONIA
    TEST 3: Repair -> verification passes -> positive -> PNEUMONIA
    TEST 4: Repair -> verification passes -> negative -> NO_PNEUMONIA
    TEST 5: Repair -> verification fails -> HUMAN_REVIEW_REQUIRED
    TEST 6: High uncertainty / escalation -> HUMAN_REVIEW_REQUIRED
    TEST 7: Borderline OOD -> safe routing & HUMAN_REVIEW_REQUIRED if not released
    TEST 8: Severe OOD -> HUMAN_REVIEW_REQUIRED
    TEST 9: Processing error -> HUMAN_REVIEW_REQUIRED
    TEST 10: Prediction withheld -> HUMAN_REVIEW_REQUIRED
    TEST 11: Prediction released consistency -> PNEUMONIA or NO_PNEUMONIA
    TEST 12: Dashboard displays new classification while all existing sections remain available
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cxr_reliability.contracts.base_model import BaseModelResult
from cxr_reliability.contracts.common import AgentName
from cxr_reliability.contracts.decision import Action, DecisionResult
from cxr_reliability.contracts.ood import OODLevel, OODResult
from cxr_reliability.contracts.pipeline import (
    FinalClassification,
    PipelineOutput,
    PipelineResult,
    PipelineState,
    PredictionSummary,
    ReliabilityLabel,
)
from cxr_reliability.contracts.quality import (
    DefectType,
    QualityFlags,
    QualityLevel,
    QualityResult,
)
from cxr_reliability.contracts.repair import RepairResult, RepairStep
from cxr_reliability.contracts.uncertainty import UncertaintyLevel, UncertaintyResult
from cxr_reliability.contracts.verification import (
    NextStep,
    VerificationResult,
    VerificationStatus,
)
from cxr_reliability.models.base_model import ModelForward
from cxr_reliability.pipeline.orchestrator import ReliabilityPipeline


# ── Helpers & Fixtures ───────────────────────────────────────────────────────


def make_quality(level=QualityLevel.GOOD, blur=False) -> QualityResult:
    return QualityResult(
        agent=AgentName.QUALITY,
        version="0.7.0",
        laplacian_variance=300.0 if not blur else 40.0,
        blur_pct=5.0 if not blur else 50.0,
        snr_db=30.0,
        mean_intensity=128.0,
        histogram_std=45.0,
        flags=QualityFlags(blur=blur, noise=False, exposure=False),
        overall=level,
        score=0.9 if level == QualityLevel.GOOD else 0.4,
        label=level.value,
        reasoning=f"Quality evaluated as {level.value}",
    )


def make_base_model_forward(raw_prob=0.85, model_id="densenet121-res224-nih") -> ModelForward:
    res = BaseModelResult(
        agent=AgentName.BASE_MODEL,
        version="0.4.0",
        model_id=model_id,
        target_pathology="Pneumonia",
        pneumonia_logit=raw_prob,
        pneumonia_probability=raw_prob,
        raw_pneumonia_score=raw_prob,
        all_pathology_outputs={"Pneumonia": raw_prob},
        score=raw_prob,
        label="Pneumonia" if raw_prob >= 0.522161 else "Non-Pneumonia",
        reasoning=f"Base model raw score: {raw_prob:.4f}",
    )
    return ModelForward(
        result=res,
        raw_probs=torch.full((18,), raw_prob),
        features=torch.ones(1024),
        inference_ms=10.0,
    )


def make_ood(level=OODLevel.IN_DISTRIBUTION, dist=10.0) -> OODResult:
    return OODResult(
        agent=AgentName.OOD,
        version="0.5.0",
        mahalanobis_distance=dist,
        mahalanobis_threshold=25.0,
        level=level,
        score=0.2,
        label=level.value,
        reasoning=f"OOD evaluated as {level.value}",
    )


def make_uncertainty(level=UncertaintyLevel.LOW, conf=0.90) -> UncertaintyResult:
    return UncertaintyResult(
        agent=AgentName.UNCERTAINTY,
        version="0.6.0",
        raw_model_score=0.85,
        confidence=conf,
        entropy=0.15 if level == UncertaintyLevel.LOW else 0.65,
        normalized_entropy=0.15 if level == UncertaintyLevel.LOW else 0.65,
        uncertainty_level=level,
        score=conf,
        label=level.value,
        reasoning=f"Uncertainty evaluated as {level.value}",
    )


def make_decision(action=Action.ACCEPT, rule_id="RULE-ACCEPT-01") -> DecisionResult:
    return DecisionResult(
        agent=AgentName.DECISION,
        version="v1-0.8.0",
        action=action,
        rule_id=rule_id,
        driving_signals={"quality": "good", "ood": "in_distribution"},
        reliability_label="accepted" if action == Action.ACCEPT else "needs_human_review",
        score=1.0,
        label=action.value,
        reasoning=f"Action decided: {action.value}",
    )


def make_verification(verified=True, next_step=NextStep.RELEASE, delta_conf=0.20):
    return VerificationResult(
        agent=AgentName.VERIFICATION,
        version="0.10.0",
        verified=verified,
        status=VerificationStatus.VERIFIED if verified else VerificationStatus.ESCALATE,
        next_step=next_step,
        delta_confidence=delta_conf,
        delta_quality=15.0,
        delta_ood=-2.0,
        label_flipped=False,
        min_confidence_gain_used=0.15,
        threshold_is_provisional=True,
        score=delta_conf,
        label="verified" if verified else "failed",
        reasoning=f"Verification verdict: next_step={next_step.value}",
    )


def build_mock_pipeline(
    quality_res=None,
    model_forward=None,
    ood_res=None,
    uncertainty_res=None,
    decision_res=None,
    repair_img=None,
    repair_res=None,
    verif_res=None,
    after_model_forward=None,
) -> ReliabilityPipeline:
    mock_q = MagicMock()
    mock_q.run.return_value = quality_res or make_quality()

    mock_bm = MagicMock()
    mock_bm.run.return_value = model_forward or make_base_model_forward(0.85)
    mock_bm.calibrator = None

    mock_ood = MagicMock()
    mock_ood.run.return_value = ood_res or make_ood()

    mock_unc = MagicMock()
    mock_unc.run.return_value = uncertainty_res or make_uncertainty()

    mock_dec = MagicMock()
    dec = decision_res or make_decision(Action.ACCEPT)
    mock_dec.decide.return_value = dec
    mock_dec.evaluate.return_value = dec

    mock_rep = MagicMock()
    mock_rep.run.return_value = (
        repair_img if repair_img is not None else np.zeros((224, 224), dtype=np.uint8),
        repair_res
        or RepairResult(
            agent=AgentName.REPAIR,
            version="0.9.0",
            repaired=True,
            repair_applied=True,
            defects_detected=[DefectType.BLUR],
            steps=[RepairStep(method="unsharp_mask", defect="blur", parameters={})],
            label="repaired",
            reasoning="Applied unsharp mask",
        ),
    )

    mock_ver = MagicMock()
    mock_ver.run.return_value = verif_res or make_verification()

    if after_model_forward is not None:
        # Second call to base_model returns after_model_forward
        mock_bm.run.side_effect = [
            model_forward or make_base_model_forward(0.85),
            after_model_forward,
        ]

    return ReliabilityPipeline(
        quality_agent=mock_q,
        base_model=mock_bm,
        ood_agent=mock_ood,
        uncertainty_agent=mock_unc,
        decision_agent=mock_dec,
        repair_agent=mock_rep,
        verification_agent=mock_ver,
    )


# ── TEST 1: Accepted positive prediction -> PNEUMONIA ─────────────────────────


def test_1_accepted_positive():
    pipeline = build_mock_pipeline(
        model_forward=make_base_model_forward(raw_prob=0.88),  # > 0.522161 threshold
        decision_res=make_decision(Action.ACCEPT),
    )
    result = pipeline.predict(np.zeros((224, 224), dtype=np.uint8))

    assert result.pipeline_state == PipelineState.ACCEPT
    assert result.final_action == Action.ACCEPT
    assert result.reliability_label == ReliabilityLabel.ACCEPTED
    assert result.needs_human_review is False
    assert result.prediction is not None
    assert result.prediction.positive is True
    assert result.final_classification == FinalClassification.PNEUMONIA


# ── TEST 2: Accepted negative prediction -> NO_PNEUMONIA ───────────────────────


def test_2_accepted_negative():
    pipeline = build_mock_pipeline(
        model_forward=make_base_model_forward(raw_prob=0.20),  # < 0.522161 threshold
        decision_res=make_decision(Action.ACCEPT),
    )
    result = pipeline.predict(np.zeros((224, 224), dtype=np.uint8))

    assert result.pipeline_state == PipelineState.ACCEPT
    assert result.final_action == Action.ACCEPT
    assert result.reliability_label == ReliabilityLabel.ACCEPTED
    assert result.needs_human_review is False
    assert result.prediction is not None
    assert result.prediction.positive is False
    assert result.final_classification == FinalClassification.NO_PNEUMONIA


# ── TEST 3: Repair -> verification passes -> positive -> PNEUMONIA ────────────


def test_3_repair_verified_positive():
    pipeline = build_mock_pipeline(
        quality_res=make_quality(level=QualityLevel.POOR, blur=True),
        model_forward=make_base_model_forward(raw_prob=0.45),
        decision_res=make_decision(Action.REPAIR, rule_id="RULE-REPAIR-01"),
        after_model_forward=make_base_model_forward(raw_prob=0.78),  # > threshold
        verif_res=make_verification(verified=True, next_step=NextStep.RELEASE, delta_conf=0.25),
    )
    result = pipeline.predict(np.zeros((224, 224), dtype=np.uint8))

    assert result.pipeline_state == PipelineState.VERIFIED
    assert result.final_action == Action.ACCEPT
    assert result.reliability_label == ReliabilityLabel.ACCEPTED_AFTER_REPAIR
    assert result.needs_human_review is False
    assert result.prediction is not None
    assert result.prediction.positive is True
    assert result.final_classification == FinalClassification.PNEUMONIA


# ── TEST 4: Repair -> verification passes -> negative -> NO_PNEUMONIA ─────────


def test_4_repair_verified_negative():
    pipeline = build_mock_pipeline(
        quality_res=make_quality(level=QualityLevel.POOR, blur=True),
        model_forward=make_base_model_forward(raw_prob=0.45),
        decision_res=make_decision(Action.REPAIR, rule_id="RULE-REPAIR-01"),
        after_model_forward=make_base_model_forward(raw_prob=0.15),  # < threshold
        verif_res=make_verification(verified=True, next_step=NextStep.RELEASE, delta_conf=0.25),
    )
    result = pipeline.predict(np.zeros((224, 224), dtype=np.uint8))

    assert result.pipeline_state == PipelineState.VERIFIED
    assert result.final_action == Action.ACCEPT
    assert result.reliability_label == ReliabilityLabel.ACCEPTED_AFTER_REPAIR
    assert result.needs_human_review is False
    assert result.prediction is not None
    assert result.prediction.positive is False
    assert result.final_classification == FinalClassification.NO_PNEUMONIA


# ── TEST 5: Repair -> verification fails -> HUMAN_REVIEW_REQUIRED ─────────────


def test_5_repair_verification_fails():
    pipeline = build_mock_pipeline(
        quality_res=make_quality(level=QualityLevel.POOR, blur=True),
        decision_res=make_decision(Action.REPAIR),
        verif_res=make_verification(verified=False, next_step=NextStep.ESCALATE, delta_conf=-0.10),
    )
    result = pipeline.predict(np.zeros((224, 224), dtype=np.uint8))

    assert result.needs_human_review is True
    assert result.prediction is None
    assert result.final_classification == FinalClassification.HUMAN_REVIEW_REQUIRED


# ── TEST 6: High uncertainty / escalation -> HUMAN_REVIEW_REQUIRED ───────────


def test_6_high_uncertainty_escalation():
    pipeline = build_mock_pipeline(
        uncertainty_res=make_uncertainty(level=UncertaintyLevel.HIGH, conf=0.51),
        decision_res=make_decision(Action.ESCALATE, rule_id="RULE-UNCERTAINTY-ESCALATE"),
    )
    result = pipeline.predict(np.zeros((224, 224), dtype=np.uint8))

    assert result.pipeline_state == PipelineState.ESCALATE
    assert result.final_action == Action.ESCALATE
    assert result.needs_human_review is True
    assert result.prediction is None
    assert result.final_classification == FinalClassification.HUMAN_REVIEW_REQUIRED


# ── TEST 7: Borderline OOD -> safe routing & HUMAN_REVIEW_REQUIRED ────────────


def test_7_borderline_ood():
    pipeline = build_mock_pipeline(
        ood_res=make_ood(level=OODLevel.BORDERLINE, dist=26.5),
        decision_res=make_decision(Action.ESCALATE, rule_id="RULE-NEAR-OOD-ESCALATE"),
    )
    result = pipeline.predict(np.zeros((224, 224), dtype=np.uint8))

    assert result.needs_human_review is True
    assert result.prediction is None
    assert result.final_classification == FinalClassification.HUMAN_REVIEW_REQUIRED


# ── TEST 8: Severe OOD -> HUMAN_REVIEW_REQUIRED ───────────────────────────────


def test_8_severe_ood():
    pipeline = build_mock_pipeline(
        ood_res=make_ood(level=OODLevel.SEVERE, dist=95.0),
        decision_res=make_decision(Action.REJECT, rule_id="RULE-FAR-OOD-REJECT"),
    )
    result = pipeline.predict(np.zeros((224, 224), dtype=np.uint8))

    assert result.pipeline_state == PipelineState.REJECT
    assert result.final_action == Action.REJECT
    assert result.needs_human_review is True
    assert result.prediction is None
    assert result.final_classification == FinalClassification.HUMAN_REVIEW_REQUIRED


# ── TEST 9: Processing error -> HUMAN_REVIEW_REQUIRED ─────────────────────────


def test_9_processing_error():
    pipeline = build_mock_pipeline()
    # Passing empty image triggers input validation error
    result = pipeline.predict(np.array([], dtype=np.uint8))

    assert result.pipeline_state == PipelineState.ERROR
    assert result.needs_human_review is True
    assert result.prediction is None
    assert result.final_classification == FinalClassification.HUMAN_REVIEW_REQUIRED


# ── TEST 10: Prediction withheld -> HUMAN_REVIEW_REQUIRED ─────────────────────


def test_10_prediction_withheld_invariant():
    res = PipelineResult(
        audit_id="test_withheld",
        pipeline_state=PipelineState.ESCALATE,
        final_action=Action.ESCALATE,
        reliability_label=ReliabilityLabel.NEEDS_HUMAN_REVIEW,
        needs_human_review=True,
        prediction=None,
    )
    assert res.final_classification == FinalClassification.HUMAN_REVIEW_REQUIRED

    # Disagreeing classification must raise ValueError
    with pytest.raises(ValueError, match="final_classification cannot release diagnosis"):
        PipelineResult(
            audit_id="test_withheld_invalid",
            pipeline_state=PipelineState.ESCALATE,
            final_action=Action.ESCALATE,
            reliability_label=ReliabilityLabel.NEEDS_HUMAN_REVIEW,
            needs_human_review=True,
            final_classification=FinalClassification.PNEUMONIA,
            prediction=None,
        )


# ── TEST 11: Prediction released consistency ──────────────────────────────────


def test_11_prediction_released_consistency():
    pred_pos = PredictionSummary(
        pneumonia_probability=0.88,
        positive=True,
        decision_threshold=0.522161,
        source_model_id="densenet121",
    )
    res_pos = PipelineResult(
        audit_id="test_rel_pos",
        pipeline_state=PipelineState.ACCEPT,
        final_action=Action.ACCEPT,
        reliability_label=ReliabilityLabel.ACCEPTED,
        needs_human_review=False,
        prediction=pred_pos,
    )
    assert res_pos.final_classification == FinalClassification.PNEUMONIA

    pred_neg = PredictionSummary(
        pneumonia_probability=0.15,
        positive=False,
        decision_threshold=0.522161,
        source_model_id="densenet121",
    )
    res_neg = PipelineResult(
        audit_id="test_rel_neg",
        pipeline_state=PipelineState.ACCEPT,
        final_action=Action.ACCEPT,
        reliability_label=ReliabilityLabel.ACCEPTED,
        needs_human_review=False,
        prediction=pred_neg,
    )
    assert res_neg.final_classification == FinalClassification.NO_PNEUMONIA


# ── TEST 12: Dashboard renders classification & preserves existing sections ───


def test_12_dashboard_renders_classification_and_preserves_sections():
    import cxr_reliability.dashboard.components as comp
    assert hasattr(comp, "render_pneumonia_classification")
    assert hasattr(comp, "render_final_result")
    assert hasattr(comp, "render_base_model_section")
    assert hasattr(comp, "render_quality_section")
    assert hasattr(comp, "render_ood_section")
    assert hasattr(comp, "render_uncertainty_section")
    assert hasattr(comp, "render_decision_section")
    assert hasattr(comp, "render_repair_section")
    assert hasattr(comp, "render_verification_section")
    assert hasattr(comp, "render_agent_trace")

    # Verify rendering does not raise errors on all three outcomes
    res_pneumonia = PipelineResult(
        audit_id="ui_pos",
        pipeline_state=PipelineState.ACCEPT,
        final_action=Action.ACCEPT,
        reliability_label=ReliabilityLabel.ACCEPTED,
        needs_human_review=False,
        prediction=PredictionSummary(
            pneumonia_probability=0.85,
            positive=True,
            decision_threshold=0.522161,
            source_model_id="densenet121",
        ),
    )
    res_no_pneumonia = PipelineResult(
        audit_id="ui_neg",
        pipeline_state=PipelineState.ACCEPT,
        final_action=Action.ACCEPT,
        reliability_label=ReliabilityLabel.ACCEPTED,
        needs_human_review=False,
        prediction=PredictionSummary(
            pneumonia_probability=0.15,
            positive=False,
            decision_threshold=0.522161,
            source_model_id="densenet121",
        ),
    )
    res_human_review = PipelineResult(
        audit_id="ui_hr",
        pipeline_state=PipelineState.ESCALATE,
        final_action=Action.ESCALATE,
        reliability_label=ReliabilityLabel.NEEDS_HUMAN_REVIEW,
        needs_human_review=True,
        prediction=None,
    )

    with patch("streamlit.subheader"), patch("streamlit.markdown") as mock_md:
        comp.render_pneumonia_classification(res_pneumonia)
        assert any("PNEUMONIA DETECTED" in str(call) for call in mock_md.call_args_list)

    with patch("streamlit.subheader"), patch("streamlit.markdown") as mock_md:
        comp.render_pneumonia_classification(res_no_pneumonia)
        assert any("PNEUMONIA NOT DETECTED" in str(call) for call in mock_md.call_args_list)

    with patch("streamlit.subheader"), patch("streamlit.markdown") as mock_md:
        comp.render_pneumonia_classification(res_human_review)
        assert any("HUMAN REVIEW REQUIRED" in str(call) for call in mock_md.call_args_list)
