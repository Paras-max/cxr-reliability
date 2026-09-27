"""
tests/unit/test_orchestrator.py
Phase 11 — Pipeline Orchestrator Test Suite.

Contains exactly 21 test items:
    20 unit tests + 1 integration smoke test.

Test Plan Items:
    1.  test_pipeline_accept_path
    2.  test_pipeline_repair_path
    3.  test_pipeline_verified_release
    4.  test_pipeline_verification_escalate
    5.  test_pipeline_verification_reject
    6.  test_pipeline_initial_escalate
    7.  test_pipeline_initial_reject
    8.  test_ood_dependency_ordering
    9.  test_decision_agent_inputs
    10. test_verification_agent_inputs
    11. test_repair_not_applied_escalation
    12. test_original_image_preserved
    13. test_agent_failure_safe_handling
    14. test_bounded_repair_loop
    15. test_pipeline_determinism
    16. test_pipeline_result_contract_validation
    17. test_pipeline_human_readable_reasoning
    18. test_pipeline_no_clinical_claims
    19. test_pipeline_retains_intermediate_results
    20. test_pipeline_empty_or_invalid_input
    21. test_integration_smoke_pipeline (integration smoke test)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cxr_reliability.contracts.base_model import BaseModelResult
from cxr_reliability.contracts.common import AgentName, DISCLAIMER
from cxr_reliability.contracts.decision import Action, DecisionResult, DecisionState
from cxr_reliability.contracts.ood import OODLevel, OODResult
from cxr_reliability.contracts.pipeline import (
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


# ── Test Fixtures & Factories ────────────────────────────────────────────────


def make_quality_result(level=QualityLevel.GOOD, blur=False, noise=False, exp=False):
    return QualityResult(
        agent=AgentName.QUALITY,
        version="0.7.0",
        laplacian_variance=300.0 if level == QualityLevel.GOOD else 50.0,
        blur_pct=5.0 if not blur else 45.0,
        snr_db=30.0 if not noise else 12.0,
        mean_intensity=128.0 if not exp else 240.0,
        histogram_std=45.0,
        flags=QualityFlags(blur=blur, noise=noise, exposure=exp),
        overall=level,
        score=0.9 if level == QualityLevel.GOOD else 0.4,
        label=level.value,
        reasoning=f"Quality evaluated as {level.value}",
    )


def make_base_model_result(prob=0.88, model_id="densenet121-res224-nih"):
    return BaseModelResult(
        agent=AgentName.BASE_MODEL,
        version="0.4.0",
        model_id=model_id,
        target_pathology="Pneumonia",
        pneumonia_logit=prob,
        pneumonia_probability=prob,
        all_pathology_outputs={"Pneumonia": prob},
        score=prob,
        label="Pneumonia" if prob >= 0.5 else "Non-Pneumonia",
        reasoning=f"Base model raw score: {prob:.4f}",
    )


def make_model_forward(prob=0.88, model_id="densenet121-res224-nih"):
    res = make_base_model_result(prob, model_id)
    return ModelForward(
        result=res,
        raw_probs=torch.full((18,), prob),
        features=torch.ones(1024),
        inference_ms=12.5,
    )


def make_ood_result(level=OODLevel.IN_DISTRIBUTION, dist=12.0):
    return OODResult(
        agent=AgentName.OOD,
        version="0.5.0",
        mahalanobis_distance=dist,
        mahalanobis_threshold=25.0,
        level=level,
        score=dist / 50.0,
        label=level.value,
        reasoning=f"OOD assessed as {level.value}",
    )


def make_uncertainty_result(level=UncertaintyLevel.LOW, prob=0.88):
    conf = max(prob, 1.0 - prob)
    return UncertaintyResult(
        agent=AgentName.UNCERTAINTY,
        version="0.6.0",
        raw_model_score=prob,
        confidence=conf,
        entropy=0.25 if level == UncertaintyLevel.LOW else 0.85,
        normalized_entropy=0.25 if level == UncertaintyLevel.LOW else 0.85,
        uncertainty_level=level,
        score=conf,
        label=level.value,
        reasoning=f"Uncertainty assessed as {level.value}",
    )


def make_decision_result(action=Action.ACCEPT, rule_id="RULE-01"):
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


def make_repair_result(applied=True, changed=True, refused=False):
    return RepairResult(
        agent=AgentName.REPAIR,
        version="0.9.0",
        repaired=applied,
        repair_applied=applied,
        image_changed=changed,
        refused_bounds=refused,
        steps=[RepairStep(defect=DefectType.EXPOSURE, method="clahe", parameters={"clip_limit": 2.0})] if applied else [],
        score=1.0 if applied else 0.0,
        label="repaired" if applied else "skipped",
        reasoning="Repair applied" if applied else "Repair skipped",
    )


def make_verification_result(verified=True, next_step=NextStep.RELEASE, delta_conf=0.18):
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



class MockPipelineComponents:
    """Configurable collection of mock agents for pipeline testing."""

    def __init__(self):
        self.quality_agent = MagicMock()
        self.quality_agent.run.return_value = make_quality_result(QualityLevel.GOOD)

        self.base_model = MagicMock()
        self.base_model.run.return_value = make_model_forward(0.88)

        self.ood_agent = MagicMock()
        self.ood_agent.run.return_value = make_ood_result(OODLevel.IN_DISTRIBUTION)

        self.uncertainty_agent = MagicMock()
        self.uncertainty_agent.run.return_value = make_uncertainty_result(UncertaintyLevel.LOW)

        self.decision_agent = MagicMock()
        self.decision_agent.decide.return_value = make_decision_result(Action.ACCEPT)

        self.repair_agent = MagicMock()
        self.repair_agent.run.side_effect = lambda img, quality=None, decision=None, image_id=None: (
            img.copy() if hasattr(img, "copy") else img,
            make_repair_result(applied=True),
        )

        self.verification_agent = MagicMock()
        self.verification_agent.run.return_value = make_verification_result(
            verified=True, next_step=NextStep.RELEASE
        )

        self.audit_logger = MagicMock()

    def build_pipeline(self, max_repair_attempts=1) -> ReliabilityPipeline:
        cfg = {"loop_limits": {"max_repair_attempts": max_repair_attempts}}
        return ReliabilityPipeline(
            config=cfg,
            thresholds=None,
            quality_agent=self.quality_agent,
            base_model=self.base_model,
            uncertainty_agent=self.uncertainty_agent,
            ood_agent=self.ood_agent,
            decision_agent=self.decision_agent,
            repair_agent=self.repair_agent,
            verification_agent=self.verification_agent,
            audit_logger=self.audit_logger,
        )


@pytest.fixture
def mocks():
    return MockPipelineComponents()


@pytest.fixture
def synthetic_image():
    # 2D grayscale uint8 array
    return np.full((224, 224), 128, dtype=np.uint8)


# ── 1. ACCEPT Path ───────────────────────────────────────────────────────────


def test_pipeline_accept_path(mocks, synthetic_image):
    """Test 1: Good quality, in-distribution, low uncertainty -> ACCEPT with released prediction."""
    pipeline = mocks.build_pipeline()
    result = pipeline.predict(synthetic_image, input_id="test_accept_01")

    assert result.pipeline_state == PipelineState.ACCEPT
    assert result.final_action == Action.ACCEPT
    assert result.reliability_label == ReliabilityLabel.ACCEPTED
    assert result.needs_human_review is False
    assert result.prediction is not None
    assert result.prediction.pneumonia_probability == pytest.approx(0.88)
    assert result.prediction.positive is True
    assert result.audit_id == "test_accept_01"
    assert result.repair is None
    assert result.verification is None


# ── 2. REPAIR Path ───────────────────────────────────────────────────────────


def test_pipeline_repair_path(mocks, synthetic_image):
    """Test 2: Degraded quality -> REPAIR -> re-inference -> VERIFIED with released prediction."""
    mocks.quality_agent.run.side_effect = [
        make_quality_result(QualityLevel.DEGRADED, exp=True),  # before
        make_quality_result(QualityLevel.GOOD),                # after
    ]
    mocks.base_model.run.side_effect = [
        make_model_forward(0.60),  # before
        make_model_forward(0.85),  # after
    ]
    mocks.decision_agent.decide.return_value = make_decision_result(Action.REPAIR, "R-REPAIR-01")
    mocks.verification_agent.run.return_value = make_verification_result(
        verified=True, next_step=NextStep.RELEASE, delta_conf=0.25
    )

    pipeline = mocks.build_pipeline()
    result = pipeline.predict(synthetic_image, input_id="test_repair_01")

    assert result.pipeline_state == PipelineState.VERIFIED
    assert result.final_action == Action.ACCEPT
    assert result.reliability_label == ReliabilityLabel.ACCEPTED_AFTER_REPAIR
    assert result.needs_human_review is False
    assert result.prediction is not None
    assert result.prediction.pneumonia_probability == pytest.approx(0.85)
    assert result.repair is not None
    assert result.repair.repair_applied is True
    assert result.verification is not None
    assert result.verification.verified is True
    assert result.after_repair_quality is not None
    assert result.after_repair_base_model is not None


# ── 3. VERIFIED Release ──────────────────────────────────────────────────────


def test_pipeline_verified_release(mocks, synthetic_image):
    """Test 3: Verification agent returning NextStep.RELEASE sets ACCEPTED_AFTER_REPAIR."""
    mocks.decision_agent.decide.return_value = make_decision_result(Action.REPAIR)
    mocks.verification_agent.run.return_value = make_verification_result(
        verified=True, next_step=NextStep.RELEASE
    )

    pipeline = mocks.build_pipeline()
    result = pipeline.predict(synthetic_image)

    assert result.pipeline_state == PipelineState.VERIFIED
    assert result.reliability_label == ReliabilityLabel.ACCEPTED_AFTER_REPAIR
    assert result.needs_human_review is False
    assert result.prediction is not None


# ── 4. Verification ESCALATE ─────────────────────────────────────────────────


def test_pipeline_verification_escalate(mocks, synthetic_image):
    """Test 4: Repair executed but verification fails with ESCALATE -> prediction withheld."""
    mocks.decision_agent.decide.return_value = make_decision_result(Action.REPAIR)
    mocks.verification_agent.run.return_value = make_verification_result(
        verified=False, next_step=NextStep.ESCALATE
    )

    pipeline = mocks.build_pipeline()
    result = pipeline.predict(synthetic_image)

    assert result.pipeline_state == PipelineState.ESCALATE
    assert result.final_action == Action.ESCALATE
    assert result.reliability_label == ReliabilityLabel.NEEDS_HUMAN_REVIEW
    assert result.needs_human_review is True
    assert result.prediction is None
    assert result.verification is not None
    assert result.verification.next_step == NextStep.ESCALATE


# ── 5. Verification REJECT ───────────────────────────────────────────────────


def test_pipeline_verification_reject(mocks, synthetic_image):
    """Test 5: Repair executed but verification determines REJECT -> prediction withheld."""
    mocks.decision_agent.decide.return_value = make_decision_result(Action.REPAIR)
    mocks.verification_agent.run.return_value = make_verification_result(
        verified=False, next_step=NextStep.REJECT
    )

    pipeline = mocks.build_pipeline()
    result = pipeline.predict(synthetic_image)

    assert result.pipeline_state == PipelineState.REJECT
    assert result.final_action == Action.REJECT
    assert result.reliability_label == ReliabilityLabel.NEEDS_HUMAN_REVIEW
    assert result.needs_human_review is True
    assert result.prediction is None
    assert result.verification.next_step == NextStep.REJECT


# ── 6. Initial ESCALATE ──────────────────────────────────────────────────────


def test_pipeline_initial_escalate(mocks, synthetic_image):
    """Test 6: Decision Agent returns ESCALATE -> prediction withheld, needs_human_review=True."""
    mocks.decision_agent.decide.return_value = make_decision_result(
        Action.ESCALATE, rule_id="R-BORDERLINE-01"
    )

    pipeline = mocks.build_pipeline()
    result = pipeline.predict(synthetic_image)

    assert result.pipeline_state == PipelineState.ESCALATE
    assert result.final_action == Action.ESCALATE
    assert result.reliability_label == ReliabilityLabel.NEEDS_HUMAN_REVIEW
    assert result.needs_human_review is True
    assert result.prediction is None
    assert result.repair is None


# ── 7. Initial REJECT ────────────────────────────────────────────────────────


def test_pipeline_initial_reject(mocks, synthetic_image):
    """Test 7: Decision Agent returns REJECT (severe OOD) -> prediction withheld."""
    mocks.decision_agent.decide.return_value = make_decision_result(
        Action.REJECT, rule_id="R-SEVERE-OOD-01"
    )

    pipeline = mocks.build_pipeline()
    result = pipeline.predict(synthetic_image)

    assert result.pipeline_state == PipelineState.REJECT
    assert result.final_action == Action.REJECT
    assert result.reliability_label == ReliabilityLabel.NEEDS_HUMAN_REVIEW
    assert result.needs_human_review is True
    assert result.prediction is None
    assert result.repair is None


# ── 8. OOD Dependency Ordering ───────────────────────────────────────────────


def test_ood_dependency_ordering(mocks, synthetic_image):
    """Test 8: OOD Agent receives mid-layer features from Base Model, not raw image."""
    dummy_features = torch.randn(1024)
    dummy_probs = torch.full((18,), 0.77)
    fwd = ModelForward(
        result=make_base_model_result(0.77),
        raw_probs=dummy_probs,
        features=dummy_features,
        inference_ms=15.0,
    )
    mocks.base_model.run.return_value = fwd

    pipeline = mocks.build_pipeline()
    pipeline.predict(synthetic_image)

    mocks.ood_agent.run.assert_called_once()
    call_kwargs = mocks.ood_agent.run.call_args.kwargs
    assert "features" in call_kwargs
    assert torch.equal(call_kwargs["features"], dummy_features)
    assert torch.equal(call_kwargs["raw_probs"], dummy_probs)


# ── 9. Decision Agent Inputs ─────────────────────────────────────────────────


def test_decision_agent_inputs(mocks, synthetic_image):
    """Test 9: Decision Agent receives outputs from Quality, OOD, Uncertainty agents and DecisionState."""
    q_out = make_quality_result(QualityLevel.GOOD)
    ood_out = make_ood_result(OODLevel.IN_DISTRIBUTION)
    unc_out = make_uncertainty_result(UncertaintyLevel.LOW)

    mocks.quality_agent.run.return_value = q_out
    mocks.ood_agent.run.return_value = ood_out
    mocks.uncertainty_agent.run.return_value = unc_out

    pipeline = mocks.build_pipeline()
    pipeline.predict(synthetic_image)

    mocks.decision_agent.decide.assert_called_once()
    call_kwargs = mocks.decision_agent.decide.call_args.kwargs
    assert call_kwargs["quality"] == q_out
    assert call_kwargs["ood"] == ood_out
    assert call_kwargs["uncertainty"] == unc_out
    assert isinstance(call_kwargs["state"], DecisionState)
    assert call_kwargs["state"].stage == "initial"


# ── 10. Verification Agent Inputs ────────────────────────────────────────────


def test_verification_agent_inputs(mocks, synthetic_image):
    """Test 10: Verification Agent receives both before and after signals when repair runs."""
    q_before = make_quality_result(QualityLevel.DEGRADED, exp=True)
    q_after = make_quality_result(QualityLevel.GOOD)
    mocks.quality_agent.run.side_effect = [q_before, q_after]

    bm_before = make_model_forward(0.55)
    bm_after = make_model_forward(0.85)
    mocks.base_model.run.side_effect = [bm_before, bm_after]

    mocks.decision_agent.decide.return_value = make_decision_result(Action.REPAIR)
    mocks.verification_agent.run.return_value = make_verification_result(True)

    pipeline = mocks.build_pipeline()
    pipeline.predict(synthetic_image)

    mocks.verification_agent.run.assert_called_once()
    kwargs = mocks.verification_agent.run.call_args.kwargs
    assert kwargs["quality_before"] == q_before
    assert kwargs["quality_after"] == q_after
    assert kwargs["base_model_before"].pneumonia_probability == pytest.approx(0.55)
    assert kwargs["base_model_after"].pneumonia_probability == pytest.approx(0.85)
    assert kwargs["action"] == Action.REPAIR


# ── 11. Repair Not Applied Safe Escalation ───────────────────────────────────


def test_repair_not_applied_escalation(mocks, synthetic_image):
    """Test 11: If repair_applied is False, re-inference and verification are skipped; escalates safely."""
    mocks.decision_agent.decide.return_value = make_decision_result(Action.REPAIR)
    unapplied_repair = make_repair_result(applied=False, changed=False)
    unapplied_repair.skipped_reason = "Noise level below repair threshold"
    mocks.repair_agent.run.side_effect = None
    mocks.repair_agent.run.return_value = (synthetic_image, unapplied_repair)

    pipeline = mocks.build_pipeline()
    result = pipeline.predict(synthetic_image)

    # Verification must NOT be called
    mocks.verification_agent.run.assert_not_called()
    # Base model called only once for initial screening, not for after-repair
    assert mocks.base_model.run.call_count == 1

    assert result.pipeline_state == PipelineState.ESCALATE
    assert result.final_action == Action.ESCALATE
    assert result.reliability_label == ReliabilityLabel.NEEDS_HUMAN_REVIEW
    assert result.needs_human_review is True
    assert result.prediction is None
    assert result.repair.repair_applied is False


# ── 12. Non-Destructive Image Handling ───────────────────────────────────────


def test_original_image_preserved(mocks):
    """Test 12: Original input image array and tensor are never modified in-place."""
    orig_np = np.random.randint(50, 200, size=(224, 224), dtype=np.uint8)
    orig_np_copy = orig_np.copy()

    pipeline = mocks.build_pipeline()
    pipeline.predict(orig_np)

    # Bit-identical check
    assert np.array_equal(orig_np, orig_np_copy)

    # Check torch tensor input
    orig_tensor = torch.rand(1, 1, 224, 224) * 2048.0 - 1024.0
    orig_tensor_copy = orig_tensor.clone()
    pipeline.predict(orig_tensor)
    assert torch.equal(orig_tensor, orig_tensor_copy)


# ── 13. Safe Agent Failure Handling ──────────────────────────────────────────


def test_agent_failure_safe_handling(mocks, synthetic_image):
    """Test 13: Pipeline catches agent exceptions and returns safe ERROR/ESCALATE state without crashing."""
    mocks.base_model.run.side_effect = RuntimeError("CUDA out of memory simulation")

    pipeline = mocks.build_pipeline()
    result = pipeline.predict(synthetic_image)

    assert result.pipeline_state == PipelineState.ERROR
    assert result.final_action == Action.ESCALATE
    assert result.reliability_label == ReliabilityLabel.NEEDS_HUMAN_REVIEW
    assert result.needs_human_review is True
    assert result.prediction is None
    assert "CUDA out of memory" in (result.error_message or "")


# ── 14. Bounded Repair Loop ──────────────────────────────────────────────────


def test_bounded_repair_loop(mocks, synthetic_image):
    """Test 14: Repair loop respects max_repair_attempts limit and never loops infinitely."""
    pipeline = mocks.build_pipeline(max_repair_attempts=1)
    assert pipeline.max_repair_attempts == 1

    # Simulate decision agent requesting repair
    mocks.decision_agent.decide.return_value = make_decision_result(Action.REPAIR)
    result = pipeline.predict(synthetic_image)

    # With max_repair_attempts = 1, repair is attempted at most once
    assert mocks.repair_agent.run.call_count == 1
    assert result.repair_attempts == 1


# ── 15. Pipeline Determinism ─────────────────────────────────────────────────


def test_pipeline_determinism(mocks, synthetic_image):
    """Test 15: Two sequential runs with identical inputs produce identical actions, states, and scores."""
    pipeline = mocks.build_pipeline()
    res1 = pipeline.predict(synthetic_image, input_id="det_1")
    res2 = pipeline.predict(synthetic_image, input_id="det_2")

    assert res1.final_action == res2.final_action
    assert res1.pipeline_state == res2.pipeline_state
    assert res1.reliability_label == res2.reliability_label
    assert res1.needs_human_review == res2.needs_human_review
    assert res1.prediction.pneumonia_probability == res2.prediction.pneumonia_probability


# ── 16. PipelineResult Contract Validation ───────────────────────────────────


def test_pipeline_result_contract_validation():
    """Test 16: PipelineResult validator enforces that review flag matches label and withholds prediction."""
    # Valid accepted result
    pred = PredictionSummary(pneumonia_probability=0.8, positive=True, decision_threshold=0.5, source_model_id="dense")
    res = PipelineResult(
        audit_id="aud_1",
        pipeline_state=PipelineState.ACCEPT,
        final_action=Action.ACCEPT,
        reliability_label=ReliabilityLabel.ACCEPTED,
        needs_human_review=False,
        prediction=pred,
    )
    assert res.needs_human_review is False

    # Disagreeing review flag must raise ValueError
    with pytest.raises(ValueError, match="needs_human_review must be True"):
        PipelineResult(
            audit_id="aud_2",
            pipeline_state=PipelineState.ESCALATE,
            final_action=Action.ESCALATE,
            reliability_label=ReliabilityLabel.NEEDS_HUMAN_REVIEW,
            needs_human_review=False,  # wrong
            prediction=None,
        )

    # Releasing prediction when review is needed must raise ValueError
    with pytest.raises(ValueError, match="Prediction must be withheld"):
        PipelineResult(
            audit_id="aud_3",
            pipeline_state=PipelineState.ESCALATE,
            final_action=Action.ESCALATE,
            reliability_label=ReliabilityLabel.NEEDS_HUMAN_REVIEW,
            needs_human_review=True,
            prediction=pred,  # should be None
        )

    # Conversion to PipelineOutput
    output = res.to_pipeline_output()
    assert isinstance(output, PipelineOutput)
    assert output.audit_id == "aud_1"
    assert output.reliability_label == ReliabilityLabel.ACCEPTED


# ── 17. Human-Readable Reasoning ─────────────────────────────────────────────


def test_pipeline_human_readable_reasoning(mocks, synthetic_image):
    """Test 17: PipelineResult contains a coherent, non-empty reason explaining the decision."""
    pipeline = mocks.build_pipeline()
    result = pipeline.predict(synthetic_image)

    assert result.reason != ""
    assert "Decision Agent" in result.reason or "Accepted" in result.reason


# ── 18. Zero Clinical Claims ─────────────────────────────────────────────────


def test_pipeline_no_clinical_claims(mocks, synthetic_image):
    """Test 18: Output reason and disclaimer explicitly declare research prototype status."""
    pipeline = mocks.build_pipeline()
    result = pipeline.predict(synthetic_image)

    assert "research prototype" in result.disclaimer.lower()
    assert "research prototype" in result.reason.lower() or "not a clinical diagnosis" in result.reason.lower()


# ── 19. Retention of Intermediate Results ────────────────────────────────────


def test_pipeline_retains_intermediate_results(mocks, synthetic_image):
    """Test 19: All intermediate agent results are preserved in PipelineResult for full auditability."""
    mocks.decision_agent.decide.return_value = make_decision_result(Action.REPAIR)
    pipeline = mocks.build_pipeline()
    result = pipeline.predict(synthetic_image)

    assert result.quality is not None
    assert result.base_model is not None
    assert result.ood is not None
    assert result.uncertainty is not None
    assert len(result.decision_history) >= 1
    assert result.repair is not None
    assert result.verification is not None
    assert result.after_repair_quality is not None
    assert result.after_repair_base_model is not None
    assert result.after_repair_ood is not None
    assert result.after_repair_uncertainty is not None
    assert "quality" in result.intermediate_latencies
    assert "base_model" in result.intermediate_latencies


# ── 20. Empty or Invalid Input Handling ──────────────────────────────────────


def test_pipeline_empty_or_invalid_input(mocks):
    """Test 20: Pipeline gracefully rejects None or empty array without accepting."""
    pipeline = mocks.build_pipeline()

    # None input
    res_none = pipeline.predict(None)
    assert res_none.pipeline_state == PipelineState.ERROR
    assert res_none.final_action == Action.REJECT
    assert res_none.needs_human_review is True
    assert res_none.prediction is None

    # Empty array
    empty_arr = np.array([])
    res_empty = pipeline.predict(empty_arr)
    assert res_empty.pipeline_state == PipelineState.ERROR
    assert res_empty.needs_human_review is True
    assert res_empty.prediction is None

    # Non-existent file path
    res_missing = pipeline.predict("/nonexistent/file/cxr.png")
    assert res_missing.pipeline_state == PipelineState.ERROR
    assert res_missing.needs_human_review is True


# ── 21. Integration Smoke Test ───────────────────────────────────────────────


def test_integration_smoke_pipeline():
    """Test 21: Integration smoke test connecting real Quality Agent, mock Base Model,
    real Uncertainty Agent, real Decision Agent, real Repair Agent, and real Verification Agent.
    """
    from cxr_reliability.agents.quality import QualityAgent
    from cxr_reliability.agents.uncertainty import UncertaintyAgent
    from cxr_reliability.agents.decision.rules import RuleTableDecisionAgent
    from cxr_reliability.agents.repair import RepairAgent
    from cxr_reliability.agents.verification import VerificationAgent
    from cxr_reliability.config.thresholds import DecisionThresholds
    from cxr_reliability.config.pipeline_config import ExecutionConfig

    quality_agent = QualityAgent()
    uncertainty_agent = UncertaintyAgent()
    decision_agent = RuleTableDecisionAgent(
        thresholds=DecisionThresholds(borderline_margin=0.05),
        execution=ExecutionConfig(),
        thresholds_version="v0_smoke_test",
    )
    repair_agent = RepairAgent()
    verification_agent = VerificationAgent()

    # Base model mock (to avoid downloading/loading 100MB DenseNet during smoke test)
    mock_bm = MagicMock()
    mock_bm.run.return_value = make_model_forward(0.92)

    # OOD agent mock (lightweight)
    mock_ood = MagicMock()
    mock_ood.run.return_value = make_ood_result(OODLevel.IN_DISTRIBUTION, dist=10.0)

    pipeline = ReliabilityPipeline(
        quality_agent=quality_agent,
        base_model=mock_bm,
        ood_agent=mock_ood,
        uncertainty_agent=uncertainty_agent,
        decision_agent=decision_agent,
        repair_agent=repair_agent,
        verification_agent=verification_agent,
    )

    # Synthetic realistic chest X-ray image (224x224 grayscale uint8 with reasonable contrast)
    img = np.zeros((224, 224), dtype=np.uint8)
    img[50:180, 50:180] = 120
    img[80:150, 80:150] = 160

    result = pipeline.predict(img, input_id="smoke_test_01")

    assert isinstance(result, PipelineResult)
    assert result.audit_id == "smoke_test_01"
    assert result.final_action in (Action.ACCEPT, Action.REPAIR, Action.ESCALATE, Action.REJECT)
    assert result.disclaimer == DISCLAIMER
    assert result.total_latency_ms is not None and result.total_latency_ms > 0

    # Pipeline output conversion check
    output = result.to_pipeline_output()
    assert isinstance(output, PipelineOutput)
    assert output.audit_id == "smoke_test_01"
