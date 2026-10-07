"""Unit tests for Initial vs Final Pneumonia Probability feature.

Verifies all 11 test requirements:
1. Initial probability is correctly captured.
2. Final probability is correctly captured.
3. Delta calculation is correct (final - initial).
4. Negative probability delta is handled correctly.
5. Positive probability delta is handled correctly.
6. Zero delta is handled correctly.
7. Classification uses the existing 0.522161 threshold.
8. Repair path uses the fresh post-repair DenseNet inference.
9. No-repair path does not perform an unnecessary second inference.
10. Human-review cases do not incorrectly release a final diagnosis.
11. Evaluation utility and serialization preservation.
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
from cxr_reliability.contracts.common import AgentName
from cxr_reliability.contracts.decision import Action, DecisionResult
from cxr_reliability.contracts.ood import OODLevel, OODResult
from cxr_reliability.contracts.pipeline import (
    FinalClassification,
    PipelineOutput,
    PipelineResult,
    PipelineState,
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
from cxr_reliability.evaluation.agent_evaluation import analyze_probability_shift
from cxr_reliability.models.base_model import ModelForward
from cxr_reliability.pipeline.orchestrator import ReliabilityPipeline


# ── Fixtures & Mock Helpers ───────────────────────────────────────────────────

def make_quality(level=QualityLevel.GOOD) -> QualityResult:
    return QualityResult(
        agent=AgentName.QUALITY,
        version="0.7.0",
        laplacian_variance=300.0,
        blur_pct=5.0,
        snr_db=30.0,
        mean_intensity=128.0,
        histogram_std=45.0,
        flags=QualityFlags(blur=False, noise=False, exposure=False),
        overall=level,
        score=0.9,
        label=level.value,
        reasoning="Good quality",
    )


def make_base_model_forward(raw_prob=0.85) -> ModelForward:
    res = BaseModelResult(
        agent=AgentName.BASE_MODEL,
        version="0.4.0",
        model_id="densenet121-res224-nih",
        target_pathology="Pneumonia",
        pneumonia_logit=raw_prob,
        pneumonia_probability=raw_prob,
        raw_pneumonia_score=raw_prob,
        all_pathology_outputs={"Pneumonia": raw_prob},
        score=raw_prob,
        label="Pneumonia" if raw_prob >= 0.522161 else "No Pneumonia",
        reasoning=f"Score: {raw_prob:.4f}",
    )
    return ModelForward(
        result=res,
        raw_probs=torch.full((18,), raw_prob),
        features=torch.ones(1024),
        inference_ms=10.0,
    )


def make_ood() -> OODResult:
    return OODResult(
        agent=AgentName.OOD,
        version="0.5.0",
        mahalanobis_distance=10.0,
        mahalanobis_threshold=25.0,
        level=OODLevel.IN_DISTRIBUTION,
        score=0.2,
        label=OODLevel.IN_DISTRIBUTION.value,
        reasoning="In distribution",
    )


def make_uncertainty() -> UncertaintyResult:
    return UncertaintyResult(
        agent=AgentName.UNCERTAINTY,
        version="0.6.0",
        raw_model_score=0.85,
        confidence=0.90,
        entropy=0.15,
        normalized_entropy=0.15,
        uncertainty_level=UncertaintyLevel.LOW,
        score=0.90,
        label=UncertaintyLevel.LOW.value,
        reasoning="Low uncertainty",
    )


def build_pipeline_with_mocks(
    initial_prob: float = 0.85,
    action: Action = Action.ACCEPT,
    repair_applied: bool = False,
    after_prob: float | None = None,
    next_step: NextStep = NextStep.RELEASE,
) -> tuple[ReliabilityPipeline, MagicMock]:
    mock_q = MagicMock()
    mock_q.run.return_value = make_quality(QualityLevel.GOOD if not repair_applied else QualityLevel.POOR)

    mock_bm = MagicMock()
    mock_bm.calibrator = None

    if repair_applied and after_prob is not None:
        mock_bm.run.side_effect = [
            make_base_model_forward(initial_prob),
            make_base_model_forward(after_prob),
        ]
    else:
        mock_bm.run.return_value = make_base_model_forward(initial_prob)

    mock_ood = MagicMock()
    mock_ood.run.return_value = make_ood()

    mock_unc = MagicMock()
    mock_unc.run.return_value = make_uncertainty()

    mock_dec = MagicMock()
    mock_dec.decide.return_value = DecisionResult(
        agent=AgentName.DECISION,
        version="v1-0.8.0",
        action=action,
        rule_id="TEST-RULE",
        driving_signals={},
        reliability_label="accepted" if action == Action.ACCEPT else "needs_human_review",
        score=1.0,
        label=action.value,
        reasoning=f"Action: {action.value}",
    )

    mock_rep = MagicMock()
    mock_rep.run.return_value = (
        np.zeros((224, 224), dtype=np.uint8),
        RepairResult(
            agent=AgentName.REPAIR,
            version="0.9.0",
            repaired=True,
            repair_applied=repair_applied,
            defects_detected=[DefectType.BLUR],
            steps=[RepairStep(method="unsharp_mask", defect="blur", parameters={})],
            label="repaired",
            reasoning="Repair test",
        ),
    )

    mock_ver = MagicMock()
    mock_ver.run.return_value = VerificationResult(
        agent=AgentName.VERIFICATION,
        version="0.10.0",
        verified=(next_step == NextStep.RELEASE),
        status=VerificationStatus.VERIFIED if next_step == NextStep.RELEASE else VerificationStatus.ESCALATE,
        next_step=next_step,
        delta_confidence=0.10,
        delta_quality=15.0,
        delta_ood=-2.0,
        label_flipped=False,
        min_confidence_gain_used=0.05,
        threshold_is_provisional=True,
        score=0.10,
        label="verified" if next_step == NextStep.RELEASE else "escalate",
        reasoning="Verification test",
    )

    pipeline = ReliabilityPipeline(
        quality_agent=mock_q,
        base_model=mock_bm,
        uncertainty_agent=mock_unc,
        ood_agent=mock_ood,
        decision_agent=mock_dec,
        repair_agent=mock_rep,
        verification_agent=mock_ver,
    )
    return pipeline, mock_bm


# ── Test Cases ───────────────────────────────────────────────────────────────

def test_1_initial_probability_correctly_captured():
    """TEST 1: Initial DenseNet probability is captured accurately from first inference."""
    raw_p = 0.523041725
    pipe, mock_bm = build_pipeline_with_mocks(initial_prob=raw_p, action=Action.ACCEPT)
    img = np.zeros((224, 224), dtype=np.uint8)

    result = pipe.predict(img)

    assert result.initial_pneumonia_probability is not None
    assert result.initial_pneumonia_probability == pytest.approx(raw_p)
    assert result.initial_classification == "Pneumonia"

    # Contract propagation
    output = result.to_pipeline_output()
    assert output.initial_pneumonia_probability == pytest.approx(raw_p)
    assert output.initial_classification == "Pneumonia"


def test_2_final_probability_correctly_captured_no_repair():
    """TEST 2: Final probability in no-repair path matches initial model output."""
    raw_p = 0.450000000
    pipe, _ = build_pipeline_with_mocks(initial_prob=raw_p, action=Action.ACCEPT)
    img = np.zeros((224, 224), dtype=np.uint8)

    result = pipe.predict(img)

    assert result.final_pneumonia_probability == pytest.approx(raw_p)
    assert result.initial_classification == "No Pneumonia"
    assert result.final_classification == FinalClassification.NO_PNEUMONIA


def test_3_delta_calculation_correct_on_repair():
    """TEST 3: Delta calculation is strictly final - initial."""
    init_p = 0.5230417
    after_p = 0.5230160
    expected_delta = after_p - init_p

    pipe, _ = build_pipeline_with_mocks(
        initial_prob=init_p,
        action=Action.REPAIR,
        repair_applied=True,
        after_prob=after_p,
        next_step=NextStep.RELEASE,
    )
    img = np.zeros((224, 224), dtype=np.uint8)
    result = pipe.predict(img)

    assert result.initial_pneumonia_probability == pytest.approx(init_p)
    assert result.final_pneumonia_probability == pytest.approx(after_p)
    assert result.probability_delta == pytest.approx(expected_delta)


def test_4_negative_probability_delta_handled():
    """TEST 4: Negative probability delta (decrease after repair) is handled without clipping/errors."""
    init_p = 0.600000
    after_p = 0.540000
    pipe, _ = build_pipeline_with_mocks(
        initial_prob=init_p,
        action=Action.REPAIR,
        repair_applied=True,
        after_prob=after_p,
        next_step=NextStep.RELEASE,
    )
    img = np.zeros((224, 224), dtype=np.uint8)
    result = pipe.predict(img)

    assert result.probability_delta < 0
    assert result.probability_delta == pytest.approx(-0.060000)
    assert result.final_pneumonia_probability == pytest.approx(0.540000)


def test_5_positive_probability_delta_handled():
    """TEST 5: Positive probability delta is handled correctly."""
    init_p = 0.400000
    after_p = 0.480000
    pipe, _ = build_pipeline_with_mocks(
        initial_prob=init_p,
        action=Action.REPAIR,
        repair_applied=True,
        after_prob=after_p,
        next_step=NextStep.RELEASE,
    )
    img = np.zeros((224, 224), dtype=np.uint8)
    result = pipe.predict(img)

    assert result.probability_delta > 0
    assert result.probability_delta == pytest.approx(+0.080000)


def test_6_zero_delta_handled_on_no_repair():
    """TEST 6: Zero delta is correctly stored on ACCEPT, ESCALATE, and REJECT without repair."""
    for action in (Action.ACCEPT, Action.ESCALATE, Action.REJECT):
        pipe, _ = build_pipeline_with_mocks(initial_prob=0.75, action=action)
        result = pipe.predict(np.zeros((224, 224), dtype=np.uint8))
        assert result.probability_delta == 0.0
        assert result.initial_pneumonia_probability == pytest.approx(0.75)
        assert result.final_pneumonia_probability == pytest.approx(0.75)


def test_7_classification_uses_existing_0_522161_threshold():
    """TEST 7: Classification rule strictly applies probability >= 0.522161 -> Pneumonia, else No Pneumonia."""
    thresh = 0.522161

    # Just below threshold
    p_below = 0.522160
    pipe_below, _ = build_pipeline_with_mocks(initial_prob=p_below, action=Action.ACCEPT)
    res_below = pipe_below.predict(np.zeros((224, 224), dtype=np.uint8))
    assert res_below.initial_classification == "No Pneumonia"
    assert res_below.final_classification == FinalClassification.NO_PNEUMONIA

    # Exactly at threshold
    p_exact = 0.522161
    pipe_exact, _ = build_pipeline_with_mocks(initial_prob=p_exact, action=Action.ACCEPT)
    res_exact = pipe_exact.predict(np.zeros((224, 224), dtype=np.uint8))
    assert res_exact.initial_classification == "Pneumonia"
    assert res_exact.final_classification == FinalClassification.PNEUMONIA

    # Above threshold
    p_above = 0.522162
    pipe_above, _ = build_pipeline_with_mocks(initial_prob=p_above, action=Action.ACCEPT)
    res_above = pipe_above.predict(np.zeros((224, 224), dtype=np.uint8))
    assert res_above.initial_classification == "Pneumonia"
    assert res_above.final_classification == FinalClassification.PNEUMONIA


def test_8_repair_path_uses_fresh_densenet_inference():
    """TEST 8: Repair path executes and uses the fresh post-repair DenseNet inference."""
    init_p = 0.510000
    after_p = 0.550000
    pipe, mock_bm = build_pipeline_with_mocks(
        initial_prob=init_p,
        action=Action.REPAIR,
        repair_applied=True,
        after_prob=after_p,
        next_step=NextStep.RELEASE,
    )
    result = pipe.predict(np.zeros((224, 224), dtype=np.uint8))

    assert mock_bm.run.call_count == 2
    assert result.initial_pneumonia_probability == pytest.approx(init_p)
    assert result.final_pneumonia_probability == pytest.approx(after_p)
    assert result.after_repair_base_model is not None


def test_9_no_repair_path_does_not_perform_second_inference():
    """TEST 9: Non-repair paths (ACCEPT, ESCALATE, REJECT) call DenseNet inference exactly ONCE."""
    for action in (Action.ACCEPT, Action.ESCALATE, Action.REJECT):
        pipe, mock_bm = build_pipeline_with_mocks(initial_prob=0.60, action=action)
        pipe.predict(np.zeros((224, 224), dtype=np.uint8))
        assert mock_bm.run.call_count == 1, f"Expected exactly 1 inference for {action.value}"


def test_10_human_review_cases_do_not_incorrectly_release_final_diagnosis():
    """TEST 10: Human review cases preserve probabilities for audit but withhold prediction."""
    pipe, _ = build_pipeline_with_mocks(initial_prob=0.88, action=Action.ESCALATE)
    result = pipe.predict(np.zeros((224, 224), dtype=np.uint8))

    assert result.needs_human_review is True
    assert result.prediction is None
    assert result.final_classification == FinalClassification.HUMAN_REVIEW_REQUIRED

    # Audit fields are preserved
    assert result.initial_pneumonia_probability == pytest.approx(0.88)
    assert result.final_pneumonia_probability == pytest.approx(0.88)
    assert result.probability_delta == 0.0
    assert result.initial_classification == "Pneumonia"

    # Strict contract invariant: cannot set diagnosis while needs_human_review=True
    with pytest.raises(ValueError, match="final_classification cannot release diagnosis when needs_human_review is True"):
        PipelineOutput(
            audit_id="test_invalid",
            reliability_label=ReliabilityLabel.NEEDS_HUMAN_REVIEW,
            final_action=Action.ESCALATE,
            needs_human_review=True,
            final_classification=FinalClassification.PNEUMONIA,
        )


def test_11_evaluation_probability_shift_analysis():
    """TEST 11: Evaluation utility analyze_probability_shift functions mathematically."""
    p_init = [0.5230, 0.5076, 0.5063, 0.7000]
    p_final = [0.5230, 0.5077, 0.5065, 0.6500]

    stats = analyze_probability_shift(p_init, p_final, threshold=0.522161)
    assert stats["n_samples"] == 4
    assert stats["operating_threshold"] == 0.522161
    assert stats["positive_shifts_count"] == 2
    assert stats["negative_shifts_count"] == 1
    assert stats["zero_shifts_count"] == 1
    assert stats["classification_concordance_count"] == 4
    assert stats["classification_concordance_pct"] == 100.0
