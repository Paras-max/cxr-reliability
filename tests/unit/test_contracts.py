"""Tests for agent/pipeline contracts (real schema code, so real tests)."""
import pytest
from pydantic import ValidationError

from cxr_reliability.contracts.common import DISCLAIMER, AgentName, AgentResult
from cxr_reliability.contracts.decision import Action
from cxr_reliability.contracts.pipeline import PipelineOutput, ReliabilityLabel
from cxr_reliability.contracts.uncertainty import ConfidenceLevel, UncertaintyResult


def test_agent_result_requires_reasoning():
    with pytest.raises(ValidationError):
        AgentResult(agent=AgentName.QUALITY, version="0", label="x", reasoning="")


def test_contracts_reject_unknown_fields():
    with pytest.raises(ValidationError):
        AgentResult(agent=AgentName.QUALITY, version="0", label="x", reasoning="y", surprise=1)


def test_confidence_cannot_be_below_one_half():
    with pytest.raises(ValidationError):
        UncertaintyResult(
            version="0", label="l", reasoning="r", probability=0.5, confidence=0.4,
            binary_entropy=0.69, level=ConfidenceLevel.LOW,
        )


def test_review_flag_must_match_label():
    with pytest.raises(ValidationError):
        PipelineOutput(
            audit_id="a", reliability_label=ReliabilityLabel.NEEDS_HUMAN_REVIEW,
            final_action=Action.REJECT, needs_human_review=False,
        )
    with pytest.raises(ValidationError):
        PipelineOutput(
            audit_id="a", reliability_label=ReliabilityLabel.ACCEPTED,
            final_action=Action.ACCEPT, needs_human_review=True,
        )


def test_pipeline_output_always_carries_disclaimer():
    out = PipelineOutput(
        audit_id="a", reliability_label=ReliabilityLabel.NEEDS_HUMAN_REVIEW,
        final_action=Action.REJECT, needs_human_review=True,
    )
    assert out.disclaimer == DISCLAIMER
    assert out.prediction is None
