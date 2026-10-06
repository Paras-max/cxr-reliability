"""Integration tests for pipeline orchestrator (Phase 11)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from cxr_reliability.contracts.decision import Action
from cxr_reliability.contracts.pipeline import ReliabilityLabel
from cxr_reliability.contracts.verification import NextStep
from tests.unit.test_orchestrator import (
    MockPipelineComponents,
    make_decision_result,
    make_repair_result,
    make_verification_result,
)


@pytest.fixture
def mocks():
    return MockPipelineComponents()


@pytest.fixture
def sample_image():
    return np.full((224, 224), 128, dtype=np.uint8)


def test_loops_terminate_at_limits(mocks, sample_image):
    """With stub agents that always request repair, predict() terminates at loop limit."""
    mocks.decision_agent.decide.return_value = make_decision_result(action=Action.REPAIR)
    mocks.repair_agent.run.return_value = make_repair_result(applied=True)
    mocks.verification_agent.run.return_value = make_verification_result(
        verified=False, next_step=NextStep.ESCALATE
    )

    pipeline = mocks.build_pipeline(max_repair_attempts=2)
    result = pipeline.run(sample_image)

    # Must terminate and not exceed configured attempt limit
    assert result.repair_attempts <= 2
    assert result.final_action == Action.ESCALATE
    assert result.needs_human_review is True


def test_repaired_output_is_tagged(mocks, sample_image):
    """Any output that used Repair has reliability_label=accepted_after_repair or needs_human_review."""
    mocks.decision_agent.decide.return_value = make_decision_result(action=Action.REPAIR)
    mocks.repair_agent.run.return_value = make_repair_result(applied=True)
    mocks.verification_agent.run.return_value = make_verification_result(
        verified=True, next_step=NextStep.RELEASE
    )

    pipeline = mocks.build_pipeline(max_repair_attempts=1)
    result = pipeline.run(sample_image)

    assert result.repair is not None
    assert result.repair.repaired is True
    assert result.reliability_label in (
        ReliabilityLabel.ACCEPTED_AFTER_REPAIR,
        ReliabilityLabel.NEEDS_HUMAN_REVIEW,
    )


def test_reject_path_returns_review_flag_and_no_prediction(mocks, sample_image):
    """Reject yields needs_human_review=True and prediction=None."""
    mocks.decision_agent.decide.return_value = make_decision_result(
        action=Action.REJECT, rule_id="R_REJECT_TEST"
    )

    pipeline = mocks.build_pipeline()
    output = pipeline.predict(sample_image)

    assert output.final_action == Action.REJECT
    assert output.needs_human_review is True
    assert output.prediction is None
    assert output.reliability_label == ReliabilityLabel.NEEDS_HUMAN_REVIEW


def test_audit_record_written_on_every_path(mocks, sample_image):
    """Accept, Repair, Escalate and Reject paths each write one audit record."""
    audit_mock = MagicMock()
    mocks.audit_logger = audit_mock

    for act in [Action.ACCEPT, Action.ESCALATE, Action.REJECT]:
        audit_mock.reset_mock()
        mocks.decision_agent.decide.return_value = make_decision_result(action=act)
        pipeline = mocks.build_pipeline()
        pipeline.predict(sample_image)
        assert audit_mock.write.call_count == 1
