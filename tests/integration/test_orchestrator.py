"""Specification tests for pipeline orchestrator. Skipped until Phase 6.

Each test below states a requirement from the PRD; bodies are intentionally unwritten.
"""
import pytest

pytestmark = pytest.mark.skip(reason="Phase 6: pipeline orchestrator not implemented")



def test_loops_terminate_at_limits():
    """With stub agents that always request repair/escalation, predict() terminates."""
    raise NotImplementedError


def test_repaired_output_is_tagged():
    """Any output that used Repair has reliability_label=accepted_after_repair or needs_human_review."""
    raise NotImplementedError


def test_reject_path_returns_review_flag_and_no_prediction():
    """Reject yields needs_human_review=True and prediction=None."""
    raise NotImplementedError


def test_audit_record_written_on_every_path():
    """Accept, Repair, Escalate and Reject paths each write one audit record."""
    raise NotImplementedError
