"""Specification tests for audit logger. Skipped until Phase 6.

Each test below states a requirement from the PRD; bodies are intentionally unwritten.
"""
import pytest

pytestmark = pytest.mark.skip(reason="Phase 6: audit logger not implemented")



def test_one_record_per_inference():
    """Each predict() call appends exactly one JSONL line."""
    raise NotImplementedError


def test_record_roundtrip():
    """A written record reads back equal."""
    raise NotImplementedError


def test_record_contains_signals_action_and_deltas():
    """Records include signals, action, rule id and verification deltas (PRD section 4)."""
    raise NotImplementedError
