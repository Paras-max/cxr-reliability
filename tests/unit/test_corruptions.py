"""Specification tests for corruption library. Skipped until Phase 3.

Each test below states a requirement from the PRD; bodies are intentionally unwritten.
"""
import pytest

pytestmark = pytest.mark.skip(reason="Phase 3: corruption library not implemented")



def test_same_seed_same_output():
    """Identical seed and severity give identical output."""
    raise NotImplementedError


def test_severity_monotonic():
    """Higher severity always degrades the corresponding metric more."""
    raise NotImplementedError
