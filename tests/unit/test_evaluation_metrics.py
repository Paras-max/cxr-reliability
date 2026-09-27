"""Specification tests for evaluation metrics. Skipped until Phase 2/7.

Each test below states a requirement from the PRD; bodies are intentionally unwritten.
"""
import pytest

pytestmark = pytest.mark.skip(reason="Phase 2/7: evaluation metrics not implemented")



def test_ece_zero_for_perfectly_calibrated_toy_case():
    """ECE is 0 for a constructed perfectly calibrated set."""
    raise NotImplementedError


def test_recovery_pct_formula():
    """recovery_pct matches the PRD formula on hand-computed values."""
    raise NotImplementedError


def test_action_breakdown_counts_sum_to_total():
    """Bucket counts sum to the number of inputs."""
    raise NotImplementedError


def test_bootstrap_resamples_by_patient():
    """Bootstrap resamples whole patients, never individual images."""
    raise NotImplementedError
