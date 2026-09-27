"""Specification tests for patient-level splits. Skipped until Phase 1.

Each test below states a requirement from the PRD; bodies are intentionally unwritten.
"""
import pytest

pytestmark = pytest.mark.skip(reason="Phase 1: patient-level splits not implemented")



def test_no_patient_overlap_across_splits():
    """No patient_id appears in more than one split."""
    raise NotImplementedError


def test_official_test_list_preserved():
    """The official test list is used unchanged as the test split."""
    raise NotImplementedError


def test_manifest_hash_stable():
    """Rewriting the same manifest gives the same hash."""
    raise NotImplementedError
