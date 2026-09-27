"""Specification tests for FastAPI backend. Skipped until Phase 8.

Each test below states a requirement from the PRD; bodies are intentionally unwritten.
"""
import pytest

pytestmark = pytest.mark.skip(reason="Phase 8: FastAPI backend not implemented")



def test_health_endpoint():
    """GET /v1/health returns 200 with the disclaimer."""
    raise NotImplementedError


def test_predict_returns_schema_and_disclaimer():
    """POST /v1/predict returns a valid PredictResponse containing the disclaimer."""
    raise NotImplementedError


def test_invalid_upload_rejected():
    """Non-image or oversize uploads return 4xx."""
    raise NotImplementedError
