"""API request/response schemas.

Responsibility:
    Wire-level response models. The prediction payload reuses PipelineOutput so the API
    cannot drift from the pipeline contract. Every response carries the disclaimer.

Input:
    n/a

Output:
    PredictResponse, HealthResponse, ConfigResponse, ErrorResponse.

Dependencies:
    pydantic, contracts.pipeline

Implementation phase: P8
"""

from __future__ import annotations

from pydantic import BaseModel

from cxr_reliability.contracts.common import DISCLAIMER
from cxr_reliability.contracts.pipeline import PipelineOutput


class PredictResponse(BaseModel):
    result: PipelineOutput


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str
    disclaimer: str = DISCLAIMER


class ConfigResponse(BaseModel):
    fast_model_id: str
    escalation_model_id: str
    thresholds_version: str
    thresholds_provenance: str
    unresolved_thresholds: list[str]
    disclaimer: str = DISCLAIMER


class ErrorResponse(BaseModel):
    detail: str
