"""Base Model contract (PRD FR-3).

Responsibility:
    Schema for the pneumonia classifier output. Feature vectors are passed in memory
    (models.base_model.ModelForward), not serialized into audit records.

Input:
    n/a

Output:
    BaseModelResult.

Dependencies:
    contracts.common

Implementation phase: P0
"""

from __future__ import annotations

from typing import Any
from pydantic import Field, model_validator

from .common import AgentName, AgentResult


class BaseModelResult(AgentResult):
    agent: AgentName = AgentName.BASE_MODEL
    model_id: str
    weights_sha256: str | None = None
    target_pathology: str = "Pneumonia"
    pneumonia_logit: float
    pneumonia_probability: float = Field(ge=0, le=1)
    raw_pneumonia_score: float = Field(default=0.0, ge=0, le=1)
    calibrated_probability: float | None = Field(default=None, ge=0, le=1)
    # Audit only. Phase 2 must verify whether TorchXRayVision returns logits or probabilities.
    all_pathology_outputs: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _sync_raw_scores(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "raw_pneumonia_score" not in data or data["raw_pneumonia_score"] is None:
                if "pneumonia_probability" in data and data["pneumonia_probability"] is not None:
                    data["raw_pneumonia_score"] = data["pneumonia_probability"]
                elif "pneumonia_logit" in data and data["pneumonia_logit"] is not None:
                    data["raw_pneumonia_score"] = data["pneumonia_logit"]
            if "pneumonia_probability" not in data or data["pneumonia_probability"] is None:
                if "raw_pneumonia_score" in data and data["raw_pneumonia_score"] is not None:
                    data["pneumonia_probability"] = data["raw_pneumonia_score"]
        return data

