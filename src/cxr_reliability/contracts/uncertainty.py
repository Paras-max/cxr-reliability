"""Uncertainty Agent contract (PRD FR-4) — Phase 6.

Responsibility:
    Schema for confidence, predictive entropy, normalized entropy, and categorical
    uncertainty levels computed from base model output.

Input:
    n/a

Output:
    UncertaintyResult, UncertaintyLevel, ConfidenceLevel.

Dependencies:
    contracts.common

Implementation phase: P6 (extended from P0)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from .common import AgentName, AgentResult


class UncertaintyLevel(str, Enum):
    """Uncertainty categories for decision agent routing."""
    LOW = "LOW"
    HIGH = "HIGH"


class ConfidenceLevel(str, Enum):
    """Legacy PRD confidence level alias."""
    HIGH = "high"
    BORDERLINE = "borderline"
    LOW = "low"


class UncertaintyResult(AgentResult):
    agent: AgentName = AgentName.UNCERTAINTY
    # Primary fields (Phase 6)
    raw_model_score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(ge=0.5, le=1.0)   # max(p, 1 - p) on the Pneumonia output
    entropy: float = Field(default=0.0, ge=0.0)
    normalized_entropy: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty_level: UncertaintyLevel = UncertaintyLevel.HIGH
    method: str = "binary_confidence_entropy"
    thresholds: dict[str, Any] = Field(default_factory=dict)
    image_id: str | None = None
    timestamp: str | None = None

    # Legacy fields for backwards-compatibility with Phase 0 contracts & tests
    probability: float = Field(default=0.5, ge=0.0, le=1.0)
    binary_entropy: float = Field(default=0.0, ge=0.0)
    level: ConfidenceLevel | UncertaintyLevel | str = ConfidenceLevel.LOW

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        # If probability provided but not raw_model_score
        if "probability" in values and "raw_model_score" not in values:
            values["raw_model_score"] = values["probability"]
        elif "raw_model_score" in values and "probability" not in values:
            values["probability"] = values["raw_model_score"]

        # If binary_entropy provided but not entropy
        if "binary_entropy" in values and "entropy" not in values:
            values["entropy"] = values["binary_entropy"]
        elif "entropy" in values and "binary_entropy" not in values:
            values["binary_entropy"] = values["entropy"]

        # Sync level and uncertainty_level
        if "level" in values and "uncertainty_level" not in values:
            lvl = values["level"]
            if isinstance(lvl, str):
                lvl_upper = lvl.upper()
                if lvl_upper in ("LOW", "HIGH"):
                    values["uncertainty_level"] = UncertaintyLevel(lvl_upper)
                else:
                    values["uncertainty_level"] = UncertaintyLevel.HIGH
            elif isinstance(lvl, ConfidenceLevel):
                if lvl == ConfidenceLevel.HIGH:
                    values["uncertainty_level"] = UncertaintyLevel.LOW  # High confidence = Low uncertainty
                else:
                    values["uncertainty_level"] = UncertaintyLevel.HIGH
        elif "uncertainty_level" in values and "level" not in values:
            ulvl = values["uncertainty_level"]
            if ulvl == UncertaintyLevel.LOW:
                values["level"] = ConfidenceLevel.HIGH
            else:
                values["level"] = ConfidenceLevel.LOW

        # Default label to uncertainty_level value if not specified
        if "label" not in values:
            ul = values.get("uncertainty_level", UncertaintyLevel.HIGH)
            values["label"] = ul.value if isinstance(ul, UncertaintyLevel) else str(ul)

        # Default score to normalized_entropy or confidence
        if "score" not in values:
            values["score"] = values.get("confidence", 0.5)

        return values
