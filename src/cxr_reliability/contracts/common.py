"""Shared contract primitives.

Responsibility:
    Base envelope returned by every agent (score + label + human-readable reasoning),
    agent names, and the research-prototype disclaimer. Enforces PRD section 4
    Explainability: no agent result may have an empty reasoning string.

Input:
    n/a

Output:
    AgentResult, AgentName, StrictModel, DISCLAIMER.

Dependencies:
    pydantic

Implementation phase: P0
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER = (
    "Research prototype. Not a clinical diagnostic system and not validated for clinical use."
)


class AgentName(str, Enum):
    QUALITY = "quality"
    OOD = "ood"
    BASE_MODEL = "base_model"
    UNCERTAINTY = "uncertainty"
    DECISION = "decision"
    REPAIR = "repair"
    VERIFICATION = "verification"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentResult(StrictModel):
    """Envelope common to all seven agents."""

    agent: AgentName
    version: str
    score: float | None = None
    label: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    latency_ms: float | None = Field(default=None, ge=0)
    thresholds_version: str | None = None
