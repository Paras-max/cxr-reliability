"""Quality Agent contract (PRD FR-1) — Phase 7.

Responsibility:
    Schema for blur / noise / exposure measurements, defect flags and overall verdict.

Input:
    n/a

Output:
    QualityResult, QualityFlags, QualityLevel, DefectType.

Dependencies:
    contracts.common

Implementation phase: P7 (extended from P0)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from .common import AgentName, AgentResult, StrictModel


class QualityLevel(str, Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"


class DefectType(str, Enum):
    BLUR = "blur"
    NOISE = "noise"
    EXPOSURE = "exposure"


class QualityFlags(StrictModel):
    blur: bool
    noise: bool
    exposure: bool


class QualityResult(AgentResult):
    agent: AgentName = AgentName.QUALITY
    laplacian_variance: float
    blur_pct: float = Field(ge=0, le=100)
    snr_db: float
    mean_intensity: float
    histogram_std: float
    flags: QualityFlags
    overall: QualityLevel
    # None when repair bounds are not yet calibrated (PRD section 11 open question).
    repairable: bool | None = None
    near_threshold: bool = False
    image_id: str | None = None
    component_statuses: dict[str, str] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_overall(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        if "overall" in values:
            ov = values["overall"]
            if isinstance(ov, str):
                ov_lower = ov.lower()
                if ov_lower in ("good", "poor", "degraded"):
                    values["overall"] = QualityLevel(ov_lower)
        if "label" not in values:
            ov = values.get("overall", QualityLevel.GOOD)
            values["label"] = ov.value if isinstance(ov, QualityLevel) else str(ov)
        return values
