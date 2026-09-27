"""OOD Agent contract (PRD FR-2).

Responsibility:
    Schema for Mahalanobis distance, energy score and the OOD verdict.

Input:
    n/a

Output:
    OODResult, OODLevel.

Dependencies:
    contracts.common

Implementation phase: P5 (extended from P0 stub)

Changes in P5:
    - energy_score made optional (float | None) because TorchXRayVision
      applies sigmoid internally and does not expose raw logits. Energy score
      is therefore an optional secondary signal.
    - Added mahalanobis_threshold, method, energy_is_approx fields for
      full auditability.
    - OODLevel retained as-is (IN_DISTRIBUTION / BORDERLINE / SEVERE).
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from .common import AgentName, AgentResult


class OODLevel(str, Enum):
    IN_DISTRIBUTION = "in_distribution"
    BORDERLINE = "borderline"
    SEVERE = "severe"


class OODResult(AgentResult):
    agent: AgentName = AgentName.OOD

    # Primary OOD signal — Mahalanobis distance
    mahalanobis_distance: float = Field(ge=0)
    mahalanobis_threshold: float | None = Field(default=None, ge=0)
    method: str = "mahalanobis"

    # Secondary signal — Energy score (OPTIONAL)
    # None when energy_enabled=False or when true logits are unavailable.
    # When computed, this is an approximation via inverse-sigmoid because
    # TorchXRayVision applies sigmoid internally.
    energy_score: float | None = None
    energy_is_approx: bool = True  # True = computed from probs via pseudo-logit

    # In-distribution percentile of this distance vs reference distribution
    in_distribution_percentile: float | None = Field(default=None, ge=0, le=100)

    # Feature layer used for extraction
    feature_layer: str | None = None

    # Whether Mahalanobis and energy signals agree on OOD verdict
    detectors_agree: bool | None = None

    # Final OOD level
    level: OODLevel
