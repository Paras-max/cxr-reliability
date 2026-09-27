"""Verification Agent contract (PRD FR-7) — Phase 10.

Responsibility:
    Schema for comparing reliability signals before and after repair / escalation,
    and producing the structured verification verdict (release / escalate / reject).

    Authoritative Signal Sources:
    Uses existing QualityResult, OODResult, BaseModelResult, and UncertaintyResult
    contracts as the primary source of before/after data. Scalar fields in
    SignalBundle serve as convenience/compatibility summaries.

Implementation phase: P10 (extended from P0 stub)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from .base_model import BaseModelResult
from .common import AgentName, AgentResult, StrictModel
from .decision import Action
from .ood import OODLevel, OODResult
from .quality import QualityLevel, QualityResult
from .uncertainty import UncertaintyLevel, UncertaintyResult


class NextStep(str, Enum):
    """Next pipeline routing action following verification."""

    RELEASE = "release"
    ESCALATE = "escalate"
    REJECT = "reject"
    NOT_APPLICABLE = "not_applicable"


class VerificationStatus(str, Enum):
    """Categorical outcome of the verification evaluation."""

    VERIFIED = "verified"
    ESCALATE = "escalate"
    REJECT = "reject"
    NOT_APPLICABLE = "not_applicable"


class SignalBundle(StrictModel):
    """
    Container of reliability signals before or after repair/escalation.

    Authoritative Sources:
    - quality: QualityResult
    - ood: OODResult
    - base_model: BaseModelResult
    - uncertainty: UncertaintyResult

    Scalar fields provide convenience and backwards compatibility with Phase 0 stubs.
    """

    # Primary authoritative agent results
    quality: QualityResult | None = None
    ood: OODResult | None = None
    base_model: BaseModelResult | None = None
    uncertainty: UncertaintyResult | None = None

    # Convenience / compatibility scalar signals
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.5, le=1.0)
    quality_score: float | None = None  # e.g., SNR dB or (100 - blur_pct)
    ood_score: float | None = None  # Mahalanobis distance

    @model_validator(mode="before")
    @classmethod
    def _extract_from_agent_results(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        # Extract probability from base_model if not explicitly supplied
        bm = values.get("base_model")
        if bm is not None:
            prob = getattr(bm, "pneumonia_probability", None)
            if prob is not None and values.get("probability") is None:
                values["probability"] = float(prob)

        # Extract confidence from uncertainty (or base_model) if not explicitly supplied
        unc = values.get("uncertainty")
        if unc is not None:
            conf = getattr(unc, "confidence", None)
            if conf is not None and values.get("confidence") is None:
                values["confidence"] = float(conf)
        elif values.get("confidence") is None and values.get("probability") is not None:
            p = float(values["probability"])
            values["confidence"] = max(p, 1.0 - p)

        # Extract ood_score from ood if not explicitly supplied
        ood = values.get("ood")
        if ood is not None:
            dist = getattr(ood, "mahalanobis_distance", None)
            if dist is not None and values.get("ood_score") is None:
                values["ood_score"] = float(dist)

        # Extract quality_score from quality if not explicitly supplied
        q = values.get("quality")
        if q is not None:
            if values.get("quality_score") is None:
                # Use snr_db as default scalar quality score if available
                snr = getattr(q, "snr_db", None)
                if snr is not None:
                    values["quality_score"] = float(snr)
                else:
                    blur = getattr(q, "blur_pct", None)
                    if blur is not None:
                        values["quality_score"] = float(100.0 - blur)

        return values

    def get_probability(self) -> float | None:
        if self.probability is not None:
            return self.probability
        if self.base_model is not None:
            return self.base_model.pneumonia_probability
        return None

    def get_confidence(self) -> float | None:
        if self.confidence is not None:
            return self.confidence
        if self.uncertainty is not None:
            return self.uncertainty.confidence
        prob = self.get_probability()
        if prob is not None:
            return max(prob, 1.0 - prob)
        return None

    def get_ood_distance(self) -> float | None:
        if self.ood_score is not None:
            return self.ood_score
        if self.ood is not None:
            return self.ood.mahalanobis_distance
        return None

    def get_ood_level(self) -> OODLevel | None:
        if self.ood is not None:
            return self.ood.level
        return None

    def get_quality_level(self) -> QualityLevel | None:
        if self.quality is not None:
            return self.quality.overall
        return None

    def get_uncertainty_level(self) -> UncertaintyLevel | None:
        if self.uncertainty is not None:
            return self.uncertainty.uncertainty_level
        return None


class VerificationResult(AgentResult):
    """
    Verification evaluation record comparing pre-repair and post-repair states.

    NOTE: The Verification Agent evaluates whether repair produced a useful change
    in reliability signals according to configured provisional development thresholds.
    It does NOT make clinical claims or declare diagnostic correctness.
    """

    agent: AgentName = AgentName.VERIFICATION
    verified: bool
    status: VerificationStatus = VerificationStatus.ESCALATE
    next_step: NextStep = NextStep.ESCALATE

    # Signal Deltas (PRD FR-7)
    delta_confidence: float
    delta_quality: float
    delta_ood: float
    delta_entropy: float | None = None
    label_flipped: bool  # True if predicted binary label changed

    # Threshold Audit
    min_confidence_gain_used: float
    threshold_is_provisional: bool = True

    # Audit & Provenance Fields
    action: Action | None = None
    repair_applied: bool = False

    # Authoritative before/after state summaries
    quality_before: QualityResult | None = None
    quality_after: QualityResult | None = None
    quality_level_before: QualityLevel | None = None
    quality_level_after: QualityLevel | None = None

    ood_level_before: OODLevel | None = None
    ood_level_after: OODLevel | None = None

    uncertainty_level_before: UncertaintyLevel | None = None
    uncertainty_level_after: UncertaintyLevel | None = None

    probability_before: float | None = None
    probability_after: float | None = None
    confidence_before: float | None = None
    confidence_after: float | None = None

    quality_metric_deltas: dict[str, float] = Field(default_factory=dict)
