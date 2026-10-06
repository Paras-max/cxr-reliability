"""End-to-end pipeline output contract (PRD FR-8).

Responsibility:
    Schema for what the system returns per image: a prediction with a reliability label,
    or a needs-human-review flag, plus the full agent trace. Enforces that the
    needs_human_review flag and the reliability label can never disagree.

Input:
    n/a

Output:
    PipelineResult, PipelineState, PipelineOutput, PredictionSummary, ReliabilityLabel.

Dependencies:
    contracts.* (all agent contracts)

Implementation phase: P0
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from .base_model import BaseModelResult
from .common import DISCLAIMER, StrictModel
from .decision import Action, DecisionResult
from .ood import OODResult
from .quality import QualityResult
from .repair import RepairResult
from .uncertainty import UncertaintyResult
from .verification import VerificationResult


class FinalClassification(str, Enum):
    PNEUMONIA = "PNEUMONIA"
    NO_PNEUMONIA = "NO_PNEUMONIA"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"


class ReliabilityLabel(str, Enum):
    ACCEPTED = "accepted"
    ACCEPTED_AFTER_REPAIR = "accepted_after_repair"
    ESCALATED_MODEL = "escalated_model"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class PredictionSummary(StrictModel):
    # Primary probability output (calibrated when available, or raw if uncalibrated)
    pneumonia_probability: float = Field(ge=0, le=1)
    # Raw uncalibrated model sigmoid score (for margin analysis and audit)
    raw_model_score: float | None = Field(default=None, ge=0, le=1)
    # Post-hoc Platt-calibrated event probability
    calibrated_probability: float | None = Field(default=None, ge=0, le=1)
    # Binary classification decision based on validated operating threshold
    positive: bool | None = None
    # Validated raw-scale decision threshold (0.522161 from Phase 4.5 calibration)
    decision_threshold: float | None = Field(default=0.522161, ge=0, le=1)
    raw_decision_threshold: float | None = Field(default=0.522161, ge=0, le=1)
    # Calibrated-scale operating threshold derived dynamically from Platt calibrator
    calibrated_decision_threshold: float | None = Field(default=None, ge=0, le=1)
    source_model_id: str

    @model_validator(mode="before")
    @classmethod
    def _sync_scores(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "raw_model_score" not in data or data["raw_model_score"] is None:
                if "pneumonia_probability" in data and data["pneumonia_probability"] is not None:
                    data["raw_model_score"] = data["pneumonia_probability"]
        return data



class PipelineState(str, Enum):
    """Execution state of the end-to-end reliability pipeline."""

    ACCEPT = "accept"
    REPAIR = "repair"
    VERIFIED = "verified"
    ESCALATE = "escalate"
    REJECT = "reject"
    ERROR = "error"


class PipelineOutput(StrictModel):
    audit_id: str
    prediction: PredictionSummary | None = None   # withheld when review is needed
    reliability_label: ReliabilityLabel
    final_action: Action
    needs_human_review: bool
    final_classification: FinalClassification = Field(default=FinalClassification.HUMAN_REVIEW_REQUIRED)

    quality: QualityResult | None = None
    ood: OODResult | None = None
    base_model: BaseModelResult | None = None
    uncertainty: UncertaintyResult | None = None
    decision_history: list[DecisionResult] = Field(default_factory=list)
    repair: RepairResult | None = None
    verification: VerificationResult | None = None
    escalation_base_model: BaseModelResult | None = None

    total_latency_ms: float | None = Field(default=None, ge=0)
    disclaimer: str = DISCLAIMER

    @model_validator(mode="before")
    @classmethod
    def _deduce_final_classification(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "final_classification" not in data or data["final_classification"] is None:
                needs_hr = data.get("needs_human_review", False)
                pred = data.get("prediction")
                if needs_hr or pred is None:
                    data["final_classification"] = FinalClassification.HUMAN_REVIEW_REQUIRED
                else:
                    pos = getattr(pred, "positive", None) if hasattr(pred, "positive") else (pred.get("positive") if isinstance(pred, dict) else None)
                    if pos is True:
                        data["final_classification"] = FinalClassification.PNEUMONIA
                    elif pos is False:
                        data["final_classification"] = FinalClassification.NO_PNEUMONIA
                    else:
                        prob = getattr(pred, "pneumonia_probability", None) if hasattr(pred, "pneumonia_probability") else (pred.get("pneumonia_probability") if isinstance(pred, dict) else None)
                        if prob is None:
                            prob = getattr(pred, "raw_model_score", None) if hasattr(pred, "raw_model_score") else (pred.get("raw_model_score") if isinstance(pred, dict) else None)
                        thresh = getattr(pred, "decision_threshold", 0.522161) if hasattr(pred, "decision_threshold") else (pred.get("decision_threshold", 0.522161) if isinstance(pred, dict) else 0.522161)
                        if thresh is None:
                            thresh = 0.522161
                        if prob is not None and prob >= thresh:
                            data["final_classification"] = FinalClassification.PNEUMONIA
                        elif prob is not None and prob < thresh:
                            data["final_classification"] = FinalClassification.NO_PNEUMONIA
                        else:
                            data["final_classification"] = FinalClassification.HUMAN_REVIEW_REQUIRED
        return data

    @model_validator(mode="after")
    def _review_flag_matches_label(self) -> PipelineOutput:
        is_review_label = self.reliability_label == ReliabilityLabel.NEEDS_HUMAN_REVIEW
        if self.needs_human_review != is_review_label:
            raise ValueError("needs_human_review must be True exactly when the label is needs_human_review")
        if self.needs_human_review and self.prediction is not None:
            raise ValueError("Prediction must be withheld (None) whenever needs_human_review is True")
        if self.final_classification in (FinalClassification.PNEUMONIA, FinalClassification.NO_PNEUMONIA):
            if self.needs_human_review or self.prediction is None:
                raise ValueError("final_classification cannot release diagnosis when needs_human_review is True")
        elif self.final_classification == FinalClassification.HUMAN_REVIEW_REQUIRED:
            if not self.needs_human_review:
                raise ValueError("final_classification HUMAN_REVIEW_REQUIRED requires needs_human_review to be True")
        return self


class PipelineResult(StrictModel):
    """
    Comprehensive structured output contract produced by the ReliabilityPipeline.

    Captures the full end-to-end execution trace: before and after repair states,
    intermediate agent results, routing states, latencies, and reasons.
    """

    audit_id: str
    pipeline_state: PipelineState
    final_action: Action
    reliability_label: ReliabilityLabel
    needs_human_review: bool
    final_classification: FinalClassification = Field(default=FinalClassification.HUMAN_REVIEW_REQUIRED)

    prediction: PredictionSummary | None = None  # withheld when review is needed or error
    quality: QualityResult | None = None
    ood: OODResult | None = None
    base_model: BaseModelResult | None = None
    uncertainty: UncertaintyResult | None = None
    decision_history: list[DecisionResult] = Field(default_factory=list)
    repair: RepairResult | None = None
    verification: VerificationResult | None = None

    after_repair_quality: QualityResult | None = None
    after_repair_ood: OODResult | None = None
    after_repair_base_model: BaseModelResult | None = None
    after_repair_uncertainty: UncertaintyResult | None = None

    repair_attempts: int = Field(default=0, ge=0)
    total_latency_ms: float | None = Field(default=None, ge=0)
    intermediate_latencies: dict[str, float] = Field(default_factory=dict)
    reason: str = ""
    disclaimer: str = DISCLAIMER
    error_message: str | None = None
    repaired_image_png: bytes | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _deduce_final_classification(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "final_classification" not in data or data["final_classification"] is None:
                needs_hr = data.get("needs_human_review", False)
                pred = data.get("prediction")
                if needs_hr or pred is None:
                    data["final_classification"] = FinalClassification.HUMAN_REVIEW_REQUIRED
                else:
                    pos = getattr(pred, "positive", None) if hasattr(pred, "positive") else (pred.get("positive") if isinstance(pred, dict) else None)
                    if pos is True:
                        data["final_classification"] = FinalClassification.PNEUMONIA
                    elif pos is False:
                        data["final_classification"] = FinalClassification.NO_PNEUMONIA
                    else:
                        prob = getattr(pred, "pneumonia_probability", None) if hasattr(pred, "pneumonia_probability") else (pred.get("pneumonia_probability") if isinstance(pred, dict) else None)
                        if prob is None:
                            prob = getattr(pred, "raw_model_score", None) if hasattr(pred, "raw_model_score") else (pred.get("raw_model_score") if isinstance(pred, dict) else None)
                        thresh = getattr(pred, "decision_threshold", 0.522161) if hasattr(pred, "decision_threshold") else (pred.get("decision_threshold", 0.522161) if isinstance(pred, dict) else 0.522161)
                        if thresh is None:
                            thresh = 0.522161
                        if prob is not None and prob >= thresh:
                            data["final_classification"] = FinalClassification.PNEUMONIA
                        elif prob is not None and prob < thresh:
                            data["final_classification"] = FinalClassification.NO_PNEUMONIA
                        else:
                            data["final_classification"] = FinalClassification.HUMAN_REVIEW_REQUIRED
        return data

    @model_validator(mode="after")
    def _validate_pipeline_result(self) -> PipelineResult:
        is_review_label = self.reliability_label == ReliabilityLabel.NEEDS_HUMAN_REVIEW
        if self.needs_human_review != is_review_label:
            raise ValueError(
                "needs_human_review must be True exactly when reliability_label is NEEDS_HUMAN_REVIEW"
            )
        if self.needs_human_review and self.prediction is not None:
            raise ValueError(
                "Prediction must be withheld (None) whenever needs_human_review is True"
            )
        if self.final_classification in (FinalClassification.PNEUMONIA, FinalClassification.NO_PNEUMONIA):
            if self.needs_human_review or self.prediction is None:
                raise ValueError("final_classification cannot release diagnosis when needs_human_review is True")
        elif self.final_classification == FinalClassification.HUMAN_REVIEW_REQUIRED:
            if not self.needs_human_review:
                raise ValueError("final_classification HUMAN_REVIEW_REQUIRED requires needs_human_review to be True")
        return self

    def to_pipeline_output(self) -> PipelineOutput:
        """Convert PipelineResult into a PipelineOutput contract for external consumers."""
        return PipelineOutput(
            audit_id=self.audit_id,
            prediction=self.prediction,
            reliability_label=self.reliability_label,
            final_action=self.final_action,
            needs_human_review=self.needs_human_review,
            final_classification=self.final_classification,
            quality=self.quality,
            ood=self.ood,
            base_model=self.base_model,
            uncertainty=self.uncertainty,
            decision_history=self.decision_history,
            repair=self.repair,
            verification=self.verification,
            total_latency_ms=self.total_latency_ms,
            disclaimer=self.disclaimer,
        )

