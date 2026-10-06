"""Uncertainty estimator — Phase 6.

Responsibility
--------------
High-level estimator combining confidence and predictive entropy into a
deterministic, structured uncertainty verdict with human-readable reasoning.

Safety and Calibration Disclaimer
---------------------------------
Research prototype only. The uncertainty measures and provisional levels
reflect model output ambiguity, NOT clinical diagnostic correctness or patient
risk. Post-hoc probability calibration (Phase 4.5) is deferred; reported
confidence and entropy thresholds are uncalibrated development defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import torch

from .confidence import compute_binary_confidence
from .entropy import compute_binary_entropy, compute_normalized_entropy


class UncertaintyLevel(str, Enum):
    """Categorical uncertainty level for multi-agent routing."""
    LOW = "LOW"
    HIGH = "HIGH"


@dataclass
class UncertaintyConfig:
    """
    Configuration and provisional development thresholds for uncertainty estimation.

    NOTE: These thresholds are initial development defaults and are NOT clinically calibrated.
    """
    enabled: bool = True
    epsilon: float = 1e-8
    entropy_low_threshold: float = 0.25      # Normalized entropy <= 0.25 is low
    entropy_high_threshold: float = 0.60     # Normalized entropy >= 0.60 is high
    confidence_low_threshold: float = 0.60   # Confidence <= 0.60 is low
    confidence_high_threshold: float = 0.85  # Confidence >= 0.85 is high
    high_if_low_confidence_or_high_entropy: bool = True
    method: str = "binary_confidence_entropy"
    is_calibrated: bool = False
    note: str = "Provisional development thresholds — uncalibrated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UncertaintyEvaluation:
    """Structured evaluation output from UncertaintyEstimator."""
    raw_model_score: float
    confidence: float
    entropy: float
    normalized_entropy: float
    uncertainty_level: UncertaintyLevel
    method: str
    thresholds: dict[str, Any]
    reasoning: str
    image_id: str | None = None
    timestamp: str | None = None


class UncertaintyEstimator:
    """
    Estimator combining predictive entropy and confidence.

    Rule:
    -----
    1. LOW uncertainty:
       confidence >= confidence_high_threshold AND normalized_entropy <= entropy_low_threshold
    2. HIGH uncertainty:
       Everything else.
    """

    def __init__(self, config: UncertaintyConfig | None = None) -> None:
        self.config = config or UncertaintyConfig()

    def evaluate(
        self,
        score: float | torch.Tensor | np.ndarray,
        image_id: str | None = None,
    ) -> UncertaintyEvaluation:
        """
        Evaluate uncertainty for a single Pneumonia score.

        Parameters
        ----------
        score : float, torch.Tensor, or np.ndarray
            Pneumonia output probability in [0.0, 1.0].
        image_id : str, optional
            Identifier for audit logging.

        Returns
        -------
        UncertaintyEvaluation
        """
        # ── Extract scalar value with validation ──────────────────────────────
        p = self._extract_scalar(score)

        # ── Compute metrics ───────────────────────────────────────────────────
        conf = float(compute_binary_confidence(p))
        ent = float(compute_binary_entropy(p, epsilon=self.config.epsilon))
        norm_ent = float(compute_normalized_entropy(p, epsilon=self.config.epsilon))

        # ── Determine Uncertainty Level ───────────────────────────────────────
        c_low = self.config.confidence_low_threshold
        c_high = self.config.confidence_high_threshold
        e_low = self.config.entropy_low_threshold
        e_high = self.config.entropy_high_threshold

        if conf >= c_high and norm_ent <= e_low:
            level = UncertaintyLevel.LOW
        else:
            level = UncertaintyLevel.HIGH

        # ── Generate Human-Readable Reasoning (No Clinical Claims) ────────────
        reasoning = self._build_reasoning(
            p=p,
            conf=conf,
            norm_ent=norm_ent,
            level=level,
            c_low=c_low,
            c_high=c_high,
            e_low=e_low,
            e_high=e_high,
        )

        now_iso = datetime.now(timezone.utc).isoformat()

        return UncertaintyEvaluation(
            raw_model_score=p,
            confidence=conf,
            entropy=ent,
            normalized_entropy=norm_ent,
            uncertainty_level=level,
            method=self.config.method,
            thresholds=self.config.to_dict(),
            reasoning=reasoning,
            image_id=image_id,
            timestamp=now_iso,
        )

    @staticmethod
    def _extract_scalar(val: float | torch.Tensor | np.ndarray) -> float:
        if isinstance(val, torch.Tensor):
            if torch.isnan(val).any():
                raise ValueError("Score tensor contains NaN.")
            if torch.isinf(val).any():
                raise ValueError("Score tensor contains Inf.")
            if val.numel() != 1:
                raise ValueError(f"Expected single scalar score, got tensor shape {tuple(val.shape)}")
            p = float(val.item())
        elif isinstance(val, np.ndarray):
            if np.isnan(val).any():
                raise ValueError("Score array contains NaN.")
            if np.isinf(val).any():
                raise ValueError("Score array contains Inf.")
            if val.size != 1:
                raise ValueError(f"Expected single scalar score, got array shape {val.shape}")
            p = float(val.item())
        elif isinstance(val, (int, float)):
            p = float(val)
            if np.isnan(p):
                raise ValueError("Score is NaN.")
            if np.isinf(p):
                raise ValueError("Score is Inf.")
        else:
            raise TypeError(f"Unsupported score type: {type(val).__name__}")

        if p < 0.0 or p > 1.0:
            raise ValueError(f"Score must be in [0.0, 1.0], got {p}")
        return p

    @staticmethod
    def _build_reasoning(
        p: float,
        conf: float,
        norm_ent: float,
        level: UncertaintyLevel,
        c_low: float,
        c_high: float,
        e_low: float,
        e_high: float,
    ) -> str:
        """Construct a strictly non-clinical explainability summary."""
        if level == UncertaintyLevel.LOW:
            return (
                f"Model output is relatively confident (confidence: {conf:.3f} >= {c_high:.2f}) "
                f"and predictive entropy is low (normalized: {norm_ent:.3f} <= {e_low:.2f}). "
                f"Uncertainty is LOW."
            )
        reasons = []
        if conf < c_high:
            reasons.append(f"confidence is below high-confidence threshold ({conf:.3f} < {c_high:.2f})")
        if norm_ent > e_low:
            reasons.append(f"predictive entropy is above low-entropy threshold ({norm_ent:.3f} > {e_low:.2f})")
        reason_str = " or ".join(reasons) if reasons else "uncertainty threshold criteria not met"
        return (
            f"Model output is uncertain because {reason_str}. "
            f"Raw score: {p:.3f}. Uncertainty is HIGH."
        )
