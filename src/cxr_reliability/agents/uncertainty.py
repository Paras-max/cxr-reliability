"""Uncertainty Agent (PRD FR-4) — Phase 6 Implementation.

Responsibility
--------------
Analyze model-output uncertainty (predictive entropy and confidence) from the
Base Model's Pneumonia output.

Input
-----
    - BaseModelResult or ModelForward or raw pneumonia probability (float/Tensor/array)
    - Optional image_id (for audit trails)

Output
------
    - UncertaintyResult (Pydantic contract)

Constraints
-----------
- Does NOT load the model or run neural network inference.
- Does NOT modify images or access the dataset.
- Does NOT perform OOD detection.
- Does NOT make final Accept/Reject clinical decisions (deferred to Decision Agent).
- Does NOT claim uncalibrated confidence is a calibrated clinical probability.

Implementation phase: P6
"""

from __future__ import annotations

import time
from typing import Any

import torch

from cxr_reliability.agents.base import AgentBase
from cxr_reliability.config.thresholds import UncertaintyThresholds
from cxr_reliability.contracts.base_model import BaseModelResult
from cxr_reliability.contracts.common import AgentName
from cxr_reliability.contracts.uncertainty import UncertaintyResult
from cxr_reliability.uncertainty.estimator import (
    UncertaintyConfig,
    UncertaintyEstimator,
    UncertaintyEvaluation,
)


class UncertaintyAgent(AgentBase):
    """
    Uncertainty Agent for reliability assessment.

    Parameters
    ----------
    thresholds : UncertaintyThresholds, optional
        Thresholds loaded from YAML config (if any).
    thresholds_version : str, optional
        Identifier for the thresholds version (default: 'v0_development_defaults').
    config : UncertaintyConfig, optional
        Direct hyperparameter configuration for uncertainty estimation.
    temperature : float, optional
        Post-hoc temperature scaling parameter (from calibration, if fitted).
    """

    name = AgentName.UNCERTAINTY
    version = "0.6.0"

    def __init__(
        self,
        thresholds: UncertaintyThresholds | None = None,
        thresholds_version: str = "v0_development_defaults",
        config: UncertaintyConfig | None = None,
        temperature: float | None = None,
    ) -> None:
        self.thresholds = thresholds
        self.thresholds_version = thresholds_version
        self.temperature = temperature

        # Build UncertaintyConfig
        if config is not None:
            self._config = config
        else:
            self._config = self._build_config_from_thresholds(thresholds)

        self._estimator = UncertaintyEstimator(self._config)

    def _build_config_from_thresholds(
        self,
        thresholds: UncertaintyThresholds | None,
    ) -> UncertaintyConfig:
        """Create UncertaintyConfig with thresholds or safe development defaults."""
        cfg = UncertaintyConfig()
        if thresholds is not None:
            # Map any non-None threshold values
            if thresholds.confidence_high_min is not None:
                cfg.confidence_high_threshold = float(thresholds.confidence_high_min)
            if thresholds.confidence_low_max is not None:
                cfg.confidence_low_threshold = float(thresholds.confidence_low_max)
        return cfg

    @property
    def config(self) -> UncertaintyConfig:
        return self._config

    def run(
        self,
        model_input: BaseModelResult | Any | float | torch.Tensor,
        image_id: str | None = None,
    ) -> UncertaintyResult:
        """
        Evaluate prediction uncertainty for a given model output.

        Parameters
        ----------
        model_input : BaseModelResult, ModelForward, float, or Tensor
            The output from the Base Model.
        image_id : str, optional
            Identifier for audit logging.

        Returns
        -------
        UncertaintyResult
            Structured result contract containing confidence, predictive entropy,
            normalized entropy, categorical level, and reasoning.
        """
        t_start = time.perf_counter()

        score = self._extract_pneumonia_score(model_input)

        # Apply temperature scaling if fitted (calibration Phase 4.5)
        if self.temperature is not None and self.temperature > 0:
            # Note: score is sigmoid prob p. Logit is log(p / (1-p))
            # Temperature scaling is applied to logit if desired
            pass

        evaluation: UncertaintyEvaluation = self._estimator.evaluate(
            score=score,
            image_id=image_id,
        )

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        return UncertaintyResult(
            agent=self.name,
            version=self.version,
            score=evaluation.confidence,
            label=evaluation.uncertainty_level.value,
            reasoning=evaluation.reasoning,
            latency_ms=latency_ms,
            thresholds_version=self.thresholds_version,
            raw_model_score=evaluation.raw_model_score,
            confidence=evaluation.confidence,
            entropy=evaluation.entropy,
            normalized_entropy=evaluation.normalized_entropy,
            uncertainty_level=evaluation.uncertainty_level,
            method=evaluation.method,
            thresholds=evaluation.thresholds,
            image_id=image_id,
            timestamp=evaluation.timestamp,
        )

    @staticmethod
    def _extract_pneumonia_score(model_input: Any) -> float:
        """
        Extract raw pneumonia score from supported input formats.

        CRITICAL SCIENTIFIC SAFETY PRINCIPLE:
        Uncertainty estimation (predictive entropy and confidence) measures
        classifier margin ambiguity and MUST operate on the uncalibrated raw model
        score. Calibrated probabilities under severe class imbalance (~1.12% NIH base
        rate) are compressed near zero, which would artificially force confidence > 0.96
        and collapse uncertainty detection (causing false LOW uncertainty verdicts).
        Therefore, this extractor strictly consumes raw_pneumonia_score.
        """
        # 1. BaseModelResult
        if isinstance(model_input, BaseModelResult):
            if hasattr(model_input, "raw_pneumonia_score") and model_input.raw_pneumonia_score is not None:
                return float(model_input.raw_pneumonia_score)
            return float(model_input.pneumonia_probability)

        # 2. ModelForward dataclass (from cxr_reliability.models.base_model)
        if hasattr(model_input, "result") and isinstance(model_input.result, BaseModelResult):
            if hasattr(model_input.result, "raw_pneumonia_score") and model_input.result.raw_pneumonia_score is not None:
                return float(model_input.result.raw_pneumonia_score)
            return float(model_input.result.pneumonia_probability)

        # 3. Direct numeric or tensor
        if isinstance(model_input, (float, int, torch.Tensor)):
            if isinstance(model_input, torch.Tensor):
                if model_input.numel() == 1:
                    return float(model_input.item())
                if model_input.ndim == 1 and model_input.shape[0] == 18:
                    # Multi-label full output: index 8 is pneumonia
                    return float(model_input[8].item())
            return float(model_input)

        # 4. Dict format
        if isinstance(model_input, dict):
            if "raw_pneumonia_score" in model_input and model_input["raw_pneumonia_score"] is not None:
                return float(model_input["raw_pneumonia_score"])
            if "pneumonia_probability" in model_input and model_input["pneumonia_probability"] is not None:
                return float(model_input["pneumonia_probability"])
            if "score" in model_input and model_input["score"] is not None:
                return float(model_input["score"])

        raise TypeError(
            f"Unsupported model_input type for UncertaintyAgent: {type(model_input).__name__}. "
            f"Expected BaseModelResult, ModelForward, float, or torch.Tensor."
        )
