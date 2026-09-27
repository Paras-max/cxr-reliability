"""OOD Agent (PRD FR-2) — Phase 5 Implementation.

Responsibility
--------------
    Primary:  Mahalanobis distance of the Base Model's mid-layer feature
              vector from the in-distribution mean and covariance.
    Optional: Approximate energy score (secondary cross-check).
    Returns:  OODResult with distance, energy, level, detector agreement,
              and human-readable reasoning.

Input
-----
    - features    : torch.Tensor shape (1024,) from ModelForward.features
    - raw_probs   : torch.Tensor shape (18,) from ModelForward.raw_probs
                    (used for optional energy score; these are sigmoid
                    probabilities — TXV does not expose raw logits)
    - stats_dir   : Path to pre-fitted OOD reference statistics directory
                    (produced by scripts/fit_ood_reference.py)

Output
------
    OODResult (see contracts/ood.py)

Energy Score Note
-----------------
    TorchXRayVision applies sigmoid internally. The model does NOT expose
    raw logits. Energy score is computed via inverse-sigmoid approximation
    and is disabled by default. See ood/energy.py for full documentation.

Dependencies
------------
    cxr_reliability.ood.detector, contracts.ood, config.thresholds

Implementation phase: P5
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch

from cxr_reliability.agents.base import AgentBase
from cxr_reliability.config.thresholds import OODThresholds
from cxr_reliability.contracts.common import AgentName
from cxr_reliability.contracts.ood import OODLevel, OODResult
from cxr_reliability.ood.detector import OODConfig, OODDetector

logger = logging.getLogger(__name__)


# ── Level conversion ─────────────────────────────────────────────────────────

_LEVEL_MAP: dict[str, OODLevel] = {
    "in_distribution": OODLevel.IN_DISTRIBUTION,
    "borderline": OODLevel.BORDERLINE,
    "severe": OODLevel.SEVERE,
}


class OODAgent(AgentBase):
    """
    OOD Agent for the reliability-aware multi-agent pipeline.

    Parameters
    ----------
    stats_dir          : Path to directory containing reference_stats.npz
                         (output of scripts/fit_ood_reference.py)
    thresholds         : OODThresholds loaded from configs/thresholds/*.yaml
    thresholds_version : string identifier for the threshold set (for audit)
    config             : OODConfig — all detector hyperparameters
                         (default: built from thresholds)

    Usage
    -----
        agent = OODAgent(stats_dir, thresholds, thresholds_version)
        result = agent.run(features=forward.features, raw_probs=forward.raw_probs)
    """

    name = AgentName.OOD
    version = "0.5.0"

    def __init__(
        self,
        stats_dir: Path | str,
        thresholds: OODThresholds,
        thresholds_version: str,
        config: OODConfig | None = None,
    ) -> None:
        self.stats_dir = Path(stats_dir) if stats_dir else None  # type: ignore[arg-type]
        self.thresholds = thresholds
        self.thresholds_version = thresholds_version

        # Build OODConfig from thresholds if not provided
        self._config = config or self._build_config_from_thresholds(thresholds)

        # Lazy-loaded detector — loaded on first run() call
        self._detector: OODDetector | None = None

    def _build_config_from_thresholds(self, thresholds: OODThresholds) -> OODConfig:
        """Map OODThresholds fields to OODConfig."""
        cfg = OODConfig()

        if thresholds.in_distribution_percentile is not None:
            cfg.threshold_percentile = thresholds.in_distribution_percentile

        if thresholds.mahalanobis_borderline is not None:
            cfg.mahalanobis_borderline = thresholds.mahalanobis_borderline

        if thresholds.mahalanobis_severe is not None:
            cfg.mahalanobis_severe = thresholds.mahalanobis_severe

        return cfg

    def _get_detector(self) -> OODDetector:
        """Lazily load the OOD detector and reference statistics."""
        if self._detector is None:
            if self.stats_dir is None or not self.stats_dir.exists():
                raise RuntimeError(
                    f"OOD reference statistics not found at: {self.stats_dir}\n"
                    "Run scripts/fit_ood_reference.py first to generate them.\n"
                    "Command: python scripts/fit_ood_reference.py --split train"
                )
            logger.info("Loading OOD reference stats from %s", self.stats_dir)
            self._detector = OODDetector(config=self._config)
            self._detector.load_stats(self.stats_dir)
        return self._detector

    @property
    def is_stats_available(self) -> bool:
        """Return True if reference statistics exist at stats_dir."""
        if self.stats_dir is None:
            return False
        return (Path(self.stats_dir) / "reference_stats.npz").exists()

    @property
    def is_development_mode(self) -> bool:
        """Return True if the loaded reference statistics are marked as development/provisional."""
        meta = self.get_stats_metadata()
        return bool(meta.get("is_development", False)) or "ood_dev" in str(self.stats_dir or "")

    def get_stats_metadata(self) -> dict[str, Any]:
        """Return metadata loaded from the reference stats directory, if available."""
        if self.stats_dir is None:
            return {}
        meta_file = Path(self.stats_dir) / "metadata.json"
        if meta_file.exists():
            try:
                import json
                with open(meta_file, encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return {}
        return {}

    def run(
        self,
        features: Any,
        raw_probs: Any = None,
        image_id: str = "",
    ) -> OODResult:
        """
        Run OOD detection on a single feature vector.

        Parameters
        ----------
        features  : torch.Tensor shape (1024,) — from ModelForward.features
        raw_probs : torch.Tensor shape (18,)   — from ModelForward.raw_probs
                    Optional; used only if energy_enabled=True.
                    These are sigmoid probabilities, NOT logits.
        image_id  : optional image identifier for logging

        Returns
        -------
        OODResult — structured result with level, distances, and reasoning
        """
        t_start = time.perf_counter()

        detector = self._get_detector()
        prediction = detector.predict(features, raw_probs=raw_probs)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # Map string level to OODLevel enum
        level = _LEVEL_MAP.get(prediction.level, OODLevel.BORDERLINE)

        # Build score: 0.0 = clearly in-distribution, 1.0 = severely OOD
        # Normalized as a rough fraction of the borderline threshold
        threshold = prediction.mahalanobis_threshold
        if threshold > 0:
            raw_score = min(prediction.mahalanobis_distance / threshold, 5.0) / 5.0
        else:
            raw_score = 0.0

        # Compose the OODResult (matches existing contract)
        result = OODResult(
            agent=AgentName.OOD,
            version=self.version,
            score=raw_score,
            label=level.value,
            reasoning=prediction.reasoning,
            latency_ms=latency_ms,
            thresholds_version=self.thresholds_version,

            # OOD-specific fields
            mahalanobis_distance=prediction.mahalanobis_distance,
            mahalanobis_threshold=prediction.mahalanobis_threshold,
            method=prediction.method,
            energy_score=prediction.energy_score,
            energy_is_approx=prediction.energy_is_approx,
            in_distribution_percentile=None,   # filled by evaluate_ood.py if needed
            feature_layer=self._config.feature_layer,
            detectors_agree=prediction.detectors_agree,
            level=level,
        )

        logger.debug(
            "OOD: image_id=%s dist=%.4f threshold=%.4f level=%s latency=%.1fms",
            image_id or "<unknown>",
            prediction.mahalanobis_distance,
            prediction.mahalanobis_threshold,
            level.value,
            latency_ms,
        )

        return result

