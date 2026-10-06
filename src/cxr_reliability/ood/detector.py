"""High-level OOD Detector interface — Phase 5.

Responsibility
--------------
Wraps the Mahalanobis distance computation and optional energy score into a
unified OODDetector interface that:
    - Fits reference statistics from a feature matrix (training data only)
    - Loads pre-fitted statistics from disk (for inference)
    - Scores a single feature vector
    - Returns a structured OODPrediction with human-readable reasoning

OOD Level Mapping
-----------------
    IN_DISTRIBUTION : mahalanobis_distance <= mahalanobis_borderline threshold
    BORDERLINE      : borderline < distance <= severe threshold
    SEVERE          : distance > severe threshold

    If only one threshold is configured, distances above it are marked SEVERE.

Human-Readable Reasoning
-------------------------
The reasoning strings are carefully worded to reflect distributional shift,
NOT clinical diagnosis:

    IN_DISTRIBUTION:
        "Feature distance (X.XX) is within the expected in-distribution
         range (threshold: Y.YY). No evidence of distribution shift."

    BORDERLINE:
        "Feature distance (X.XX) is moderately elevated relative to the
         training distribution. Potential distribution shift; downstream
         confidence should be interpreted with caution."

    SEVERE:
        "Feature distance (X.XX) substantially exceeds the distribution
         threshold (Y.YY). This image may differ significantly from the
         training distribution; downstream results should be treated as
         low-reliability."

These wordings must NOT be changed to imply diagnosis or certainty.

Scientific Constraints
----------------------
- Only training data is used to fit reference statistics.
- Validation data may be used to select thresholds (NOT test data).
- The OOD level is a reliability signal, not a clinical finding.
- Energy score is secondary and optional.
- Thresholds are provisional; one value is not universally optimal.

Implementation phase: P5
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cxr_reliability.ood.energy import energy_score_safe
from cxr_reliability.ood.mahalanobis import (
    batch_mahalanobis_distances,
    mahalanobis_distance,
)
from cxr_reliability.ood.statistics import (
    DEFAULT_LAMBDA_REG,
    OODReferenceStats,
    compute_percentile_threshold,
    fit_reference_stats,
    load_stats,
    save_stats,
)

logger = logging.getLogger(__name__)


@dataclass
class OODConfig:
    """
    Configuration for the OOD Detector.

    All fields correspond to the ood: section of pipeline.yaml.
    """

    enabled: bool = True
    method: str = "mahalanobis"

    # Covariance regularization: Σ_reg = Σ + lambda_reg * I
    covariance_regularization: float = DEFAULT_LAMBDA_REG

    # Threshold method: "percentile" or "fixed"
    threshold_method: str = "percentile"
    threshold_percentile: float = 99.0

    # Absolute threshold values (set by fit_ood_reference.py after fitting)
    # If None and threshold_method == "percentile", these are computed during fit.
    mahalanobis_borderline: float | None = None
    mahalanobis_severe: float | None = None

    # Energy score (secondary, optional)
    energy_enabled: bool = False
    energy_temperature: float = 1.0

    # Feature layer (for provenance/metadata only; not used in computation)
    feature_layer: str = "features.norm5"
    feature_dim: int = 1024

    def validate(self) -> None:
        """Raise ValueError for invalid configuration values."""
        if self.covariance_regularization <= 0:
            raise ValueError(
                f"covariance_regularization must be > 0, "
                f"got {self.covariance_regularization}"
            )
        if not (0 < self.threshold_percentile <= 100):
            raise ValueError(
                f"threshold_percentile must be in (0, 100], "
                f"got {self.threshold_percentile}"
            )
        if self.threshold_method not in ("percentile", "fixed"):
            raise ValueError(
                f"threshold_method must be 'percentile' or 'fixed', "
                f"got '{self.threshold_method}'"
            )
        if self.energy_temperature <= 0:
            raise ValueError(
                f"energy_temperature must be > 0, got {self.energy_temperature}"
            )


@dataclass
class OODPrediction:
    """
    Structured result of a single OOD detection query.

    Fields
    ------
    is_ood              : True if mahalanobis_distance > mahalanobis_borderline
    level               : "in_distribution" | "borderline" | "severe"
    mahalanobis_distance: the computed distance
    mahalanobis_threshold: the threshold used for the primary decision
    method              : always "mahalanobis" for the primary detector
    energy_score        : optional secondary signal (None if disabled or failed)
    energy_is_approx    : True if energy score is a pseudo-logit approximation
    detectors_agree     : True if both Maha and energy agree on OOD verdict
                          (None if energy_score is None)
    reasoning           : human-readable explanation (non-diagnostic)
    latency_ms          : computation time in milliseconds
    """

    is_ood: bool
    level: str                          # "in_distribution" | "borderline" | "severe"
    mahalanobis_distance: float
    mahalanobis_threshold: float
    method: str = "mahalanobis"
    energy_score: float | None = None
    energy_is_approx: bool = True
    detectors_agree: bool | None = None
    reasoning: str = ""
    latency_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def _build_reasoning(
    level: str,
    distance: float,
    borderline_threshold: float,
    severe_threshold: float | None,
    energy_score: float | None,
    energy_is_approx: bool,
) -> str:
    """Build a human-readable, scientifically cautious reasoning string."""

    dist_str = f"{distance:.4f}"
    thr_str = f"{borderline_threshold:.4f}"

    if level == "in_distribution":
        text = (
            f"Feature distance ({dist_str}) is within the expected "
            f"in-distribution range (threshold: {thr_str}). "
            "No evidence of distribution shift detected for this image."
        )
    elif level == "borderline":
        sev_str = f"{severe_threshold:.4f}" if severe_threshold is not None else "not set"
        text = (
            f"Feature distance ({dist_str}) is moderately elevated relative "
            f"to the training distribution (borderline threshold: {thr_str}, "
            f"severe threshold: {sev_str}). "
            "Potential distribution shift; downstream predictions should be "
            "interpreted with caution."
        )
    else:  # severe
        text = (
            f"Feature distance ({dist_str}) substantially exceeds the "
            f"distribution threshold ({thr_str}). "
            "This image may differ significantly from the training distribution. "
            "Downstream predictions should be treated as low-reliability. "
            "This is a distribution-shift signal, NOT a clinical diagnosis."
        )

    if energy_score is not None:
        approx_note = " (approximation — TXV provides probabilities, not logits)" if energy_is_approx else ""
        text += (
            f" Secondary energy score{approx_note}: {energy_score:.4f}."
        )

    return text


class OODDetector:
    """
    High-level OOD Detector for the reliability-aware pipeline.

    Usage for inference (fitted stats pre-computed):
        detector = OODDetector(config)
        detector.load_stats(stats_dir)
        prediction = detector.predict(feature_vector)

    Usage for fitting:
        detector = OODDetector(config)
        detector.fit(feature_matrix)
        detector.save(output_dir)

    The detector maintains reference statistics in memory. Calling
    load_stats() avoids recomputing statistics every inference call.

    Scientific Constraints
    ----------------------
    - fit() must only be called with TRAINING features.
    - Thresholds derived from validation data are provisional; they must
      not be confused with calibrated operating points.
    - is_ood=True does not mean the image is clinically abnormal.
    """

    def __init__(self, config: OODConfig | None = None) -> None:
        self.config: OODConfig = config or OODConfig()
        self.config.validate()
        self._stats: OODReferenceStats | None = None
        self._is_fitted: bool = False

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(
        self,
        feature_matrix: np.ndarray,
        model_id: str = "densenet121-res224-nih",
        split: str = "train",
    ) -> OODDetector:
        """
        Fit the reference distribution from a feature matrix.

        Parameters
        ----------
        feature_matrix : np.ndarray, shape (N, D)
            Feature vectors extracted from TRAINING images only.
        model_id : str
            Model identifier for provenance metadata.
        split : str
            Should always be "train". Documented for reproducibility.

        Returns
        -------
        self — for method chaining
        """
        if split != "train":
            logger.warning(
                "OODDetector.fit() called with split='%s'. "
                "Reference statistics should ONLY be fitted on training data. "
                "Using any other split violates the scientific protocol.",
                split,
            )

        self._stats = fit_reference_stats(
            feature_matrix=feature_matrix,
            lambda_reg=self.config.covariance_regularization,
            model_id=model_id,
            split=split,
        )
        self._is_fitted = True
        logger.info(
            "OODDetector fitted: N=%d, D=%d",
            self._stats.n_samples, self._stats.feature_dim,
        )
        return self

    def set_thresholds_from_validation(
        self,
        val_distances: np.ndarray,
        percentile: float | None = None,
    ) -> tuple[float, float]:
        """
        Set borderline and severe thresholds from validation distances.

        IMPORTANT: This uses VALIDATION data distances to characterize the
        in-distribution tail. The result is a provisional threshold; it does
        NOT use any test data.

        Parameters
        ----------
        val_distances : 1D array of Mahalanobis distances for validation images
        percentile    : percentile to use (default: config.threshold_percentile)

        Returns
        -------
        (borderline_threshold, severe_threshold)
        """
        p = percentile or self.config.threshold_percentile

        # borderline = chosen percentile (e.g. 95th or 99th)
        borderline = compute_percentile_threshold(val_distances, p)

        # severe = a stricter cut (e.g., 99.9th or 3σ equivalent)
        # Use min(p + (100-p)/2, 99.9) to place it conservatively above borderline
        severe_pct = min(p + (100.0 - p) / 2.0, 99.9)
        severe = compute_percentile_threshold(val_distances, severe_pct)

        self.config.mahalanobis_borderline = borderline
        self.config.mahalanobis_severe = severe

        logger.info(
            "OOD thresholds set from validation (N=%d, %.1f-pctile=%.4f, %.1f-pctile=%.4f)",
            len(val_distances), p, borderline, severe_pct, severe,
        )
        return borderline, severe

    def save(
        self,
        output_dir: Path,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Save fitted statistics to disk."""
        self._require_fitted()
        meta = {
            "threshold_method": self.config.threshold_method,
            "threshold_percentile": self.config.threshold_percentile,
            "mahalanobis_borderline": self.config.mahalanobis_borderline,
            "mahalanobis_severe": self.config.mahalanobis_severe,
            "energy_enabled": self.config.energy_enabled,
            "energy_temperature": self.config.energy_temperature,
            "feature_layer": self.config.feature_layer,
        }
        if extra_metadata:
            meta.update(extra_metadata)
        return save_stats(self._stats, output_dir, extra_metadata=meta)  # type: ignore[arg-type]

    def load_stats(self, stats_dir: Path) -> OODDetector:
        """Load pre-fitted statistics from disk."""
        self._stats = load_stats(stats_dir)

        # Also try to load thresholds from metadata
        meta_path = Path(stats_dir) / "metadata.json"
        if meta_path.exists():
            import json
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("mahalanobis_borderline") is not None:
                self.config.mahalanobis_borderline = float(meta["mahalanobis_borderline"])
            if meta.get("mahalanobis_severe") is not None:
                self.config.mahalanobis_severe = float(meta["mahalanobis_severe"])
            if meta.get("threshold_percentile") is not None:
                self.config.threshold_percentile = float(meta["threshold_percentile"])

        self._is_fitted = True
        return self

    # ── Scoring ──────────────────────────────────────────────────────────────

    def score(
        self,
        feature_vector: np.ndarray | torch.Tensor,
    ) -> dict[str, float | None]:
        """
        Return raw distance scores for a feature vector (no level/threshold).

        Returns
        -------
        dict with:
            mahalanobis_distance : float
            energy_score         : float | None
        """
        self._require_fitted()
        dist = mahalanobis_distance(feature_vector, self._stats)  # type: ignore[arg-type]
        return {"mahalanobis_distance": dist, "energy_score": None}

    def predict(
        self,
        feature_vector: np.ndarray | torch.Tensor,
        raw_probs: np.ndarray | torch.Tensor | None = None,
    ) -> OODPrediction:
        """
        Score a feature vector and return an OODPrediction with level assignment.

        Parameters
        ----------
        feature_vector : (D,) or (1, D) feature vector from the Base Model
        raw_probs      : (optional) model sigmoid probabilities for energy score

        Returns
        -------
        OODPrediction
        """
        self._require_fitted()
        t_start = time.perf_counter()

        # ── Primary: Mahalanobis distance ─────────────────────────────────
        dist = mahalanobis_distance(feature_vector, self._stats)  # type: ignore[arg-type]

        # ── Secondary: Energy score (optional) ────────────────────────────
        energy: float | None = None
        energy_is_approx = True
        if self.config.energy_enabled and raw_probs is not None:
            energy = energy_score_safe(
                raw_probs, temperature=self.config.energy_temperature, is_logits=False
            )

        # ── Level assignment ───────────────────────────────────────────────
        borderline_thr = self._get_borderline_threshold()
        severe_thr = self.config.mahalanobis_severe

        if dist <= borderline_thr:
            level = "in_distribution"
            is_ood = False
        elif severe_thr is not None and dist > severe_thr:
            level = "severe"
            is_ood = True
        else:
            level = "borderline"
            is_ood = True

        # ── Detector agreement ─────────────────────────────────────────────
        detectors_agree: bool | None = None
        if energy is not None:
            # Energy: less negative = more OOD. No absolute threshold defined yet.
            # Agreement: check directional consistency only.
            # (A proper threshold for energy would require proper calibration.)
            detectors_agree = is_ood == (energy > -10.0)  # rough heuristic

        # ── Reasoning ─────────────────────────────────────────────────────
        reasoning = _build_reasoning(
            level=level,
            distance=dist,
            borderline_threshold=borderline_thr,
            severe_threshold=severe_thr,
            energy_score=energy,
            energy_is_approx=energy_is_approx,
        )

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return OODPrediction(
            is_ood=is_ood,
            level=level,
            mahalanobis_distance=dist,
            mahalanobis_threshold=borderline_thr,
            method="mahalanobis",
            energy_score=energy,
            energy_is_approx=energy_is_approx,
            detectors_agree=detectors_agree,
            reasoning=reasoning,
            latency_ms=latency_ms,
        )

    def analyze(
        self,
        feature_vector: np.ndarray | torch.Tensor,
        image_id: str = "",
        raw_probs: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Full analysis returning a serializable dict (for logging/audit).

        Parameters
        ----------
        feature_vector : Base Model feature vector
        image_id       : optional image identifier for logging
        raw_probs      : optional model sigmoid probabilities for energy score

        Returns
        -------
        dict with all OOD fields plus image metadata
        """
        prediction = self.predict(feature_vector, raw_probs=raw_probs)

        return {
            "image_id": image_id,
            "is_ood": prediction.is_ood,
            "level": prediction.level,
            "mahalanobis_distance": round(prediction.mahalanobis_distance, 6),
            "mahalanobis_threshold": round(prediction.mahalanobis_threshold, 6),
            "method": prediction.method,
            "energy_score": (
                round(prediction.energy_score, 6)
                if prediction.energy_score is not None else None
            ),
            "energy_is_approx": prediction.energy_is_approx,
            "detectors_agree": prediction.detectors_agree,
            "reasoning": prediction.reasoning,
            "latency_ms": round(prediction.latency_ms, 3),
            "feature_layer": self.config.feature_layer,
            "feature_dim": self.config.feature_dim,
            "n_reference_samples": (
                self._stats.n_samples if self._stats else None
            ),
            "disclaimer": (
                "Research prototype. OOD level is a distribution-shift signal, "
                "not a clinical diagnosis."
            ),
        }

    def batch_score(
        self,
        feature_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Mahalanobis distances for a batch of feature vectors.

        Parameters
        ----------
        feature_matrix : (N, D)

        Returns
        -------
        np.ndarray (N,) — distances
        """
        self._require_fitted()
        return batch_mahalanobis_distances(feature_matrix, self._stats)  # type: ignore[arg-type]

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def stats(self) -> OODReferenceStats | None:
        return self._stats

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ── Internal ─────────────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if not self._is_fitted or self._stats is None:
            raise RuntimeError(
                "OODDetector has not been fitted. "
                "Call fit() with training features, or load_stats() from disk."
            )

    def _get_borderline_threshold(self) -> float:
        if self.config.mahalanobis_borderline is not None:
            return self.config.mahalanobis_borderline
        # Default fallback — should not normally be reached after fitting
        logger.warning(
            "mahalanobis_borderline not set. Using large default (1e9). "
            "Run fit_ood_reference.py to compute a proper threshold."
        )
        return 1e9
