"""OOD statistics fitting and threshold calibration — Phase 5.

Responsibility
--------------
    Fit the in-distribution feature mean and (regularized) covariance on
    the training split ONLY, save them as an artifact, then derive the
    borderline/severe distance thresholds from percentiles on the validation
    split.

    IMPORTANT Scientific Constraints:
    - Reference statistics are fitted on TRAINING data only.
    - Thresholds are selected from VALIDATION data only.
    - TEST data is NEVER used for fitting or threshold selection.
    - NIH validation images are in-distribution NIH data, NOT true OOD.

Input
-----
    Feature vectors from the fast model on the training split;
    validation distances for threshold calibration.

Output
------
    Statistics artifact (npz + metadata.json) and calibrated OODThresholds.

Dependencies
------------
    numpy, scipy, cxr_reliability.ood (statistics, detector)

Implementation phase: P5
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from cxr_reliability.config.thresholds import OODThresholds
from cxr_reliability.ood.statistics import (
    OODReferenceStats,
    compute_percentile_threshold,
    fit_reference_stats,
    save_stats,
)

logger = logging.getLogger(__name__)


def fit_gaussian_stats(
    features: np.ndarray,
    output_path: Path,
    lambda_reg: float = 1e-5,
    model_id: str = "densenet121-res224-nih",
    split: str = "train",
    extra_metadata: dict | None = None,
) -> OODReferenceStats:
    """
    Fit a multivariate Gaussian reference distribution and save artifacts.

    SCIENTIFIC CONSTRAINT: features must come from the TRAINING split only.

    Parameters
    ----------
    features     : np.ndarray shape (N, D) — training feature matrix
    output_path  : directory where reference_stats.npz + metadata.json are saved
    lambda_reg   : Tikhonov regularization constant (must be > 0)
    model_id     : base model identifier for provenance
    split        : should always be "train"; logged for reproducibility
    extra_metadata: additional fields to include in metadata.json

    Returns
    -------
    OODReferenceStats — validated fitted statistics

    Raises
    ------
    ValueError  : invalid input (NaN, wrong shape, etc.)
    RuntimeError: Cholesky decomposition fails (lambda_reg too small)
    """
    if split != "train":
        logger.warning(
            "fit_gaussian_stats() called with split='%s'. "
            "Reference statistics MUST be fitted on training data only. "
            "Proceeding, but this violates the scientific protocol.",
            split,
        )

    logger.info(
        "Fitting OOD Gaussian reference stats: split=%s, N=%d, D=%d, λ=%.2e",
        split, features.shape[0], features.shape[1], lambda_reg,
    )

    stats = fit_reference_stats(
        feature_matrix=features,
        lambda_reg=lambda_reg,
        model_id=model_id,
        split=split,
    )

    output_path = Path(output_path)
    save_stats(stats, output_dir=output_path, extra_metadata=extra_metadata)

    logger.info(
        "OOD reference stats saved to %s (N=%d, D=%d)",
        output_path, stats.n_samples, stats.feature_dim,
    )

    return stats


def calibrate_ood_thresholds(
    id_val_scores: np.ndarray,
    percentile: float = 99.0,
) -> OODThresholds:
    """
    Derive OOD thresholds from in-distribution validation distances.

    IMPORTANT: This function uses VALIDATION distances (in-distribution NIH
    images) to characterize the tail of the in-distribution distance
    distribution. The resulting thresholds are PROVISIONAL operating points.

    NIH validation images are NOT a true OOD dataset. This function does NOT
    compute OOD AUROC or claim the thresholds are scientifically calibrated.

    Parameters
    ----------
    id_val_scores : np.ndarray (N,) — Mahalanobis distances for NIH validation
                    images. These are in-distribution samples.
    percentile    : float in (0, 100] — e.g. 95.0 or 99.0
                    The borderline threshold = percentile of validation distances.
                    Default 99.0 (conservative).

    Returns
    -------
    OODThresholds with mahalanobis_borderline and mahalanobis_severe set.
    The returned thresholds should be saved to a YAML file; they do NOT
    modify the v0_prd_defaults.yaml.

    Scientific Note
    ---------------
    Using the 99th percentile of in-distribution distances means that
    approximately 1% of in-distribution images will be flagged as OOD
    under ideal conditions. This false-positive rate is a design choice,
    not a measurement of detector quality.
    """
    if id_val_scores.ndim != 1 or len(id_val_scores) == 0:
        raise ValueError(
            f"id_val_scores must be a non-empty 1D array, got shape {id_val_scores.shape}"
        )

    logger.info(
        "Calibrating OOD thresholds from %d validation distances "
        "(these are IN-DISTRIBUTION NIH validation images — NOT true OOD).",
        len(id_val_scores),
    )

    # Borderline: chosen percentile
    borderline = compute_percentile_threshold(id_val_scores, percentile)

    # Severe: halfway between borderline percentile and 99.9th percentile
    severe_pct = min(percentile + (100.0 - percentile) / 2.0, 99.9)
    severe = compute_percentile_threshold(id_val_scores, severe_pct)

    logger.info(
        "Validation distance statistics: "
        "min=%.4f max=%.4f mean=%.4f median=%.4f std=%.4f",
        float(np.min(id_val_scores)),
        float(np.max(id_val_scores)),
        float(np.mean(id_val_scores)),
        float(np.median(id_val_scores)),
        float(np.std(id_val_scores)),
    )
    logger.info(
        "Threshold selection: %.1f-pctile=%.4f (borderline), "
        "%.1f-pctile=%.4f (severe)",
        percentile, borderline, severe_pct, severe,
    )

    thresholds = OODThresholds(
        in_distribution_percentile=percentile,
        mahalanobis_borderline=borderline,
        mahalanobis_severe=severe,
        energy_borderline=None,   # energy thresholds not calibrated
        energy_severe=None,
    )

    return thresholds
