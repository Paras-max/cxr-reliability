"""In-distribution reference statistics fitting and persistence — Phase 5.

Responsibility
--------------
Fit a multivariate Gaussian reference distribution from training feature
vectors extracted by the Base Model. Stores the sufficient statistics
(mean vector and regularized covariance matrix) and their Cholesky
decomposition for use by the Mahalanobis distance computation.

Numerical Approach
------------------
Direct matrix inversion of the covariance matrix (np.linalg.inv) is
numerically unstable when:
    - The feature dimension (1024) is large relative to the number of
      samples used for local batches.
    - The covariance matrix is nearly singular (features are correlated).

Instead we use:
    1. Tikhonov regularization:   Σ_reg = Σ + λ·I
       where λ (covariance_regularization) is configurable.
    2. Cholesky decomposition of Σ_reg.
    3. For Mahalanobis computation: solve the triangular system via
       scipy.linalg.cho_solve rather than computing Σ_reg⁻¹ explicitly.

This approach is standard in the Mahalanobis-distance OOD literature
(Lee et al., 2018, "A Simple Unified Framework for Detecting OOD Samples").

Scientific Constraints
----------------------
- Fitted ONLY on training data. Validation and test data are never used
  to fit reference statistics.
- No fabricated accuracy claims.
- NaN/Inf inputs cause a clear RuntimeError.

Implementation phase: P5
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg

logger = logging.getLogger(__name__)

# Default regularization strength — configurable per Phase 16 spec.
DEFAULT_LAMBDA_REG: float = 1e-5


@dataclass
class OODReferenceStats:
    """
    Fitted in-distribution reference statistics.

    Fields
    ------
    mean_vector         : shape (D,) — sample mean of training features
    covariance_matrix   : shape (D, D) — unregularized sample covariance
    regularized_cov     : shape (D, D) — Σ + λI (used for distance computation)
    cholesky_factor     : shape (D, D) — lower-triangular L s.t. LL^T = Σ_reg
                          Used for numerically stable Mahalanobis computation.
    n_samples           : number of training images used to fit
    feature_dim         : D (expected 1024)
    lambda_reg          : the regularization constant λ used
    fit_timestamp_utc   : ISO-8601 UTC timestamp of fitting
    model_id            : base model identifier
    split               : dataset split used for fitting (should always be "train")
    """

    mean_vector: np.ndarray           # (D,)
    covariance_matrix: np.ndarray     # (D, D)
    regularized_cov: np.ndarray       # (D, D)
    cholesky_factor: np.ndarray       # (D, D) lower-triangular
    n_samples: int
    feature_dim: int
    lambda_reg: float
    fit_timestamp_utc: str = ""
    model_id: str = "densenet121-res224-nih"
    split: str = "train"

    def validate(self) -> None:
        """
        Raise ValueError if any statistic is NaN, Inf, or dimensionally
        inconsistent. Called automatically by fit_reference_stats().
        """
        D = self.feature_dim

        if self.mean_vector.shape != (D,):
            raise ValueError(
                f"mean_vector shape {self.mean_vector.shape} != ({D},)"
            )
        if self.covariance_matrix.shape != (D, D):
            raise ValueError(
                f"covariance_matrix shape {self.covariance_matrix.shape} != ({D}, {D})"
            )
        if self.regularized_cov.shape != (D, D):
            raise ValueError(
                f"regularized_cov shape {self.regularized_cov.shape} != ({D}, {D})"
            )
        if self.cholesky_factor.shape != (D, D):
            raise ValueError(
                f"cholesky_factor shape {self.cholesky_factor.shape} != ({D}, {D})"
            )

        for name, arr in [
            ("mean_vector", self.mean_vector),
            ("covariance_matrix", self.covariance_matrix),
            ("regularized_cov", self.regularized_cov),
            ("cholesky_factor", self.cholesky_factor),
        ]:
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"OOD reference statistics contain NaN or Inf in '{name}'. "
                    "This indicates a numerical failure during fitting."
                )

        if self.n_samples < D:
            logger.warning(
                "Number of reference samples (%d) < feature dimension (%d). "
                "The sample covariance will be rank-deficient. "
                "Regularization (λ=%.2e) is essential in this case.",
                self.n_samples, D, self.lambda_reg,
            )


def fit_reference_stats(
    feature_matrix: np.ndarray,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    model_id: str = "densenet121-res224-nih",
    split: str = "train",
) -> OODReferenceStats:
    """
    Fit a multivariate Gaussian reference distribution from training features.

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (N, D)
        Feature vectors from N training images. D should be 1024.
        Must not contain NaN or Inf.
    lambda_reg : float
        Tikhonov regularization constant λ added to the diagonal of the
        sample covariance: Σ_reg = Σ + λ·I.
        Must be > 0. Default: 1e-5 (configurable).
    model_id : str
        Identifier of the base model (for metadata provenance).
    split : str
        Dataset split used (must be "train" for correct scientific practice).

    Returns
    -------
    OODReferenceStats — validated fitted statistics

    Raises
    ------
    ValueError
        - feature_matrix is not 2D
        - feature_matrix contains NaN or Inf
        - lambda_reg <= 0
        - Regularized covariance is not positive definite (Cholesky fails)
    RuntimeError
        - Cholesky decomposition fails despite regularization (degenerate data)
    """
    if feature_matrix.ndim != 2:
        raise ValueError(
            f"feature_matrix must be 2D (N, D), got shape {feature_matrix.shape}"
        )

    n_samples, feature_dim = feature_matrix.shape
    logger.info(
        "Fitting OOD reference stats: N=%d, D=%d, λ=%.2e, split='%s'",
        n_samples, feature_dim, lambda_reg, split,
    )

    if not np.all(np.isfinite(feature_matrix)):
        n_bad = int(np.sum(~np.isfinite(feature_matrix)))
        raise ValueError(
            f"feature_matrix contains {n_bad} NaN/Inf value(s). "
            "Cannot fit reference statistics on invalid data."
        )

    if lambda_reg <= 0:
        raise ValueError(
            f"lambda_reg must be > 0, got {lambda_reg}. "
            "A positive regularization constant is required for numerical stability."
        )

    # ── Sample mean ──────────────────────────────────────────────────────────
    mean_vector: np.ndarray = np.mean(feature_matrix, axis=0)  # (D,)

    # ── Sample covariance (unbiased, ddof=1) ─────────────────────────────────
    # rowvar=False: each column is a variable, each row is an observation
    cov_matrix: np.ndarray = np.cov(feature_matrix.T, ddof=1)  # (D, D)
    # For N=1 or D=1 edge cases, np.cov may return a scalar
    cov_matrix = np.atleast_2d(cov_matrix)

    if not np.all(np.isfinite(cov_matrix)):
        raise ValueError(
            "Sample covariance matrix contains NaN/Inf after computation. "
            "Check feature_matrix for degenerate (constant) features."
        )

    # ── Tikhonov regularization: Σ_reg = Σ + λ·I ────────────────────────────
    reg_cov: np.ndarray = cov_matrix + lambda_reg * np.eye(feature_dim)

    if not np.all(np.isfinite(reg_cov)):
        raise ValueError(
            "Regularized covariance matrix contains NaN/Inf. "
            "This should not occur; check for extremely large feature values."
        )

    # ── Cholesky decomposition for numerical stability ───────────────────────
    # scipy.linalg.cholesky with lower=True returns L such that L @ L.T = Σ_reg.
    # This will raise LinAlgError if the matrix is not positive definite,
    # which indicates that lambda_reg is too small.
    try:
        chol_factor: np.ndarray = scipy.linalg.cholesky(reg_cov, lower=True)
    except scipy.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"Cholesky decomposition of the regularized covariance matrix "
            f"failed (not positive definite). This usually means lambda_reg "
            f"({lambda_reg:.2e}) is too small for this feature dimension ({feature_dim}) "
            f"and sample count ({n_samples}). Try increasing covariance_regularization. "
            f"Original error: {exc}"
        ) from exc

    logger.info("Cholesky decomposition succeeded. Reference stats are valid.")

    timestamp = datetime.now(timezone.utc).isoformat()

    stats = OODReferenceStats(
        mean_vector=mean_vector,
        covariance_matrix=cov_matrix,
        regularized_cov=reg_cov,
        cholesky_factor=chol_factor,
        n_samples=n_samples,
        feature_dim=feature_dim,
        lambda_reg=lambda_reg,
        fit_timestamp_utc=timestamp,
        model_id=model_id,
        split=split,
    )

    # Validate all fields for consistency and finiteness
    stats.validate()

    return stats


# ── Persistence ──────────────────────────────────────────────────────────────

def save_stats(
    stats: OODReferenceStats,
    output_dir: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """
    Save OOD reference statistics to disk.

    Saves two files:
        <output_dir>/reference_stats.npz  — numpy arrays (mean, cov, etc.)
        <output_dir>/metadata.json        — provenance and configuration

    Parameters
    ----------
    stats        : fitted OODReferenceStats
    output_dir   : directory where files will be written (created if needed)
    extra_metadata : additional key-value pairs appended to metadata.json

    Returns
    -------
    dict with keys 'stats_path' and 'metadata_path'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "reference_stats.npz"
    meta_path = output_dir / "metadata.json"

    # Save arrays
    np.savez_compressed(
        stats_path,
        mean_vector=stats.mean_vector,
        covariance_matrix=stats.covariance_matrix,
        regularized_cov=stats.regularized_cov,
        cholesky_factor=stats.cholesky_factor,
    )

    # Compute a content hash of the stats file for audit purposes
    sha = hashlib.sha256()
    with open(stats_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    stats_sha256 = sha.hexdigest()

    metadata: dict[str, Any] = {
        "format_version": "1.0",
        "feature_dim": stats.feature_dim,
        "n_reference_samples": stats.n_samples,
        "split_used": stats.split,
        "model_id": stats.model_id,
        "lambda_reg": stats.lambda_reg,
        "fit_timestamp_utc": stats.fit_timestamp_utc,
        "stats_file": "reference_stats.npz",
        "stats_sha256": stats_sha256,
        "scientific_note": (
            "Reference statistics fitted on TRAINING data only. "
            "Validation and test data were NOT used for fitting. "
            "These statistics support the OOD reliability signal "
            "and do NOT constitute a clinical diagnosis."
        ),
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Saved OOD reference stats to %s", output_dir)
    return {"stats_path": stats_path, "metadata_path": meta_path}


def load_stats(stats_dir: Path) -> OODReferenceStats:
    """
    Load OOD reference statistics previously saved by save_stats().

    Parameters
    ----------
    stats_dir : directory containing reference_stats.npz and metadata.json

    Returns
    -------
    OODReferenceStats — validated

    Raises
    ------
    FileNotFoundError : if stats_dir or required files do not exist
    RuntimeError      : if loaded stats fail validation
    """
    stats_dir = Path(stats_dir)
    stats_path = stats_dir / "reference_stats.npz"
    meta_path = stats_dir / "metadata.json"

    if not stats_dir.exists():
        raise FileNotFoundError(
            f"OOD stats directory not found: {stats_dir}\n"
            "Run scripts/fit_ood_reference.py to generate reference statistics."
        )
    if not stats_path.exists():
        raise FileNotFoundError(
            f"OOD stats file not found: {stats_path}\n"
            "Run scripts/fit_ood_reference.py to generate reference statistics."
        )

    data = np.load(stats_path)
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            metadata = json.load(fh)

    mean_vector = data["mean_vector"]
    feature_dim = int(mean_vector.shape[0])

    stats = OODReferenceStats(
        mean_vector=mean_vector,
        covariance_matrix=data["covariance_matrix"],
        regularized_cov=data["regularized_cov"],
        cholesky_factor=data["cholesky_factor"],
        n_samples=int(metadata.get("n_reference_samples", 0)),
        feature_dim=feature_dim,
        lambda_reg=float(metadata.get("lambda_reg", DEFAULT_LAMBDA_REG)),
        fit_timestamp_utc=metadata.get("fit_timestamp_utc", ""),
        model_id=metadata.get("model_id", ""),
        split=metadata.get("split_used", ""),
    )

    stats.validate()
    logger.info(
        "Loaded OOD reference stats: N=%d, D=%d, model=%s, split=%s",
        stats.n_samples, stats.feature_dim, stats.model_id, stats.split,
    )
    return stats


def compute_percentile_threshold(
    distances: np.ndarray,
    percentile: float,
) -> float:
    """
    Compute a distance threshold at the given percentile of a distance array.

    IMPORTANT: This function is intended to be called on VALIDATION distances
    only (never on test distances). Validation distances characterize the
    in-distribution tail; the resulting threshold is a provisional operating
    point, not a scientifically calibrated value.

    Parameters
    ----------
    distances  : 1D array of Mahalanobis distances
    percentile : float in (0, 100] — e.g. 95.0, 97.5, 99.0

    Returns
    -------
    float — the threshold value at the given percentile
    """
    if distances.ndim != 1 or len(distances) == 0:
        raise ValueError(
            f"distances must be a non-empty 1D array, got shape {distances.shape}"
        )
    if not (0 < percentile <= 100):
        raise ValueError(
            f"percentile must be in (0, 100], got {percentile}"
        )
    return float(np.percentile(distances, percentile))
