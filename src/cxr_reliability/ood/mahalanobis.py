"""Mahalanobis distance computation — Phase 5.

Responsibility
--------------
Compute the Mahalanobis distance between a feature vector and the fitted
in-distribution reference distribution.

Mathematical Definition
-----------------------
    d(x) = sqrt( (x - μ)ᵀ · Σ⁻¹ · (x - μ) )

where:
    x  = query feature vector, shape (D,)
    μ  = reference mean vector, shape (D,)
    Σ  = regularized covariance matrix (Σ + λI), shape (D, D)

Numerical Implementation
------------------------
Direct computation via np.linalg.inv(Σ) is avoided because:
    - Matrix inversion amplifies numerical errors in near-singular matrices.
    - The 1024×1024 covariance matrix is likely highly correlated.

Instead, using the Cholesky factor L (lower triangular, L·Lᵀ = Σ):
    1. Solve the triangular system:  L·z = (x - μ)     [forward substitution]
    2. d(x) = sqrt( zᵀ·z ) = sqrt( (x-μ)ᵀ L⁻ᵀ L⁻¹ (x-μ) )

This is equivalent to the true Mahalanobis distance but avoids explicit
matrix inversion. scipy.linalg.solve_triangular is used for this.

This approach is numerically stable and well-established in the OOD
detection literature (Lee et al., 2018).

Scientific Constraints
----------------------
- Returns a non-negative float.
- Validates input dimension, dtype, and finite-ness.
- Does NOT interpret the distance as a probability or diagnosis.

Implementation phase: P5
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.linalg
import torch

from cxr_reliability.ood.statistics import OODReferenceStats

logger = logging.getLogger(__name__)


def _to_numpy_1d(feature_vector: np.ndarray | torch.Tensor, expected_dim: int) -> np.ndarray:
    """
    Convert a feature vector to a validated float64 numpy 1D array.

    Parameters
    ----------
    feature_vector : numpy array or torch Tensor of shape (D,) or (1, D)
    expected_dim   : expected feature dimension D

    Returns
    -------
    numpy array of shape (D,), dtype float64

    Raises
    ------
    TypeError  : unsupported input type
    ValueError : wrong shape, non-finite values
    """
    if isinstance(feature_vector, torch.Tensor):
        vec = feature_vector.detach().cpu().numpy()
    elif isinstance(feature_vector, np.ndarray):
        vec = feature_vector
    else:
        raise TypeError(
            f"feature_vector must be np.ndarray or torch.Tensor, "
            f"got {type(feature_vector)}"
        )

    vec = np.squeeze(vec)  # Handle (1, D) → (D,)
    vec = vec.astype(np.float64)

    if vec.ndim != 1:
        raise ValueError(
            f"feature_vector must be 1D after squeeze, got shape {vec.shape}. "
            "Expected a single feature vector of shape (D,) or (1, D)."
        )

    if vec.shape[0] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: expected {expected_dim}, "
            f"got {vec.shape[0]}. "
            "Ensure the feature vector comes from the same model and layer "
            "that was used to fit the reference statistics."
        )

    if not np.all(np.isfinite(vec)):
        n_bad = int(np.sum(~np.isfinite(vec)))
        raise ValueError(
            f"feature_vector contains {n_bad} NaN/Inf value(s). "
            "Cannot compute Mahalanobis distance on invalid features."
        )

    return vec


def mahalanobis_distance(
    feature_vector: np.ndarray | torch.Tensor,
    stats: OODReferenceStats,
) -> float:
    """
    Compute the Mahalanobis distance from the fitted reference distribution.

    d(x) = sqrt( (x - μ)ᵀ · Σ_reg⁻¹ · (x - μ) )

    The computation is performed using the Cholesky factor stored in `stats`
    to avoid explicit matrix inversion:
        z = L⁻¹ · (x - μ)    (solve triangular system)
        d = sqrt( zᵀz )

    Parameters
    ----------
    feature_vector : np.ndarray or torch.Tensor, shape (D,) or (1, D)
        Feature vector for a single image.
    stats : OODReferenceStats
        Fitted reference statistics (must have valid cholesky_factor).

    Returns
    -------
    float — Mahalanobis distance (non-negative)

    Raises
    ------
    TypeError  : unsupported input type
    ValueError : dimension mismatch, NaN/Inf in features
    """
    vec = _to_numpy_1d(feature_vector, expected_dim=stats.feature_dim)

    # Centred feature: (x - μ)
    delta = vec - stats.mean_vector.astype(np.float64)

    # Solve L·z = delta  (forward substitution: L is lower-triangular)
    # scipy.linalg.solve_triangular is more stable than np.linalg.solve
    z = scipy.linalg.solve_triangular(
        stats.cholesky_factor.astype(np.float64),
        delta,
        lower=True,
        check_finite=True,
    )

    # d = sqrt( zᵀz )
    distance = float(np.sqrt(np.dot(z, z)))

    if not np.isfinite(distance):
        raise RuntimeError(
            f"Mahalanobis distance computation produced a non-finite result: {distance}. "
            "This may indicate a numerical issue with the covariance matrix. "
            "Try increasing covariance_regularization (lambda_reg)."
        )

    return distance


def batch_mahalanobis_distances(
    feature_matrix: np.ndarray,
    stats: OODReferenceStats,
) -> np.ndarray:
    """
    Compute Mahalanobis distances for a batch of feature vectors.

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (N, D)
        Batch of N feature vectors.
    stats : OODReferenceStats

    Returns
    -------
    np.ndarray, shape (N,) — Mahalanobis distance for each row
    """
    if feature_matrix.ndim != 2:
        raise ValueError(
            f"feature_matrix must be 2D (N, D), got shape {feature_matrix.shape}"
        )

    n, d = feature_matrix.shape
    if d != stats.feature_dim:
        raise ValueError(
            f"Feature dimension mismatch: stats.feature_dim={stats.feature_dim}, "
            f"feature_matrix has {d} columns."
        )

    if not np.all(np.isfinite(feature_matrix)):
        n_bad = int(np.sum(~np.isfinite(feature_matrix)))
        raise ValueError(
            f"feature_matrix contains {n_bad} NaN/Inf values. "
            "Cannot compute batch Mahalanobis distances."
        )

    feat = feature_matrix.astype(np.float64)
    mu = stats.mean_vector.astype(np.float64)
    L = stats.cholesky_factor.astype(np.float64)

    # Delta matrix: (N, D)
    delta = feat - mu[np.newaxis, :]

    # Solve L·Z = deltaᵀ  → Z shape (D, N)
    # Then distances[i] = sqrt( Z[:, i] · Z[:, i] )
    Z = scipy.linalg.solve_triangular(L, delta.T, lower=True, check_finite=True)

    # ||Z[:, i]||² = sum of squares along axis 0
    distances = np.sqrt(np.sum(Z ** 2, axis=0))

    if not np.all(np.isfinite(distances)):
        n_bad = int(np.sum(~np.isfinite(distances)))
        raise RuntimeError(
            f"Batch Mahalanobis computation produced {n_bad} non-finite distance(s). "
            "Check covariance_regularization value."
        )

    return distances
