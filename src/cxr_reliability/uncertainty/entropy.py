"""Predictive entropy calculation — Phase 6.

Responsibility
--------------
Compute binary predictive entropy and normalized entropy with numerical safeguards.

Mathematical Definition
-----------------------
For a binary probability p in [0, 1]:
    H(p) = -p * ln(p) - (1 - p) * ln(1 - p)   [in nats]

Normalized Entropy:
    H_norm(p) = H(p) / ln(2)                   [in [0, 1]]

Properties:
- Peak: at p = 0.5, H(0.5) = ln(2) ≈ 0.693147 nats, H_norm(0.5) = 1.0.
- Minima: as p -> 0 or p -> 1, H(p) -> 0, H_norm(p) -> 0.
- Symmetry: H(p) = H(1 - p).

Numerical Safeguards:
- p is clamped to [epsilon, 1 - epsilon] where epsilon > 0 (default: 1e-8)
  to avoid ln(0) = -inf and NaN products.
- Exact boundary inputs (p == 0.0 or p == 1.0) explicitly evaluate to 0.0.
"""

from __future__ import annotations

import math

import numpy as np
import torch

_LN2 = math.log(2.0)


def compute_binary_entropy(
    p: float | np.ndarray | torch.Tensor,
    epsilon: float = 1e-8,
) -> float | np.ndarray:
    """
    Compute binary predictive entropy in nats: H(p) = -p*ln(p) - (1-p)*ln(1-p).

    Parameters
    ----------
    p : float, np.ndarray, or torch.Tensor
        Probability score(s) in [0.0, 1.0].
    epsilon : float, optional
        Small positive constant for clamping to avoid ln(0). Default 1e-8.

    Returns
    -------
    float or np.ndarray
        Predictive entropy in nats in [0.0, ln(2)].

    Raises
    ------
    ValueError
        If p contains NaN, Inf, or values outside [0.0, 1.0], or has
        unsupported dimensionality (> 1D), or epsilon <= 0.
    """
    if epsilon <= 0.0 or epsilon >= 0.5:
        raise ValueError(f"epsilon must be in (0, 0.5), got {epsilon}")

    # ── PyTorch Tensor ────────────────────────────────────────────────────────
    if isinstance(p, torch.Tensor):
        if torch.isnan(p).any():
            raise ValueError("Input probability tensor contains NaN.")
        if torch.isinf(p).any():
            raise ValueError("Input probability tensor contains Inf.")
        if p.numel() == 1:
            val = float(p.item())
            return _compute_scalar_entropy(val, epsilon)
        if p.ndim > 1:
            raise ValueError(f"Input tensor must be 0D or 1D, got shape {tuple(p.shape)}")
        arr = p.detach().cpu().numpy()
        return _compute_array_entropy(arr, epsilon)

    # ── NumPy array ───────────────────────────────────────────────────────────
    if isinstance(p, np.ndarray):
        if np.isnan(p).any():
            raise ValueError("Input probability array contains NaN.")
        if np.isinf(p).any():
            raise ValueError("Input probability array contains Inf.")
        if p.ndim == 0:
            return _compute_scalar_entropy(float(p), epsilon)
        if p.ndim > 1:
            raise ValueError(f"Input array must be 0D or 1D, got shape {p.shape}")
        return _compute_array_entropy(p, epsilon)

    # ── Scalar float/int ──────────────────────────────────────────────────────
    if not isinstance(p, (int, float)):
        raise TypeError(f"Expected float, np.ndarray, or torch.Tensor, got {type(p).__name__}")

    val = float(p)
    if np.isnan(val):
        raise ValueError("Input probability is NaN.")
    if np.isinf(val):
        raise ValueError("Input probability is Inf.")

    return _compute_scalar_entropy(val, epsilon)


def compute_normalized_entropy(
    p: float | np.ndarray | torch.Tensor,
    epsilon: float = 1e-8,
) -> float | np.ndarray:
    """
    Compute normalized binary predictive entropy in [0.0, 1.0].

    H_norm(p) = H(p) / ln(2)
    """
    raw_h = compute_binary_entropy(p, epsilon=epsilon)
    if isinstance(raw_h, np.ndarray):
        norm_h = raw_h / _LN2
        return np.clip(norm_h, 0.0, 1.0).astype(np.float64)
    norm_val = raw_h / _LN2
    return float(max(0.0, min(1.0, norm_val)))


def _compute_scalar_entropy(val: float, epsilon: float) -> float:
    if val < 0.0 or val > 1.0:
        raise ValueError(f"Probability must be in [0.0, 1.0], got {val}")
    if val == 0.0 or val == 1.0:
        return 0.0
    p_clamped = max(epsilon, min(1.0 - epsilon, val))
    q_clamped = 1.0 - p_clamped
    h = -p_clamped * math.log(p_clamped) - q_clamped * math.log(q_clamped)
    return float(max(0.0, h))


def _compute_array_entropy(arr: np.ndarray, epsilon: float) -> np.ndarray:
    if (arr < 0.0).any() or (arr > 1.0).any():
        raise ValueError("All probabilities must be in [0.0, 1.0].")
    # Clamp values
    p_clamped = np.clip(arr, epsilon, 1.0 - epsilon)
    q_clamped = 1.0 - p_clamped
    h = -p_clamped * np.log(p_clamped) - q_clamped * np.log(q_clamped)
    # Exact 0 and 1 evaluate to 0.0
    h[arr == 0.0] = 0.0
    h[arr == 1.0] = 0.0
    return np.maximum(0.0, h).astype(np.float64)
