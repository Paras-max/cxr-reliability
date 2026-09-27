"""Confidence estimation — Phase 6.

Responsibility
--------------
Compute model prediction confidence from binary/multi-label sigmoid output.

Mathematical Definition
-----------------------
For a binary classification task or a target pathology score p in [0, 1]:
    confidence(p) = max(p, 1 - p)

Range:
    confidence in [0.5, 1.0]

Distinction of Concepts (Scientific Safety)
-------------------------------------------
1. Raw Model Score:
   The raw continuous output from the model (for TorchXRayVision, this is a
   sigmoid probability at index 8 for Pneumonia).
2. Derived Binary Confidence:
   max(p, 1 - p). Represents the model's certainty in its preferred binary
   direction (either Pneumonia if p >= 0.5 or Non-Pneumonia if p < 0.5).
3. Calibrated Probability:
   A probability whose empirical frequency matches the true underlying event
   rate. Post-hoc calibration (Phase 4.5) is deferred. The derived confidence
   is NOT claimed to be clinically calibrated.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_binary_confidence(
    p: float | np.ndarray | torch.Tensor,
) -> float | np.ndarray:
    """
    Compute binary confidence as max(p, 1 - p).

    Parameters
    ----------
    p : float, np.ndarray, or torch.Tensor
        Probability score(s) in [0.0, 1.0].

    Returns
    -------
    float or np.ndarray
        Confidence in [0.5, 1.0].

    Raises
    ------
    ValueError
        If p contains NaN, Inf, or values strictly outside [0.0, 1.0], or has
        unsupported dimensionality (greater than 1D).
    """
    # ── Convert PyTorch Tensor if provided ────────────────────────────────────
    if isinstance(p, torch.Tensor):
        if torch.isnan(p).any():
            raise ValueError("Input probability tensor contains NaN.")
        if torch.isinf(p).any():
            raise ValueError("Input probability tensor contains Inf.")
        if p.numel() == 1:
            val = float(p.item())
            return _compute_scalar_confidence(val)
        if p.ndim > 1:
            raise ValueError(f"Input tensor must be 0D or 1D, got shape {tuple(p.shape)}")
        arr = p.detach().cpu().numpy()
        return _compute_array_confidence(arr)

    # ── Convert NumPy array if provided ───────────────────────────────────────
    if isinstance(p, np.ndarray):
        if np.isnan(p).any():
            raise ValueError("Input probability array contains NaN.")
        if np.isinf(p).any():
            raise ValueError("Input probability array contains Inf.")
        if p.ndim == 0:
            return _compute_scalar_confidence(float(p))
        if p.ndim > 1:
            raise ValueError(f"Input array must be 0D or 1D, got shape {p.shape}")
        return _compute_array_confidence(p)

    # ── Scalar float/int ──────────────────────────────────────────────────────
    if not isinstance(p, (int, float)):
        raise TypeError(f"Expected float, np.ndarray, or torch.Tensor, got {type(p).__name__}")

    val = float(p)
    if np.isnan(val):
        raise ValueError("Input probability is NaN.")
    if np.isinf(val):
        raise ValueError("Input probability is Inf.")

    return _compute_scalar_confidence(val)


def _compute_scalar_confidence(val: float) -> float:
    if val < 0.0 or val > 1.0:
        raise ValueError(f"Probability must be in [0.0, 1.0], got {val}")
    return float(max(val, 1.0 - val))


def _compute_array_confidence(arr: np.ndarray) -> np.ndarray:
    if (arr < 0.0).any() or (arr > 1.0).any():
        raise ValueError("All probabilities must be in [0.0, 1.0].")
    return np.maximum(arr, 1.0 - arr).astype(np.float64)
