"""Repair recovery measurement (PRD section 7).

Responsibility:
    recovery_pct = (variance_after_repair - variance_blurred) / (variance_original -
    variance_blurred) x 100 on deliberately corrupted images, plus SSIM / PSNR versus the
    original and whether the base-model prediction recovers.

Input:
    Original, corrupted and repaired images with known labels.

Output:
    Recovery percentages and curves by severity.

Dependencies:
    scikit-image, numpy
"""

from __future__ import annotations

import numpy as np


def laplacian_recovery_pct(var_original: float, var_corrupted: float, var_repaired: float) -> float:
    """
    Compute recovery percentage per PRD section 7:
    (variance_after_repair - variance_blurred) / (variance_original - variance_blurred) * 100
    """
    denom = var_original - var_corrupted
    if abs(denom) < 1e-8:
        return 0.0
    return float((var_repaired - var_corrupted) / denom * 100.0)


def structural_recovery(original: np.ndarray, corrupted: np.ndarray, repaired: np.ndarray) -> dict[str, float]:
    """Compute PSNR and structural recovery metrics between original, corrupted, and repaired."""
    mse_orig_repaired = float(np.mean((original.astype(float) - repaired.astype(float)) ** 2))
    mse_orig_corrupted = float(np.mean((original.astype(float) - corrupted.astype(float)) ** 2))

    psnr_repaired = 10.0 * np.log10(255.0**2 / (mse_orig_repaired + 1e-10))
    psnr_corrupted = 10.0 * np.log10(255.0**2 / (mse_orig_corrupted + 1e-10))

    return {
        "psnr_corrupted": float(psnr_corrupted),
        "psnr_repaired": float(psnr_repaired),
        "psnr_gain": float(psnr_repaired - psnr_corrupted),
    }
