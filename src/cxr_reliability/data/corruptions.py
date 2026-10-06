"""Synthetic corruption library with known severity (Phase 3).

Responsibility:
    Generate seeded blur, noise, exposure and combined degradations at graded severities.
    Provides the ground truth needed to calibrate Quality thresholds (ROC), Repair bounds
    and the Verification margin, since NIH has no natural defect labels.

Input:
    Clean image array, defect type, severity, seed.

Output:
    Corrupted image array plus a record of the exact parameters applied.

Dependencies:
    numpy, cv2 / scipy.ndimage
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_blur(image: np.ndarray, severity: float, seed: int = 42) -> np.ndarray:
    """
    Apply Gaussian blur with monotonic severity in [0.0, 1.0].
    Higher severity produces larger blur (lower Laplacian variance).
    """
    if severity <= 0.0:
        return image.copy()

    # Map severity 0..1 to odd kernel size 3..31 and sigma 1..15
    ksize = int(3 + np.clip(severity, 0.0, 1.0) * 28)
    if ksize % 2 == 0:
        ksize += 1
    sigma = 0.5 + severity * 10.0

    blurred = cv2.GaussianBlur(image.astype(np.float32), (ksize, ksize), sigmaX=sigma)
    return np.clip(blurred, 0, 255).astype(image.dtype)


def apply_noise(image: np.ndarray, severity: float, seed: int = 42) -> np.ndarray:
    """
    Add zero-mean Gaussian noise with standard deviation proportional to severity in [0.0, 1.0].
    Higher severity produces lower SNR (dB).
    """
    if severity <= 0.0:
        return image.copy()

    rng = np.random.RandomState(seed)
    std = severity * 50.0  # Max std ~50 on [0, 255] scale
    noise = rng.normal(0, std, size=image.shape).astype(np.float32)

    corrupted = image.astype(np.float32) + noise
    return np.clip(corrupted, 0, 255).astype(image.dtype)


def apply_exposure_shift(image: np.ndarray, severity: float, seed: int = 42) -> np.ndarray:
    """
    Shift image exposure (underexposure if negative, overexposure if positive).
    """
    if severity == 0.0:
        return image.copy()

    shift = severity * 100.0  # Shift up to +/- 100 intensity values
    shifted = image.astype(np.float32) + shift
    return np.clip(shifted, 0, 255).astype(image.dtype)


def apply_combined(image: np.ndarray, severities: dict[str, float], seed: int = 42) -> np.ndarray:
    """Apply sequential blur, noise, and exposure degradations."""
    out = image.copy()
    if "blur" in severities and severities["blur"] > 0:
        out = apply_blur(out, severities["blur"], seed=seed)
    if "noise" in severities and severities["noise"] > 0:
        out = apply_noise(out, severities["noise"], seed=seed + 1)
    if "exposure" in severities and severities["exposure"] != 0:
        out = apply_exposure_shift(out, severities["exposure"], seed=seed + 2)
    return out
