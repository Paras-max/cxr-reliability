"""Blur repair module (PRD FR-6) — Phase 9.

Responsibility:
    Apply conservative sharpening of existing image structures using unsharp masking
    with conservative parameters intended to limit excessive sharpening artifacts.

    IMPORTANT SAFETY & TECHNICAL LIMITATION:
    Unsharp masking cannot reconstruct information lost through blur, cannot recover
    missing anatomical details, and does not perform generative deblurring. It applies
    a high-frequency contrast enhancement to existing boundaries.

Implementation phase: P9
"""

from __future__ import annotations

import numpy as np
from skimage.filters import unsharp_mask


def repair_blur(
    image: np.ndarray,
    radius: float = 1.0,
    amount: float = 0.5,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """
    Apply conservative unsharp masking to a 2D uint8 image.

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image array with shape (H, W).
    radius : float
        Radius of Gaussian blur used for unsharp mask (default 1.0).
    amount : float
        Sharpening strength factor (default 0.5).

    Returns
    -------
    tuple[np.ndarray, dict[str, float | int | str]]
        Repaired uint8 image array and dict of recorded operation parameters.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"repair_blur expects uint8 image, got {image.dtype}")
    if image.ndim != 2:
        raise ValueError(f"repair_blur expects 2D array, got shape {image.shape}")

    # Scale to [0.0, 1.0] float for skimage unsharp_mask
    img_f = image.astype(np.float32) / 255.0

    sharpened = unsharp_mask(
        img_f,
        radius=float(radius),
        amount=float(amount),
        preserve_range=False,
    )

    # Rescale and clip strictly to [0, 255] uint8
    repaired = np.clip(np.round(sharpened * 255.0), 0.0, 255.0).astype(np.uint8)

    params: dict[str, float | int | str] = {
        "method": "unsharp_mask",
        "radius": float(radius),
        "amount": float(amount),
    }
    return repaired, params
