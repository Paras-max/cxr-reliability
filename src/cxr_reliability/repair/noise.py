"""Noise repair module (PRD FR-6) — Phase 9.

Responsibility:
    Apply Non-Local Means (NL-Means) denoising using conservative development defaults.
    Attenuates patch-wise stochastic noise without intentionally removing meaningful
    anatomical structures.

Implementation phase: P9
"""

from __future__ import annotations

import cv2
import numpy as np


def repair_noise(
    image: np.ndarray,
    h: float = 3.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """
    Apply Fast Non-Local Means Denoising to a 2D uint8 image.

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image array with shape (H, W).
    h : float
        Filter strength controlling denoising power. Provisional default is 3.0.
    template_window_size : int
        Size in pixels of template patch used to compute weights (must be odd, default 7).
    search_window_size : int
        Size in pixels of window used to compute weighted average (must be odd, default 21).

    Returns
    -------
    tuple[np.ndarray, dict[str, float | int | str]]
        Denoised uint8 image array and dict of recorded operation parameters.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"repair_noise expects uint8 image, got {image.dtype}")
    if image.ndim != 2:
        raise ValueError(f"repair_noise expects 2D array, got shape {image.shape}")

    # Ensure window sizes are odd
    t_size = int(template_window_size)
    if t_size % 2 == 0:
        t_size += 1
    s_size = int(search_window_size)
    if s_size % 2 == 0:
        s_size += 1

    # Apply fast Non-Local Means
    repaired = cv2.fastNlMeansDenoising(
        src=image,
        dst=None,
        h=float(h),
        templateWindowSize=t_size,
        searchWindowSize=s_size,
    )

    params: dict[str, float | int | str] = {
        "h": float(h),
        "template_window_size": t_size,
        "search_window_size": s_size,
    }
    return repaired, params
