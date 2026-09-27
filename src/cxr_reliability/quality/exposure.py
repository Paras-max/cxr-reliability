"""Exposure assessment module — Phase 7.

Responsibility
--------------
Assess image exposure using photometric intensity distribution statistics.

Mathematical Definition
-----------------------
1. Mean Intensity:
       mu_intensity = mean(I) in [0.0, 255.0]

2. Histogram Standard Deviation:
       histogram_std = std(I)
   Measures dynamic range utilization and contrast spread.

3. Saturation Fractions:
       under_exposed_fraction = mean(I < 10.0)
       over_exposed_fraction  = mean(I > 245.0)

Thresholds (PRD v0 defaults):
- exposure_mean_min: 20.0  (underexposure / too dark if mean < 20.0)
- exposure_mean_max: 235.0 (overexposure / washed out if mean > 235.0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExposureResult:
    """Component result for exposure assessment."""
    mean_intensity: float
    histogram_std: float
    under_exposed_fraction: float
    over_exposed_fraction: float
    exposure_mean_min: float
    exposure_mean_max: float
    is_underexposed: bool
    is_overexposed: bool
    is_near_threshold: bool
    status: str  # "good" | "poor" | "degraded"
    reasoning: str


def compute_exposure_statistics(image: np.ndarray) -> dict[str, float]:
    """
    Compute photometric statistics for exposure evaluation.

    Parameters
    ----------
    image : 2D np.ndarray in [0, 255]

    Returns
    -------
    dict with keys: mean_intensity, histogram_std, under_exposed_fraction, over_exposed_fraction
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

    img_f = image.astype(np.float64)
    mean_val = float(np.mean(img_f))
    std_val = float(np.std(img_f))
    under_frac = float(np.mean(img_f < 10.0))
    over_frac = float(np.mean(img_f > 245.0))

    return {
        "mean_intensity": mean_val,
        "histogram_std": std_val,
        "under_exposed_fraction": under_frac,
        "over_exposed_fraction": over_frac,
    }


def evaluate_exposure(
    image: np.ndarray,
    exposure_mean_min: float = 20.0,
    exposure_mean_max: float = 235.0,
    borderline_margin: float = 10.0,
) -> ExposureResult:
    """
    Evaluate image exposure against configured development thresholds.

    Parameters
    ----------
    image : 2D np.ndarray
    exposure_mean_min : float
        Minimum acceptable mean intensity (default: 20.0).
    exposure_mean_max : float
        Maximum acceptable mean intensity (default: 235.0).
    borderline_margin : float
        Margin in intensity units for near-threshold detection (default: 10.0).

    Returns
    -------
    ExposureResult
    """
    stats = compute_exposure_statistics(image)
    mean_val = stats["mean_intensity"]
    std_val = stats["histogram_std"]
    under_frac = stats["under_exposed_fraction"]
    over_frac = stats["over_exposed_fraction"]

    is_under = mean_val < exposure_mean_min
    is_over = mean_val > exposure_mean_max

    near_low = abs(mean_val - exposure_mean_min) <= borderline_margin
    near_high = abs(mean_val - exposure_mean_max) <= borderline_margin
    is_near = near_low or near_high

    if is_under:
        status = "poor"
        reasoning = (
            f"Image is underexposed with mean intensity {mean_val:.1f} < "
            f"{exposure_mean_min:.1f} (dark pixel fraction: {under_frac * 100:.1f}%)."
        )
    elif is_over:
        status = "poor"
        reasoning = (
            f"Image is overexposed with mean intensity {mean_val:.1f} > "
            f"{exposure_mean_max:.1f} (saturated pixel fraction: {over_frac * 100:.1f}%)."
        )
    elif is_near:
        status = "degraded"
        bound_str = f"lower bound {exposure_mean_min:.1f}" if near_low else f"upper bound {exposure_mean_max:.1f}"
        reasoning = (
            f"Image exposure (mean {mean_val:.1f}) is near the {bound_str} "
            f"(margin ±{borderline_margin:.1f})."
        )
    else:
        status = "good"
        reasoning = (
            f"Image exposure is acceptable with mean intensity {mean_val:.1f} "
            f"within [{exposure_mean_min:.1f}, {exposure_mean_max:.1f}] "
            f"(contrast std: {std_val:.1f})."
        )

    return ExposureResult(
        mean_intensity=mean_val,
        histogram_std=std_val,
        under_exposed_fraction=under_frac,
        over_exposed_fraction=over_frac,
        exposure_mean_min=exposure_mean_min,
        exposure_mean_max=exposure_mean_max,
        is_underexposed=is_under,
        is_overexposed=is_over,
        is_near_threshold=is_near,
        status=status,
        reasoning=reasoning,
    )
