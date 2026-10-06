"""Blur detection module — Phase 7.

Responsibility
--------------
Detect image blur using Laplacian variance and compute blur percentage.

Mathematical Definition
-----------------------
1. Laplacian Operator:
   Discrete 2D second spatial derivative of the image:
       L(I) = (d^2 I / dx^2) + (d^2 I / dy^2)
   Convolved with standard 3x3 kernel:
       [[0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]]

2. Laplacian Variance (Metric):
       laplacian_variance = Var(L(I))
   Higher variance indicates sharper transitions and high-frequency details.
   Lower variance indicates smoothed transitions (blur).

3. Blur Percentage (PRD FR-1 formula):
       blur_pct = 100 * (1 - min(laplacian_variance / reference_max_variance, 1.0))
   where reference_max_variance represents the expected high-frequency content of
   a reference sharp chest radiograph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage

# Standard 3x3 discrete Laplacian kernel
_LAPLACIAN_KERNEL = np.array(
    [[0.0, 1.0, 0.0],
     [1.0, -4.0, 1.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


@dataclass
class BlurResult:
    """Component result for blur assessment."""
    laplacian_variance: float
    blur_pct: float
    threshold: float
    reference_max_variance: float
    is_blurred: bool
    is_near_threshold: bool
    status: str  # "good" | "poor" | "degraded"
    reasoning: str


def compute_laplacian_variance(image: np.ndarray) -> float:
    """
    Compute variance of the discrete 2D Laplacian of a grayscale image.

    Parameters
    ----------
    image : 2D np.ndarray (float64 or float32 in [0, 255])

    Returns
    -------
    float
        Laplacian variance >= 0.0.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

    img_f = image.astype(np.float64)
    # 2D discrete convolution with reflect boundary conditions
    laplacian = scipy.ndimage.convolve(img_f, _LAPLACIAN_KERNEL, mode="reflect")
    var = float(np.var(laplacian))
    return max(0.0, var)


def compute_blur_pct(
    laplacian_variance: float,
    reference_max_variance: float = 500.0,
) -> float:
    """
    Compute blur percentage: 100 * (1 - min(var / ref_max, 1.0)).
    """
    if reference_max_variance <= 0:
        raise ValueError(f"reference_max_variance must be > 0, got {reference_max_variance}")
    ratio = min(max(0.0, laplacian_variance) / reference_max_variance, 1.0)
    return float(100.0 * (1.0 - ratio))


def evaluate_blur(
    image: np.ndarray,
    threshold: float = 100.0,
    reference_max_variance: float = 500.0,
    borderline_margin_pct: float = 0.15,
) -> BlurResult:
    """
    Evaluate image blur against configured development thresholds.

    Parameters
    ----------
    image : 2D np.ndarray
    threshold : float
        Minimum acceptable Laplacian variance (PRD v0 default: 100.0).
    reference_max_variance : float
        Variance reference for 100% sharp scale (default: 500.0).
    borderline_margin_pct : float
        Relative band around threshold for near-threshold detection.

    Returns
    -------
    BlurResult
    """
    var = compute_laplacian_variance(image)
    blur_pct = compute_blur_pct(var, reference_max_variance=reference_max_variance)

    is_blurred = var < threshold
    margin = threshold * borderline_margin_pct
    is_near = abs(var - threshold) <= margin

    if is_blurred:
        status = "poor"
        reasoning = (
            f"Image shows low Laplacian variance ({var:.1f} < {threshold:.1f}, "
            f"blur: {blur_pct:.1f}%), suggesting reduced high-frequency detail."
        )
    elif is_near:
        status = "degraded"
        reasoning = (
            f"Image Laplacian variance ({var:.1f}) is near the blur threshold "
            f"({threshold:.1f} ± {margin:.1f})."
        )
    else:
        status = "good"
        reasoning = (
            f"Image sharpness is acceptable with Laplacian variance {var:.1f} "
            f">= {threshold:.1f} (blur: {blur_pct:.1f}%)."
        )

    return BlurResult(
        laplacian_variance=var,
        blur_pct=blur_pct,
        threshold=threshold,
        reference_max_variance=reference_max_variance,
        is_blurred=is_blurred,
        is_near_threshold=is_near,
        status=status,
        reasoning=reasoning,
    )
