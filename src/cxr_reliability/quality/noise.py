"""Noise estimation module — Phase 7.

Responsibility
--------------
Estimate image noise from high-frequency spatial residuals and calculate
Signal-to-Noise Ratio (SNR in dB).

Mathematical Definition
-----------------------
1. Spatial Residual:
   Denoise image with a deterministic 2D spatial smoothing filter (Gaussian sigma=1.0):
       I_smooth = Gaussian(I, sigma=1.0)
       R = I - I_smooth
   R represents the high-frequency residual containing noise and micro-texture.

2. Noise Power (sigma_noise):
       sigma_noise = std(R)
   Measures high-frequency residual dispersion.

3. Signal Measure (mu_signal):
       mu_signal = mean(I)

4. Signal-to-Noise Ratio (SNR in dB):
       SNR_dB = 20 * log10( max(mu_signal, eps) / (sigma_noise + eps) )

Numerical Safeguards:
- When sigma_noise == 0.0 (e.g., synthetic uniform/noiseless pattern), SNR_dB is
  cleanly capped at +100.0 dB to prevent inf/NaN.
- eps = 1e-6 prevents division by zero and log10(0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.ndimage


@dataclass
class NoiseResult:
    """Component result for noise assessment."""
    snr_db: float
    noise_std: float
    signal_mean: float
    threshold: float
    is_noisy: bool
    is_near_threshold: bool
    status: str  # "good" | "poor" | "degraded"
    reasoning: str


def estimate_noise_std(image: np.ndarray, filter_size: int = 3) -> float:
    """
    Estimate image noise standard deviation using high-frequency residual and
    robust Median Absolute Deviation (MAD), preventing edge contamination.

    Parameters
    ----------
    image : 2D np.ndarray
    filter_size : int
        Median filter size for edge-preserving smoothing (default 3).

    Returns
    -------
    float
        Estimated noise standard deviation >= 0.0.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

    img_f = image.astype(np.float64)
    # Edge-preserving median smoothing
    smoothed = scipy.ndimage.median_filter(img_f, size=filter_size, mode="reflect")
    residual = img_f - smoothed

    # Robust MAD: sigma = MAD / 0.6745
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    if mad > 1e-8:
        noise_std = mad / 0.6745
    else:
        noise_std = float(np.std(residual))
    return max(0.0, float(noise_std))


def compute_snr_db(
    image: np.ndarray,
    noise_std: float | None = None,
    eps: float = 1e-6,
) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in decibels (dB).

    Parameters
    ----------
    image : 2D np.ndarray
    noise_std : float, optional
        Precomputed noise standard deviation. If None, estimated via Gaussian residual.
    eps : float
        Small positive constant for numerical stability.

    Returns
    -------
    float
        SNR in dB. If noise_std == 0.0, returns 100.0 dB.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

    if noise_std is None:
        noise_std = estimate_noise_std(image)

    if noise_std <= 1e-8:
        return 100.0  # Clean ceiling for zero-noise images

    signal_mean = float(np.mean(image))
    effective_signal = max(signal_mean, eps)
    effective_noise = max(noise_std, eps)

    ratio = effective_signal / effective_noise
    snr_db = 20.0 * math.log10(ratio)
    return float(snr_db)


def evaluate_noise(
    image: np.ndarray,
    threshold_db: float = 15.0,
    borderline_margin_db: float = 3.0,
) -> NoiseResult:
    """
    Evaluate image noise against configured development threshold.

    Parameters
    ----------
    image : 2D np.ndarray
    threshold_db : float
        Minimum acceptable SNR in dB (PRD v0 default: 15.0 dB).
    borderline_margin_db : float
        Margin around threshold for near-threshold detection (default: 3.0 dB).

    Returns
    -------
    NoiseResult
    """
    noise_std = estimate_noise_std(image)
    signal_mean = float(np.mean(image))
    snr_db = compute_snr_db(image, noise_std=noise_std)

    is_noisy = snr_db < threshold_db
    is_near = abs(snr_db - threshold_db) <= borderline_margin_db

    if is_noisy:
        status = "poor"
        reasoning = (
            f"Estimated high-frequency residual is elevated (SNR: {snr_db:.1f} dB < "
            f"{threshold_db:.1f} dB, noise std: {noise_std:.2f})."
        )
    elif is_near:
        status = "degraded"
        reasoning = (
            f"Image SNR ({snr_db:.1f} dB) is near the noise threshold "
            f"({threshold_db:.1f} ± {borderline_margin_db:.1f} dB)."
        )
    else:
        status = "good"
        reasoning = (
            f"Image noise is acceptable with SNR {snr_db:.1f} dB >= {threshold_db:.1f} dB "
            f"(noise std: {noise_std:.2f})."
        )

    return NoiseResult(
        snr_db=snr_db,
        noise_std=noise_std,
        signal_mean=signal_mean,
        threshold=threshold_db,
        is_noisy=is_noisy,
        is_near_threshold=is_near,
        status=status,
        reasoning=reasoning,
    )
