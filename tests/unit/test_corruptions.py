"""Tests for synthetic corruption library (Phase 3)."""

from __future__ import annotations

import numpy as np

from cxr_reliability.data.corruptions import (
    apply_blur,
    apply_combined,
    apply_noise,
)
from cxr_reliability.quality.blur import compute_laplacian_variance
from cxr_reliability.quality.noise import compute_snr_db


def _create_sample_image() -> np.ndarray:
    """Create a structured test pattern image (gradient + edges)."""
    x = np.linspace(50, 200, 128, dtype=np.uint8)
    img = np.tile(x, (128, 1))
    # Add high-frequency checkerboard features in center
    img[32:96, 32:96] = ((np.arange(64)[:, None] + np.arange(64)[None, :]) % 2) * 150 + 50
    return img


def test_same_seed_same_output():
    """Identical seed and severity give identical output."""
    img = _create_sample_image()

    noisy1 = apply_noise(img, severity=0.5, seed=1234)
    noisy2 = apply_noise(img, severity=0.5, seed=1234)
    np.testing.assert_array_equal(noisy1, noisy2)

    blurred1 = apply_blur(img, severity=0.5, seed=1234)
    blurred2 = apply_blur(img, severity=0.5, seed=1234)
    np.testing.assert_array_equal(blurred1, blurred2)

    comb1 = apply_combined(img, {"blur": 0.3, "noise": 0.3}, seed=1234)
    comb2 = apply_combined(img, {"blur": 0.3, "noise": 0.3}, seed=1234)
    np.testing.assert_array_equal(comb1, comb2)


def test_severity_monotonic():
    """Higher severity always degrades the corresponding metric more."""
    img = _create_sample_image()

    # Blur: higher severity should decrease Laplacian variance monotonically
    var_clean = compute_laplacian_variance(img)
    var_low = compute_laplacian_variance(apply_blur(img, severity=0.2))
    var_mid = compute_laplacian_variance(apply_blur(img, severity=0.5))
    var_high = compute_laplacian_variance(apply_blur(img, severity=0.9))

    assert var_clean > var_low > var_mid > var_high

    # Noise: higher severity should decrease SNR (dB) monotonically
    snr_clean = compute_snr_db(img)
    snr_low = compute_snr_db(apply_noise(img, severity=0.2, seed=42))
    snr_mid = compute_snr_db(apply_noise(img, severity=0.5, seed=42))
    snr_high = compute_snr_db(apply_noise(img, severity=0.9, seed=42))

    assert snr_clean > snr_low > snr_mid > snr_high
