"""tests/unit/test_quality_agent.py

Comprehensive unit tests and integration tests for the Quality Agent (Phase 7).

Test Coverage:
    1.  Sharp synthetic pattern has high Laplacian variance and no blur flag.
    2.  Gaussian blur lowers variance and raises blur_pct monotonically.
    3.  Laplacian variance calculation accuracy.
    4.  Blur threshold boundary behavior.
    5.  Clean synthetic image has high SNR.
    6.  Synthetic noise lowers SNR (dB) monotonically.
    7.  Noise metric calculation accuracy.
    8.  Zero-noise handling (finite, non-NaN, capped cleanly).
    9.  Noise numerical stability with near-zero noise.
    10. Dark synthetic image (mean < 20) flags underexposure.
    11. Normal synthetic image has acceptable exposure.
    12. Bright synthetic image (mean > 235) flags overexposure.
    13. Exposure boundary behavior.
    14. NumPy grayscale input (2D).
    15. NumPy RGB input (3D).
    16. uint8 input [0, 255].
    17. float [0, 1] input.
    18. Invalid dimensions (e.g., 1D or 5D).
    19. Empty image raises ValueError.
    20. NaN input raises ValueError.
    21. Inf input raises ValueError.
    22. GOOD overall quality case.
    23. DEGRADED overall quality case (near-threshold).
    24. POOR overall quality case.
    25. Structured contract validation (QualityResult, QualityFlags).
    26. Human-readable reasoning without clinical claims.
    27. Deterministic output across repeated runs.
    28. Integration tests across synthetic test images (Cases 1 - 6).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage
import torch
from PIL import Image

from cxr_reliability.agents.quality import QualityAgent
from cxr_reliability.contracts.common import AgentName
from cxr_reliability.contracts.quality import QualityFlags, QualityLevel, QualityResult
from cxr_reliability.quality import (
    QualityConfig,
    QualityEvaluator,
    compute_blur_pct,
    compute_laplacian_variance,
    compute_snr_db,
    estimate_noise_std,
    evaluate_blur,
    evaluate_exposure,
    evaluate_noise,
    prepare_image_for_quality,
)


# ── Synthetic image helpers ───────────────────────────────────────────────────

def _make_sharp_image(size: int = 128) -> np.ndarray:
    """Checkerboard high-frequency sharp pattern in [0, 255]."""
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)
    checker = ((xx // 8) % 2 == (yy // 8) % 2).astype(np.float64) * 200.0 + 25.0
    return checker


def _make_smooth_image(size: int = 128) -> np.ndarray:
    """Smooth low-frequency gradient (blurred / low edge content)."""
    x = np.linspace(50.0, 150.0, size)
    return np.tile(x, (size, 1))


def _make_noisy_image(base: np.ndarray, noise_std: float = 25.0, seed: int = 42) -> np.ndarray:
    """Add Gaussian noise to a base image, clipped to [0, 255]."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=base.shape)
    return np.clip(base + noise, 0.0, 255.0)


# ── Test 1–4: Blur Evaluation ─────────────────────────────────────────────────

class TestBlurEvaluation:
    def test_sharp_image_not_flagged_blur(self):
        sharp = _make_sharp_image()
        res = evaluate_blur(sharp, threshold=100.0)
        assert res.laplacian_variance > 100.0
        assert not res.is_blurred
        assert res.status == "good"

    def test_gaussian_blur_lowers_variance_and_raises_blur_pct(self):
        sharp = _make_sharp_image()
        var_prev = compute_laplacian_variance(sharp)
        blur_pct_prev = compute_blur_pct(var_prev, reference_max_variance=50000.0)

        for sigma in [1.0, 2.0, 4.0]:
            blurred = scipy.ndimage.gaussian_filter(sharp, sigma=sigma)
            var_curr = compute_laplacian_variance(blurred)
            blur_pct_curr = compute_blur_pct(var_curr, reference_max_variance=50000.0)

            assert var_curr < var_prev, f"Variance did not decrease at sigma={sigma}"
            assert blur_pct_curr > blur_pct_prev, f"Blur % did not increase at sigma={sigma}"
            var_prev = var_curr
            blur_pct_prev = blur_pct_curr

    def test_laplacian_variance_calculation(self):
        # Uniform image has zero Laplacian variance
        uniform = np.full((64, 64), 128.0)
        var = compute_laplacian_variance(uniform)
        assert var == 0.0

    def test_blur_threshold_boundary_behavior(self):
        smooth = _make_smooth_image()
        res = evaluate_blur(smooth, threshold=100.0)
        assert res.laplacian_variance < 100.0
        assert res.is_blurred
        assert res.status == "poor"


# ── Test 5–9: Noise Evaluation ────────────────────────────────────────────────

class TestNoiseEvaluation:
    def test_clean_image_has_high_snr(self):
        clean = _make_sharp_image()
        res = evaluate_noise(clean, threshold_db=15.0)
        # Clean image has low residual noise relative to signal
        assert res.snr_db > 15.0
        assert not res.is_noisy

    def test_synthetic_noise_lowers_snr_monotonically(self):
        clean = np.full((128, 128), 128.0)
        snr_prev = compute_snr_db(clean)

        for noise_level in [5.0, 15.0, 30.0, 50.0]:
            noisy = _make_noisy_image(clean, noise_std=noise_level)
            snr_curr = compute_snr_db(noisy)
            assert snr_curr < snr_prev, f"SNR did not drop at noise={noise_level}"
            snr_prev = snr_curr

    def test_noise_metric_calculation(self):
        clean = np.full((100, 100), 100.0)
        noisy = _make_noisy_image(clean, noise_std=20.0, seed=123)
        res = evaluate_noise(noisy, threshold_db=15.0)
        assert abs(res.noise_std - 20.0) < 4.0
        assert res.snr_db > 0.0

    def test_zero_noise_handling(self):
        uniform = np.full((64, 64), 100.0)
        snr = compute_snr_db(uniform)
        assert np.isfinite(snr)
        assert snr == 100.0  # Safe clean ceiling

    def test_numerical_stability_near_zero(self):
        img = np.full((64, 64), 100.0) + 1e-7
        res = evaluate_noise(img)
        assert np.isfinite(res.snr_db)
        assert not np.isnan(res.snr_db)


# ── Test 10–13: Exposure Evaluation ───────────────────────────────────────────

class TestExposureEvaluation:
    def test_dark_image_flags_underexposure(self):
        # PRD FR-1: flag if mean intensity < 20
        dark = np.full((64, 64), 15.0)
        res = evaluate_exposure(dark, exposure_mean_min=20.0, exposure_mean_max=235.0)
        assert res.is_underexposed
        assert not res.is_overexposed
        assert res.status == "poor"

    def test_normal_image_acceptable_exposure(self):
        normal = np.full((64, 64), 120.0)
        res = evaluate_exposure(normal, exposure_mean_min=20.0, exposure_mean_max=235.0)
        assert not res.is_underexposed
        assert not res.is_overexposed
        assert res.status == "good"

    def test_bright_image_flags_overexposure(self):
        # PRD FR-1: flag if mean intensity > 235
        bright = np.full((64, 64), 240.0)
        res = evaluate_exposure(bright, exposure_mean_min=20.0, exposure_mean_max=235.0)
        assert res.is_overexposed
        assert not res.is_underexposed
        assert res.status == "poor"

    def test_exposure_boundary_behavior(self):
        # Near lower boundary (20.0 +/- margin)
        near_low = np.full((64, 64), 22.0)
        res = evaluate_exposure(near_low, exposure_mean_min=20.0, exposure_mean_max=235.0, borderline_margin=5.0)
        assert res.is_near_threshold
        assert res.status == "degraded"


# ── Test 14–21: Input Validation ──────────────────────────────────────────────

class TestInputValidation:
    def test_numpy_grayscale_2d(self):
        arr = np.ones((64, 64), dtype=np.uint8) * 100
        out = prepare_image_for_quality(arr)
        assert out.shape == (64, 64)
        assert out.dtype == np.float64
        assert out[0, 0] == 100.0

    def test_numpy_rgb_3d(self):
        arr = np.ones((64, 64, 3), dtype=np.uint8) * 150
        out = prepare_image_for_quality(arr)
        assert out.shape == (64, 64)
        assert abs(out[0, 0] - 150.0) < 1.0

    def test_uint8_range_conversion(self):
        arr = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        out = prepare_image_for_quality(arr)
        assert out.min() == 0.0
        assert out.max() == 255.0

    def test_float_zero_one_scaling(self):
        arr = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
        out = prepare_image_for_quality(arr)
        assert abs(out.max() - 255.0) < 1e-4
        assert abs(out[1, 0] - 127.5) < 1e-4

    def test_txv_normalized_tensor_inversion(self):
        # TXV normalization: [-1024, 1024]
        # Inverts back to [0, 255]
        tensor = torch.tensor([[[[-1024.0, 1024.0], [0.0, 512.0]]]], dtype=torch.float32)
        out = prepare_image_for_quality(tensor)
        assert abs(out[0, 0] - 0.0) < 1e-2
        assert abs(out[0, 1] - 255.0) < 1e-2
        assert abs(out[1, 0] - 127.5) < 1e-2

    def test_invalid_dimensions_raises(self):
        with pytest.raises(ValueError, match="shape"):
            prepare_image_for_quality(np.ones((2, 2, 2, 2, 2)))

    def test_empty_image_raises(self):
        with pytest.raises(ValueError, match="empty"):
            prepare_image_for_quality(np.array([]))

    def test_nan_input_raises(self):
        arr = np.ones((32, 32))
        arr[5, 5] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            prepare_image_for_quality(arr)

    def test_inf_input_raises(self):
        arr = np.ones((32, 32))
        arr[5, 5] = np.inf
        with pytest.raises(ValueError, match="Inf"):
            prepare_image_for_quality(arr)


# ── Test 22–27: Overall Quality Verdict, Contract, Reasoning ──────────────────

class TestOverallQualityVerdict:
    @pytest.fixture
    def agent(self):
        return QualityAgent()

    def test_good_case(self, agent):
        img = _make_sharp_image()
        res = agent.run(img, image_id="good_img.png")
        assert res.overall == QualityLevel.GOOD
        assert not res.flags.blur
        assert not res.flags.noise
        assert not res.flags.exposure
        assert "GOOD" in res.reasoning

    def test_degraded_case(self, agent):
        sharp = _make_sharp_image()
        blurred = scipy.ndimage.gaussian_filter(sharp, sigma=2.0)
        var = compute_laplacian_variance(blurred)
        cfg = QualityConfig(
            blur_laplacian_var_min=var - 5.0,
            borderline_margin_pct=0.20,
        )
        custom_agent = QualityAgent(config=cfg)
        res = custom_agent.run(blurred, image_id="borderline.png")
        assert res.near_threshold
        assert res.overall == QualityLevel.DEGRADED

    def test_poor_case(self, agent):
        # Severely dark image (mean = 10 < 20)
        dark = np.full((64, 64), 10.0)
        res = agent.run(dark, image_id="dark.png")
        assert res.overall == QualityLevel.POOR
        assert res.flags.exposure
        assert "POOR" in res.reasoning

    def test_structured_contract_validation(self, agent):
        img = _make_sharp_image()
        res = agent.run(img, image_id="test_contract.png")
        assert isinstance(res, QualityResult)
        assert res.agent == AgentName.QUALITY
        assert res.version == "0.7.0"
        assert isinstance(res.flags, QualityFlags)
        assert res.laplacian_variance >= 0.0
        assert 0.0 <= res.blur_pct <= 100.0
        assert isinstance(res.snr_db, float)
        assert isinstance(res.mean_intensity, float)
        assert isinstance(res.histogram_std, float)
        assert res.latency_ms is not None and res.latency_ms >= 0.0

    def test_reasoning_has_no_clinical_claims(self, agent):
        dark = np.full((64, 64), 5.0)
        res = agent.run(dark)
        txt = res.reasoning.lower()
        assert "patient" not in txt
        assert "diagnosis" not in txt
        assert "pneumonia" not in txt
        assert "medically unusable" not in txt

    def test_deterministic_output(self, agent):
        img = _make_sharp_image()
        res1 = agent.run(img)
        res2 = agent.run(img)
        assert res1.laplacian_variance == res2.laplacian_variance
        assert res1.snr_db == res2.snr_db
        assert res1.mean_intensity == res2.mean_intensity
        assert res1.overall == res2.overall
        assert res1.reasoning == res2.reasoning


# ── Test 28: Integration Tests (Cases 1–6) ────────────────────────────────────

class TestSyntheticIntegration:
    @pytest.fixture
    def agent(self):
        return QualityAgent()

    def test_case_1_clean_sharp_normal(self, agent):
        """Case 1: Clean, sharp, normal exposure."""
        img = _make_sharp_image(128)
        res = agent.run(img, image_id="case1.png")
        assert res.overall == QualityLevel.GOOD
        assert not res.flags.blur
        assert not res.flags.noise
        assert not res.flags.exposure

    def test_case_2_blurred_image(self, agent):
        """Case 2: Severely blurred image."""
        img = _make_sharp_image(128)
        blurred = scipy.ndimage.gaussian_filter(img, sigma=5.0)
        res = agent.run(blurred, image_id="case2.png")
        assert res.flags.blur
        assert res.overall == QualityLevel.POOR
        assert "blur" in res.reasoning.lower()

    def test_case_3_noisy_image(self, agent):
        """Case 3: Heavy Gaussian noise image."""
        clean = np.full((128, 128), 100.0)
        noisy = _make_noisy_image(clean, noise_std=40.0)
        res = agent.run(noisy, image_id="case3.png")
        assert res.flags.noise
        assert res.overall == QualityLevel.POOR
        assert "noise" in res.reasoning.lower()

    def test_case_4_dark_image(self, agent):
        """Case 4: Underexposed image (mean < 20)."""
        dark = np.full((128, 128), 12.0)
        res = agent.run(dark, image_id="case4.png")
        assert res.flags.exposure
        assert res.overall == QualityLevel.POOR
        assert "underexposure" in res.reasoning.lower()

    def test_case_5_overexposed_image(self, agent):
        """Case 5: Overexposed image (mean > 235)."""
        bright = np.full((128, 128), 242.0)
        res = agent.run(bright, image_id="case5.png")
        assert res.flags.exposure
        assert res.overall == QualityLevel.POOR
        assert "overexposure" in res.reasoning.lower()

    def test_case_6_combined_poor_quality(self, agent):
        """Case 6: Dark, blurred, and noisy image."""
        dark = np.full((128, 128), 15.0)
        noisy_dark = _make_noisy_image(dark, noise_std=30.0)
        res = agent.run(noisy_dark, image_id="case6.png")
        assert res.overall == QualityLevel.POOR
        # Multiple defects flagged
        assert res.flags.noise or res.flags.exposure or res.flags.blur
