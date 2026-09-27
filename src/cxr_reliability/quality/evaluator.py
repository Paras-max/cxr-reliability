"""Quality evaluator and input standardizer — Phase 7.

Responsibility
--------------
Orchestrate blur, noise, and exposure evaluations into a unified QualityResult.
Provides rigorous input validation and normalization across image types.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .blur import BlurResult, evaluate_blur
from .exposure import ExposureResult, evaluate_exposure
from .noise import NoiseResult, evaluate_noise


@dataclass
class QualityConfig:
    """Configuration and development thresholds for the Quality Agent."""
    enabled: bool = True
    blur_laplacian_var_min: float = 100.0        # PRD FR-1: flag blur if var < 100
    reference_max_variance: float = 500.0        # Reference sharp image variance
    snr_db_min: float = 15.0                     # PRD FR-1: flag noise if SNR < 15 dB
    exposure_mean_min: float = 20.0              # PRD FR-1: flag if mean intensity < 20
    exposure_mean_max: float = 235.0             # PRD FR-1: flag if mean intensity > 235
    borderline_margin_pct: float = 0.15          # Margin percentage for near-threshold detection
    degraded_if_any_component_degraded: bool = True
    poor_if_any_component_poor: bool = True
    note: str = "Provisional development defaults — not clinically calibrated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_image_for_quality(image: Any) -> np.ndarray:
    """
    Standardize various image input formats into a 2D float64 array in [0.0, 255.0].

    Parameters
    ----------
    image : np.ndarray, torch.Tensor, PIL.Image, or Path/str
        Input image.

    Returns
    -------
    np.ndarray
        2D float64 array of shape (H, W) in [0.0, 255.0].

    Raises
    ------
    ValueError
        If image contains NaN, Inf, has invalid dimensions, or is empty.
    TypeError
        If image type is unsupported.
    """
    # ── 1. File Path ──────────────────────────────────────────────────────────
    if isinstance(image, (str, Path)):
        p = Path(image)
        if not p.exists():
            raise FileNotFoundError(f"Image path does not exist: {p}")
        pil_img = Image.open(p)
        return prepare_image_for_quality(pil_img)

    # ── 2. PIL Image ──────────────────────────────────────────────────────────
    if isinstance(image, Image.Image):
        gray_pil = image.convert("L")
        arr = np.array(gray_pil, dtype=np.float64)
        return arr

    # ── 3. PyTorch Tensor ─────────────────────────────────────────────────────
    if isinstance(image, torch.Tensor):
        if torch.isnan(image).any():
            raise ValueError("Input image tensor contains NaN.")
        if torch.isinf(image).any():
            raise ValueError("Input image tensor contains Inf.")
        if image.numel() == 0:
            raise ValueError("Input image tensor cannot be empty.")
        arr = image.detach().cpu().numpy()
        return prepare_image_for_quality(arr)

    # ── 4. NumPy Array ────────────────────────────────────────────────────────
    if isinstance(image, np.ndarray):
        if np.isnan(image).any():
            raise ValueError("Input image array contains NaN.")
        if np.isinf(image).any():
            raise ValueError("Input image array contains Inf.")
        if image.size == 0:
            raise ValueError("Input image array cannot be empty.")

        arr = image.astype(np.float64)

        # Handle dimensions: (1, 1, H, W) or (1, H, W)
        if arr.ndim == 4:
            if arr.shape[0] == 1 and arr.shape[1] in (1, 3, 4):
                arr = arr[0]  # reduce to (C, H, W)
            elif arr.shape[0] == 1 and arr.shape[3] in (1, 3, 4):
                arr = arr[0]  # reduce to (H, W, C)
            else:
                raise ValueError(f"Unsupported 4D image shape: {arr.shape}")

        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]  # (1, H, W) -> (H, W)
            elif arr.shape[2] == 1:
                arr = arr[:, :, 0]  # (H, W, 1) -> (H, W)
            elif arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
                arr = np.transpose(arr, (1, 2, 0))  # (C, H, W) -> (H, W, C)

            if arr.ndim == 3 and arr.shape[2] in (3, 4):
                r = arr[:, :, 0]
                g = arr[:, :, 1]
                b = arr[:, :, 2]
                arr = 0.299 * r + 0.587 * g + 0.114 * b
            elif arr.ndim == 3:
                raise ValueError(f"Unsupported 3D image shape: {arr.shape}")

        if arr.ndim != 2:
            raise ValueError(f"Image must be 2D after channel reduction, got shape {arr.shape}")

        # Check for zero-sized dimensions
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"Zero-sized image dimension: {arr.shape}")

        # ── Handle value ranges ───────────────────────────────────────────────
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))

        # Check if TorchXRayVision normalized range [-1024, 1024]
        if min_v < -50.0 and max_v <= 1024.0:
            # Invert: pixel_uint8 = (pixel_txv + 1024.0) * (255.0 / 2048.0)
            arr = (arr + 1024.0) * (255.0 / 2048.0)
            arr = np.clip(arr, 0.0, 255.0)
        elif min_v < 0.0:
            raise ValueError(f"Negative pixel intensity values detected without TXV encoding: min={min_v}")
        elif max_v <= 1.0 and max_v > 0.0:
            # Scaled [0.0, 1.0] -> [0.0, 255.0]
            arr = arr * 255.0
        elif max_v > 255.0 and max_v <= 65535.0:
            # 16-bit radiograph -> scale to [0, 255]
            arr = (arr / 65535.0) * 255.0
        elif max_v > 65535.0:
            raise ValueError(f"Unsupported high pixel values: max={max_v}")

        return arr

    raise TypeError(f"Unsupported image input type: {type(image).__name__}")


class QualityEvaluator:
    """
    High-level orchestrator evaluating blur, noise, and exposure.
    """

    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()

    def evaluate(
        self,
        image: Any,
        image_id: str | None = None,
    ) -> tuple[BlurResult, NoiseResult, ExposureResult, str, str, bool]:
        """
        Evaluate all quality dimensions.

        Returns
        -------
        tuple containing:
            blur_res       : BlurResult
            noise_res      : NoiseResult
            exposure_res   : ExposureResult
            overall_status : "good" | "degraded" | "poor"
            reasoning      : human-readable explanation
            near_threshold : bool (True if any metric is within borderline margin)
        """
        std_img = prepare_image_for_quality(image)

        # 1. Blur
        blur_res = evaluate_blur(
            std_img,
            threshold=self.config.blur_laplacian_var_min,
            reference_max_variance=self.config.reference_max_variance,
            borderline_margin_pct=self.config.borderline_margin_pct,
        )

        # 2. Noise
        noise_res = evaluate_noise(
            std_img,
            threshold_db=self.config.snr_db_min,
            borderline_margin_db=self.config.snr_db_min * self.config.borderline_margin_pct,
        )

        # 3. Exposure
        margin_exposure = (self.config.exposure_mean_max - self.config.exposure_mean_min) * 0.05
        exposure_res = evaluate_exposure(
            std_img,
            exposure_mean_min=self.config.exposure_mean_min,
            exposure_mean_max=self.config.exposure_mean_max,
            borderline_margin=margin_exposure,
        )

        # ── Overall Verdict ───────────────────────────────────────────────────
        is_poor = (
            blur_res.is_blurred
            or noise_res.is_noisy
            or exposure_res.is_underexposed
            or exposure_res.is_overexposed
        )
        is_near = (
            blur_res.is_near_threshold
            or noise_res.is_near_threshold
            or exposure_res.is_near_threshold
        )

        if is_poor and self.config.poor_if_any_component_poor:
            overall_status = "poor"
        elif is_near and self.config.degraded_if_any_component_degraded:
            overall_status = "degraded"
        else:
            overall_status = "good"

        # ── Explainability Reasoning ──────────────────────────────────────────
        defects = []
        if blur_res.is_blurred:
            defects.append(f"blur (var: {blur_res.laplacian_variance:.1f} < {blur_res.threshold:.1f})")
        if noise_res.is_noisy:
            defects.append(f"noise (SNR: {noise_res.snr_db:.1f} dB < {noise_res.threshold:.1f} dB)")
        if exposure_res.is_underexposed:
            defects.append(f"underexposure (mean: {exposure_res.mean_intensity:.1f} < {exposure_res.exposure_mean_min:.1f})")
        if exposure_res.is_overexposed:
            defects.append(f"overexposure (mean: {exposure_res.mean_intensity:.1f} > {exposure_res.exposure_mean_max:.1f})")

        if defects:
            reasoning = f"Image quality is POOR due to {', '.join(defects)}."
        elif is_near:
            reasons = []
            if blur_res.is_near_threshold:
                reasons.append(f"sharpness near threshold ({blur_res.laplacian_variance:.1f})")
            if noise_res.is_near_threshold:
                reasons.append(f"noise near threshold ({noise_res.snr_db:.1f} dB)")
            if exposure_res.is_near_threshold:
                reasons.append(f"exposure near boundary ({exposure_res.mean_intensity:.1f})")
            reasoning = f"Image quality is DEGRADED with borderline indicators: {', '.join(reasons)}."
        else:
            reasoning = (
                f"Image quality is GOOD: sharpness acceptable (var: {blur_res.laplacian_variance:.1f}), "
                f"noise acceptable (SNR: {noise_res.snr_db:.1f} dB), "
                f"exposure within normal bounds (mean: {exposure_res.mean_intensity:.1f})."
            )

        return blur_res, noise_res, exposure_res, overall_status, reasoning, is_near
