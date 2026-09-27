"""Quality assessment module — Phase 7.

Responsibility
--------------
Assess whether a chest X-ray has image-quality degradations (blur, noise, exposure)
that may affect downstream model reliability.

Modules:
- blur: Laplacian variance and blur percentage
- noise: high-frequency residual noise estimation and SNR (dB)
- exposure: photometric mean intensity and histogram spread
- evaluator: input standardization and high-level quality orchestrator
"""

from __future__ import annotations

from .blur import BlurResult, compute_blur_pct, compute_laplacian_variance, evaluate_blur
from .exposure import (
    ExposureResult,
    compute_exposure_statistics,
    evaluate_exposure,
)
from .noise import NoiseResult, compute_snr_db, estimate_noise_std, evaluate_noise
from .evaluator import QualityConfig, QualityEvaluator, prepare_image_for_quality

__all__ = [
    "BlurResult",
    "compute_blur_pct",
    "compute_laplacian_variance",
    "evaluate_blur",
    "ExposureResult",
    "compute_exposure_statistics",
    "evaluate_exposure",
    "NoiseResult",
    "compute_snr_db",
    "estimate_noise_std",
    "evaluate_noise",
    "QualityConfig",
    "QualityEvaluator",
    "prepare_image_for_quality",
]
