"""Synthetic corruption library with known severity.

Responsibility:
    Generate seeded blur, noise, exposure and combined degradations at graded severities.
    Provides the ground truth needed to calibrate Quality thresholds (ROC), Repair bounds
    and the Verification margin, since NIH has no natural defect labels.

Input:
    Clean image array, defect type, severity, seed.

Output:
    Corrupted image array plus a record of the exact parameters applied.

Dependencies:
    numpy, opencv-python-headless, scikit-image

Implementation phase: P3
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def apply_blur(image: "np.ndarray", severity: float, seed: int) -> "np.ndarray":
    raise NotImplementedError("Phase 3")


def apply_noise(image: "np.ndarray", severity: float, seed: int) -> "np.ndarray":
    raise NotImplementedError("Phase 3")


def apply_exposure_shift(image: "np.ndarray", severity: float, seed: int) -> "np.ndarray":
    raise NotImplementedError("Phase 3")


def apply_combined(image: "np.ndarray", severities: dict[str, float], seed: int) -> "np.ndarray":
    raise NotImplementedError("Phase 3")
