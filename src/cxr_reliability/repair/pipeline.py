"""Deterministic sequential repair pipeline (PRD FR-6) — Phase 9.

Responsibility:
    Execute quality repairs in a fixed, deterministic sequence based on detected defects.

    Multi-Repair Execution Order:
        1. Exposure correction (CLAHE)
        2. Noise reduction (Non-Local Means)
        3. Conservative sharpening (Unsharp Masking)

    IMPORTANT DOCUMENTATION NOTE:
    This ordering is a deterministic engineering choice for this research prototype
    and is not presented as a clinically validated processing order.

Implementation phase: P9
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cxr_reliability.contracts.quality import DefectType
from cxr_reliability.contracts.repair import RepairStep

from .blur import repair_blur
from .exposure import repair_exposure
from .noise import repair_noise


@dataclass
class RepairConfig:
    """
    Configurable development defaults for the Repair Agent.

    NOTE: All values are provisional development parameters and are NOT
    clinically validated or calibrated thresholds.
    """

    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    nl_means_h: float = 3.0
    nl_means_template_size: int = 7
    nl_means_search_size: int = 21
    unsharp_radius: float = 1.0
    unsharp_amount: float = 0.5


# Deterministic engineering execution order: Exposure -> Noise -> Blur
REPAIR_EXECUTION_ORDER: tuple[DefectType, ...] = (
    DefectType.EXPOSURE,
    DefectType.NOISE,
    DefectType.BLUR,
)


def apply_sequential_repairs(
    image: np.ndarray,
    defects: Sequence[DefectType],
    config: RepairConfig | None = None,
) -> tuple[np.ndarray, list[RepairStep]]:
    """
    Apply requested repairs in deterministic order: Exposure -> Noise -> Blur.

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image array with shape (H, W).
    defects : Sequence[DefectType]
        Collection of defect types to repair.
    config : RepairConfig, optional
        Configuration for repair operations (provisional development defaults).

    Returns
    -------
    tuple[np.ndarray, list[RepairStep]]
        Repaired uint8 image array and list of recorded RepairStep contracts.
    """
    cfg = config or RepairConfig()
    current_img = image.copy()
    steps: list[RepairStep] = []

    defects_set = set(defects)

    for defect in REPAIR_EXECUTION_ORDER:
        if defect not in defects_set:
            continue

        if defect == DefectType.EXPOSURE:
            current_img, params = repair_exposure(
                current_img,
                clip_limit=cfg.clahe_clip_limit,
                tile_grid_size=cfg.clahe_tile_grid_size,
            )
            steps.append(
                RepairStep(
                    defect=DefectType.EXPOSURE,
                    method="clahe",
                    parameters=params,
                )
            )

        elif defect == DefectType.NOISE:
            current_img, params = repair_noise(
                current_img,
                h=cfg.nl_means_h,
                template_window_size=cfg.nl_means_template_size,
                search_window_size=cfg.nl_means_search_size,
            )
            steps.append(
                RepairStep(
                    defect=DefectType.NOISE,
                    method="nl_means",
                    parameters=params,
                )
            )

        elif defect == DefectType.BLUR:
            current_img, params = repair_blur(
                current_img,
                radius=cfg.unsharp_radius,
                amount=cfg.unsharp_amount,
            )
            steps.append(
                RepairStep(
                    defect=DefectType.BLUR,
                    method="unsharp_mask",
                    parameters=params,
                )
            )

    return current_img, steps
