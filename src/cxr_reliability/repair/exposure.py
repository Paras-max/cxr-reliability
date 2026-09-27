"""Exposure repair module (PRD FR-6) — Phase 9.

Responsibility:
    Apply configurable local contrast enhancement using CLAHE with bounded parameters.
    Does not claim clinical validation, anatomical preservation guarantees, or diagnostic
    improvement.

Implementation phase: P9
"""

from __future__ import annotations

import cv2
import numpy as np


def repair_exposure(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """
    Apply CLAHE local contrast enhancement to a 2D uint8 image.

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image array with shape (H, W).
    clip_limit : float
        Provisional development threshold limiting contrast amplification (default 2.0).
    tile_grid_size : tuple[int, int]
        Grid size for histogram equalization blocks (default (8, 8)).

    Returns
    -------
    tuple[np.ndarray, dict[str, float | int | str]]
        Repaired uint8 image array and dict of recorded operation parameters.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"repair_exposure expects uint8 image, got {image.dtype}")
    if image.ndim != 2:
        raise ValueError(f"repair_exposure expects 2D array, got shape {image.shape}")

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])),
    )
    repaired = clahe.apply(image)

    params: dict[str, float | int | str] = {
        "clip_limit": float(clip_limit),
        "tile_grid_h": int(tile_grid_size[0]),
        "tile_grid_w": int(tile_grid_size[1]),
    }
    return repaired, params
