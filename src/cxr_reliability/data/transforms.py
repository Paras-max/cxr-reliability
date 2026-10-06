"""Image preprocessing transforms (Phase 4).

Implements the TXV preprocessing pipeline and a quality-grayscale
converter stub for the future Quality Agent (Phase 5).

Implementation phase: P4
"""

from __future__ import annotations

import numpy as np


def to_txv_input(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Convert a numpy image array to TXV-normalized format.

    Parameters
    ----------
    image       : H x W uint8 or float array (grayscale)
    target_size : resize to (target_size x target_size)

    Returns
    -------
    numpy array shape (1, 1, target_size, target_size) in [-1024, 1024]

    Note: For loading from a file path, prefer
    models.preprocessing.load_image_for_txv() which handles
    all format conversions and error handling automatically.
    """
    from PIL import Image as PILImage

    arr = np.array(image, dtype=np.float32)

    # If already 2D grayscale, normalise
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        pil = PILImage.fromarray(np.uint8(arr)).convert("L")
    elif arr.ndim == 2:
        pil = PILImage.fromarray(np.uint8(arr.clip(0, 255))).convert("L")
    else:
        raise ValueError(f"Unexpected image shape: {arr.shape}")

    pil = pil.resize((target_size, target_size), PILImage.LANCZOS)
    out = np.array(pil, dtype=np.float32)

    # Normalise [0,255] → [-1024, 1024]
    out = out * (2048.0 / 255.0) - 1024.0

    # (1, 1, H, W)
    return out[np.newaxis, np.newaxis, :, :]


def to_quality_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Return a standardised uint8 grayscale view of an image.
    Used by the Quality Agent (Phase 5) for blur/artefact detection.

    Implementation: stub — will be completed in Phase 5.
    """
    # Phase 5 stub: basic uint8 conversion for now
    arr = np.array(image, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr.clip(0, 255)
    else:
        arr = (arr * 255).clip(0, 255)
    return arr.astype(np.uint8)
