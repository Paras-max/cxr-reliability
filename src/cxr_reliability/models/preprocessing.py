"""Image preprocessing for TorchXRayVision models (Phase 4).

Responsibility:
    Convert raw chest X-ray images (PNG/JPG/JPEG) into the tensor format
    expected by TorchXRayVision models.

EXACT PREPROCESSING PIPELINE (documented as required by PRD):
    1. Load image from disk using Pillow.
    2. Convert to grayscale (L mode) — CXR images are single-channel.
    3. Resize to target_size x target_size using LANCZOS resampling.
    4. Convert to float32 numpy array in [0.0, 255.0].
    5. Normalize to TXV HU-like range:
           pixel_txv = (pixel_uint8 / 255.0) * 2048.0 - 1024.0
       This maps:  0   → -1024.0
                   255 →  +1024.0
       TorchXRayVision expects this range for all its pretrained models.
       Do NOT use ImageNet mean/std normalization here.
    6. Add batch and channel dimensions: shape (1, 1, H, W).
    7. Convert to torch.Tensor (float32).

IMPORTANT:
    The original image file is NEVER modified.
    The tensor is created in memory only.

Dependencies:
    Pillow, numpy, torch

Implementation phase: P4
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

# Supported input formats
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# TXV normalization constants (derived from [0,255] uint8 → [-1024,1024] range)
_TXV_SCALE = 2048.0 / 255.0   # multiply factor
_TXV_SHIFT = -1024.0           # add after scaling

# Default input size for densenet121-res224-nih
DEFAULT_TARGET_SIZE = 224


def load_image_for_txv(
    image_path: Path | str,
    target_size: int = DEFAULT_TARGET_SIZE,
) -> torch.Tensor:
    """
    Load a chest X-ray image and return a model-ready tensor.

    Parameters
    ----------
    image_path  : path to a PNG, JPG, or JPEG file
    target_size : resize to (target_size × target_size); default 224 for
                  densenet121-res224-nih

    Returns
    -------
    torch.Tensor of shape (1, 1, target_size, target_size), dtype=float32
    Values are in the range [-1024.0, 1024.0] (TXV HU-like normalization).

    Raises
    ------
    FileNotFoundError  : if image_path does not exist
    ValueError         : if file extension is not supported
    RuntimeError       : if the image cannot be decoded (corrupt file)
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}\n"
            "Check that the image path in the CSV matches the dataset on disk."
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: '{path.suffix}'. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        img = Image.open(path)
        # Convert to grayscale — chest X-rays are single-channel
        img = img.convert("L")
        # Resize with high-quality LANCZOS resampling
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.LANCZOS)
    except UnidentifiedImageError as exc:
        raise RuntimeError(
            f"Cannot decode image (file may be corrupt): {path}\n"
            f"Original error: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load image {path}: {exc}"
        ) from exc

    # float32 array in [0.0, 255.0]
    arr = np.array(img, dtype=np.float32)

    # Normalize to TXV HU-like range [-1024, 1024]
    # Formula: pixel_txv = pixel_uint8 * (2048/255) - 1024
    arr = arr * _TXV_SCALE + _TXV_SHIFT

    # Shape: (1, 1, H, W)  →  batch=1, channels=1, height, width
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    return tensor


def prepare_image_for_txv(
    image: Any,
    target_size: int = DEFAULT_TARGET_SIZE,
) -> torch.Tensor:
    """
    Prepare an image (file path, PIL Image, numpy array, or torch Tensor)
    into a model-ready tensor for TorchXRayVision models.

    Ensures single-channel grayscale, resized to (target_size, target_size),
    normalized to TXV HU-like range [-1024.0, 1024.0], and shaped as (1, 1, H, W).
    Does NOT double-normalize if the input is already in TXV HU range.

    Parameters
    ----------
    image : Path, str, PIL.Image, np.ndarray, or torch.Tensor
    target_size : int, default 224

    Returns
    -------
    torch.Tensor of shape (1, 1, target_size, target_size), dtype=float32
    """
    # 1. File path or str
    if isinstance(image, (str, Path)):
        return load_image_for_txv(image, target_size=target_size)

    # 2. PIL Image
    if isinstance(image, Image.Image):
        img = image.convert("L")
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        arr = arr * _TXV_SCALE + _TXV_SHIFT
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()

    # 3. Torch Tensor
    if isinstance(image, torch.Tensor):
        if image.numel() == 0:
            raise ValueError("Input image tensor cannot be empty.")
        if torch.isnan(image).any() or torch.isinf(image).any():
            raise ValueError("Input image tensor contains NaN or Inf.")

        # Check if already a 4D model-ready TXV tensor in [-1024, 1024]
        if (
            image.ndim == 4
            and image.shape[0] == 1
            and image.shape[1] == 1
            and image.shape[2] == target_size
            and image.shape[3] == target_size
            and float(image.min()) < -100.0
        ):
            return image.detach().clone().float()

        arr = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Input image array cannot be empty.")
        if np.isnan(image).any() or np.isinf(image).any():
            raise ValueError("Input image array contains NaN or Inf.")

        # Check if already a 4D model-ready TXV array in [-1024, 1024]
        if (
            image.ndim == 4
            and image.shape[0] == 1
            and image.shape[1] == 1
            and image.shape[2] == target_size
            and image.shape[3] == target_size
            and float(image.min()) < -100.0
        ):
            return torch.from_numpy(image.copy()).float()

        arr = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    # Process numpy array to 2D grayscale in [0, 255]
    # Squeeze batch / channel dims if present
    while arr.ndim > 2 and (arr.shape[0] == 1 or arr.shape[-1] == 1):
        arr = np.squeeze(arr)

    # If RGB/RGBA, convert to grayscale
    if arr.ndim == 3:
        if arr.shape[2] in (3, 4):
            arr = np.mean(arr[:, :, :3], axis=2)
        elif arr.shape[0] in (3, 4):
            arr = np.mean(arr[:3, :, :], axis=0)
        else:
            arr = arr[:, :, 0]

    if arr.ndim != 2:
        raise ValueError(f"Cannot reduce image array with shape {image.shape} to 2D grayscale.")

    # Check value range and standardize to [0.0, 255.0]
    arr_min, arr_max = float(np.min(arr)), float(np.max(arr))
    if arr.dtype == np.uint8:
        arr_f32 = arr.astype(np.float32)
    elif 0.0 <= arr_min and arr_max <= 1.0:
        arr_f32 = (arr * 255.0).astype(np.float32)
    elif arr_min < -100.0 and arr_max <= 1024.0:
        # Array is already in TXV HU range [-1024, 1024] but might need resizing
        # Invert TXV formula: pixel_uint8 = (pixel_txv + 1024) * (255 / 2048)
        arr_f32 = np.clip((arr + 1024.0) * (255.0 / 2048.0), 0.0, 255.0).astype(np.float32)
    else:
        arr_f32 = np.clip(arr, 0.0, 255.0).astype(np.float32)

    # Resize using Pillow with LANCZOS to match load_image_for_txv exactly
    if arr_f32.shape != (target_size, target_size):
        pil_img = Image.fromarray(arr_f32.astype(np.uint8), mode="L")
        pil_img = pil_img.resize((target_size, target_size), Image.LANCZOS)
        arr_f32 = np.array(pil_img, dtype=np.float32)

    # Apply TXV normalization
    arr_txv = arr_f32 * _TXV_SCALE + _TXV_SHIFT
    return torch.from_numpy(arr_txv).unsqueeze(0).unsqueeze(0).float()



def image_info(image_path: Path | str) -> dict:
    """
    Return basic metadata about an image without running inference.

    Returns dict with: path, exists, format, mode, size_px, file_size_bytes
    """
    path = Path(image_path)
    info: dict = {
        "path": str(path),
        "exists": path.exists(),
        "file_size_bytes": path.stat().st_size if path.exists() else None,
        "format": None,
        "mode": None,
        "size_px": None,
    }
    if path.exists():
        try:
            with Image.open(path) as img:
                info["format"] = img.format
                info["mode"] = img.mode
                info["size_px"] = img.size  # (width, height)
        except Exception:
            pass
    return info
