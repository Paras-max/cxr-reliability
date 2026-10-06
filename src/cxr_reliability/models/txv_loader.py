"""TorchXRayVision weight loading (Phase 4).

Responsibility:
    Load a pretrained TorchXRayVision model by ID, place it in eval mode on the
    requested device, and return it with a SHA-256 of the weight file for
    audit/reproducibility purposes.

Weight cache location:
    TorchXRayVision stores downloaded weights in:
        ~/.torchxrayvision/models_data/   (Linux/macOS)
        C:\\Users\\<user>\\.torchxrayvision\\models_data\\  (Windows)

    After the first download, subsequent loads are instant (no network needed).

Supported model IDs (Phase 4 implements fast; escalation stub left for Phase 5):
    densenet121-res224-nih      — FAST model (14-class NIH-trained DenseNet121)
    resnet50-res512-all         — ESCALATION model (stub, Phase 5)

Dependencies:
    torch, torchxrayvision

Implementation phase: P4
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
import torchxrayvision as xrv

# Default cache directory used by TorchXRayVision
_TXV_CACHE_DIR = Path.home() / ".torchxrayvision" / "models_data"

# Mapping model_id -> TorchXRayVision weights string
_MODEL_WEIGHTS_MAP: dict[str, str] = {
    "densenet121-res224-nih": "densenet121-res224-nih",
    "resnet50-res512-all": "resnet50-res512-all",
}

# Mapping model_id -> expected local weight filename (for hash verification)
_WEIGHT_FILENAMES: dict[str, str] = {
    "densenet121-res224-nih": (
        "nih-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
    ),
    "resnet50-res512-all": "resnet50-res512-all-auc84.pt",
}


def load_txv_model(
    model_id: str,
    weights_dir: Path | None = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Load a pretrained TorchXRayVision model in eval mode.

    Parameters
    ----------
    model_id    : one of the supported model IDs (see _MODEL_WEIGHTS_MAP)
    weights_dir : override the TXV cache directory (None = use default)
    device      : 'cpu' or 'cuda' (or 'cuda:0', etc.)

    Returns
    -------
    torch.nn.Module in eval mode, moved to the requested device

    Notes
    -----
    On first call TorchXRayVision downloads the weights (~90 MB for DenseNet121)
    from GitHub releases and caches them in ~/.torchxrayvision/models_data/.
    Subsequent calls are instant.

    If the download is interrupted, delete the partial .pt file from the cache
    directory and retry.
    """
    if model_id not in _MODEL_WEIGHTS_MAP:
        raise ValueError(
            f"Unsupported model_id '{model_id}'. "
            f"Supported: {list(_MODEL_WEIGHTS_MAP.keys())}"
        )

    # Point TXV at a custom cache dir if provided
    cache_dir = weights_dir or _TXV_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    weights_key = _MODEL_WEIGHTS_MAP[model_id]

    # Set Windows console encoding to avoid TXV's Unicode progress-bar crash
    if os.name == "nt":
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    try:
        if "densenet" in model_id:
            model = xrv.models.DenseNet(
                weights=weights_key,
                cache_dir=str(cache_dir),
            )
        elif "resnet" in model_id:
            model = xrv.models.ResNet(
                weights=weights_key,
                cache_dir=str(cache_dir),
            )
        else:
            raise ValueError(f"Unknown architecture for model_id '{model_id}'")
    except RuntimeError as exc:
        weight_file = cache_dir / _WEIGHT_FILENAMES.get(model_id, "")
        if weight_file.exists() and weight_file.stat().st_size < 1_000_000:
            raise RuntimeError(
                f"Weight file appears corrupt or incomplete: {weight_file}\n"
                f"Delete it and re-run to trigger a fresh download.\n"
                f"Original error: {exc}"
            ) from exc
        raise RuntimeError(
            f"Failed to load model '{model_id}': {exc}\n"
            f"Ensure network access for the first download, or manually place "
            f"the weight file at: {cache_dir}"
        ) from exc

    model.eval()
    dev = torch.device(device)
    model = model.to(dev)
    return model


def weights_sha256(model_id: str, weights_dir: Path | None = None) -> str:
    """
    Compute the SHA-256 hex digest of the cached weight file.

    Returns empty string if the file does not exist (e.g. before first download).
    """
    cache_dir = weights_dir or _TXV_CACHE_DIR
    filename = _WEIGHT_FILENAMES.get(model_id, "")
    if not filename:
        return ""
    weight_path = cache_dir / filename
    if not weight_path.exists():
        return ""
    sha = hashlib.sha256()
    with open(weight_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()
