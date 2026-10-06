"""Model factory — creates and caches model agents (Phase 4).

Responsibility:
    Provide a single entry point for creating model agents without
    scattering model IDs and device-selection logic across the codebase.

    Designed to support both the FAST model (Phase 4) and the ESCALATION
    model (Phase 5) without requiring the caller to know which class to
    instantiate.

Supported models:
    FAST        : densenet121-res224-nih   (DenseNet121, 14-class NIH)
    ESCALATION  : resnet50-res512-all      (ResNet50, multi-dataset) [Phase 5]

Device selection:
    Automatically uses CUDA if available; falls back to CPU.
    Can be overridden via the device parameter or CXR_DEVICE in settings.

Dependencies:
    torch, models.base_model, config.settings

Implementation phase: P4
"""

from __future__ import annotations

from pathlib import Path

import torch

from cxr_reliability.models.base_model import BaseModelAgent
from cxr_reliability.models.feature_hook import FEATURE_LAYER_NAME

# Registered model IDs and their metadata
_MODEL_REGISTRY: dict[str, dict] = {
    "densenet121-res224-nih": {
        "role": "fast",
        "class": "DenseNet",
        "target_size": 224,
        "feature_dim": 1024,
        "description": "DenseNet-121 pretrained on NIH ChestX-ray14 (14 classes)",
    },
    "resnet50-res512-all": {
        "role": "escalation",
        "class": "ResNet",
        "target_size": 512,
        "feature_dim": 2048,
        "description": "ResNet-50 pretrained on NIH+CheXpert+PadChest+MIMIC",
    },
}


def detect_device() -> str:
    """
    Return 'cuda' if a CUDA GPU is available, otherwise 'cpu'.
    Prints device information to stdout.
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Device       : {device.upper()}")
        print("  CUDA available: True")
        print(f"  GPU name     : {gpu_name}")
    else:
        device = "cpu"
        print("  Device       : CPU")
        print("  CUDA available: False")
    return device


def get_model(
    model_id: str = "densenet121-res224-nih",
    device: str | None = None,
    weights_dir: Path | None = None,
    feature_layer: str | None = None,
    load: bool = True,
) -> BaseModelAgent:
    """
    Create and (optionally) load a BaseModelAgent.

    Parameters
    ----------
    model_id     : model identifier (see _MODEL_REGISTRY)
    device       : 'cpu', 'cuda', or None (auto-detect)
    weights_dir  : override TXV weight cache directory
    feature_layer: layer name for feature extraction (default: features.norm5)
    load         : if True, call load_model() before returning

    Returns
    -------
    BaseModelAgent — ready for inference if load=True

    Notes
    -----
    The ESCALATION model (resnet50-res512-all) is registered but its
    full implementation is deferred to Phase 5. Calling get_model() with
    the escalation ID will raise NotImplementedError until Phase 5.
    """
    if model_id not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_id '{model_id}'. "
            f"Registered models: {list(_MODEL_REGISTRY.keys())}"
        )

    meta = _MODEL_REGISTRY[model_id]

    if meta["role"] == "escalation":
        raise NotImplementedError(
            f"The escalation model '{model_id}' is registered but not yet "
            f"implemented. It will be built in Phase 5."
        )

    resolved_device = device or detect_device()

    agent = BaseModelAgent(
        model_id=model_id,
        target_pathology="Pneumonia",
        feature_layer=feature_layer or FEATURE_LAYER_NAME,
        weights_dir=weights_dir,
        device=resolved_device,
    )

    if load:
        agent.load_model()

    return agent


def list_models() -> list[dict]:
    """Return metadata for all registered models."""
    return [
        {"model_id": mid, **meta}
        for mid, meta in _MODEL_REGISTRY.items()
    ]
