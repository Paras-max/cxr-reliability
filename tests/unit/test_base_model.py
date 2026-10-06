"""
tests/unit/test_base_model.py
Unit tests for Phase 4 Base Model Integration.

Test coverage:
    1. model_factory.detect_device()     — returns valid device string
    2. preprocessing.load_image_for_txv — output shape, dtype, value range
    3. preprocessing.load_image_for_txv — handles missing file gracefully
    4. preprocessing.load_image_for_txv — handles unsupported format gracefully
    5. feature_hook.FeatureExtractor    — correct layer navigation
    6. base_model.PNEUMONIA_OUTPUT_INDEX — equals 8 (verified mapping)
    7. base_model.BaseModelAgent        — model_info() structure
    8. base_model.BaseModelAgent.run()  — output structure with real model
       (marked slow — skipped unless -m slow or real model is available)

Usage:
    python -m pytest tests/unit/test_base_model.py -v
    python -m pytest tests/unit/test_base_model.py -v -m slow   # includes model test
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Make src/ importable
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cxr_reliability.models.base_model import (
    PNEUMONIA_LABEL_NAME,
    PNEUMONIA_OUTPUT_INDEX,
    TXV_NIH_PATHOLOGY_LABELS,
    BaseModelAgent,
)
from cxr_reliability.models.model_factory import detect_device
from cxr_reliability.models.preprocessing import load_image_for_txv

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_png(tmp_path: Path) -> Path:
    """Create a temporary 1024x1024 grayscale PNG for testing."""
    arr = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    path = tmp_path / "test_image.png"
    img.save(path)
    return path


@pytest.fixture
def tmp_small_png(tmp_path: Path) -> Path:
    """Create a small 64x64 PNG to test resizing."""
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[10:20, 10:20] = 200
    img = Image.fromarray(arr, mode="L")
    path = tmp_path / "small.png"
    img.save(path)
    return path


# ── Test 1: Device detection ──────────────────────────────────────────────────

def test_detect_device_returns_string(capsys):
    device = detect_device()
    assert isinstance(device, str)
    assert device in ("cpu", "cuda")


def test_detect_device_matches_torch():
    device = detect_device()
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == expected


# ── Test 2: Preprocessing — correct output ────────────────────────────────────

def test_load_image_shape(tmp_png: Path):
    tensor = load_image_for_txv(tmp_png, target_size=224)
    assert tensor.shape == (1, 1, 224, 224), f"Expected (1,1,224,224), got {tensor.shape}"


def test_load_image_dtype(tmp_png: Path):
    tensor = load_image_for_txv(tmp_png)
    assert tensor.dtype == torch.float32


def test_load_image_value_range(tmp_png: Path):
    tensor = load_image_for_txv(tmp_png)
    assert tensor.min().item() >= -1024.0 - 1e-3, "Min value below -1024"
    assert tensor.max().item() <= 1024.0 + 1e-3, "Max value above 1024"


def test_load_image_normalization_endpoints(tmp_path: Path):
    """Pure black image (0) → -1024; pure white (255) → +1024."""
    black = Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L")
    white = Image.fromarray(np.full((32, 32), 255, dtype=np.uint8), mode="L")
    black_path = tmp_path / "black.png"
    white_path = tmp_path / "white.png"
    black.save(black_path)
    white.save(white_path)

    t_black = load_image_for_txv(black_path, target_size=32)
    t_white = load_image_for_txv(white_path, target_size=32)
    assert abs(t_black.min().item() - (-1024.0)) < 1.0, "Black image should map to ~-1024"
    assert abs(t_white.max().item() - 1024.0) < 1.0, "White image should map to ~+1024"


def test_load_image_resizes_correctly(tmp_small_png: Path):
    """Image smaller than target_size should be upsampled correctly."""
    tensor = load_image_for_txv(tmp_small_png, target_size=224)
    assert tensor.shape == (1, 1, 224, 224)


# ── Test 3: Preprocessing — error handling ────────────────────────────────────

def test_load_image_missing_file():
    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_image_for_txv(Path("/nonexistent/path/image.png"))


def test_load_image_unsupported_format(tmp_path: Path):
    bad = tmp_path / "image.bmp"
    bad.write_bytes(b"\x00" * 100)
    with pytest.raises(ValueError, match="Unsupported image format"):
        load_image_for_txv(bad)


def test_load_image_corrupt_file(tmp_path: Path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a valid png file at all")
    with pytest.raises(RuntimeError, match="Cannot decode|Failed to load"):
        load_image_for_txv(corrupt)


# ── Test 4: Pneumonia index mapping ──────────────────────────────────────────

def test_pneumonia_index_is_8():
    """Critical: Pneumonia must be at index 8 in densenet121-res224-nih."""
    assert PNEUMONIA_OUTPUT_INDEX == 8, (
        f"Pneumonia index changed! Expected 8, got {PNEUMONIA_OUTPUT_INDEX}. "
        "Re-probe the model architecture before changing this."
    )


def test_pneumonia_label_at_index_8():
    assert TXV_NIH_PATHOLOGY_LABELS[PNEUMONIA_OUTPUT_INDEX] == PNEUMONIA_LABEL_NAME


def test_label_list_length():
    assert len(TXV_NIH_PATHOLOGY_LABELS) == 18, (
        f"Expected 18 output labels, got {len(TXV_NIH_PATHOLOGY_LABELS)}"
    )


# ── Test 5: Feature hook ──────────────────────────────────────────────────────

def test_feature_hook_layer_navigation():
    """FeatureExtractor should find features.norm5 in a mock module."""
    import torch.nn as nn

    from cxr_reliability.models.feature_hook import FeatureExtractor

    class MockNorm(nn.Module):
        def forward(self, x): return x

    class MockFeatures(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm5 = MockNorm()

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = MockFeatures()

    model = MockModel()
    extractor = FeatureExtractor(model, "features.norm5")
    # Should not raise
    layer = extractor._get_layer("features.norm5")
    assert isinstance(layer, MockNorm)


def test_feature_hook_invalid_layer():
    import torch.nn as nn

    from cxr_reliability.models.feature_hook import FeatureExtractor

    model = nn.Linear(10, 10)
    extractor = FeatureExtractor(model, "nonexistent_layer")
    with pytest.raises(ValueError, match="not found"):
        extractor._get_layer("nonexistent_layer")


# ── Test 6: BaseModelAgent attributes ────────────────────────────────────────

def test_base_model_agent_default_params():
    agent = BaseModelAgent()
    assert agent.model_id == "densenet121-res224-nih"
    assert agent.target_pathology == "Pneumonia"
    assert agent.device == "cpu"


def test_base_model_agent_run_raises_before_load():
    agent = BaseModelAgent()
    dummy_tensor = torch.zeros(1, 1, 224, 224)
    with pytest.raises(RuntimeError, match="Model not loaded"):
        agent.run(dummy_tensor)


def test_base_model_agent_wrong_tensor_shape():
    """run() should raise ValueError for wrong tensor dimensions."""
    agent = BaseModelAgent()
    # Manually set a dummy model to bypass load check
    import torch.nn as nn
    agent._model = nn.Identity()
    bad_tensor = torch.zeros(224, 224)  # missing batch and channel dims
    with pytest.raises((ValueError, RuntimeError)):
        agent.run(bad_tensor)


# ── Test 7: Integration test (slow — uses real model) ────────────────────────

@pytest.mark.slow
def test_full_inference_with_real_model(tmp_png: Path):
    """
    Load the real pretrained model and run inference on a synthetic image.
    Requires network access on first run (downloads ~28 MB weights).
    Skip with:  pytest -m 'not slow'
    """
    agent = BaseModelAgent(device="cpu")
    agent.load_model()

    tensor = load_image_for_txv(tmp_png, target_size=224)
    fwd = agent.run(tensor)

    # Output structure checks
    assert fwd.raw_probs.shape == (18,)
    assert fwd.features.shape == (1024,)
    assert 0.0 <= fwd.result.pneumonia_probability <= 1.0
    assert hasattr(fwd.result, "raw_pneumonia_score")
    assert 0.0 <= fwd.result.raw_pneumonia_score <= 1.0
    assert fwd.result.raw_pneumonia_score == fwd.result.pneumonia_probability
    if agent.is_calibrated:
        assert fwd.result.calibrated_probability is not None
        assert 0.0 <= fwd.result.calibrated_probability <= 1.0
    assert fwd.result.label in ("Pneumonia", "Non-Pneumonia")
    assert fwd.inference_ms > 0
    assert len(fwd.result.all_pathology_outputs) == 14  # 14 named labels

    # Pneumonia must be in the output dict
    assert "Pneumonia" in fwd.result.all_pathology_outputs


@pytest.mark.slow
def test_model_info_structure():
    """Verify model_info() returns all required keys."""
    agent = BaseModelAgent(device="cpu")
    agent.load_model()
    info = agent.model_info()

    required_keys = [
        "model_name", "model_class", "pretrained_weight_id",
        "txv_training_dataset", "output_labels", "num_output_classes",
        "target_pathology", "target_pathology_index", "output_type",
        "input_size", "input_normalization", "feature_layer",
        "feature_dim", "device", "torchxrayvision_version",
        "pytorch_version", "research_disclaimer",
    ]
    for key in required_keys:
        assert key in info, f"Missing key in model_info(): '{key}'"

    assert info["target_pathology_index"] == 8
    assert info["feature_dim"] == 1024
