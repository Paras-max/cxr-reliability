"""tests/unit/test_ood_dev_reference.py
Unit tests for the configurable development OOD reference mode.

Verifies:
1. Deterministic subset selection (seed reproducibility and uniqueness).
2. Scientific constraint: training-only fitting; test split strictly prohibited.
3. Separate development output directory (artifacts/ood_dev).
4. Metadata explicitly identifies development/provisional status and sample count.
5. OODAgent recognizes development mode and exposes metadata.
6. Dashboard discovers and loads development OOD artifact cleanly.
7. Full research OOD configuration remains completely available and default when not in dev mode.
8. End-to-end dev fitting smoke test with synthetic inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
for p in (str(_SRC), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fit_ood_reference import _load_split_image_ids, _parse_args, select_deterministic_subset

from cxr_reliability.agents.ood import OODAgent
from cxr_reliability.calibration.ood_fit import fit_gaussian_stats
from cxr_reliability.config.thresholds import OODThresholds
from cxr_reliability.dashboard.app import discover_available_ood_stats, resolve_ood_stats_path

# ── 1. Deterministic Subset Selection ─────────────────────────────────────────


def test_deterministic_subset_selection():
    """Test 1: Subset selection must be strictly deterministic with fixed seed."""
    image_ids = [f"img_{i:04d}.png" for i in range(100)]

    # Same seed -> identical results
    sub1 = select_deterministic_subset(image_ids, n_samples=25, seed=42)
    sub2 = select_deterministic_subset(image_ids, n_samples=25, seed=42)
    assert sub1 == sub2
    assert len(sub1) == 25
    assert len(set(sub1)) == 25  # No duplicates

    # Different seed -> different selection
    sub3 = select_deterministic_subset(image_ids, n_samples=25, seed=999)
    assert sub1 != sub3
    assert len(sub3) == 25

    # Target >= len -> returns all
    sub_all = select_deterministic_subset(image_ids, n_samples=150, seed=42)
    assert sub_all == image_ids

    # None target -> returns all
    sub_none = select_deterministic_subset(image_ids, n_samples=None, seed=42)
    assert sub_none == image_ids


# ── 2. Scientific Constraint: Training Only (No Test) ────────────────────────


def test_training_only_manifest_constraint(tmp_path: Path):
    """Test 2: Test split must be strictly prohibited from OOD reference fitting."""
    # Attempting to request test split raises PermissionError
    with pytest.raises(PermissionError, match="SCIENTIFIC CONSTRAINT VIOLATION"):
        _load_split_image_ids(tmp_path, "test")

    with pytest.raises(PermissionError, match="SCIENTIFIC CONSTRAINT VIOLATION"):
        _load_split_image_ids(tmp_path, "TEST")


# ── 3. Separate Development Output Directory ──────────────────────────────────


def test_separate_development_output_directory():
    """Test 3: Dev mode defaults to artifacts/ood_dev without overriding full research directory."""
    with patch("sys.argv", ["fit_ood_reference.py", "--dev"]):
        args_dev = _parse_args()
        assert args_dev.dev is True
        # In main(), when args.dev is True and output_dir is "artifacts/ood", it defaults to artifacts/ood_dev
        target_dir = Path("artifacts/ood_dev") if (args_dev.dev and args_dev.output_dir == "artifacts/ood") else Path(args_dev.output_dir)
        assert target_dir == Path("artifacts/ood_dev")

    # When not in dev mode, default is artifacts/ood
    with patch("sys.argv", ["fit_ood_reference.py"]):
        args_full = _parse_args()
        assert args_full.dev is False
        assert args_full.output_dir == "artifacts/ood"

    # When explicit output_dir passed in dev mode, custom path is respected
    with patch("sys.argv", ["fit_ood_reference.py", "--dev", "--output-dir", "custom/dev_dir"]):
        args_custom = _parse_args()
        assert args_custom.dev is True
        assert args_custom.output_dir == "custom/dev_dir"


# ── 4. Metadata Identifies Development Mode ───────────────────────────────────


def test_metadata_identifies_development_mode(tmp_path: Path):
    """Test 4: Metadata must record development status, sample count, and provisional disclaimer."""
    feat = np.random.randn(20, 1024).astype(np.float32)
    dev_meta = {
        "is_development": True,
        "mode": "development",
        "status": "provisional_development",
        "n_train_samples": 20,
        "n_train_images_total_available": 78299,
        "sampling_seed": 42,
        "disclaimer": "DEVELOPMENT / PROVISIONAL reference statistics fitted on a subset.",
    }

    _stats = fit_gaussian_stats(
        features=feat,
        output_path=tmp_path,
        lambda_reg=1e-5,
        model_id="densenet121-res224-nih",
        split="train",
        extra_metadata=dev_meta,
    )

    meta_file = tmp_path / "metadata.json"
    assert meta_file.exists()

    with open(meta_file, encoding="utf-8") as fh:
        saved = json.load(fh)

    assert saved["is_development"] is True
    assert saved["status"] == "provisional_development"
    assert saved["n_train_samples"] == 20
    assert saved["sampling_seed"] == 42
    assert "PROVISIONAL" in saved["disclaimer"]


# ── 5. OODAgent Recognizes Development Mode ───────────────────────────────────


def test_ood_agent_detects_development_mode(tmp_path: Path):
    """Test 5: OODAgent exposes is_development_mode property based on metadata and path."""
    meta_dev = {"is_development": True, "n_train_samples": 500}
    (tmp_path / "metadata.json").write_text(json.dumps(meta_dev), encoding="utf-8")
    (tmp_path / "reference_stats.npz").write_bytes(b"dummy")

    thresholds = OODThresholds(in_distribution_percentile=95.0)
    agent_dev = OODAgent(stats_dir=tmp_path, thresholds=thresholds, thresholds_version="v0_test")

    assert agent_dev.is_stats_available is True
    assert agent_dev.is_development_mode is True
    assert agent_dev.get_stats_metadata()["n_train_samples"] == 500

    # Non-dev directory
    non_dev_path = tmp_path / "full_ref"
    non_dev_path.mkdir()
    meta_full = {"is_development": False, "n_train_samples": 78299}
    (non_dev_path / "metadata.json").write_text(json.dumps(meta_full), encoding="utf-8")
    (non_dev_path / "reference_stats.npz").write_bytes(b"dummy")

    agent_full = OODAgent(stats_dir=non_dev_path, thresholds=thresholds, thresholds_version="v0_test")
    assert agent_full.is_development_mode is False


# ── 6. Dashboard Discovers and Loads Development OOD Artifact ────────────────


def test_dashboard_discovers_and_loads_dev_ood_artifact(tmp_path: Path):
    """Test 6: Dashboard resolves development artifact path and passes metadata."""
    dev_dir = tmp_path / "ood_dev"
    dev_dir.mkdir()
    (dev_dir / "reference_stats.npz").write_bytes(b"dummy")
    meta = {"is_development": True, "n_train_samples": 2000}
    (dev_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    with patch("cxr_reliability.dashboard.app.DEV_OOD_STATS_PATH", dev_dir):
        resolved_path, info = resolve_ood_stats_path(dev_dir)
        assert resolved_path == dev_dir
        assert info["is_dev"] is True
        assert info["n_samples"] == 2000
        assert info["exists"] is True


# ── 7. Full Research OOD Configuration Remains Available ─────────────────────


def test_full_ood_configuration_remains_available(tmp_path: Path):
    """Test 7: When both full and dev references exist, discover_available_ood_stats returns both."""
    full_dir = tmp_path / "ood_full"
    full_dir.mkdir()
    (full_dir / "reference_stats.npz").write_bytes(b"dummy")

    dev_dir = tmp_path / "ood_dev"
    dev_dir.mkdir()
    (dev_dir / "reference_stats.npz").write_bytes(b"dummy")

    with patch("cxr_reliability.dashboard.app.FULL_OOD_STATS_PATH", full_dir), \
         patch("cxr_reliability.dashboard.app.DEV_OOD_STATS_PATH", dev_dir):
        available = discover_available_ood_stats()
        # Both full and dev are registered
        assert any("Full Research Reference" in k for k in available)
        assert any("Development Reference" in k for k in available)
        assert available["Full Research Reference (artifacts/ood)"] == full_dir
        assert available["Development Reference (artifacts/ood_dev)"] == dev_dir


# ── 8. End-to-End Dev Fitting Smoke Test ─────────────────────────────────────


def test_dev_fitting_smoke_test(tmp_path: Path):
    """Test 8: Verify end-to-end dev execution on a small mock dataset."""
    import pandas as pd
    from fit_ood_reference import main as fit_main

    # Create synthetic train and validation manifests
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_df = pd.DataFrame({"image_id": [f"train_{i}.png" for i in range(10)]})
    val_df = pd.DataFrame({"image_id": [f"val_{i}.png" for i in range(5)]})
    train_df.to_csv(data_dir / "train.csv", index=False)
    val_df.to_csv(data_dir / "validation.csv", index=False)

    out_dir = tmp_path / "artifacts" / "ood_dev"

    # Mock model and feature extraction to avoid running DenseNet in unit test
    def mock_extract(image_ids, dataset_root, model_agent, max_samples=None, log_every=500):
        n = min(len(image_ids), max_samples or len(image_ids))
        return np.random.randn(n, 1024).astype(np.float32), image_ids[:n]

    args = argparse.Namespace(
        data_dir=str(data_dir),
        dataset_root=str(tmp_path),
        output_dir=str(out_dir),
        dev=True,
        dev_train_samples=4,
        dev_val_samples=3,
        seed=123,
        device="cpu",
        lambda_reg=1e-5,
        threshold_percentile=95.0,
        max_samples=None,
        smoke_test=False,
        log_every=500,
        verbose=False,
    )

    with patch("fit_ood_reference.get_model", return_value=MagicMock()), \
         patch("fit_ood_reference.extract_features", side_effect=mock_extract):
        ret = fit_main(args)
        assert ret == 0

    assert (out_dir / "reference_stats.npz").exists()
    assert (out_dir / "metadata.json").exists()

    with open(out_dir / "metadata.json", encoding="utf-8") as fh:
        meta = json.load(fh)

    assert meta["is_development"] is True
    assert meta["status"] == "provisional_development"
    assert meta["n_train_samples"] == 4
    assert meta["n_val_images_processed"] == 3
    assert meta["sampling_seed"] == 123
    assert "PROVISIONAL" in meta["disclaimer"]
