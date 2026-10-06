"""
scripts/test_base_model.py
===========================
Phase 4 — Base Model Integration Test

What this script does
---------------------
1. Detects device (CUDA or CPU).
2. Loads the pretrained densenet121-res224-nih model.
3. Prints model information (labels, architecture, weights).
4. Runs inference on ONE real image from test.csv.
5. Prints the required output block.
6. Runs inference on 10 real test images (sanity check).
7. Shows batch results — image ID, ground-truth label, raw score, time.

IMPORTANT:
    - This script uses ACTUAL images from the NIH dataset.
    - Pneumonia raw scores are NOT calibrated probabilities.
    - Do NOT use 5-10 image results to claim model accuracy.
    - This is an inference sanity check only.

RESEARCH DISCLAIMER:
    This is a research prototype. Not validated for clinical use.

Usage:
    python scripts/test_base_model.py

Configuration:
    CXR_DATASET_ROOT  — path to the NIH dataset (in .env)
    CXR_DATA_DIR      — location of data/processed/test.csv
    CXR_DEVICE        — 'cpu' or 'cuda' (or auto-detect)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ── Bootstrap ────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from cxr_reliability.config.settings import get_settings
    _s = get_settings()
    _DATASET_ROOT = (
        _s.dataset_root if _s.dataset_root.is_absolute()
        else (_PROJECT_ROOT / _s.dataset_root).resolve()
    )
    _DATA_DIR = (
        _s.data_dir if _s.data_dir.is_absolute()
        else (_PROJECT_ROOT / _s.data_dir).resolve()
    )
    _DEVICE_OVERRIDE = _s.device  # 'cpu' or 'cuda' from .env
except Exception as _e:
    import os
    print(f"[WARN] Could not load settings ({_e}). Using defaults.")
    _DATASET_ROOT = (_PROJECT_ROOT / "dataset").resolve()
    _DATA_DIR = (_PROJECT_ROOT / "data").resolve()
    _DEVICE_OVERRIDE = os.environ.get("CXR_DEVICE", None)

_TEST_CSV = _DATA_DIR / "processed" / "test.csv"
_N_SANITY_IMAGES = 10


def _resolve_image_path(rel_path: str) -> Path:
    """Resolve a relative image_path from the CSV to an absolute path."""
    return _DATASET_ROOT / rel_path


def main() -> None:
    import pandas as pd

    from cxr_reliability.models.model_factory import detect_device, get_model
    from cxr_reliability.models.preprocessing import load_image_for_txv

    print()
    print("=" * 60)
    print("  BASE MODEL TEST  —  Phase 4")
    print("=" * 60)
    print()

    # ── Validate inputs ──────────────────────────────────────────────────
    if not _TEST_CSV.exists():
        print(f"[ERROR] test.csv not found: {_TEST_CSV}")
        print("        Run scripts/create_splits.py first.")
        sys.exit(1)
    if not _DATASET_ROOT.exists():
        print(f"[ERROR] Dataset root not found: {_DATASET_ROOT}")
        sys.exit(1)

    # ── Device detection ─────────────────────────────────────────────────
    print("DEVICE INFORMATION")
    print("-" * 40)
    import torch
    if _DEVICE_OVERRIDE and _DEVICE_OVERRIDE != "cpu":
        device = _DEVICE_OVERRIDE
        print(f"  Device       : {device.upper()} (from config)")
        cuda_ok = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_ok}")
        if cuda_ok:
            print(f"  GPU name     : {torch.cuda.get_device_name(0)}")
    else:
        device = detect_device()
    print()

    # ── Load model ───────────────────────────────────────────────────────
    print("Loading model (first run downloads weights ~28 MB)...")
    t0 = time.perf_counter()
    agent = get_model(model_id="densenet121-res224-nih", device=device, load=True)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Model loaded in {load_ms:.0f} ms")
    print()

    # ── Model information ─────────────────────────────────────────────────
    info = agent.model_info()
    print("MODEL INFORMATION")
    print("-" * 40)
    print(f"  Model name       : {info['model_name']}")
    print(f"  Architecture     : {info['model_class']}")
    print(f"  Pretrained on    : {info['txv_training_dataset']}")
    print(f"  Output classes   : {info['num_output_classes']}")
    print(f"  Pathology labels : {[lbl for lbl in info['output_labels'] if lbl]}")
    print(f"  Target pathology : {info['target_pathology']} (index {info['target_pathology_index']})")
    print(f"  Output type      : {info['output_type']}")
    print(f"  Input size       : {info['input_size']}")
    print(f"  Input norm       : {info['input_normalization']}")
    print(f"  Feature layer    : {info['feature_layer']}")
    print(f"  Feature dim      : {info['feature_dim']}")
    print(f"  TXV version      : {info['torchxrayvision_version']}")
    print(f"  PyTorch version  : {info['pytorch_version']}")
    print(f"  Weights SHA-256  : {info['weights_sha256'][:32]}..." if info['weights_sha256'] != 'not_loaded' else "  Weights SHA-256  : not computed")
    print()

    # ── Load test.csv ────────────────────────────────────────────────────
    test_df = pd.read_csv(_TEST_CSV)
    print(f"Loaded test.csv — {len(test_df):,} images")
    print()

    # ── SINGLE IMAGE TEST ────────────────────────────────────────────────
    # Select one Pneumonia image and one Non-Pneumonia for clear demonstration
    pneu_rows = test_df[test_df["label"] == 1]
    non_pneu_rows = test_df[test_df["label"] == 0]
    sample_row = pneu_rows.iloc[0] if not pneu_rows.empty else test_df.iloc[0]

    img_path = _resolve_image_path(sample_row["image_path"])
    tensor = load_image_for_txv(img_path)
    fwd = agent.run(tensor)

    print("=" * 60)
    print("  BASE MODEL TEST  —  Single Image")
    print("=" * 60)
    print(f"  Image          : {sample_row['image_id']}")
    print(f"  Image path     : {img_path}")
    print(f"  Ground-truth   : {sample_row['label_name']} (label={sample_row['label']})")
    print(f"  Model          : {agent.model_id}")
    print(f"  Device         : {device}")
    print(f"  Input shape    : {tuple(tensor.shape)}")
    print(f"  Prediction     : {fwd.result.label}")
    print(f"  Pneumonia raw score : {fwd.result.pneumonia_probability:.6f}")
    print(f"  Raw output shape    : {tuple(fwd.raw_probs.shape)}")
    print(f"  Feature vector shape: {tuple(fwd.features.shape)}")
    print(f"  Inference time      : {fwd.inference_ms:.1f} ms")
    print("=" * 60)
    print()

    # ── TOP-K RAW OUTPUTS ─────────────────────────────────────────────────
    print("ALL MODEL OUTPUTS (sorted by raw score):")
    print("-" * 40)
    sorted_outputs = sorted(
        fwd.result.all_pathology_outputs.items(),
        key=lambda x: x[1], reverse=True
    )
    for label, score in sorted_outputs:
        marker = " <-- TARGET" if label == "Pneumonia" else ""
        print(f"  {label:<22s}: {score:.4f}{marker}")
    print()

    # ── SANITY CHECK: 10 IMAGES ────────────────────────────────────────────
    print(f"SANITY CHECK: {_N_SANITY_IMAGES} images from test.csv")
    print("NOTE: This is an inference check only. Do NOT interpret as accuracy.")
    print("-" * 70)

    # Mix of Pneumonia and Non-Pneumonia
    n_pneu = min(3, len(pneu_rows))
    n_non = _N_SANITY_IMAGES - n_pneu
    sample_pneu = pneu_rows.head(n_pneu)
    sample_non = non_pneu_rows.head(n_non)
    sample_batch = pd.concat([sample_pneu, sample_non]).reset_index(drop=True)

    print(f"  {'Image ID':<30} {'GT Label':<15} {'Raw Score':>10} {'Pred':>15} {'ms':>8}")
    print("  " + "-" * 84)

    for _, row in sample_batch.iterrows():
        try:
            path = _resolve_image_path(row["image_path"])
            t = load_image_for_txv(path)
            fwd_i = agent.run(t)
            score = fwd_i.result.pneumonia_probability
            pred = fwd_i.result.label
            ms_str = f"{fwd_i.inference_ms:.1f}"
        except Exception as exc:
            score, pred, ms_str = None, f"ERROR: {exc}", "N/A"

        gt_label = row.get("label_name", str(row.get("label", "?")))
        score_str = f"{score:.4f}" if score is not None else "ERROR"
        print(f"  {row['image_id']:<30} {gt_label:<15} {score_str:>10} {str(pred):>15} {ms_str:>8}")

    print()
    print("IMPORTANT: The above is an inference sanity check only.")
    print("  Do NOT calculate accuracy from 10 images.")
    print("  Proper evaluation will be done in Phase 5 on the full test set.")
    print()
    print("Phase 4 base model test complete.")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(1)
    except Exception as exc:
        import traceback
        print(f"\n[FATAL ERROR] {exc}")
        traceback.print_exc()
        sys.exit(2)
