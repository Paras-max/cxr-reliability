"""Pneumonia calibration workflow (Phase 4.5).

Orchestrates:
    Part 1 — Run Base Model on validation.csv, cache predictions
    Part 2 — Threshold analysis (200 candidates, 3 strategies)
    Part 3 — Probability calibration (Platt + Isotonic)
    Part 4 — Save calibration_results.json
    Part 5 — Print summary to terminal

Usage:
    python scripts/calibrate_pneumonia.py
    python scripts/calibrate_pneumonia.py --force-recompute
    python scripts/calibrate_pneumonia.py --strategy youden

IMPORTANT — DATA LEAKAGE POLICY:
    All fitting uses ONLY data/processed/validation.csv.
    test.csv is NEVER touched in this script.

Implementation phase: P4.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_reliability.calibration.probability_calibration import (
    ProbabilityCalibrator,
    plot_calibration_curve,
)
from cxr_reliability.calibration.threshold_calibration import (
    ThresholdAnalyzer,
    plot_threshold_analysis,
)
from cxr_reliability.config.settings import get_settings
from cxr_reliability.models.model_factory import get_model
from cxr_reliability.models.preprocessing import load_image_for_txv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  PART 1 — Validation Predictions
# ═══════════════════════════════════════════════════════════════════════

def run_validation_inference(
    val_csv_path: Path,
    dataset_root: Path,
    output_pred_csv: Path,
    max_samples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run Base Model on validation.csv images (or a sampled subset).

    Saves predictions to output_pred_csv with columns:
        image_id, patient_id, ground_truth, raw_pneumonia_score

    The model is loaded once and each image is processed sequentially
    through BaseModelAgent.run().
    """
    logger.info("Loading validation metadata from %s", val_csv_path)
    df = pd.read_csv(val_csv_path)
    if max_samples is not None and max_samples < len(df):
        logger.info(
            "Smoke test mode: sampling %d images (seed=%d) from %d total",
            max_samples, seed, len(df),
        )
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    n_total = len(df)
    logger.info("Validation set: %d images", n_total)

    # ── Load model ────────────────────────────────────────────────────
    logger.info("Loading Base Model (densenet121-res224-nih)...")
    agent = get_model(model_id="densenet121-res224-nih", load=True)
    logger.info("Model loaded on device=%s", agent.device)

    # ── Iterate ───────────────────────────────────────────────────────
    results: list[dict] = []
    t_start = time.time()
    errors = 0
    prog_step = 50 if n_total <= 1000 else 500

    for idx, row in df.iterrows():
        image_rel_path = row["image_path"]  # e.g. images_001\images\00000001_000.png
        image_full_path = dataset_root / image_rel_path
        ground_truth = int(row["label"])     # 1 = Pneumonia, 0 = Non-Pneumonia

        try:
            tensor = load_image_for_txv(image_full_path)
            fwd = agent.run(tensor)
            raw_score = fwd.result.pneumonia_probability
        except Exception as exc:
            logger.warning("Error on image %s: %s", row["image_id"], exc)
            raw_score = float("nan")
            errors += 1

        results.append({
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "ground_truth": ground_truth,
            "raw_pneumonia_score": float(raw_score),
        })

        # Progress reporting
        done = idx + 1  # type: ignore[operator]
        if done % prog_step == 0 or done == n_total:
            elapsed = time.time() - t_start
            speed = done / elapsed if elapsed > 0 else 0
            eta_s = (n_total - done) / speed if speed > 0 else 0
            logger.info(
                "  [%d/%d] %.1f%%  |  %.1f img/s  |  ETA %dm %ds  |  errors=%d",
                done, n_total, (done / n_total) * 100,
                speed, int(eta_s // 60), int(eta_s % 60), errors,
            )

    pred_df = pd.DataFrame(results)
    output_pred_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_pred_csv, index=False)
    elapsed = time.time() - t_start
    logger.info(
        "Inference complete: %d images in %.1fs (%.1f img/s), %d errors",
        n_total, elapsed, n_total / elapsed if elapsed > 0 else 0, errors,
    )
    logger.info("Saved: %s", output_pred_csv)
    return pred_df


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Pneumonia Calibration (Phase 4.5)")
    parser.add_argument(
        "--force-recompute", action="store_true",
        help="Force re-inference even if validation_predictions.csv exists",
    )
    parser.add_argument(
        "--strategy", type=str, default="f1_optimal",
        choices=["f1_optimal", "youden", "balanced_accuracy"],
        help="Threshold selection strategy (default: f1_optimal)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of validation images for smoke testing (e.g. 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling when --max-samples is set (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: outputs/calibration, or outputs/calibration_smoke_test if --max-samples is set)",
    )
    args = parser.parse_args()

    settings = get_settings()
    dataset_root = settings.dataset_root
    val_csv_path = settings.data_dir / "processed" / "validation.csv"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.max_samples:
        output_dir = settings.outputs_dir / "calibration_smoke_test"
    else:
        output_dir = settings.outputs_dir / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_csv_path = output_dir / "validation_predictions.csv"

    # ── PART 1: Validation Predictions (with caching) ─────────────────
    if pred_csv_path.exists() and not args.force_recompute:
        logger.info(
            "Found cached predictions at %s — loading (use --force-recompute to re-run)",
            pred_csv_path,
        )
        pred_df = pd.read_csv(pred_csv_path)
    else:
        pred_df = run_validation_inference(
            val_csv_path=val_csv_path,
            dataset_root=dataset_root,
            output_pred_csv=pred_csv_path,
            max_samples=args.max_samples,
            seed=args.seed,
        )

    # Drop any NaN rows from errors
    n_before = len(pred_df)
    pred_df = pred_df.dropna(subset=["raw_pneumonia_score"])
    n_dropped = n_before - len(pred_df)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with NaN scores (inference errors)", n_dropped)

    y_true = pred_df["ground_truth"].values.astype(int)
    y_score = pred_df["raw_pneumonia_score"].values.astype(float)

    # ── PART 2: Threshold Analysis ────────────────────────────────────
    logger.info("Running threshold analysis (strategy=%s)...", args.strategy)
    analyzer = ThresholdAnalyzer(strategy=args.strategy)
    threshold_result = analyzer.fit(y_true, y_score)

    threshold_df_path = output_dir / "threshold_results.csv"
    threshold_result.thresholds_df.to_csv(threshold_df_path, index=False)
    logger.info("Saved: %s", threshold_df_path)

    threshold_plot_path = output_dir / "threshold_analysis.png"
    plot_threshold_analysis(
        result=threshold_result,
        output_path=threshold_plot_path,
        selected_threshold=threshold_result.project_validation_threshold,
    )

    # ── PART 3: Probability Calibration ───────────────────────────────
    logger.info("Fitting probability calibrators (Platt scaling + Isotonic)...")
    calibrator = ProbabilityCalibrator()
    calib_result = calibrator.fit(y_true, y_score)
    calibrator.save(output_dir)
    logger.info("Saved: platt_calibrator.joblib, isotonic_calibrator.joblib")

    calib_plot_path = output_dir / "calibration_curve.png"
    plot_calibration_curve(result=calib_result, output_path=calib_plot_path)

    # ── PART 4: Save calibration_results.json ─────────────────────────
    raw_stats = {
        "mean":   float(np.mean(y_score)),
        "std":    float(np.std(y_score)),
        "min":    float(np.min(y_score)),
        "max":    float(np.max(y_score)),
        "median": float(np.median(y_score)),
        "p25":    float(np.percentile(y_score, 25)),
        "p75":    float(np.percentile(y_score, 75)),
        "p95":    float(np.percentile(y_score, 95)),
    }

    summary = {
        "raw_score_statistics": raw_stats,
        "threshold_candidates": threshold_result.candidates,
        "selected_threshold": threshold_result.project_validation_threshold,
        "selection_method": threshold_result.selected_strategy,
        "roc_auc": threshold_result.roc_auc,
        "pr_auc": threshold_result.pr_auc,
        "raw_ECE": calib_result.raw.ece,
        "calibrated_ECE": calib_result.platt.ece,
        "isotonic_ECE": calib_result.isotonic.ece,
        "raw_Brier": calib_result.raw.brier_score,
        "calibrated_Brier": calib_result.platt.brier_score,
        "isotonic_Brier": calib_result.isotonic.brier_score,
        "calibration_method": "platt",
        "number_of_validation_images": threshold_result.n_total,
        "number_of_positive_cases": threshold_result.n_positive,
        "number_of_negative_cases": threshold_result.n_negative,
        "is_smoke_test": args.max_samples is not None,
    }

    json_path = output_dir / "calibration_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s", json_path)

    # ── PART 5: Terminal Summary ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("   PNEUMONIA BASE MODEL CALIBRATION SUMMARY  (Phase 4.5)")
    print("=" * 70)
    print(f"  Validation Images        : {threshold_result.n_total:,}")
    print(f"    Pneumonia (Positive)   : {threshold_result.n_positive:,}  "
          f"({threshold_result.n_positive / threshold_result.n_total * 100:.2f}%)")
    print(f"    Non-Pneumonia (Neg.)   : {threshold_result.n_negative:,}")
    print("-" * 70)
    print("  GLOBAL PERFORMANCE")
    print(f"    ROC-AUC                : {threshold_result.roc_auc:.4f}")
    print(f"    PR-AUC                 : {threshold_result.pr_auc:.4f}")
    print("-" * 70)
    print("  RAW SCORE STATISTICS")
    print(f"    Mean / Median          : {raw_stats['mean']:.4f} / {raw_stats['median']:.4f}")
    print(f"    Min / Max              : {raw_stats['min']:.4f} / {raw_stats['max']:.4f}")
    print(f"    P25 / P75 / P95        : {raw_stats['p25']:.4f} / {raw_stats['p75']:.4f} / {raw_stats['p95']:.4f}")
    print("-" * 70)
    print("  THRESHOLD CANDIDATES")
    for strat, cand in threshold_result.candidates.items():
        sel = "  ◀ SELECTED" if strat == threshold_result.selected_strategy else ""
        print(f"    {strat:<20}: thr={cand['threshold']:.4f}  "
              f"F1={cand['f1']:.4f}  P={cand.get('precision', 0):.4f}  "
              f"R={cand['recall']:.4f}{sel}")
    print("-" * 70)
    sel_thr = threshold_result.project_validation_threshold
    print(f"  project_validation_threshold = {sel_thr:.4f}  ({threshold_result.selected_strategy})")
    print("-" * 70)
    print("  CALIBRATION RESULTS")
    print(f"    Raw Scores             : ECE={calib_result.raw.ece:.4f}  "
          f"Brier={calib_result.raw.brier_score:.4f}")
    print(f"    Platt Scaling          : ECE={calib_result.platt.ece:.4f}  "
          f"Brier={calib_result.platt.brier_score:.4f}")
    print(f"    Isotonic Regression    : ECE={calib_result.isotonic.ece:.4f}  "
          f"Brier={calib_result.isotonic.brier_score:.4f}")
    print("=" * 70)
    print("  Output files in: outputs/calibration/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
