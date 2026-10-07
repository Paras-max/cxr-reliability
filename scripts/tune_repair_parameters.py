"""Controlled Repair Agent Parameter Tuning Experiment.

Responsibility:
    Executes a multi-parameter sweep for Blur, Noise, and Exposure repair methods:
    - Uses ONLY the patient-isolated development/validation split (data/processed/validation.csv).
    - Enforces ZERO data leakage with the frozen test split (data/processed/test.csv).
    - Runs the frozen TorchXRayVision DenseNet-121 Base Model before and after repair.
    - Evaluates objective image metrics (Laplacian, SNR, PSNR, RMSE, photometric balance).
    - Evaluates model-aware metrics (raw score delta, confidence delta, prediction stability,
      Correct->Correct, Correct->Incorrect, Incorrect->Correct, Incorrect->Incorrect).
    - Outputs:
        outputs/repair_parameter_tuning.csv
        outputs/repair_parameter_tuning_summary.json
        docs/REPAIR_AGENT_TUNING_REPORT.md

Constraints:
    - DO NOT change Base Model weights or decision thresholds.
    - DO NOT use the frozen test set for tuning.
    - DO NOT fabricate improvements.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Ensure project src is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_reliability.agents.quality import QualityAgent
from cxr_reliability.data.corruptions import apply_exposure_shift, apply_noise
from cxr_reliability.models.base_model import BaseModelAgent
from cxr_reliability.models.preprocessing import prepare_image_for_txv
from cxr_reliability.quality.blur import evaluate_blur
from cxr_reliability.quality.exposure import compute_exposure_statistics
from cxr_reliability.quality.noise import compute_snr_db, estimate_noise_std
from cxr_reliability.repair.blur import repair_blur
from cxr_reliability.repair.exposure import repair_exposure
from cxr_reliability.repair.noise import repair_noise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("repair_tuning")

OPERATING_THRESHOLD = 0.522161


def compute_psnr(orig: np.ndarray, rep: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB between two uint8 images."""
    mse = float(np.mean((orig.astype(np.float64) - rep.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * math.log10((255.0 ** 2) / mse))


def run_model_inference(model: BaseModelAgent, image_u8: np.ndarray) -> tuple[float, float, int]:
    """Run frozen model inference on a uint8 image array and return (score, confidence, prediction)."""
    tensor = prepare_image_for_txv(image_u8)
    with torch.no_grad():
        fwd = model.run(tensor)
    raw_score = float(fwd.result.raw_pneumonia_score)
    confidence = float(max(raw_score, 1.0 - raw_score))
    prediction = int(raw_score >= OPERATING_THRESHOLD)
    return raw_score, confidence, prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair Agent Parameter Sweep on Validation Split")
    parser.add_argument("--val-csv", type=str, default="data/processed/validation.csv")
    parser.add_argument("--test-csv", type=str, default="data/processed/test.csv")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--output-csv", type=str, default="outputs/repair_parameter_tuning.csv")
    parser.add_argument("--output-json", type=str, default="outputs/repair_parameter_tuning_summary.json")
    parser.add_argument("--output-report", type=str, default="docs/REPAIR_AGENT_TUNING_REPORT.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    val_csv_path = Path(args.val_csv)
    test_csv_path = Path(args.test_csv)
    dataset_dir = Path(args.dataset_dir)
    output_csv_path = Path(args.output_csv)
    output_json_path = Path(args.output_json)
    output_report_path = Path(args.output_report)

    logger.info("Initializing Repair Agent Tuning Experiment...")
    rng = np.random.RandomState(args.seed)

    # 1. Verify zero data leakage
    df_val = pd.read_csv(val_csv_path)
    df_test = pd.read_csv(test_csv_path)
    val_patients = set(df_val["patient_id"])
    test_patients = set(df_test["patient_id"])
    overlap_patients = val_patients.intersection(test_patients)
    val_images = set(df_val["image_id"])
    test_images = set(df_test["image_id"])
    overlap_images = val_images.intersection(test_images)

    logger.info("Data Leakage Check: Overlap Patients = %d, Overlap Images = %d", len(overlap_patients), len(overlap_images))
    if len(overlap_patients) > 0 or len(overlap_images) > 0:
        raise ValueError("CRITICAL: Data leakage detected between validation and test partitions!")

    # 2. Load Model and Quality Agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading frozen DenseNet-121 Base Model on device '%s'...", device)
    model = BaseModelAgent(device=device)
    model.load_model()
    quality_agent = QualityAgent()

    # 3. Construct Cohorts from validation.csv
    logger.info("Selecting representative validation images...")
    val_existing = []
    for _, row in df_val.iterrows():
        p = dataset_dir / str(row["image_path"]).replace("\\", "/")
        if p.exists():
            val_existing.append(row)

    df_val_exist = pd.DataFrame(val_existing)
    logger.info("Total accessible validation images: %d", len(df_val_exist))

    # Blur Cohort: natural NIH blur (Laplacian variance < 100)
    # Stratified 10 Positive, 10 Negative
    pos_val = df_val_exist[df_val_exist["label"] == 1]
    neg_val = df_val_exist[df_val_exist["label"] == 0]

    blur_candidates_pos = []
    for _, row in pos_val.iterrows():
        p = dataset_dir / str(row["image_path"]).replace("\\", "/")
        img = np.array(Image.open(p).convert("L"))
        q = quality_agent.run(img)
        if q.laplacian_variance < 100.0:
            blur_candidates_pos.append(row)
        if len(blur_candidates_pos) >= 10:
            break

    blur_candidates_neg = []
    for _, row in neg_val.iterrows():
        p = dataset_dir / str(row["image_path"]).replace("\\", "/")
        img = np.array(Image.open(p).convert("L"))
        q = quality_agent.run(img)
        if q.laplacian_variance < 100.0:
            blur_candidates_neg.append(row)
        if len(blur_candidates_neg) >= 10:
            break

    df_blur_cohort = pd.concat([pd.DataFrame(blur_candidates_pos), pd.DataFrame(blur_candidates_neg)]).reset_index(drop=True)
    logger.info("Constructed Blur Cohort: N=%d (Pos=%d, Neg=%d)", len(df_blur_cohort), len(blur_candidates_pos), len(blur_candidates_neg))

    # Noise & Exposure Cohorts: Clean reference images (good quality)
    clean_candidates_pos = []
    clean_candidates_neg = []
    for _, row in pos_val.iterrows():
        if row["image_id"] in df_blur_cohort["image_id"].values:
            continue
        p = dataset_dir / str(row["image_path"]).replace("\\", "/")
        img = np.array(Image.open(p).convert("L"))
        q = quality_agent.run(img)
        if q.overall.value == "good":
            clean_candidates_pos.append(row)
        if len(clean_candidates_pos) >= 5:
            break

    for _, row in neg_val.iterrows():
        if row["image_id"] in df_blur_cohort["image_id"].values:
            continue
        p = dataset_dir / str(row["image_path"]).replace("\\", "/")
        img = np.array(Image.open(p).convert("L"))
        q = quality_agent.run(img)
        if q.overall.value == "good":
            clean_candidates_neg.append(row)
        if len(clean_candidates_neg) >= 5:
            break

    df_clean_cohort = pd.concat([pd.DataFrame(clean_candidates_pos), pd.DataFrame(clean_candidates_neg)]).reset_index(drop=True)
    logger.info("Constructed Clean Synthetic Base Cohort: N=%d (Pos=%d, Neg=%d)", len(df_clean_cohort), len(clean_candidates_pos), len(clean_candidates_neg))

    tuning_records: list[dict[str, Any]] = []

    # =========================================================================
    # PHASE 3: BLUR REPAIR PARAMETER SWEEP
    # =========================================================================
    logger.info("Starting Phase 3: Blur Repair Parameter Sweep...")
    blur_grid = [
        {"radius": r, "amount": a}
        for r in [1.0, 2.0, 3.0]
        for a in [0.25, 0.5, 0.75, 1.0]
    ]

    for _, row in df_blur_cohort.iterrows():
        img_id = str(row["image_id"])
        gt_label = int(row["label"])
        p = dataset_dir / str(row["image_path"]).replace("\\", "/")
        orig_img = np.array(Image.open(p).convert("L"))

        q_before = quality_agent.run(orig_img)
        lap_before = float(q_before.laplacian_variance)
        mean_before = float(np.mean(orig_img))
        std_before = float(np.std(orig_img))
        raw_before, conf_before, pred_before = run_model_inference(model, orig_img)
        corr_before = int(pred_before == gt_label)

        for cfg in blur_grid:
            r = cfg["radius"]
            a = cfg["amount"]
            rep_img, params = repair_blur(orig_img, radius=r, amount=a)

            q_after = quality_agent.run(rep_img)
            lap_after = float(q_after.laplacian_variance)
            mean_after = float(np.mean(rep_img))
            std_after = float(np.std(rep_img))
            raw_after, conf_after, pred_after = run_model_inference(model, rep_img)
            corr_after = int(pred_after == gt_label)

            range_violation = bool(np.min(rep_img) < 0 or np.max(rep_img) > 255)
            lap_delta = lap_after - lap_before
            conf_delta = conf_after - conf_before
            raw_delta = raw_after - raw_before
            flip = bool(pred_before != pred_after)

            c_to_c = bool(corr_before == 1 and corr_after == 1)
            c_to_i = bool(corr_before == 1 and corr_after == 0)
            i_to_c = bool(corr_before == 0 and corr_after == 1)
            i_to_i = bool(corr_before == 0 and corr_after == 0)

            tuning_records.append({
                "defect_type": "blur",
                "cohort_nature": "natural_nih",
                "image_id": img_id,
                "ground_truth": gt_label,
                "parameter_label": f"radius={r}_amount={a}",
                "param_1_name": "radius",
                "param_1_val": r,
                "param_2_name": "amount",
                "param_2_val": a,
                "quality_metric_before": lap_before,
                "quality_metric_after": lap_after,
                "quality_metric_delta": lap_delta,
                "mean_intensity_before": mean_before,
                "mean_intensity_after": mean_after,
                "intensity_delta": mean_after - mean_before,
                "range_violation": range_violation,
                "snr_delta": None,
                "psnr_clean": None,
                "rmse_clean": None,
                "raw_score_before": raw_before,
                "raw_score_after": raw_after,
                "raw_score_delta": raw_delta,
                "confidence_before": conf_before,
                "confidence_after": conf_after,
                "confidence_delta": conf_delta,
                "prediction_before": pred_before,
                "prediction_after": pred_after,
                "prediction_flip": flip,
                "correct_to_correct": c_to_c,
                "correct_to_incorrect": c_to_i,
                "incorrect_to_correct": i_to_c,
                "incorrect_to_incorrect": i_to_i,
            })

    # =========================================================================
    # PHASE 4: NOISE REPAIR PARAMETER SWEEP
    # =========================================================================
    logger.info("Starting Phase 4: Noise Repair Parameter Sweep...")
    noise_grid = [
        {"h": h_val, "template_window_size": 7, "search_window_size": 21}
        for h_val in [3.0, 5.0, 7.0, 10.0, 15.0]
    ]

    for _, row in df_clean_cohort.iterrows():
        img_id = str(row["image_id"])
        gt_label = int(row["label"])
        p = dataset_dir / str(row["image_path"]).replace("\\", "/")
        orig_img = np.array(Image.open(p).convert("L"))

        # Clean measurements
        snr_clean = float(compute_snr_db(orig_img))
        raw_clean, conf_clean, pred_clean = run_model_inference(model, orig_img)

        # Synthetic corruption: sigma = 17.5 (severity = 0.35 * 50)
        corr_img = apply_noise(orig_img, severity=0.35, seed=int(rng.randint(0, 10000)))
        snr_corr = float(compute_snr_db(corr_img))
        raw_corr, conf_corr, pred_corr = run_model_inference(model, corr_img)
        corr_before = int(pred_corr == gt_label)

        for cfg in noise_grid:
            h_val = cfg["h"]
            rep_img, params = repair_noise(
                corr_img,
                h=h_val,
                template_window_size=cfg["template_window_size"],
                search_window_size=cfg["search_window_size"],
            )

            snr_rep = float(compute_snr_db(rep_img))
            snr_delta = snr_rep - snr_corr
            psnr_clean = compute_psnr(orig_img, rep_img)
            rmse_clean = float(np.sqrt(np.mean((orig_img.astype(float) - rep_img.astype(float)) ** 2)))
            mapd_corr = float(np.mean(np.abs(rep_img.astype(float) - corr_img.astype(float))))

            raw_rep, conf_rep, pred_rep = run_model_inference(model, rep_img)
            corr_after = int(pred_rep == gt_label)

            range_violation = bool(np.min(rep_img) < 0 or np.max(rep_img) > 255)
            conf_delta = conf_rep - conf_corr
            raw_delta = raw_rep - raw_corr
            flip = bool(pred_corr != pred_rep)

            c_to_c = bool(corr_before == 1 and corr_after == 1)
            c_to_i = bool(corr_before == 1 and corr_after == 0)
            i_to_c = bool(corr_before == 0 and corr_after == 1)
            i_to_i = bool(corr_before == 0 and corr_after == 0)

            tuning_records.append({
                "defect_type": "noise",
                "cohort_nature": "synthetic_gaussian_sigma_17.5",
                "image_id": img_id,
                "ground_truth": gt_label,
                "parameter_label": f"h={h_val}_t7_s21",
                "param_1_name": "h",
                "param_1_val": h_val,
                "param_2_name": "template_window_size",
                "param_2_val": 7,
                "quality_metric_before": snr_corr,
                "quality_metric_after": snr_rep,
                "quality_metric_delta": snr_delta,
                "mean_intensity_before": float(np.mean(corr_img)),
                "mean_intensity_after": float(np.mean(rep_img)),
                "intensity_delta": float(np.mean(rep_img) - np.mean(corr_img)),
                "range_violation": range_violation,
                "snr_delta": snr_delta,
                "psnr_clean": psnr_clean,
                "rmse_clean": rmse_clean,
                "mapd_from_corrupted": mapd_corr,
                "raw_score_before": raw_corr,
                "raw_score_after": raw_rep,
                "raw_score_delta": raw_delta,
                "confidence_before": conf_corr,
                "confidence_after": conf_rep,
                "confidence_delta": conf_delta,
                "prediction_before": pred_corr,
                "prediction_after": pred_rep,
                "prediction_flip": flip,
                "correct_to_correct": c_to_c,
                "correct_to_incorrect": c_to_i,
                "incorrect_to_correct": i_to_c,
                "incorrect_to_incorrect": i_to_i,
            })

    # =========================================================================
    # PHASE 5: EXPOSURE REPAIR PARAMETER SWEEP
    # =========================================================================
    logger.info("Starting Phase 5: Exposure Repair Parameter Sweep...")
    exp_grid = [
        {"clip_limit": clip, "tile_grid_size": grid}
        for clip in [1.0, 2.0, 3.0, 4.0]
        for grid in [(4, 4), (8, 8), (16, 16)]
    ]

    for _, row in df_clean_cohort.iterrows():
        img_id = str(row["image_id"])
        gt_label = int(row["label"])
        p = dataset_dir / str(row["image_path"]).replace("\\", "/")
        orig_img = np.array(Image.open(p).convert("L"))

        # Clean measurements
        stats_orig = compute_exposure_statistics(orig_img)
        raw_clean, conf_clean, pred_clean = run_model_inference(model, orig_img)

        # Synthetic corruption: underexposure shift = -30 units (severity = -0.30)
        corr_img = apply_exposure_shift(orig_img, severity=-0.30, seed=int(rng.randint(0, 10000)))
        stats_corr = compute_exposure_statistics(corr_img)
        raw_corr, conf_corr, pred_corr = run_model_inference(model, corr_img)
        corr_before = int(pred_corr == gt_label)

        for cfg in exp_grid:
            clip = cfg["clip_limit"]
            grid = cfg["tile_grid_size"]
            rep_img, params = repair_exposure(corr_img, clip_limit=clip, tile_grid_size=grid)

            stats_rep = compute_exposure_statistics(rep_img)
            psnr_clean = compute_psnr(orig_img, rep_img)
            rmse_clean = float(np.sqrt(np.mean((orig_img.astype(float) - rep_img.astype(float)) ** 2)))
            mae_clean = float(np.mean(np.abs(orig_img.astype(float) - rep_img.astype(float))))

            raw_rep, conf_rep, pred_rep = run_model_inference(model, rep_img)
            corr_after = int(pred_rep == gt_label)

            range_violation = bool(np.min(rep_img) < 0 or np.max(rep_img) > 255)
            conf_delta = conf_rep - conf_corr
            raw_delta = raw_rep - raw_corr
            flip = bool(pred_corr != pred_rep)

            c_to_c = bool(corr_before == 1 and corr_after == 1)
            c_to_i = bool(corr_before == 1 and corr_after == 0)
            i_to_c = bool(corr_before == 0 and corr_after == 1)
            i_to_i = bool(corr_before == 0 and corr_after == 0)

            tuning_records.append({
                "defect_type": "exposure",
                "cohort_nature": "synthetic_underexposure_neg30",
                "image_id": img_id,
                "ground_truth": gt_label,
                "parameter_label": f"clip={clip}_grid={grid[0]}x{grid[1]}",
                "param_1_name": "clip_limit",
                "param_1_val": clip,
                "param_2_name": "tile_grid_size",
                "param_2_val": f"{grid[0]}x{grid[1]}",
                "quality_metric_before": stats_corr["mean_intensity"],
                "quality_metric_after": stats_rep["mean_intensity"],
                "quality_metric_delta": stats_rep["mean_intensity"] - stats_corr["mean_intensity"],
                "mean_intensity_before": stats_corr["mean_intensity"],
                "mean_intensity_after": stats_rep["mean_intensity"],
                "intensity_delta": stats_rep["mean_intensity"] - stats_corr["mean_intensity"],
                "dark_fraction_rep": stats_rep["under_exposed_fraction"],
                "bright_fraction_rep": stats_rep["over_exposed_fraction"],
                "std_rep": stats_rep["histogram_std"],
                "range_violation": range_violation,
                "snr_delta": None,
                "psnr_clean": psnr_clean,
                "rmse_clean": rmse_clean,
                "mae_clean": mae_clean,
                "raw_score_before": raw_corr,
                "raw_score_after": raw_rep,
                "raw_score_delta": raw_delta,
                "confidence_before": conf_corr,
                "confidence_after": conf_rep,
                "confidence_delta": conf_delta,
                "prediction_before": pred_corr,
                "prediction_after": pred_rep,
                "prediction_flip": flip,
                "correct_to_correct": c_to_c,
                "correct_to_incorrect": c_to_i,
                "incorrect_to_correct": i_to_c,
                "incorrect_to_incorrect": i_to_i,
            })

    # Save detailed CSV
    df_results = pd.DataFrame(tuning_records)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv_path, index=False)
    logger.info("Saved image-level tuning records (%d rows) to %s", len(df_results), output_csv_path)

    # =========================================================================
    # COMPILE SUMMARY STATISTICS & MULTI-METRIC COMPARISON
    # =========================================================================
    summary: dict[str, Any] = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "val_csv": str(val_csv_path),
            "test_csv": str(test_csv_path),
            "seed": args.seed,
            "total_tuning_images": len(df_blur_cohort) + len(df_clean_cohort),
            "blur_cohort_size": len(df_blur_cohort),
            "clean_cohort_size": len(df_clean_cohort),
            "leakage_patients": 0,
            "leakage_images": 0,
        },
        "sweeps": {},
    }

    # Group by defect_type and parameter_label
    for defect, df_defect in df_results.groupby("defect_type"):
        group_summaries = []
        for param_lbl, sub in df_defect.groupby("parameter_label"):
            n_cases = len(sub)
            q_deltas = sub["quality_metric_delta"].values
            conf_deltas = sub["confidence_delta"].values
            raw_deltas = sub["raw_score_delta"].values
            flips = int(sub["prediction_flip"].sum())
            stability_pct = float((n_cases - flips) / n_cases * 100)

            c2c = int(sub["correct_to_correct"].sum())
            c2i = int(sub["correct_to_incorrect"].sum())
            i2c = int(sub["incorrect_to_correct"].sum())
            i2i = int(sub["incorrect_to_incorrect"].sum())

            pct_improved = float(np.mean(q_deltas > 0) * 100)
            pct_unchanged = float(np.mean(q_deltas == 0) * 100)
            pct_worsened = float(np.mean(q_deltas < 0) * 100)

            rec = {
                "parameter_label": param_lbl,
                "n_cases": n_cases,
                "quality_delta_mean": float(np.mean(q_deltas)),
                "quality_delta_median": float(np.median(q_deltas)),
                "pct_improved": pct_improved,
                "pct_unchanged": pct_unchanged,
                "pct_worsened": pct_worsened,
                "prediction_stability_pct": stability_pct,
                "flips_count": flips,
                "conf_delta_mean": float(np.mean(conf_deltas)),
                "raw_score_delta_mean": float(np.mean(raw_deltas)),
                "correct_to_correct": c2c,
                "correct_to_incorrect": c2i,
                "incorrect_to_correct": i2c,
                "incorrect_to_incorrect": i2i,
            }

            if defect == "noise":
                rec["snr_delta_mean"] = float(sub["snr_delta"].mean())
                rec["psnr_clean_mean"] = float(sub["psnr_clean"].mean())
                rec["rmse_clean_mean"] = float(sub["rmse_clean"].mean())
                rec["mapd_corrupted_mean"] = float(sub["mapd_from_corrupted"].mean())

            if defect == "exposure":
                rec["mean_intensity_rep"] = float(sub["mean_intensity_after"].mean())
                rec["dark_fraction_mean"] = float(sub["dark_fraction_rep"].mean())
                rec["psnr_clean_mean"] = float(sub["psnr_clean"].mean())
                rec["rmse_clean_mean"] = float(sub["rmse_clean"].mean())

            group_summaries.append(rec)

        summary["sweeps"][defect] = group_summaries

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary JSON to %s", output_json_path)

    # Print summary table to console
    print("\n" + "=" * 90)
    print("REPAIR PARAMETER TUNING SUMMARY TABLE")
    print("=" * 90)
    for defect, sweeps in summary["sweeps"].items():
        print(f"\n--- DEFECT TYPE: {defect.upper()} ---")
        df_sw = pd.DataFrame(sweeps)
        cols_to_show = [
            c for c in ["parameter_label", "quality_delta_mean", "prediction_stability_pct", "flips_count", "incorrect_to_correct", "correct_to_incorrect", "conf_delta_mean"]
            if c in df_sw.columns
        ]
        if defect == "noise":
            cols_to_show.extend([c for c in ["snr_delta_mean", "psnr_clean_mean", "rmse_clean_mean"] if c in df_sw.columns])
        print(df_sw[cols_to_show].to_string(index=False))

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
