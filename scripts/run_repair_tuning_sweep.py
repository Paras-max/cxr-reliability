"""Repair Agent Parameter Sweep and Tuning Script.

Executes controlled, reproducible parameter sweeps for Blur, Noise, and Exposure
repairs using a patient-isolated validation tuning cohort (zero test set leakage).
Evaluates image quality, model prediction stability, confidence deltas, and
diagnostic transitions with the frozen DenseNet-121 Base Model.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import unsharp_mask

# Set up project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_reliability.data.corruptions import apply_exposure_shift, apply_noise
from cxr_reliability.models.base_model import BaseModelAgent
from cxr_reliability.models.preprocessing import prepare_image_for_txv
from cxr_reliability.quality.blur import compute_laplacian_variance, evaluate_blur
from cxr_reliability.quality.evaluator import prepare_image_for_quality
from cxr_reliability.quality.exposure import compute_exposure_statistics, evaluate_exposure
from cxr_reliability.quality.noise import compute_snr_db, evaluate_noise
from cxr_reliability.repair.blur import repair_blur
from cxr_reliability.repair.exposure import repair_exposure
from cxr_reliability.repair.noise import repair_noise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("repair_tuning_sweep")

OPERATING_THRESHOLD = 0.522161
RANDOM_SEED = 42


def compute_psnr(ref: np.ndarray, target: np.ndarray) -> float:
    """Compute PSNR between reference and target uint8 images."""
    mse = float(np.mean((ref.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse <= 1e-10:
        return 100.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def compute_mae(ref: np.ndarray, target: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(ref.astype(np.float64) - target.astype(np.float64))))


def compute_rmse(ref: np.ndarray, target: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    mse = float(np.mean((ref.astype(np.float64) - target.astype(np.float64)) ** 2))
    return float(math.sqrt(mse))


def compute_confidence(raw_score: float) -> float:
    """Compute derived binary confidence max(p, 1-p)."""
    return float(max(raw_score, 1.0 - raw_score))


def select_tuning_cohort(
    val_csv_path: Path,
    test_csv_path: Path,
    dataset_dir: Path,
    seed: int = RANDOM_SEED,
) -> dict[str, list[dict[str, Any]]]:
    """
    Select representative, patient-isolated tuning images from validation.csv.
    Guarantees zero overlap with test.csv.
    """
    logger.info("Loading validation and test partitions for leakage verification...")
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    test_image_ids = set(test_df["image_id"])
    test_patient_ids = set(test_df["patient_id"])

    # Double check patient-level split
    val_patient_ids = set(val_df["patient_id"])
    patient_overlap = val_patient_ids.intersection(test_patient_ids)
    if patient_overlap:
        raise RuntimeError(f"Patient leakage detected between val and test: {len(patient_overlap)} patients!")
    logger.info("Patient-isolation verified: 0 patient overlap between validation and test.")

    pos_val = val_df[val_df["label"] == 1].copy()
    neg_val = val_df[val_df["label"] == 0].copy()

    # 1. Select Blur cohort: naturally blurred (laplacian_variance < 100.0)
    logger.info("Scanning validation set for naturally blurred candidates...")
    blur_candidates_pos: list[dict[str, Any]] = []
    blur_candidates_neg: list[dict[str, Any]] = []

    # Iterate over pos and neg until we have 10 each
    for _, row in pos_val.sample(frac=1.0, random_state=seed).iterrows():
        if len(blur_candidates_pos) >= 10:
            break
        img_path = dataset_dir / row["image_path"]
        if not img_path.exists():
            continue
        arr = prepare_image_for_quality(img_path)
        b = evaluate_blur(arr)
        if b.is_blurred:
            blur_candidates_pos.append({
                "image_id": row["image_id"],
                "patient_id": row["patient_id"],
                "image_path": str(img_path),
                "label": int(row["label"]),
                "laplacian_var": b.laplacian_variance,
                "is_synthetic": False,
                "corruption_type": "natural_blur",
            })

    for _, row in neg_val.sample(frac=1.0, random_state=seed).iterrows():
        if len(blur_candidates_neg) >= 10:
            break
        img_path = dataset_dir / row["image_path"]
        if not img_path.exists():
            continue
        arr = prepare_image_for_quality(img_path)
        b = evaluate_blur(arr)
        if b.is_blurred:
            blur_candidates_neg.append({
                "image_id": row["image_id"],
                "patient_id": row["patient_id"],
                "image_path": str(img_path),
                "label": int(row["label"]),
                "laplacian_var": b.laplacian_variance,
                "is_synthetic": False,
                "corruption_type": "natural_blur",
            })

    blur_cohort = blur_candidates_pos + blur_candidates_neg
    used_image_ids = {c["image_id"] for c in blur_cohort}
    logger.info(f"Selected Blur cohort: N={len(blur_cohort)} (Pos={len(blur_candidates_pos)}, Neg={len(blur_candidates_neg)})")

    # 2. Select Noise cohort: 10 Pos + 10 Neg, disjoint from Blur
    logger.info("Selecting candidates for controlled synthetic noise cohort...")
    noise_cohort: list[dict[str, Any]] = []
    pos_rem = pos_val[~pos_val["image_id"].isin(used_image_ids)]
    neg_rem = neg_val[~neg_val["image_id"].isin(used_image_ids)]

    for _, row in pos_rem.sample(n=10, random_state=seed).iterrows():
        img_path = dataset_dir / row["image_path"]
        noise_cohort.append({
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "image_path": str(img_path),
            "label": int(row["label"]),
            "is_synthetic": True,
            "corruption_type": "synthetic_gaussian_noise_0.35",
        })
        used_image_ids.add(row["image_id"])

    for _, row in neg_rem.sample(n=10, random_state=seed).iterrows():
        img_path = dataset_dir / row["image_path"]
        noise_cohort.append({
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "image_path": str(img_path),
            "label": int(row["label"]),
            "is_synthetic": True,
            "corruption_type": "synthetic_gaussian_noise_0.35",
        })
        used_image_ids.add(row["image_id"])

    logger.info(f"Selected Noise cohort: N={len(noise_cohort)} (Pos=10, Neg=10)")

    # 3. Select Exposure cohort: 10 underexposed (-0.25) + 10 overexposed (+0.25)
    logger.info("Selecting candidates for controlled synthetic exposure cohort...")
    pos_rem = pos_val[~pos_val["image_id"].isin(used_image_ids)]
    neg_rem = neg_val[~neg_val["image_id"].isin(used_image_ids)]

    exposure_cohort: list[dict[str, Any]] = []
    pos_under = pos_rem.sample(n=5, random_state=seed)
    pos_over = pos_rem[~pos_rem["image_id"].isin(set(pos_under["image_id"]))].sample(n=5, random_state=seed)
    neg_under = neg_rem.sample(n=5, random_state=seed)
    neg_over = neg_rem[~neg_rem["image_id"].isin(set(neg_under["image_id"]))].sample(n=5, random_state=seed)

    for _, row in pos_under.iterrows():
        img_path = dataset_dir / row["image_path"]
        exposure_cohort.append({
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "image_path": str(img_path),
            "label": int(row["label"]),
            "is_synthetic": True,
            "corruption_type": "synthetic_underexposure_-0.25",
            "severity": -0.25,
        })
        used_image_ids.add(row["image_id"])

    for _, row in neg_under.iterrows():
        img_path = dataset_dir / row["image_path"]
        exposure_cohort.append({
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "image_path": str(img_path),
            "label": int(row["label"]),
            "is_synthetic": True,
            "corruption_type": "synthetic_underexposure_-0.25",
            "severity": -0.25,
        })
        used_image_ids.add(row["image_id"])

    for _, row in pos_over.iterrows():
        img_path = dataset_dir / row["image_path"]
        exposure_cohort.append({
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "image_path": str(img_path),
            "label": int(row["label"]),
            "is_synthetic": True,
            "corruption_type": "synthetic_overexposure_+0.25",
            "severity": +0.25,
        })
        used_image_ids.add(row["image_id"])

    for _, row in neg_over.iterrows():
        img_path = dataset_dir / row["image_path"]
        exposure_cohort.append({
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "image_path": str(img_path),
            "label": int(row["label"]),
            "is_synthetic": True,
            "corruption_type": "synthetic_overexposure_+0.25",
            "severity": +0.25,
        })
        used_image_ids.add(row["image_id"])

    logger.info(f"Selected Exposure cohort: N={len(exposure_cohort)} (Underexposed=10, Overexposed=10)")

    # Strict Leakage Assertion
    all_tuning_ids = {c["image_id"] for c in blur_cohort + noise_cohort + exposure_cohort}
    leakage = all_tuning_ids.intersection(test_image_ids)
    if leakage:
        raise RuntimeError(f"CRITICAL ERROR: Leakage detected! {len(leakage)} tuning images are in the test set!")

    logger.info(f"Zero leakage confirmed. Total tuning images: {len(all_tuning_ids)}. Zero intersection with test set.")
    return {
        "blur": blur_cohort,
        "noise": noise_cohort,
        "exposure": exposure_cohort,
    }


def evaluate_model_on_image(model: BaseModelAgent, img_u8: np.ndarray) -> tuple[float, float, int]:
    """Run frozen DenseNet-121 forward pass and return (raw_score, confidence, prediction)."""
    tensor = prepare_image_for_txv(img_u8)
    forward = model.run(tensor)
    raw_score = float(forward.result.pneumonia_probability)
    conf = compute_confidence(raw_score)
    pred = 1 if raw_score >= OPERATING_THRESHOLD else 0
    return raw_score, conf, pred


def get_diagnostic_transition(y_true: int, pred_before: int, pred_after: int) -> str:
    """Determine the diagnostic transition category."""
    corr_before = (pred_before == y_true)
    corr_after = (pred_after == y_true)
    if corr_before and corr_after:
        return "Correct->Correct"
    elif corr_before and not corr_after:
        return "Correct->Incorrect"
    elif not corr_before and corr_after:
        return "Incorrect->Correct"
    else:
        return "Incorrect->Incorrect"


def run_sweeps(
    cohorts: dict[str, list[dict[str, Any]]],
    model: BaseModelAgent,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute Blur, Noise, and Exposure parameter sweeps."""
    all_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "metadata": {
            "execution_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "random_seed": RANDOM_SEED,
            "operating_threshold": OPERATING_THRESHOLD,
            "model_id": model.model_id,
            "device": model.device,
        },
        "blur_sweep": {},
        "noise_sweep": {},
        "exposure_sweep": {},
        "selection_rules": {},
        "candidate_recommendations": {},
    }

    # =========================================================================
    # PHASE 3: BLUR REPAIR PARAMETER SWEEP
    # =========================================================================
    logger.info("Starting Phase 3: Blur Repair Parameter Sweep (12 configurations)...")
    blur_configs = [
        {"radius": r, "amount": a}
        for r in [1.0, 2.0, 3.0]
        for a in [0.25, 0.5, 0.75, 1.0]
    ]

    # Preload raw images and baseline model inference
    blur_preloaded = []
    for item in cohorts["blur"]:
        pil = Image.open(item["image_path"]).convert("L")
        orig_u8 = np.array(pil, dtype=np.uint8)
        lap_before = compute_laplacian_variance(orig_u8)
        mean_int_before = float(np.mean(orig_u8))
        raw_before, conf_before, pred_before = evaluate_model_on_image(model, orig_u8)
        blur_preloaded.append({
            "meta": item,
            "orig_u8": orig_u8,
            "lap_before": lap_before,
            "mean_int_before": mean_int_before,
            "raw_before": raw_before,
            "conf_before": conf_before,
            "pred_before": pred_before,
        })

    for cfg in blur_configs:
        r = cfg["radius"]
        a = cfg["amount"]
        cfg_name = f"radius={r},amount={a}"
        logger.info(f"Evaluating Blur config: {cfg_name}")

        cfg_records = []
        for sample in blur_preloaded:
            item = sample["meta"]
            orig_u8 = sample["orig_u8"]
            y_true = item["label"]

            # Apply repair
            repaired_u8, _ = repair_blur(orig_u8, radius=r, amount=a)

            # Check pixel range violations
            range_viol = bool(np.any(repaired_u8 < 0) or np.any(repaired_u8 > 255))
            lap_after = compute_laplacian_variance(repaired_u8)
            lap_delta = lap_after - sample["lap_before"]
            mean_int_after = float(np.mean(repaired_u8))
            exp_delta = mean_int_after - sample["mean_int_before"]

            # Model inference
            raw_after, conf_after, pred_after = evaluate_model_on_image(model, repaired_u8)
            raw_delta = raw_after - sample["raw_before"]
            conf_delta = conf_after - sample["conf_before"]
            flip = bool(pred_after != sample["pred_before"])
            diag_trans = get_diagnostic_transition(y_true, sample["pred_before"], pred_after)

            rec = {
                "repair_type": "blur",
                "param_name": cfg_name,
                "radius": r,
                "amount": a,
                "h": np.nan,
                "clip_limit": np.nan,
                "tile_grid": "",
                "image_id": item["image_id"],
                "patient_id": item["patient_id"],
                "ground_truth_label": y_true,
                "is_synthetic": item["is_synthetic"],
                "corruption_type": item["corruption_type"],
                "quality_metric_before": sample["lap_before"],
                "quality_metric_after": lap_after,
                "quality_metric_delta": lap_delta,
                "snr_before": np.nan,
                "snr_after": np.nan,
                "snr_delta": np.nan,
                "laplacian_var_before": sample["lap_before"],
                "laplacian_var_after": lap_after,
                "laplacian_var_delta": lap_delta,
                "mean_intensity_before": sample["mean_int_before"],
                "mean_intensity_after": mean_int_after,
                "mean_intensity_delta": exp_delta,
                "mae_vs_clean": np.nan,
                "rmse_vs_clean": np.nan,
                "psnr_vs_clean": np.nan,
                "pixel_range_violation": range_viol,
                "raw_score_before": sample["raw_before"],
                "raw_score_after": raw_after,
                "raw_score_clean": sample["raw_before"],
                "raw_score_delta": raw_delta,
                "confidence_before": sample["conf_before"],
                "confidence_after": conf_after,
                "confidence_delta": conf_delta,
                "pred_before": sample["pred_before"],
                "pred_after": pred_after,
                "pred_clean": sample["pred_before"],
                "prediction_flip": flip,
                "diagnostic_transition": diag_trans,
            }
            cfg_records.append(rec)
            all_records.append(rec)

        # Aggregate summary for this config
        lap_deltas = [r["laplacian_var_delta"] for r in cfg_records]
        conf_deltas = [r["confidence_delta"] for r in cfg_records]
        raw_deltas = [r["raw_score_delta"] for r in cfg_records]
        n_samples = len(cfg_records)
        pct_imp = float(sum(d > 0.0 for d in lap_deltas) / n_samples * 100.0)
        pct_unch = float(sum(d == 0.0 for d in lap_deltas) / n_samples * 100.0)
        pct_wors = float(sum(d < 0.0 for d in lap_deltas) / n_samples * 100.0)
        flips = sum(r["prediction_flip"] for r in cfg_records)
        stability = float((n_samples - flips) / n_samples * 100.0)

        diag_counts = {
            "Correct->Correct": sum(r["diagnostic_transition"] == "Correct->Correct" for r in cfg_records),
            "Correct->Incorrect": sum(r["diagnostic_transition"] == "Correct->Incorrect" for r in cfg_records),
            "Incorrect->Correct": sum(r["diagnostic_transition"] == "Incorrect->Correct" for r in cfg_records),
            "Incorrect->Incorrect": sum(r["diagnostic_transition"] == "Incorrect->Incorrect" for r in cfg_records),
        }

        summary["blur_sweep"][cfg_name] = {
            "radius": r,
            "amount": a,
            "n_samples": n_samples,
            "mean_laplacian_before": float(np.mean([r["laplacian_var_before"] for r in cfg_records])),
            "mean_laplacian_after": float(np.mean([r["laplacian_var_after"] for r in cfg_records])),
            "mean_laplacian_delta": float(np.mean(lap_deltas)),
            "median_laplacian_delta": float(np.median(lap_deltas)),
            "pct_improved": pct_imp,
            "pct_unchanged": pct_unch,
            "pct_worsened": pct_wors,
            "mean_exposure_delta": float(np.mean([r["mean_intensity_delta"] for r in cfg_records])),
            "range_violations_count": sum(r["pixel_range_violation"] for r in cfg_records),
            "mean_raw_score_delta": float(np.mean(raw_deltas)),
            "mean_confidence_delta": float(np.mean(conf_deltas)),
            "median_confidence_delta": float(np.median(conf_deltas)),
            "prediction_stability_pct": stability,
            "total_flips": flips,
            "diagnostic_transitions": diag_counts,
        }

    # =========================================================================
    # PHASE 4: NOISE REPAIR PARAMETER SWEEP
    # =========================================================================
    logger.info("Starting Phase 4: Noise Repair Parameter Sweep (5 configurations)...")
    noise_configs = [3.0, 5.0, 7.0, 10.0, 15.0]

    # Preload clean and corrupted noise cohort
    noise_preloaded = []
    for item in cohorts["noise"]:
        pil = Image.open(item["image_path"]).convert("L")
        clean_u8 = np.array(pil, dtype=np.uint8)
        clean_snr = compute_snr_db(clean_u8)
        raw_clean, conf_clean, pred_clean = evaluate_model_on_image(model, clean_u8)

        # Apply controlled synthetic noise: severity 0.35, seed=42
        corrupted_u8 = apply_noise(clean_u8, severity=0.35, seed=RANDOM_SEED)
        corrupted_snr = compute_snr_db(corrupted_u8)
        raw_corr, conf_corr, pred_corr = evaluate_model_on_image(model, corrupted_u8)

        noise_preloaded.append({
            "meta": item,
            "clean_u8": clean_u8,
            "clean_snr": clean_snr,
            "raw_clean": raw_clean,
            "conf_clean": conf_clean,
            "pred_clean": pred_clean,
            "corrupted_u8": corrupted_u8,
            "corrupted_snr": corrupted_snr,
            "raw_corr": raw_corr,
            "conf_corr": conf_corr,
            "pred_corr": pred_corr,
        })

    for h in noise_configs:
        cfg_name = f"h={h}"
        logger.info(f"Evaluating Noise config: {cfg_name}")

        cfg_records = []
        for sample in noise_preloaded:
            item = sample["meta"]
            clean_u8 = sample["clean_u8"]
            corr_u8 = sample["corrupted_u8"]
            y_true = item["label"]

            # Denoise using Fast NLMeans
            repaired_u8, _ = repair_noise(corr_u8, h=h, template_window_size=7, search_window_size=21)
            repaired_snr = compute_snr_db(repaired_u8)
            snr_gain = repaired_snr - sample["corrupted_snr"]

            # Distortion vs clean reference
            mae_clean = compute_mae(clean_u8, repaired_u8)
            rmse_clean = compute_rmse(clean_u8, repaired_u8)
            psnr_clean = compute_psnr(clean_u8, repaired_u8)
            lap_rep = compute_laplacian_variance(repaired_u8)
            range_viol = bool(np.any(repaired_u8 < 0) or np.any(repaired_u8 > 255))

            # Model inference on repaired image
            raw_rep, conf_rep, pred_rep = evaluate_model_on_image(model, repaired_u8)
            raw_delta = raw_rep - sample["raw_corr"]
            conf_delta = conf_rep - sample["conf_corr"]
            flip = bool(pred_rep != sample["pred_corr"])
            diag_trans = get_diagnostic_transition(y_true, sample["pred_corr"], pred_rep)

            rec = {
                "repair_type": "noise",
                "param_name": cfg_name,
                "radius": np.nan,
                "amount": np.nan,
                "h": h,
                "clip_limit": np.nan,
                "tile_grid": "",
                "image_id": item["image_id"],
                "patient_id": item["patient_id"],
                "ground_truth_label": y_true,
                "is_synthetic": True,
                "corruption_type": item["corruption_type"],
                "quality_metric_before": sample["corrupted_snr"],
                "quality_metric_after": repaired_snr,
                "quality_metric_delta": snr_gain,
                "snr_before": sample["corrupted_snr"],
                "snr_after": repaired_snr,
                "snr_delta": snr_gain,
                "laplacian_var_before": compute_laplacian_variance(corr_u8),
                "laplacian_var_after": lap_rep,
                "laplacian_var_delta": lap_rep - compute_laplacian_variance(corr_u8),
                "mean_intensity_before": float(np.mean(corr_u8)),
                "mean_intensity_after": float(np.mean(repaired_u8)),
                "mean_intensity_delta": float(np.mean(repaired_u8)) - float(np.mean(corr_u8)),
                "mae_vs_clean": mae_clean,
                "rmse_vs_clean": rmse_clean,
                "psnr_vs_clean": psnr_clean,
                "pixel_range_violation": range_viol,
                "raw_score_before": sample["raw_corr"],
                "raw_score_after": raw_rep,
                "raw_score_clean": sample["raw_clean"],
                "raw_score_delta": raw_delta,
                "confidence_before": sample["conf_corr"],
                "confidence_after": conf_rep,
                "confidence_delta": conf_delta,
                "pred_before": sample["pred_corr"],
                "pred_after": pred_rep,
                "pred_clean": sample["pred_clean"],
                "prediction_flip": flip,
                "diagnostic_transition": diag_trans,
            }
            cfg_records.append(rec)
            all_records.append(rec)

        # Aggregate summary for noise
        snr_gains = [r["snr_delta"] for r in cfg_records]
        conf_deltas = [r["confidence_delta"] for r in cfg_records]
        raw_deltas = [r["raw_score_delta"] for r in cfg_records]
        n_samples = len(cfg_records)
        pct_imp = float(sum(d > 0.05 for d in snr_gains) / n_samples * 100.0)
        pct_unch = float(sum(abs(d) <= 0.05 for d in snr_gains) / n_samples * 100.0)
        pct_wors = float(sum(d < -0.05 for d in snr_gains) / n_samples * 100.0)
        flips = sum(r["prediction_flip"] for r in cfg_records)
        stability = float((n_samples - flips) / n_samples * 100.0)

        diag_counts = {
            "Correct->Correct": sum(r["diagnostic_transition"] == "Correct->Correct" for r in cfg_records),
            "Correct->Incorrect": sum(r["diagnostic_transition"] == "Correct->Incorrect" for r in cfg_records),
            "Incorrect->Correct": sum(r["diagnostic_transition"] == "Incorrect->Correct" for r in cfg_records),
            "Incorrect->Incorrect": sum(r["diagnostic_transition"] == "Incorrect->Incorrect" for r in cfg_records),
        }

        summary["noise_sweep"][cfg_name] = {
            "h": h,
            "n_samples": n_samples,
            "mean_clean_snr": float(np.mean([s["clean_snr"] for s in noise_preloaded])),
            "mean_corrupted_snr": float(np.mean([s["corrupted_snr"] for s in noise_preloaded])),
            "mean_repaired_snr": float(np.mean([r["snr_after"] for r in cfg_records])),
            "mean_snr_gain": float(np.mean(snr_gains)),
            "median_snr_gain": float(np.median(snr_gains)),
            "pct_improved": pct_imp,
            "pct_unchanged": pct_unch,
            "pct_worsened": pct_wors,
            "mean_mae_vs_clean": float(np.mean([r["mae_vs_clean"] for r in cfg_records])),
            "mean_rmse_vs_clean": float(np.mean([r["rmse_vs_clean"] for r in cfg_records])),
            "mean_psnr_vs_clean": float(np.mean([r["psnr_vs_clean"] for r in cfg_records])),
            "mean_laplacian_after": float(np.mean([r["laplacian_var_after"] for r in cfg_records])),
            "mean_raw_score_delta": float(np.mean(raw_deltas)),
            "mean_confidence_delta": float(np.mean(conf_deltas)),
            "prediction_stability_pct": stability,
            "total_flips": flips,
            "diagnostic_transitions": diag_counts,
        }

    # =========================================================================
    # PHASE 5: EXPOSURE REPAIR PARAMETER SWEEP
    # =========================================================================
    logger.info("Starting Phase 5: Exposure Repair Parameter Sweep (12 configurations)...")
    exposure_configs = [
        {"clip_limit": clip, "tile_grid": grid}
        for clip in [1.0, 2.0, 3.0, 4.0]
        for grid in [(4, 4), (8, 8), (16, 16)]
    ]

    exposure_preloaded = []
    for item in cohorts["exposure"]:
        pil = Image.open(item["image_path"]).convert("L")
        clean_u8 = np.array(pil, dtype=np.uint8)
        clean_stats = compute_exposure_statistics(clean_u8)
        raw_clean, conf_clean, pred_clean = evaluate_model_on_image(model, clean_u8)

        # Apply controlled exposure shift
        sev = item["severity"]
        corrupted_u8 = apply_exposure_shift(clean_u8, severity=sev, seed=RANDOM_SEED)
        corr_stats = compute_exposure_statistics(corrupted_u8)
        raw_corr, conf_corr, pred_corr = evaluate_model_on_image(model, corrupted_u8)

        exposure_preloaded.append({
            "meta": item,
            "clean_u8": clean_u8,
            "clean_stats": clean_stats,
            "raw_clean": raw_clean,
            "conf_clean": conf_clean,
            "pred_clean": pred_clean,
            "corrupted_u8": corrupted_u8,
            "corr_stats": corr_stats,
            "raw_corr": raw_corr,
            "conf_corr": conf_corr,
            "pred_corr": pred_corr,
        })

    for cfg in exposure_configs:
        clip = cfg["clip_limit"]
        grid = cfg["tile_grid"]
        cfg_name = f"clip={clip},grid={grid[0]}x{grid[1]}"
        logger.info(f"Evaluating Exposure config: {cfg_name}")

        cfg_records = []
        for sample in exposure_preloaded:
            item = sample["meta"]
            clean_u8 = sample["clean_u8"]
            corr_u8 = sample["corrupted_u8"]
            y_true = item["label"]

            # Apply CLAHE repair
            repaired_u8, _ = repair_exposure(corr_u8, clip_limit=clip, tile_grid_size=grid)
            rep_stats = compute_exposure_statistics(repaired_u8)

            mae_clean = compute_mae(clean_u8, repaired_u8)
            rmse_clean = compute_rmse(clean_u8, repaired_u8)
            psnr_clean = compute_psnr(clean_u8, repaired_u8)
            lap_rep = compute_laplacian_variance(repaired_u8)
            range_viol = bool(np.any(repaired_u8 < 0) or np.any(repaired_u8 > 255))

            # Model inference
            raw_rep, conf_rep, pred_rep = evaluate_model_on_image(model, repaired_u8)
            raw_delta = raw_rep - sample["raw_corr"]
            conf_delta = conf_rep - sample["conf_corr"]
            flip = bool(pred_rep != sample["pred_corr"])
            diag_trans = get_diagnostic_transition(y_true, sample["pred_corr"], pred_rep)

            rec = {
                "repair_type": "exposure",
                "param_name": cfg_name,
                "radius": np.nan,
                "amount": np.nan,
                "h": np.nan,
                "clip_limit": clip,
                "tile_grid": f"{grid[0]}x{grid[1]}",
                "image_id": item["image_id"],
                "patient_id": item["patient_id"],
                "ground_truth_label": y_true,
                "is_synthetic": True,
                "corruption_type": item["corruption_type"],
                "quality_metric_before": sample["corr_stats"]["mean_intensity"],
                "quality_metric_after": rep_stats["mean_intensity"],
                "quality_metric_delta": rep_stats["mean_intensity"] - sample["corr_stats"]["mean_intensity"],
                "snr_before": np.nan,
                "snr_after": np.nan,
                "snr_delta": np.nan,
                "laplacian_var_before": compute_laplacian_variance(corr_u8),
                "laplacian_var_after": lap_rep,
                "laplacian_var_delta": lap_rep - compute_laplacian_variance(corr_u8),
                "mean_intensity_before": sample["corr_stats"]["mean_intensity"],
                "mean_intensity_after": rep_stats["mean_intensity"],
                "mean_intensity_delta": rep_stats["mean_intensity"] - sample["corr_stats"]["mean_intensity"],
                "mae_vs_clean": mae_clean,
                "rmse_vs_clean": rmse_clean,
                "psnr_vs_clean": psnr_clean,
                "pixel_range_violation": range_viol,
                "raw_score_before": sample["raw_corr"],
                "raw_score_after": raw_rep,
                "raw_score_clean": sample["raw_clean"],
                "raw_score_delta": raw_delta,
                "confidence_before": sample["conf_corr"],
                "confidence_after": conf_rep,
                "confidence_delta": conf_delta,
                "pred_before": sample["pred_corr"],
                "pred_after": pred_rep,
                "pred_clean": sample["pred_clean"],
                "prediction_flip": flip,
                "diagnostic_transition": diag_trans,
            }
            cfg_records.append(rec)
            all_records.append(rec)

        # Aggregate summary for exposure
        conf_deltas = [r["confidence_delta"] for r in cfg_records]
        raw_deltas = [r["raw_score_delta"] for r in cfg_records]
        n_samples = len(cfg_records)
        flips = sum(r["prediction_flip"] for r in cfg_records)
        stability = float((n_samples - flips) / n_samples * 100.0)

        diag_counts = {
            "Correct->Correct": sum(r["diagnostic_transition"] == "Correct->Correct" for r in cfg_records),
            "Correct->Incorrect": sum(r["diagnostic_transition"] == "Correct->Incorrect" for r in cfg_records),
            "Incorrect->Correct": sum(r["diagnostic_transition"] == "Incorrect->Correct" for r in cfg_records),
            "Incorrect->Incorrect": sum(r["diagnostic_transition"] == "Incorrect->Incorrect" for r in cfg_records),
        }

        summary["exposure_sweep"][cfg_name] = {
            "clip_limit": clip,
            "tile_grid": f"{grid[0]}x{grid[1]}",
            "n_samples": n_samples,
            "mean_intensity_before": float(np.mean([r["mean_intensity_before"] for r in cfg_records])),
            "mean_intensity_after": float(np.mean([r["mean_intensity_after"] for r in cfg_records])),
            "mean_intensity_delta": float(np.mean([r["mean_intensity_delta"] for r in cfg_records])),
            "mean_mae_vs_clean": float(np.mean([r["mae_vs_clean"] for r in cfg_records])),
            "mean_rmse_vs_clean": float(np.mean([r["rmse_vs_clean"] for r in cfg_records])),
            "mean_psnr_vs_clean": float(np.mean([r["psnr_vs_clean"] for r in cfg_records])),
            "mean_laplacian_after": float(np.mean([r["laplacian_var_after"] for r in cfg_records])),
            "mean_raw_score_delta": float(np.mean(raw_deltas)),
            "mean_confidence_delta": float(np.mean(conf_deltas)),
            "prediction_stability_pct": stability,
            "total_flips": flips,
            "diagnostic_transitions": diag_counts,
        }

    return all_records, summary


def multi_metric_selection(summary: dict[str, Any]) -> dict[str, Any]:
    """
    Apply multi-metric selection rules across Quality, Model Stability, Diagnostic Impact, Safety.
    """
    logger.info("Applying Multi-Metric Selection Rules...")

    # 1. BLUR SELECTION
    # Current: radius=1.0, amount=0.5
    # Requirements: improve sharpness, prevent excessive noise/edge halo, preserve model stability, no confidence drops, minimize Correct->Incorrect.
    best_blur_cfg = None
    best_blur_score = -float("inf")
    blur_evals = []

    for name, s in summary["blur_sweep"].items():
        # High Laplacian gain is good up to a point, but extreme amounts (e.g. amount=1.0 or radius=3) create halos and noise.
        # Stability and zero Correct->Incorrect are vital.
        stability = s["prediction_stability_pct"]
        c_to_inc = s["diagnostic_transitions"]["Correct->Incorrect"]
        inc_to_c = s["diagnostic_transitions"]["Incorrect->Correct"]
        mean_lap_delta = s["mean_laplacian_delta"]
        conf_delta = s["mean_confidence_delta"]

        # Composite multi-metric utility score:
        # Base: Laplacian gain (+1 per 10 variance gain)
        # Penalty for instability: -20 for each flip
        # Penalty for Correct->Incorrect: -100 per case
        # Reward for Incorrect->Correct: +100 per case
        # Penalty for aggressive oversharpening (amount > 0.5 or radius > 1) if stability drops
        utility = (
            (mean_lap_delta / 20.0)
            + (10.0 if stability == 100.0 else (stability - 100.0) * 2.0)
            - (c_to_inc * 100.0)
            + (inc_to_c * 100.0)
            + (conf_delta * 100.0)
        )
        blur_evals.append({
            "param": name,
            "laplacian_gain": mean_lap_delta,
            "stability_pct": stability,
            "conf_delta": conf_delta,
            "inc_to_corr": inc_to_c,
            "corr_to_inc": c_to_inc,
            "utility_score": utility,
        })
        if utility > best_blur_score:
            best_blur_score = utility
            best_blur_cfg = name

    # 2. NOISE SELECTION
    # Current: h=3.0
    # Problem: h=3.0 produced 0 dB SNR improvement!
    # Goal: Maximize SNR improvement while minimizing MAE/RMSE distortion vs clean reference and maintaining model stability.
    best_noise_cfg = None
    best_noise_score = -float("inf")
    noise_evals = []

    for name, s in summary["noise_sweep"].items():
        snr_gain = s["mean_snr_gain"]
        mae = s["mean_mae_vs_clean"]
        psnr = s["mean_psnr_vs_clean"]
        stability = s["prediction_stability_pct"]
        c_to_inc = s["diagnostic_transitions"]["Correct->Incorrect"]
        inc_to_c = s["diagnostic_transitions"]["Incorrect->Correct"]

        # Objective: High SNR gain, high PSNR vs clean, high stability, zero diagnostic corruption
        utility = (
            (snr_gain * 2.0)
            + (psnr / 5.0)
            - (mae * 1.5)
            + (10.0 if stability == 100.0 else (stability - 100.0) * 2.0)
            - (c_to_inc * 100.0)
            + (inc_to_c * 100.0)
        )
        noise_evals.append({
            "param": name,
            "snr_gain": snr_gain,
            "mae_vs_clean": mae,
            "psnr_vs_clean": psnr,
            "stability_pct": stability,
            "inc_to_corr": inc_to_c,
            "corr_to_inc": c_to_inc,
            "utility_score": utility,
        })
        if utility > best_noise_score:
            best_noise_score = utility
            best_noise_cfg = name

    # 3. EXPOSURE SELECTION
    # Current: clip_limit=2.0, tile_grid=8x8
    best_exposure_cfg = None
    best_exposure_score = -float("inf")
    exposure_evals = []

    for name, s in summary["exposure_sweep"].items():
        mae = s["mean_mae_vs_clean"]
        psnr = s["mean_psnr_vs_clean"]
        stability = s["prediction_stability_pct"]
        c_to_inc = s["diagnostic_transitions"]["Correct->Incorrect"]
        inc_to_c = s["diagnostic_transitions"]["Incorrect->Correct"]

        utility = (
            (psnr / 5.0)
            - (mae * 1.5)
            + (10.0 if stability == 100.0 else (stability - 100.0) * 2.0)
            - (c_to_inc * 100.0)
            + (inc_to_c * 100.0)
        )
        exposure_evals.append({
            "param": name,
            "mae_vs_clean": mae,
            "psnr_vs_clean": psnr,
            "stability_pct": stability,
            "inc_to_corr": inc_to_c,
            "corr_to_inc": c_to_inc,
            "utility_score": utility,
        })
        if utility > best_exposure_score:
            best_exposure_score = utility
            best_exposure_cfg = name

    return {
        "blur_evaluations": blur_evals,
        "best_blur": best_blur_cfg,
        "noise_evaluations": noise_evals,
        "best_noise": best_noise_cfg,
        "exposure_evaluations": exposure_evals,
        "best_exposure": best_exposure_cfg,
    }


def main() -> None:
    val_csv = PROJECT_ROOT / "data" / "processed" / "validation.csv"
    test_csv = PROJECT_ROOT / "data" / "processed" / "test.csv"
    dataset_dir = PROJECT_ROOT / "dataset"
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing DenseNet-121 Base Model Agent (frozen)...")
    model = BaseModelAgent()
    model.load_model()
    logger.info(f"Base Model loaded on {model.device}.")

    # 1. Create Tuning Cohort
    cohorts = select_tuning_cohort(val_csv, test_csv, dataset_dir, seed=RANDOM_SEED)

    # 2. Run Sweeps
    records, summary = run_sweeps(cohorts, model)

    # 3. Apply Multi-Metric Selection
    selection = multi_metric_selection(summary)
    summary["selection_analysis"] = selection

    # Record Cohort Metadata
    summary["cohort_metadata"] = {
        "blur_image_ids": [c["image_id"] for c in cohorts["blur"]],
        "noise_image_ids": [c["image_id"] for c in cohorts["noise"]],
        "exposure_image_ids": [c["image_id"] for c in cohorts["exposure"]],
        "all_tuning_image_ids": list(set(
            [c["image_id"] for c in cohorts["blur"]]
            + [c["image_id"] for c in cohorts["noise"]]
            + [c["image_id"] for c in cohorts["exposure"]]
        )),
        "test_dataset_size": len(pd.read_csv(test_csv)),
        "zero_leakage_verified": True,
    }

    # Save outputs
    csv_path = outputs_dir / "repair_parameter_tuning.csv"
    json_path = outputs_dir / "repair_parameter_tuning_summary.json"

    df_records = pd.DataFrame(records)
    df_records.to_csv(csv_path, index=False)
    logger.info(f"Saved tuning details to {csv_path} (rows={len(df_records)})")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved tuning summary to {json_path}")

    logger.info("Sweep execution completed successfully!")


if __name__ == "__main__":
    main()
