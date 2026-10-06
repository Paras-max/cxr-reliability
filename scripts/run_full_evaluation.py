"""Full Test-Set Evaluation Workflow for CXR Reliability Pipeline.

Evaluates ALL 16,724 test images from data/processed/test.csv using the
frozen ReliabilityPipeline configuration with final OOD reference artifacts (artifacts/ood)
and post-hoc Platt probability calibration (outputs/calibration/platt_calibrator.joblib).

IMPORTANT — SAFETY & INTEGRITY:
- Evaluation-only run on frozen configuration.
- Non-destructive: data/processed/test.csv is accessed read-only and never modified.
- No ground-truth labels are passed to the pipeline or used in routing decisions.
- Thresholds, calibrators, and model weights are strictly frozen.
- Checkpointed execution: flushes every 50 images to prevent data loss.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# ── Ensure project root and src are on sys.path ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_reliability.dashboard.app import FULL_OOD_STATS_PATH, load_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("full_eval")


def run_full_evaluation(resume: bool = True, flush_interval: int = 50) -> None:
    test_csv_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    dataset_dir = PROJECT_ROOT / "dataset"
    output_csv_path = PROJECT_ROOT / "outputs" / "evaluation_full_test.csv"
    output_json_path = PROJECT_ROOT / "outputs" / "evaluation_full_test_summary.json"

    logger.info("=" * 70)
    logger.info("STARTING FINAL FULL TEST-SET EVALUATION (16,724 IMAGES)")
    logger.info("=" * 70)
    logger.info("Reading test split from %s", test_csv_path)

    df_test = pd.read_csv(test_csv_path)
    n_total_test = len(df_test)
    logger.info("Loaded %d total test rows", n_total_test)

    # Initialize ReliabilityPipeline using exact app.py loader with artifacts/ood
    logger.info("Initializing ReliabilityPipeline with OOD stats: %s", FULL_OOD_STATS_PATH)
    t0_init = time.perf_counter()
    pipeline, thresholds, device, ood_info = load_pipeline(str(FULL_OOD_STATS_PATH))
    t_init = time.perf_counter() - t0_init
    logger.info(
        "Pipeline initialized in %.2fs (device: %s, OOD samples: %s, is_dev: %s)",
        t_init,
        device,
        ood_info.get("n_samples"),
        ood_info.get("is_dev"),
    )

    # Check for existing checkpoint to resume
    existing_records: list[dict[str, Any]] = []
    completed_ids: set[str] = set()

    if resume and output_csv_path.exists():
        try:
            df_existing = pd.read_csv(output_csv_path)
            existing_records = df_existing.to_dict(orient="records")
            completed_ids = set(df_existing["image_id"].astype(str))
            logger.info("Found existing checkpoint at %s with %d completed images", output_csv_path, len(completed_ids))
        except Exception as exc:
            logger.warning("Could not read existing checkpoint (%s); starting fresh", exc)
            existing_records = []
            completed_ids = set()

    records: list[dict[str, Any]] = list(existing_records)
    quality_rank = {"poor": 0, "degraded": 1, "good": 2}

    logger.info("Starting processing of remaining images (%d completed so far)...", len(completed_ids))
    t_start_loop = time.perf_counter()
    n_processed_this_session = 0

    try:
        for _idx, row in df_test.iterrows():
            img_id = str(row["image_id"])

            if img_id in completed_ids:
                continue

            pat_id = str(row["patient_id"])
            gt_label = int(row["label"])
            gt_name = str(row.get("label_name", "Unknown"))
            finding_labels = str(row.get("finding_labels", ""))
            rel_img_path = str(row["image_path"]).replace("\\", "/")
            full_img_path = dataset_dir / rel_img_path

            if not full_img_path.exists():
                logger.error("Image file missing: %s", full_img_path)
                records.append({
                    "image_id": img_id,
                    "patient_id": pat_id,
                    "ground_truth_label": gt_label,
                    "ground_truth_name": gt_name,
                    "finding_labels": finding_labels,
                    "pipeline_state": "error",
                    "final_action": "reject",
                    "reliability_label": "needs_human_review",
                    "needs_human_review": True,
                    "final_classification": "HUMAN_REVIEW_REQUIRED",
                    "prediction_released": False,
                    "prediction_positive": None,
                    "raw_model_score": None,
                    "calibrated_probability": None,
                    "quality_label": "UNKNOWN",
                    "after_repair_quality_label": None,
                    "quality_improved": False,
                    "quality_unchanged": False,
                    "quality_worsened": False,
                    "ood_level": "UNKNOWN",
                    "mahalanobis_distance": None,
                    "uncertainty_level": "UNKNOWN",
                    "repair_attempted": False,
                    "repair_applied": False,
                    "repair_name": None,
                    "verification_status": "N/A",
                    "verification_next_step": "N/A",
                    "delta_confidence": None,
                    "latency_ms": 0.0,
                    "error_message": f"File not found: {full_img_path}",
                })
                completed_ids.add(img_id)
                n_processed_this_session += 1
                continue

            try:
                res = pipeline.run(full_img_path, input_id=img_id)

                # Determine model scores (even when prediction is withheld)
                base_model_obj = res.after_repair_base_model or res.base_model
                raw_score = None
                cal_prob = None
                pred_pos = None

                if res.prediction is not None:
                    raw_score = res.prediction.raw_model_score
                    cal_prob = res.prediction.calibrated_probability
                    pred_pos = res.prediction.positive
                elif base_model_obj is not None:
                    raw_score = getattr(
                        base_model_obj,
                        "raw_pneumonia_score",
                        getattr(base_model_obj, "pneumonia_probability", None),
                    )
                    cal_prob = getattr(base_model_obj, "calibrated_probability", None)

                # Quality signals & transitions
                q_label = res.quality.overall.value if res.quality else "UNKNOWN"
                after_q_label = res.after_repair_quality.overall.value if res.after_repair_quality else None

                q_improved = False
                q_unchanged = False
                q_worsened = False
                if after_q_label and q_label in quality_rank and after_q_label in quality_rank:
                    diff_rank = quality_rank[after_q_label] - quality_rank[q_label]
                    q_improved = diff_rank > 0
                    q_unchanged = diff_rank == 0
                    q_worsened = diff_rank < 0

                # OOD signals
                ood_lvl = res.ood.level.value if res.ood else "UNKNOWN"
                ood_dist = res.ood.mahalanobis_distance if res.ood else None

                # Uncertainty signals
                u_lvl = res.uncertainty.uncertainty_level.value if res.uncertainty else "UNKNOWN"

                # Repair signals
                repair_attempted = bool(res.repair_attempts > 0)
                repair_applied = bool(res.repair and (res.repair.repair_applied or res.repair.repaired))
                repair_methods = [s.method for s in res.repair.steps] if (res.repair and res.repair.steps) else []
                repair_name = ",".join(repair_methods) if repair_methods else None

                # Verification signals
                ver_status = res.verification.verified if res.verification else "N/A"
                ver_next = res.verification.next_step.value if res.verification else "N/A"
                delta_conf = res.verification.delta_confidence if res.verification else None

                records.append({
                    "image_id": img_id,
                    "patient_id": pat_id,
                    "ground_truth_label": gt_label,
                    "ground_truth_name": gt_name,
                    "finding_labels": finding_labels,
                    "pipeline_state": res.pipeline_state.value,
                    "final_action": res.final_action.value,
                    "reliability_label": res.reliability_label.value,
                    "needs_human_review": res.needs_human_review,
                    "final_classification": res.final_classification.value,
                    "prediction_released": (res.prediction is not None and not res.needs_human_review),
                    "prediction_positive": pred_pos,
                    "raw_model_score": raw_score,
                    "calibrated_probability": cal_prob,
                    "quality_label": q_label,
                    "after_repair_quality_label": after_q_label,
                    "quality_improved": q_improved,
                    "quality_unchanged": q_unchanged,
                    "quality_worsened": q_worsened,
                    "ood_level": ood_lvl,
                    "mahalanobis_distance": ood_dist,
                    "uncertainty_level": u_lvl,
                    "repair_attempted": repair_attempted,
                    "repair_applied": repair_applied,
                    "repair_name": repair_name,
                    "verification_status": ver_status,
                    "verification_next_step": ver_next,
                    "delta_confidence": delta_conf,
                    "latency_ms": res.total_latency_ms,
                    "error_message": res.error_message,
                })

            except Exception as exc:
                logger.exception("Error processing image %s: %s", img_id, exc)
                records.append({
                    "image_id": img_id,
                    "patient_id": pat_id,
                    "ground_truth_label": gt_label,
                    "ground_truth_name": gt_name,
                    "finding_labels": finding_labels,
                    "pipeline_state": "error",
                    "final_action": "reject",
                    "reliability_label": "needs_human_review",
                    "needs_human_review": True,
                    "final_classification": "HUMAN_REVIEW_REQUIRED",
                    "prediction_released": False,
                    "prediction_positive": None,
                    "raw_model_score": None,
                    "calibrated_probability": None,
                    "quality_label": "UNKNOWN",
                    "after_repair_quality_label": None,
                    "quality_improved": False,
                    "quality_unchanged": False,
                    "quality_worsened": False,
                    "ood_level": "UNKNOWN",
                    "mahalanobis_distance": None,
                    "uncertainty_level": "UNKNOWN",
                    "repair_attempted": False,
                    "repair_applied": False,
                    "repair_name": None,
                    "verification_status": "N/A",
                    "verification_next_step": "N/A",
                    "delta_confidence": None,
                    "latency_ms": 0.0,
                    "error_message": str(exc),
                })

            completed_ids.add(img_id)
            n_processed_this_session += 1

            # Checkpoint flush
            if n_processed_this_session % flush_interval == 0:
                output_csv_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(records).to_csv(output_csv_path, index=False)
                elapsed_s = time.perf_counter() - t_start_loop
                rate = elapsed_s / n_processed_this_session if n_processed_this_session > 0 else 0.0
                remaining = n_total_test - len(records)
                eta_s = remaining * rate
                logger.info(
                    "Checkpoint [%d/%d images (%.1f%%)]: elapsed %.1fs, rate %.2fs/img, ETA %.1f min (%.2f hr)",
                    len(records),
                    n_total_test,
                    len(records) / n_total_test * 100.0,
                    elapsed_s,
                    rate,
                    eta_s / 60.0,
                    eta_s / 3600.0,
                )

    finally:
        # Final save of CSV even on keyboard interrupt or exception
        if records:
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_out = pd.DataFrame(records)
            df_out.to_csv(output_csv_path, index=False)
            logger.info("Saved final CSV to %s with %d records", output_csv_path, len(df_out))

    # Compute comprehensive summary
    df_out = pd.DataFrame(records)
    n_total = len(df_out)
    n_errors = int((df_out["pipeline_state"] == "error").sum())
    failed_ids = df_out.loc[df_out["pipeline_state"] == "error", "image_id"].tolist()
    n_successful = n_total - n_errors

    # 1. Action counts
    action_counts = df_out["final_action"].str.upper().value_counts().to_dict()
    n_accept = action_counts.get("ACCEPT", 0)
    n_repair = action_counts.get("REPAIR", 0)
    n_escalate = action_counts.get("ESCALATE", 0)
    n_reject = action_counts.get("REJECT", 0)

    # 2. Human review & prediction coverage
    n_human_review = int(df_out["needs_human_review"].sum())
    n_released = int(df_out["prediction_released"].sum())
    n_withheld = n_total - n_released
    coverage_pct = (n_released / n_total * 100.0) if n_total > 0 else 0.0

    # 3. Quality distribution
    q_counts = df_out["quality_label"].str.lower().value_counts().to_dict()
    n_q_good = q_counts.get("good", 0)
    n_q_degraded = q_counts.get("degraded", 0)
    n_q_poor = q_counts.get("poor", 0)

    # 4. OOD distribution
    ood_counts = df_out["ood_level"].str.lower().value_counts().to_dict()
    n_ood_in = ood_counts.get("in_distribution", 0)
    n_ood_borderline = ood_counts.get("borderline", 0)
    n_ood_severe = ood_counts.get("severe", 0)

    # 5. Uncertainty distribution
    u_counts = df_out["uncertainty_level"].str.upper().value_counts().to_dict()
    n_u_low = u_counts.get("LOW", 0)
    n_u_medium = u_counts.get("MEDIUM", 0)
    n_u_high = u_counts.get("HIGH", 0)

    # 6. Repair breakdown
    n_repair_attempts = int(df_out["repair_attempted"].sum())
    n_repairs_applied = int(df_out["repair_applied"].sum())
    n_repairs_skipped = n_repair_attempts - n_repairs_applied

    df_applied = df_out[df_out["repair_applied"]]
    n_q_improved = int(df_applied["quality_improved"].sum()) if not df_applied.empty else 0
    n_q_unchanged = int(df_applied["quality_unchanged"].sum()) if not df_applied.empty else 0
    n_q_worsened = int(df_applied["quality_worsened"].sum()) if not df_applied.empty else 0

    # 7. Verification breakdown
    _df_ver = df_out[df_out["verification_status"].isin([True, False, "True", "False"])]
    n_verified = int((df_out["verification_status"].isin([True, "True"])).sum())
    n_ver_escalated = int((df_out["verification_next_step"] == "escalate").sum())
    n_ver_rejected = int((df_out["verification_next_step"] == "reject").sum())

    # Confidence delta statistics
    delta_s = df_out["delta_confidence"].dropna()
    conf_stats = {
        "count": len(delta_s),
        "mean": float(delta_s.mean()) if not delta_s.empty else None,
        "median": float(delta_s.median()) if not delta_s.empty else None,
        "std": float(delta_s.std()) if len(delta_s) > 1 else None,
        "min": float(delta_s.min()) if not delta_s.empty else None,
        "max": float(delta_s.max()) if not delta_s.empty else None,
        "n_pos": int((delta_s > 0).sum()) if not delta_s.empty else 0,
        "n_neg": int((delta_s < 0).sum()) if not delta_s.empty else 0,
        "n_ge_15": int((delta_s >= 0.15).sum()) if not delta_s.empty else 0,
    }

    # 8. Classification metrics on RELEASED predictions only
    df_released = df_out[df_out["prediction_released"]]
    class_metrics: dict[str, Any] = {
        "n_total": n_total,
        "n_released": n_released,
        "n_withheld": n_withheld,
        "coverage_pct": coverage_pct,
    }

    if n_released > 0:
        y_true = df_released["ground_truth_label"].astype(int).to_numpy()
        y_pred = df_released["prediction_positive"].astype(bool).astype(int).to_numpy()

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        accuracy = float((tp + tn) / n_released)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

        class_metrics.update({
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall_sensitivity": recall,
            "specificity": specificity,
            "npv": npv,
            "f1_score": f1,
        })

    # 9. Latency distribution
    latencies = df_out.loc[df_out["latency_ms"] > 0, "latency_ms"]
    latency_stats = {
        "mean_ms": float(latencies.mean()) if not latencies.empty else 0.0,
        "median_ms": float(latencies.median()) if not latencies.empty else 0.0,
        "p95_ms": float(latencies.quantile(0.95)) if not latencies.empty else 0.0,
        "min_ms": float(latencies.min()) if not latencies.empty else 0.0,
        "max_ms": float(latencies.max()) if not latencies.empty else 0.0,
        "total_wall_time_s": time.perf_counter() - t_start_loop,
    }

    # 10. Ground truth distribution
    gt_s = df_out["ground_truth_label"].astype(int)
    n_pneumonia = int((gt_s == 1).sum())
    n_non_pneumonia = int((gt_s == 0).sum())
    prevalence = float(n_pneumonia / n_total) if n_total > 0 else 0.0

    summary = {
        "evaluation_name": "final_full_test_evaluation",
        "description": "FINAL TEST-SET RESULTS FOR THE FROZEN CONFIGURATION",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ood_reference_dir": str(FULL_OOD_STATS_PATH),
        "total_images": n_total,
        "successful_runs": n_successful,
        "errors": n_errors,
        "failed_image_ids": failed_ids,
        "actions": {
            "ACCEPT": n_accept,
            "REPAIR": n_repair,
            "ESCALATE": n_escalate,
            "REJECT": n_reject,
        },
        "human_review": {
            "needs_human_review": n_human_review,
            "predictions_withheld": n_withheld,
            "predictions_released": n_released,
            "coverage_pct": coverage_pct,
        },
        "quality_distribution": {
            "GOOD": n_q_good,
            "DEGRADED": n_q_degraded,
            "POOR": n_q_poor,
        },
        "ood_distribution": {
            "IN_DISTRIBUTION": n_ood_in,
            "BORDERLINE": n_ood_borderline,
            "SEVERE": n_ood_severe,
        },
        "uncertainty_distribution": {
            "LOW": n_u_low,
            "MEDIUM": n_u_medium,
            "HIGH": n_u_high,
        },
        "repair": {
            "repair_attempts": n_repair_attempts,
            "repairs_applied": n_repairs_applied,
            "repairs_skipped": n_repairs_skipped,
            "quality_improved": n_q_improved,
            "quality_unchanged": n_q_unchanged,
            "quality_worsened": n_q_worsened,
        },
        "verification": {
            "verified": n_verified,
            "escalated": n_ver_escalated,
            "rejected": n_ver_rejected,
            "confidence_delta": conf_stats,
        },
        "classification_on_released_predictions": class_metrics,
        "latency": latency_stats,
        "ground_truth": {
            "pneumonia_count": n_pneumonia,
            "non_pneumonia_count": n_non_pneumonia,
            "prevalence": prevalence,
        },
    }

    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Summary saved to %s", output_json_path)

    # Print summary to terminal
    print("\n" + "=" * 70)
    print("FINAL TEST-SET RESULTS FOR THE FROZEN CONFIGURATION")
    print("=" * 70)
    print(f"Total Test Images:     {n_total}")
    print(f"Successful Runs:       {n_successful}")
    print(f"Errors:                {n_errors}")
    print("-" * 70)
    print("ACTION DISTRIBUTION:")
    print(f"  ACCEPT:              {n_accept} ({n_accept / n_total * 100:.1f}%)")
    print(f"  REPAIR (terminal):   {n_repair} ({n_repair / n_total * 100:.1f}%)")
    print(f"  ESCALATE:            {n_escalate} ({n_escalate / n_total * 100:.1f}%)")
    print(f"  REJECT:              {n_reject} ({n_reject / n_total * 100:.1f}%)")
    print("-" * 70)
    print("HUMAN REVIEW & PREDICTION COVERAGE:")
    print(f"  Needs Human Review:  {n_human_review} ({n_human_review / n_total * 100:.1f}%)")
    print(f"  Predictions Released:{n_released} ({coverage_pct:.1f}%)")
    print(f"  Predictions Withheld:{n_withheld} ({n_withheld / n_total * 100:.1f}%)")
    print("-" * 70)
    print("QUALITY DISTRIBUTION:")
    print(f"  GOOD:                {n_q_good} ({n_q_good / n_total * 100:.1f}%)")
    print(f"  DEGRADED:            {n_q_degraded} ({n_q_degraded / n_total * 100:.1f}%)")
    print(f"  POOR:                {n_q_poor} ({n_q_poor / n_total * 100:.1f}%)")
    print("-" * 70)
    print("OOD DISTRIBUTION:")
    print(f"  IN_DISTRIBUTION:     {n_ood_in} ({n_ood_in / n_total * 100:.1f}%)")
    print(f"  BORDERLINE:          {n_ood_borderline} ({n_ood_borderline / n_total * 100:.1f}%)")
    print(f"  SEVERE:              {n_ood_severe} ({n_ood_severe / n_total * 100:.1f}%)")
    print("-" * 70)
    print("UNCERTAINTY DISTRIBUTION:")
    print(f"  LOW:                 {n_u_low} ({n_u_low / n_total * 100:.1f}%)")
    print(f"  MEDIUM:              {n_u_medium} ({n_u_medium / n_total * 100:.1f}%)")
    print(f"  HIGH:                {n_u_high} ({n_u_high / n_total * 100:.1f}%)")
    print("-" * 70)
    print("REPAIR & VERIFICATION:")
    print(f"  Repair Attempts:     {n_repair_attempts}")
    print(f"  Repairs Applied:     {n_repairs_applied}")
    print(f"  Repairs Skipped:     {n_repairs_skipped}")
    print(f"  Quality Improved:    {n_q_improved}")
    print(f"  Quality Unchanged:   {n_q_unchanged}")
    print(f"  Quality Worsened:    {n_q_worsened}")
    print(f"  Verified Repairs:    {n_verified}")
    print(f"  Escalated Repairs:   {n_ver_escalated}")
    print(f"  Delta Conf (Mean):   {conf_stats.get('mean')}")
    print(f"  Delta Conf (Median): {conf_stats.get('median')}")
    print(f"  Delta Conf >= 0.15:  {conf_stats.get('n_ge_15')}")
    print("-" * 70)
    print("CLASSIFICATION PERFORMANCE (Released Predictions Only):")
    if n_released > 0:
        print(f"  Accuracy:            {class_metrics.get('accuracy'):.4f}")
        print(f"  Precision:           {class_metrics.get('precision'):.4f}")
        print(f"  Recall / Sens:       {class_metrics.get('recall_sensitivity'):.4f}")
        print(f"  Specificity:         {class_metrics.get('specificity'):.4f}")
        print(f"  F1 Score:            {class_metrics.get('f1_score'):.4f}")
    else:
        print("  [No predictions released autonomously — all cases safely gated for human review]")
    print("-" * 70)
    print("LATENCY METRICS:")
    print(f"  Mean Latency:        {latency_stats['mean_ms']:.1f} ms")
    print(f"  Median Latency:      {latency_stats['median_ms']:.1f} ms")
    print(f"  P95 Latency:         {latency_stats['p95_ms']:.1f} ms")
    print(f"  Min / Max:           {latency_stats['min_ms']:.1f} ms / {latency_stats['max_ms']:.1f} ms")
    print(f"  Total Wall Time:     {latency_stats['total_wall_time_s']:.1f} s ({latency_stats['total_wall_time_s'] / 3600.0:.2f} hr)")
    print("-" * 70)
    print("GROUND TRUTH PREVALENCE:")
    print(f"  Pneumonia:           {n_pneumonia} ({prevalence * 100:.2f}%)")
    print(f"  Non-Pneumonia:       {n_non_pneumonia} ({(1 - prevalence) * 100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Test Evaluation")
    parser.add_argument("--no-resume", action="store_true", help="Start from beginning")
    parser.add_argument("--flush-interval", type=int, default=50, help="Checkpoint frequency")
    args = parser.parse_args()

    run_full_evaluation(resume=not args.no_resume, flush_interval=args.flush_interval)
