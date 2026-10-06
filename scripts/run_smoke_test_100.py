"""100-Image Evaluation Smoke Test for CXR Reliability Pipeline.

Runs the frozen ReliabilityPipeline on the first 100 rows of data/processed/test.csv
using the final OOD reference artifacts (artifacts/ood).

IMPORTANT:
- Smoke test only. NOT the final test-set evaluation.
- Strictly non-destructive: does not modify test.csv.
- Does not modify or fit any models, calibrators, or thresholds.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure src is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_reliability.dashboard.app import FULL_OOD_STATS_PATH, load_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_test_100")


def run_smoke_test() -> None:
    test_csv_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    dataset_dir = PROJECT_ROOT / "dataset"
    output_csv_path = PROJECT_ROOT / "outputs" / "evaluation_smoke_100.csv"
    output_json_path = PROJECT_ROOT / "outputs" / "evaluation_smoke_100_summary.json"

    logger.info("Reading first 100 rows from %s", test_csv_path)
    df_test = pd.read_csv(test_csv_path, nrows=100)
    logger.info("Loaded %d rows for smoke evaluation", len(df_test))

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

    records: list[dict[str, Any]] = []

    logger.info("Starting processing of 100 images...")
    t_start_all = time.perf_counter()

    for idx, row in df_test.iterrows():
        img_id = str(row["image_id"])
        pat_id = str(row["patient_id"])
        gt_label = int(row["label"])
        gt_name = str(row.get("label_name", "Unknown"))
        rel_img_path = str(row["image_path"]).replace("\\", "/")
        full_img_path = dataset_dir / rel_img_path

        if not full_img_path.exists():
            logger.error("Image file missing: %s", full_img_path)
            records.append({
                "image_id": img_id,
                "patient_id": pat_id,
                "ground_truth_label": gt_label,
                "ground_truth_name": gt_name,
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
                "ood_level": "UNKNOWN",
                "mahalanobis_distance": None,
                "uncertainty_level": "UNKNOWN",
                "repair_attempted": False,
                "repair_applied": False,
                "repair_name": None,
                "verification_status": "N/A",
                "verification_next_step": "N/A",
                "latency_ms": 0.0,
                "error_message": f"File not found: {full_img_path}",
            })
            continue

        try:
            res = pipeline.run(full_img_path, input_id=img_id)

            # Determine scores (extract even if prediction is withheld)
            base_model_obj = res.after_repair_base_model or res.base_model
            raw_score = None
            cal_prob = None
            pred_pos = None

            if res.prediction is not None:
                raw_score = res.prediction.raw_model_score
                cal_prob = res.prediction.calibrated_probability
                pred_pos = res.prediction.positive
            elif base_model_obj is not None:
                raw_score = getattr(base_model_obj, "raw_pneumonia_score", getattr(base_model_obj, "pneumonia_probability", None))
                cal_prob = getattr(base_model_obj, "calibrated_probability", None)

            # Quality info
            q_label = res.quality.overall.value if res.quality else "UNKNOWN"
            after_q_label = res.after_repair_quality.overall.value if res.after_repair_quality else None

            # OOD info
            ood_lvl = res.ood.level.value if res.ood else "UNKNOWN"
            ood_dist = res.ood.mahalanobis_distance if res.ood else None

            # Uncertainty info
            u_lvl = res.uncertainty.uncertainty_level.value if res.uncertainty else "UNKNOWN"

            # Repair info
            repair_attempted = bool(res.repair_attempts > 0)
            repair_applied = bool(res.repair and (res.repair.repair_applied or res.repair.repaired))
            repair_methods = [s.method for s in res.repair.steps] if (res.repair and res.repair.steps) else []
            repair_name = ",".join(repair_methods) if repair_methods else None

            # Verification info
            ver_status = res.verification.verified if res.verification else "N/A"
            ver_next = res.verification.next_step.value if res.verification else "N/A"

            records.append({
                "image_id": img_id,
                "patient_id": pat_id,
                "ground_truth_label": gt_label,
                "ground_truth_name": gt_name,
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
                "ood_level": ood_lvl,
                "mahalanobis_distance": ood_dist,
                "uncertainty_level": u_lvl,
                "repair_attempted": repair_attempted,
                "repair_applied": repair_applied,
                "repair_name": repair_name,
                "verification_status": ver_status,
                "verification_next_step": ver_next,
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
                "ood_level": "UNKNOWN",
                "mahalanobis_distance": None,
                "uncertainty_level": "UNKNOWN",
                "repair_attempted": False,
                "repair_applied": False,
                "repair_name": None,
                "verification_status": "N/A",
                "verification_next_step": "N/A",
                "latency_ms": 0.0,
                "error_message": str(exc),
            })

        if (idx + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_start_all
            logger.info("Processed %d/100 images (elapsed: %.1fs, avg: %.2fs/img)", idx + 1, elapsed, elapsed / (idx + 1))

    total_time = time.perf_counter() - t_start_all
    logger.info("Completed processing 100 images in %.2fs", total_time)

    # Convert to DataFrame and save
    df_out = pd.DataFrame(records)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)
    logger.info("Saved evaluation results to %s", output_csv_path)

    # Aggregate metrics
    n_total = len(df_out)
    n_errors = int((df_out["pipeline_state"] == "error").sum())
    n_successful = n_total - n_errors

    # Action counts (case-insensitive)
    action_counts = df_out["final_action"].str.upper().value_counts().to_dict()
    n_accept = action_counts.get("ACCEPT", 0)
    n_repair = action_counts.get("REPAIR", 0)
    n_escalate = action_counts.get("ESCALATE", 0)
    n_reject = action_counts.get("REJECT", 0)

    # Human review counts
    n_human_review = int(df_out["needs_human_review"].sum())

    # Repair counts
    n_repair_attempts = int(df_out["repair_attempted"].sum())
    n_verified_repairs = int((df_out["verification_status"].isin([True, "True"])).sum())

    # Latencies
    latencies = df_out.loc[df_out["latency_ms"] > 0, "latency_ms"]
    avg_latency_ms = float(latencies.mean()) if not latencies.empty else 0.0
    median_latency_ms = float(latencies.median()) if not latencies.empty else 0.0

    # Classification metrics on RELEASED predictions only
    df_released = df_out[df_out["prediction_released"]]
    n_released = len(df_released)
    release_rate = n_released / n_total if n_total > 0 else 0.0

    class_metrics: dict[str, Any] = {
        "n_total": n_total,
        "n_released": n_released,
        "n_withheld": n_total - n_released,
        "release_rate": release_rate,
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
            "ground_truth_positives": int((y_true == 1).sum()),
            "ground_truth_negatives": int((y_true == 0).sum()),
        })

    summary = {
        "evaluation_name": "100_image_smoke_test",
        "description": "Initial 100-image smoke test using the frozen pipeline and final OOD reference stats",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ood_reference_dir": str(FULL_OOD_STATS_PATH),
        "total_images": n_total,
        "successful_runs": n_successful,
        "errors": n_errors,
        "actions": {
            "ACCEPT": n_accept,
            "REPAIR": n_repair,
            "ESCALATE": n_escalate,
            "REJECT": n_reject,
        },
        "needs_human_review": n_human_review,
        "repair_attempts": n_repair_attempts,
        "verified_repairs": n_verified_repairs,
        "latency": {
            "average_ms": avg_latency_ms,
            "median_ms": median_latency_ms,
            "total_runtime_s": total_time,
        },
        "classification_on_released_predictions": class_metrics,
    }

    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Smoke test summary saved to %s", output_json_path)

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("100-IMAGE SMOKE TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Images:          {n_total}")
    print(f"Successful Runs:       {n_successful}")
    print(f"Errors:                {n_errors}")
    print("-" * 60)
    print("ACTION DISTRIBUTION:")
    print(f"  ACCEPT:              {n_accept}")
    print(f"  REPAIR:              {n_repair}")
    print(f"  ESCALATE:            {n_escalate}")
    print(f"  REJECT:              {n_reject}")
    print("-" * 60)
    print("SAFETY & HUMAN REVIEW:")
    print(f"  Needs Human Review:  {n_human_review} ({n_human_review / n_total * 100:.1f}%)")
    print(f"  Repair Attempts:     {n_repair_attempts}")
    print(f"  Verified Repairs:    {n_verified_repairs}")
    print("-" * 60)
    print("LATENCY:")
    print(f"  Average Latency:     {avg_latency_ms:.1f} ms ({avg_latency_ms / 1000.0:.2f} s)")
    print(f"  Median Latency:      {median_latency_ms:.1f} ms ({median_latency_ms / 1000.0:.2f} s)")
    print(f"  Total Wall Time:     {total_time:.1f} s")
    print("-" * 60)
    print("PREDICTION RELEASE & PERFORMANCE (Released Predictions Only):")
    print(f"  Predictions Released: {n_released}/{n_total} ({release_rate * 100:.1f}%)")
    print(f"  Predictions Withheld: {n_total - n_released} ({(n_total - n_released) / n_total * 100:.1f}%)")
    if n_released > 0:
        print(f"  Released TP: {class_metrics.get('tp')}, FP: {class_metrics.get('fp')}, TN: {class_metrics.get('tn')}, FN: {class_metrics.get('fn')}")
        print(f"  Released Accuracy:    {class_metrics.get('accuracy'):.4f}")
        print(f"  Released Precision:   {class_metrics.get('precision'):.4f}")
        print(f"  Released Recall/Sens: {class_metrics.get('recall_sensitivity'):.4f}")
        print(f"  Released Specificity: {class_metrics.get('specificity'):.4f}")
        print(f"  Released F1 Score:    {class_metrics.get('f1_score'):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
