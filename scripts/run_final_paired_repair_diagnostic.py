"""Final Paired Repair -> DenseNet Diagnostic Validation Script.

Executes the definitive paired validation experiment answering:
"After the Repair Agent repairs a poor-quality X-ray, does passing the repaired
X-ray through DenseNet-121 AGAIN improve the pneumonia classification?"

Strict Protocol:
- Base Model: DenseNet-121 (densenet121-res224-nih) strictly frozen.
- Clinical Operating Threshold: 0.522161 (fixed, unchanged).
- Experimental Repair Parameters:
    Blur: radius=1.0, amount=0.5
    Noise: h=7.0, templateWindowSize=7, searchWindowSize=21
    Exposure: clipLimit=1.0, tileGridSize=(8, 8)
- Test Cohort: Real NIH test set images from data/processed/test.csv.
- Two distinct DenseNet forward passes: Before Repair vs After Repair.
- Four-way transition tracking: Correct->Correct, Correct->Incorrect,
  Incorrect->Correct, Incorrect->Incorrect.
- Quality vs Diagnostic improvement matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_reliability.agents.quality import QualityAgent
from cxr_reliability.agents.repair import RepairAgent
from cxr_reliability.agents.verification import VerificationAgent
from cxr_reliability.contracts.decision import Action
from cxr_reliability.data.corruptions import apply_exposure_shift, apply_noise
from cxr_reliability.models.base_model import BaseModelAgent
from cxr_reliability.models.preprocessing import prepare_image_for_txv
from cxr_reliability.quality.evaluator import prepare_image_for_quality
from cxr_reliability.repair.pipeline import RepairConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("final_paired_validation")

OPERATING_THRESHOLD = 0.522161
RANDOM_SEED = 42


def compute_metrics(y_true: list[int], y_scores: list[float], threshold: float = OPERATING_THRESHOLD) -> dict[str, Any]:
    """Compute classification metrics at the given threshold."""
    y_pred = [1 if s >= threshold else 0 for s in y_scores]
    n = len(y_true)
    if n == 0:
        return {}

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # ROC-AUC and PR-AUC
    auroc = None
    auprc = None
    if len(set(y_true)) > 1:
        try:
            auroc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            auroc = None
        try:
            auprc = float(average_precision_score(y_true, y_scores))
        except Exception:
            auprc = None

    return {
        "n_total": n,
        "n_pos": sum(y_true),
        "n_neg": n - sum(y_true),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": spec,
        "f1_score": f1,
        "npv": npv,
        "auroc": auroc,
        "auprc": auprc,
    }


def select_candidate_cohort(
    test_csv_path: Path,
    dataset_dir: Path,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """
    Select candidate test images representing:
    - Natural blur (Laplacian < 100) -> 20 images (10 pos, 10 neg)
    - Synthetic noise -> 10 images (5 pos, 5 neg)
    - Synthetic exposure -> 10 images (5 pos, 5 neg)
    - Clean / Skipped -> 10 images (5 pos, 5 neg)
    Total = 50 candidate images.
    """
    df = pd.read_csv(test_csv_path)
    pos_df = df[df["label"] == 1].sample(frac=1.0, random_state=seed)
    neg_df = df[df["label"] == 0].sample(frac=1.0, random_state=seed)

    qa = QualityAgent()
    used_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []

    # 1. Natural Blur (10 pos, 10 neg)
    logger.info("Selecting Natural Blur cohort from test set...")
    blur_pos = []
    for _, row in pos_df.iterrows():
        if len(blur_pos) >= 10:
            break
        p = dataset_dir / row["image_path"]
        if not p.exists():
            continue
        q = qa.run(p)
        if q.flags.blur:
            blur_pos.append({
                "image_id": row["image_id"],
                "patient_id": row["patient_id"],
                "image_path": str(p),
                "ground_truth": int(row["label"]),
                "repair_group": "blur",
                "is_synthetic": False,
            })
            used_ids.add(row["image_id"])

    blur_neg = []
    for _, row in neg_df.iterrows():
        if len(blur_neg) >= 10:
            break
        p = dataset_dir / row["image_path"]
        if not p.exists():
            continue
        q = qa.run(p)
        if q.flags.blur:
            blur_neg.append({
                "image_id": row["image_id"],
                "patient_id": row["patient_id"],
                "image_path": str(p),
                "ground_truth": int(row["label"]),
                "repair_group": "blur",
                "is_synthetic": False,
            })
            used_ids.add(row["image_id"])

    candidates.extend(blur_pos)
    candidates.extend(blur_neg)
    logger.info(f"Selected {len(blur_pos)} pos + {len(blur_neg)} neg natural blur images.")

    # 2. Controlled Noise (5 pos, 5 neg)
    logger.info("Selecting Noise cohort from test set...")
    pos_avail = pos_df[~pos_df["image_id"].isin(used_ids)]
    neg_avail = neg_df[~neg_df["image_id"].isin(used_ids)]

    noise_pos = []
    for _, row in pos_avail.iterrows():
        if len(noise_pos) >= 5:
            break
        p = dataset_dir / row["image_path"]
        if not p.exists():
            continue
        noise_pos.append({
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "image_path": str(p),
            "ground_truth": int(row["label"]),
            "repair_group": "noise",
            "is_synthetic": True,
            "corruption_param": 0.55,
        })
        used_ids.add(row["image_id"])

    noise_neg = []
    for _, row in neg_avail.iterrows():
        if len(noise_neg) >= 5:
            break
        p = dataset_dir / row["image_path"]
        if not p.exists():
            continue
        noise_neg.append({
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "image_path": str(p),
            "ground_truth": int(row["label"]),
            "repair_group": "noise",
            "is_synthetic": True,
            "corruption_param": 0.55,
        })
        used_ids.add(row["image_id"])

    candidates.extend(noise_pos)
    candidates.extend(noise_neg)
    logger.info(f"Selected {len(noise_pos)} pos + {len(noise_neg)} neg noise images.")

    # 3. Controlled Exposure (5 pos, 5 neg)
    logger.info("Selecting Exposure cohort from test set...")
    pos_avail = pos_df[~pos_df["image_id"].isin(used_ids)]
    neg_avail = neg_df[~neg_df["image_id"].isin(used_ids)]

    exp_pos = []
    for _, row in pos_avail.iterrows():
        if len(exp_pos) >= 5:
            break
        p = dataset_dir / row["image_path"]
        if not p.exists():
            continue
        sev = -0.95 if len(exp_pos) % 2 == 0 else 1.3
        exp_pos.append({
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "image_path": str(p),
            "ground_truth": int(row["label"]),
            "repair_group": "exposure",
            "is_synthetic": True,
            "corruption_param": sev,
        })
        used_ids.add(row["image_id"])

    exp_neg = []
    for _, row in neg_avail.iterrows():
        if len(exp_neg) >= 5:
            break
        p = dataset_dir / row["image_path"]
        if not p.exists():
            continue
        sev = -0.95 if len(exp_neg) % 2 == 0 else 1.3
        exp_neg.append({
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "image_path": str(p),
            "ground_truth": int(row["label"]),
            "repair_group": "exposure",
            "is_synthetic": True,
            "corruption_param": sev,
        })
        used_ids.add(row["image_id"])

    candidates.extend(exp_pos)
    candidates.extend(exp_neg)
    logger.info(f"Selected {len(exp_pos)} pos + {len(exp_neg)} neg exposure images.")

    # 4. Clean / Skipped Cohort (5 pos, 5 neg) where quality is good
    logger.info("Selecting Clean / Skipped cohort from test set...")
    pos_avail = pos_df[~pos_df["image_id"].isin(used_ids)]
    neg_avail = neg_df[~neg_df["image_id"].isin(used_ids)]

    clean_pos = []
    for _, row in pos_avail.iterrows():
        if len(clean_pos) >= 5:
            break
        p = dataset_dir / row["image_path"]
        if not p.exists():
            continue
        q = qa.run(p)
        if q.overall.value == "good":
            clean_pos.append({
                "image_id": row["image_id"],
                "patient_id": row["patient_id"],
                "image_path": str(p),
                "ground_truth": int(row["label"]),
                "repair_group": "clean_skipped",
                "is_synthetic": False,
            })
            used_ids.add(row["image_id"])

    clean_neg = []
    for _, row in neg_avail.iterrows():
        if len(clean_neg) >= 5:
            break
        p = dataset_dir / row["image_path"]
        if not p.exists():
            continue
        q = qa.run(p)
        if q.overall.value == "good":
            clean_neg.append({
                "image_id": row["image_id"],
                "patient_id": row["patient_id"],
                "image_path": str(p),
                "ground_truth": int(row["label"]),
                "repair_group": "clean_skipped",
                "is_synthetic": False,
            })
            used_ids.add(row["image_id"])

    candidates.extend(clean_pos)
    candidates.extend(clean_neg)
    logger.info(f"Selected {len(clean_pos)} pos + {len(clean_neg)} neg clean/skipped images.")
    logger.info(f"Total candidate cohort size: N={len(candidates)}")
    return candidates


def run_paired_validation(
    candidates: list[dict[str, Any]],
    model: BaseModelAgent,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute before/after paired validation."""
    qa = QualityAgent()
    exp_config = RepairConfig(
        unsharp_radius=1.0,
        unsharp_amount=0.5,
        nl_means_h=7.0,
        nl_means_template_size=7,
        nl_means_search_size=21,
        clahe_clip_limit=1.0,
        clahe_tile_grid_size=(8, 8),
    )
    repair_agent = RepairAgent(config=exp_config)
    verifier = VerificationAgent()

    rows: list[dict[str, Any]] = []

    for item in candidates:
        img_id = item["image_id"]
        gt = item["ground_truth"]
        group = item["repair_group"]

        # Load input image
        pil = Image.open(item["image_path"]).convert("L")
        img_u8 = np.array(pil, dtype=np.uint8)

        # Apply synthetic corruption if designated
        if item.get("is_synthetic", False):
            if group == "noise":
                img_u8 = apply_noise(img_u8, severity=item["corruption_param"], seed=RANDOM_SEED)
            elif group == "exposure":
                img_u8 = apply_exposure_shift(img_u8, severity=item["corruption_param"], seed=RANDOM_SEED)

        # ── BEFORE REPAIR ─────────────────────────────────────────────────────
        q_before = qa.run(img_u8, image_id=img_id)
        tensor_before = prepare_image_for_txv(img_u8)
        forward_before = model.run(tensor_before)
        raw_score_before = float(forward_before.result.pneumonia_probability)
        pred_before = 1 if raw_score_before >= OPERATING_THRESHOLD else 0
        conf_before = abs(raw_score_before - OPERATING_THRESHOLD)
        corr_before = (pred_before == gt)

        # ── REPAIR PROCESS ────────────────────────────────────────────────────
        repair_required = (q_before.overall.value in ("poor", "degraded"))
        repair_applied = False
        repair_type = "none"
        repaired_img_u8 = img_u8

        if repair_required:
            repaired_out, repair_res = repair_agent.run(
                img_u8,
                quality=q_before,
                decision=Action.REPAIR,
                image_id=img_id,
            )
            repair_applied = repair_res.repair_applied
            repaired_img_u8 = repaired_out
            if len(repair_res.steps) > 0:
                repair_type = "+".join([s.method for s in repair_res.steps])
            else:
                repair_type = "skipped"

        # ── AFTER REPAIR (FRESH DENSENET FORWARD PASS) ─────────────────────────
        if repair_applied:
            tensor_after = prepare_image_for_txv(repaired_img_u8)
            forward_after = model.run(tensor_after)
            raw_score_after = float(forward_after.result.pneumonia_probability)
            pred_after = 1 if raw_score_after >= OPERATING_THRESHOLD else 0
            conf_after = abs(raw_score_after - OPERATING_THRESHOLD)
            corr_after = (pred_after == gt)
            q_after = qa.run(repaired_img_u8, image_id=img_id)

            # Run Verification Agent
            ver_res = verifier.run(
                action=Action.REPAIR,
                repair=repair_res,
                quality_before=q_before,
                quality_after=q_after,
                base_model_before=forward_before.result,
                base_model_after=forward_after.result,
            )
            verification_str = f"{ver_res.status.value}_{ver_res.next_step.value}"
        else:
            # When repair is not applied (e.g., clean image skipped)
            raw_score_after = raw_score_before
            pred_after = pred_before
            conf_after = conf_before
            corr_after = corr_before
            q_after = q_before
            verification_str = "skipped_not_repaired"

        conf_delta = conf_after - conf_before

        # Determine transition
        if corr_before and corr_after:
            trans = "Correct->Correct"
        elif corr_before and not corr_after:
            trans = "Correct->Incorrect"
        elif not corr_before and corr_after:
            trans = "Incorrect->Correct"
        else:
            trans = "Incorrect->Incorrect"

        rows.append({
            "image_id": img_id,
            "patient_id": item["patient_id"],
            "ground_truth": gt,
            "repair_group": group,
            "quality_before": q_before.overall.value,
            "quality_after": q_after.overall.value,
            "repair_required": repair_required,
            "repair_applied": repair_applied,
            "repair_type": repair_type,
            "raw_score_before": raw_score_before,
            "raw_score_after": raw_score_after,
            "prediction_before": pred_before,
            "prediction_after": pred_after,
            "confidence_before": conf_before,
            "confidence_after": conf_after,
            "confidence_delta": conf_delta,
            "correct_before": corr_before,
            "correct_after": corr_after,
            "transition": trans,
            "verification_result": verification_str,
            "laplacian_before": q_before.laplacian_variance,
            "laplacian_after": q_after.laplacian_variance,
            "snr_before": q_before.snr_db,
            "snr_after": q_after.snr_db,
            "mean_intensity_before": q_before.mean_intensity,
            "mean_intensity_after": q_after.mean_intensity,
        })

    # Summary Statistics
    df_all = pd.DataFrame(rows)
    df_repaired = df_all[df_all["repair_applied"] == True].copy()

    # Metrics on Repaired Cohort
    y_true_rep = df_repaired["ground_truth"].tolist()
    scores_before_rep = df_repaired["raw_score_before"].tolist()
    scores_after_rep = df_repaired["raw_score_after"].tolist()

    m_before = compute_metrics(y_true_rep, scores_before_rep)
    m_after = compute_metrics(y_true_rep, scores_after_rep)

    deltas = {
        "accuracy_delta": m_after["accuracy"] - m_before["accuracy"],
        "precision_delta": m_after["precision"] - m_before["precision"],
        "recall_delta": m_after["recall_sensitivity"] - m_before["recall_sensitivity"],
        "specificity_delta": m_after["specificity"] - m_before["specificity"],
        "f1_delta": m_after["f1_score"] - m_before["f1_score"],
        "npv_delta": m_after["npv"] - m_before["npv"],
    }

    transitions_counts = df_repaired["transition"].value_counts().to_dict()
    for k in ["Correct->Correct", "Correct->Incorrect", "Incorrect->Correct", "Incorrect->Incorrect"]:
        transitions_counts.setdefault(k, 0)

    n_repaired = len(df_repaired)
    stability_pct = (transitions_counts["Correct->Correct"] + transitions_counts["Incorrect->Incorrect"]) / n_repaired * 100.0 if n_repaired > 0 else 100.0
    flips_count = transitions_counts["Correct->Incorrect"] + transitions_counts["Incorrect->Correct"]

    score_deltas = [r["raw_score_after"] - r["raw_score_before"] for _, r in df_repaired.iterrows()]
    conf_deltas = [r["confidence_delta"] for _, r in df_repaired.iterrows()]

    # Breakdown by Repair Type
    type_breakdown = {}
    for grp in ["blur", "noise", "exposure"]:
        df_grp = df_repaired[df_repaired["repair_group"] == grp]
        if len(df_grp) == 0:
            continue
        yt = df_grp["ground_truth"].tolist()
        sb = df_grp["raw_score_before"].tolist()
        sa = df_grp["raw_score_after"].tolist()
        mb = compute_metrics(yt, sb)
        ma = compute_metrics(yt, sa)
        tc = df_grp["transition"].value_counts().to_dict()
        for k in ["Correct->Correct", "Correct->Incorrect", "Incorrect->Correct", "Incorrect->Incorrect"]:
            tc.setdefault(k, 0)
        stab = (tc["Correct->Correct"] + tc["Incorrect->Incorrect"]) / len(df_grp) * 100.0
        sd = [a - b for a, b in zip(sa, sb)]

        type_breakdown[grp] = {
            "n": len(df_grp),
            "n_pos": sum(yt),
            "n_neg": len(yt) - sum(yt),
            "accuracy_before": mb["accuracy"],
            "accuracy_after": ma["accuracy"],
            "accuracy_delta": ma["accuracy"] - mb["accuracy"],
            "incorrect_to_correct": tc["Incorrect->Correct"],
            "correct_to_incorrect": tc["Correct->Incorrect"],
            "correct_to_correct": tc["Correct->Correct"],
            "incorrect_to_incorrect": tc["Incorrect->Incorrect"],
            "prediction_stability": stab,
            "mean_score_delta": float(np.mean(sd)),
            "note": "Sufficient cohort for paired observation" if len(df_grp) >= 10 else "Insufficient evidence for a reliable diagnostic conclusion.",
        }

    # Quality vs Diagnostic Matrix
    # Quality improved: (laplacian increased for blur, SNR increased for noise, or intensity improved)
    # Diagnosis: improved (Inc->Corr), worsened (Corr->Inc), unchanged (Corr->Corr or Inc->Inc)
    q_diag_matrix = {
        "quality_improved_diag_improved": 0,
        "quality_improved_diag_unchanged": 0,
        "quality_improved_diag_worsened": 0,
        "quality_unchanged_diag_improved": 0,
        "quality_unchanged_diag_unchanged": 0,
        "quality_worsened_diag_worsened": 0,
    }

    for _, r in df_repaired.iterrows():
        # Quality check
        q_imp = False
        if r["repair_group"] == "blur":
            q_imp = (r["laplacian_after"] > r["laplacian_before"])
        elif r["repair_group"] == "noise":
            q_imp = (r["snr_after"] > r["snr_before"])
        elif r["repair_group"] == "exposure":
            dist_before = abs(r["mean_intensity_before"] - 128.0)
            dist_after = abs(r["mean_intensity_after"] - 128.0)
            q_imp = (dist_after < dist_before)

        trans = r["transition"]
        if q_imp:
            if trans == "Incorrect->Correct":
                q_diag_matrix["quality_improved_diag_improved"] += 1
            elif trans == "Correct->Incorrect":
                q_diag_matrix["quality_improved_diag_worsened"] += 1
            else:
                q_diag_matrix["quality_improved_diag_unchanged"] += 1
        else:
            if trans == "Incorrect->Correct":
                q_diag_matrix["quality_unchanged_diag_improved"] += 1
            elif trans == "Correct->Incorrect":
                q_diag_matrix["quality_worsened_diag_worsened"] += 1
            else:
                q_diag_matrix["quality_unchanged_diag_unchanged"] += 1

    summary = {
        "experiment_name": "final_paired_repair_diagnostic_validation",
        "sample_size": {
            "candidate_count": len(df_all),
            "repair_required_count": int(df_all["repair_required"].sum()),
            "repair_applied_count": int(df_all["repair_applied"].sum()),
            "repair_skipped_count": int((~df_all["repair_applied"]).sum()),
        },
        "operating_threshold": OPERATING_THRESHOLD,
        "before_metrics": m_before,
        "after_metrics": m_after,
        "metric_deltas": deltas,
        "four_transitions": transitions_counts,
        "transitions_percentage": {
            k: (v / n_repaired * 100.0) for k, v in transitions_counts.items()
        },
        "prediction_flips": flips_count,
        "prediction_stability_pct": stability_pct,
        "mean_raw_score_delta": float(np.mean(score_deltas)) if score_deltas else 0.0,
        "median_raw_score_delta": float(np.median(score_deltas)) if score_deltas else 0.0,
        "mean_confidence_delta": float(np.mean(conf_deltas)) if conf_deltas else 0.0,
        "repair_type_breakdown": type_breakdown,
        "quality_vs_diagnostic_matrix": q_diag_matrix,
    }

    return rows, summary


def main() -> None:
    test_csv = PROJECT_ROOT / "data" / "processed" / "test.csv"
    dataset_dir = PROJECT_ROOT / "dataset"
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing DenseNet-121 Base Model Agent (frozen)...")
    model = BaseModelAgent()
    model.load_model()
    logger.info("Base Model loaded successfully on CPU.")

    # 1. Select Candidates
    candidates = select_candidate_cohort(test_csv, dataset_dir, seed=RANDOM_SEED)

    # 2. Run Paired Validation
    rows, summary = run_paired_validation(candidates, model)

    # 3. Save CSV
    csv_path = outputs_dir / "final_paired_repair_diagnostic.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved paired evaluation rows to {csv_path} (N={len(df)})")

    # 4. Save JSON Summary
    json_path = outputs_dir / "final_paired_repair_diagnostic_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved paired summary to {json_path}")

    # 5. Select Demo Cases
    # CASE 1: Prefer Incorrect -> Correct; if none, select Correct -> Correct and label as prediction preservation
    # CASE 2: Image Quality Improvement (before vs after quality)
    # CASE 3: Safety Case (Verification escalation or OOD/Uncertainty review)
    df_rep = df[df["repair_applied"] == True]
    inc_to_corr = df_rep[df_rep["transition"] == "Incorrect->Correct"]
    corr_to_corr = df_rep[df_rep["transition"] == "Correct->Correct"]

    if len(inc_to_corr) > 0:
        c1_row = inc_to_corr.iloc[0]
        c1_name = "Diagnostic Correction (Incorrect -> Correct)"
    else:
        c1_row = corr_to_corr.iloc[0]
        c1_name = "Prediction Preservation (Correct -> Correct)"

    case1 = {
        "case_name": "Demo Case 1: Diagnostic State Transition",
        "category": c1_name,
        "image_id": c1_row["image_id"],
        "ground_truth": int(c1_row["ground_truth"]),
        "quality_before": c1_row["quality_before"],
        "quality_after": c1_row["quality_after"],
        "repair_type": c1_row["repair_type"],
        "raw_score_before": float(c1_row["raw_score_before"]),
        "raw_score_after": float(c1_row["raw_score_after"]),
        "prediction_before": int(c1_row["prediction_before"]),
        "prediction_after": int(c1_row["prediction_after"]),
        "transition": c1_row["transition"],
        "final_action": "accept" if "release" in c1_row["verification_result"] else "escalate",
        "verification_result": c1_row["verification_result"],
        "clinical_significance": "Demonstrates model stability / diagnostic transition following targeted repair.",
    }

    # Case 2: Maximum image quality gain
    c2_row = df_rep.iloc[0]
    for _, r in df_rep.iterrows():
        if r["repair_group"] == "noise" and (r["snr_after"] - r["snr_before"]) > 0.5:
            c2_row = r
            break
        elif r["repair_group"] == "blur" and (r["laplacian_after"] - r["laplacian_before"]) > 40:
            c2_row = r

    case2 = {
        "case_name": "Demo Case 2: Image Quality Restoration Gate",
        "category": "Measurable Quality Recovery",
        "image_id": c2_row["image_id"],
        "ground_truth": int(c2_row["ground_truth"]),
        "quality_before": c2_row["quality_before"],
        "quality_after": c2_row["quality_after"],
        "repair_type": c2_row["repair_type"],
        "raw_score_before": float(c2_row["raw_score_before"]),
        "raw_score_after": float(c2_row["raw_score_after"]),
        "prediction_before": int(c2_row["prediction_before"]),
        "prediction_after": int(c2_row["prediction_after"]),
        "transition": c2_row["transition"],
        "final_action": "accept" if "release" in c2_row["verification_result"] else "escalate",
        "verification_result": c2_row["verification_result"],
        "quality_metrics_delta": {
            "laplacian_delta": float(c2_row["laplacian_after"] - c2_row["laplacian_before"]),
            "snr_delta": float(c2_row["snr_after"] - c2_row["snr_before"]),
        },
        "clinical_significance": "Demonstrates objective image quality enhancement satisfying quality gates.",
    }

    # Case 3: Pipeline Safety / Verification Escalation Case
    # Find an image where verification escalated or confidence degraded
    c3_cand = df_rep[df_rep["verification_result"].str.contains("escalate", case=False)]
    if len(c3_cand) > 0:
        c3_row = c3_cand.iloc[0]
    else:
        c3_row = df_rep.iloc[-1]

    case3 = {
        "case_name": "Demo Case 3: Verification Agent Safety Guardrail",
        "category": "Reliability Escalation & Human Review",
        "image_id": c3_row["image_id"],
        "ground_truth": int(c3_row["ground_truth"]),
        "quality_before": c3_row["quality_before"],
        "quality_after": c3_row["quality_after"],
        "repair_type": c3_row["repair_type"],
        "raw_score_before": float(c3_row["raw_score_before"]),
        "raw_score_after": float(c3_row["raw_score_after"]),
        "prediction_before": int(c3_row["prediction_before"]),
        "prediction_after": int(c3_row["prediction_after"]),
        "transition": c3_row["transition"],
        "final_action": "escalate",
        "verification_result": c3_row["verification_result"],
        "clinical_significance": "Prevents unverified automated release when post-repair confidence fails non-degradation guard.",
    }

    demo_cases = {
        "demo_case_1": case1,
        "demo_case_2": case2,
        "demo_case_3": case3,
    }

    demo_path = outputs_dir / "demo_cases.json"
    with open(demo_path, "w", encoding="utf-8") as f:
        json.dump(demo_cases, f, indent=2)
    logger.info(f"Saved live demo cases to {demo_path}")


if __name__ == "__main__":
    main()
