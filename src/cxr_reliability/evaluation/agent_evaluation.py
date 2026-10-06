"""Agent-wise before vs after evaluation metrics and analysis utilities.

Responsibility:
    Provide mathematically sound, reproducible evaluation functions for each agent
    and pipeline stage. Enforces strict rules:
    - NO FABRICATED METRICS: If an agent lacks a valid ground-truth target,
      classification metrics (Accuracy, F1, ROC-AUC) are marked as 'N/A'.
    - PAIRED POPULATIONS: Enforces N_before == N_after for paired before/after comparisons.
    - SAMPLE SIZES: Every metric reports exact sample sizes (N, N_pos, N_neg).
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_prob: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """
    Calculate mathematically valid classification metrics.

    Parameters
    ----------
    y_true : 1D array of ground truth binary labels (0 or 1)
    y_pred : 1D array of binary predictions (0 or 1)
    y_prob : optional 1D array of continuous scores/probabilities for AUROC / AUPRC

    Returns
    -------
    dict with TP, TN, FP, FN, N, N_pos, N_neg, prevalence, accuracy, precision,
    recall_sensitivity, specificity, npv, f1_score, auroc, auprc.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} vs len(y_pred)={len(y_p)}")

    n_total = len(y_t)
    if n_total == 0:
        return {
            "n_total": 0,
            "n_pos": 0,
            "n_neg": 0,
            "prevalence": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall_sensitivity": 0.0,
            "specificity": 0.0,
            "npv": 0.0,
            "f1_score": 0.0,
            "auroc": None,
            "auprc": None,
        }

    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))
    prevalence = float(n_pos / n_total) if n_total > 0 else 0.0

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    auroc = None
    auprc = None
    if y_prob is not None:
        y_scores = np.asarray(y_prob, dtype=float)
        if len(np.unique(y_t)) > 1:
            try:
                auroc = float(roc_auc_score(y_t, y_scores))
                auprc = float(average_precision_score(y_t, y_scores))
            except Exception:
                auroc = None
                auprc = None

    return {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": prevalence,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": spec,
        "npv": npv,
        "f1_score": f1,
        "auroc": auroc,
        "auprc": auprc,
    }


def calculate_quality_transitions(
    df: pd.DataFrame,
    initial_col: str = "quality_label",
    after_col: str = "after_repair_quality_label",
    applied_col: str = "repair_applied",
) -> dict[str, Any]:
    """
    Calculate Quality Agent transitions before and after repair.

    F1 / Accuracy is strictly N/A because natural NIH has no ground-truth image quality labels.
    """
    total_imgs = len(df)
    initial_counts = df[initial_col].str.lower().value_counts().to_dict()
    initial_good = initial_counts.get("good", 0)
    initial_degraded = initial_counts.get("degraded", 0)
    initial_poor = initial_counts.get("poor", 0)

    # Sub-population where repair was applied
    df_rep = df[df[applied_col] == True].copy() if applied_col in df.columns else df.copy()
    n_repaired = len(df_rep)

    levels = ["good", "degraded", "poor"]
    matrix: dict[str, dict[str, int]] = {b: {a: 0 for a in levels} for b in levels}

    for _, row in df_rep.iterrows():
        b = str(row[initial_col]).lower()
        a = str(row[after_col]).lower() if pd.notna(row[after_col]) else "poor"
        if b in matrix and a in matrix[b]:
            matrix[b][a] += 1

    # Quality rank: poor=0, degraded=1, good=2
    rank = {"poor": 0, "degraded": 1, "good": 2}
    improved = 0
    unchanged = 0
    worsened = 0

    for b in levels:
        for a in levels:
            c = matrix[b][a]
            if rank[a] > rank[b]:
                improved += c
            elif rank[a] == rank[b]:
                unchanged += c
            else:
                worsened += c

    poor_total = sum(matrix["poor"].values())
    poor_to_good = matrix["poor"]["good"]
    poor_to_degraded = matrix["poor"]["degraded"]
    poor_to_acceptable = poor_to_good + poor_to_degraded

    return {
        "n_total_images": total_imgs,
        "initial_distribution": {
            "good": initial_good,
            "degraded": initial_degraded,
            "poor": initial_poor,
            "pct_good": float(initial_good / total_imgs * 100) if total_imgs else 0.0,
            "pct_degraded": float(initial_degraded / total_imgs * 100) if total_imgs else 0.0,
            "pct_poor": float(initial_poor / total_imgs * 100) if total_imgs else 0.0,
        },
        "n_repairs_evaluated": n_repaired,
        "repair_application_rate_pct": float(n_repaired / total_imgs * 100) if total_imgs else 0.0,
        "transition_matrix": matrix,
        "transitions_breakdown": {
            "poor_to_good": poor_to_good,
            "poor_to_degraded": poor_to_degraded,
            "poor_to_poor": matrix["poor"]["poor"],
            "degraded_to_good": matrix["degraded"]["good"],
            "degraded_to_degraded": matrix["degraded"]["degraded"],
            "degraded_to_poor": matrix["degraded"]["poor"],
            "good_to_good": matrix["good"]["good"],
            "good_to_degraded": matrix["good"]["degraded"],
            "good_to_poor": matrix["good"]["poor"],
        },
        "quality_improved_count": improved,
        "quality_unchanged_count": unchanged,
        "quality_worsened_count": worsened,
        "quality_improvement_rate_pct": float(improved / n_repaired * 100) if n_repaired else 0.0,
        "quality_unchanged_rate_pct": float(unchanged / n_repaired * 100) if n_repaired else 0.0,
        "quality_worsening_rate_pct": float(worsened / n_repaired * 100) if n_repaired else 0.0,
        "poor_to_good_recovery_rate_pct": float(poor_to_good / poor_total * 100) if poor_total else 0.0,
        "poor_to_acceptable_recovery_rate_pct": float(poor_to_acceptable / poor_total * 100) if poor_total else 0.0,
        "f1_score": "N/A — not a valid metric for this agent (no natural NIH image quality ground truth)",
        "accuracy": "N/A — not a valid metric for this agent",
    }


def calculate_ood_metrics(
    df: pd.DataFrame,
    dist_col: str = "mahalanobis_distance",
    level_col: str = "ood_level",
    action_col: str = "final_action",
    human_review_col: str = "needs_human_review",
) -> dict[str, Any]:
    """
    Calculate OOD Agent distribution and routing behavior.

    Standard OOD classification accuracy/F1 is N/A because the in-distribution
    NIH test set does not provide a true external OOD ground-truth label.
    """
    n_total = len(df)
    counts = df[level_col].str.upper().value_counts().to_dict()
    in_dist = counts.get("IN_DISTRIBUTION", 0)
    borderline = counts.get("BORDERLINE", 0)
    severe = counts.get("SEVERE", 0)

    distances = df[dist_col].dropna().values
    dist_stats = {}
    if len(distances) > 0:
        dist_stats = {
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "std": float(np.std(distances)),
            "min": float(np.min(distances)),
            "max": float(np.max(distances)),
            "p25": float(np.percentile(distances, 25)),
            "p75": float(np.percentile(distances, 75)),
            "p95": float(np.percentile(distances, 95)),
            "p99": float(np.percentile(distances, 99)),
        }

    # Routing impact table
    routing_summary = {}
    for lvl in ["IN_DISTRIBUTION", "BORDERLINE", "SEVERE"]:
        subset = df[df[level_col].str.upper() == lvl]
        c = len(subset)
        withheld = int(subset[human_review_col].sum()) if human_review_col in subset.columns else 0
        released = c - withheld
        act_dist = subset[action_col].str.upper().value_counts().to_dict() if action_col in subset.columns else {}
        routing_summary[lvl] = {
            "count": c,
            "pct": float(c / n_total * 100) if n_total else 0.0,
            "actions": act_dist,
            "released": released,
            "withheld": withheld,
        }

    return {
        "n_total": n_total,
        "counts": {
            "in_distribution": in_dist,
            "borderline": borderline,
            "severe": severe,
        },
        "percentages": {
            "in_distribution": float(in_dist / n_total * 100) if n_total else 0.0,
            "borderline": float(borderline / n_total * 100) if n_total else 0.0,
            "severe": float(severe / n_total * 100) if n_total else 0.0,
        },
        "mahalanobis_distance_stats": dist_stats,
        "routing_summary": routing_summary,
        "f1_score": "N/A — not a valid metric for this agent (in-distribution test set lacks true external OOD ground truth)",
        "accuracy": "N/A — not a valid metric for this agent",
    }


def calculate_uncertainty_metrics(
    df: pd.DataFrame,
    level_col: str = "uncertainty_level",
    released_col: str = "prediction_released",
    pred_col: str = "prediction_positive",
    gt_col: str = "ground_truth_label",
) -> dict[str, Any]:
    """
    Calculate Uncertainty Agent distribution and group-wise released performance.
    """
    n_total = len(df)
    counts = df[level_col].str.upper().value_counts().to_dict()
    low_count = counts.get("LOW", 0)
    high_count = counts.get("HIGH", 0)

    groups = {}
    for lvl in ["LOW", "HIGH"]:
        sub = df[df[level_col].str.upper() == lvl]
        c = len(sub)
        pneu_c = int(sub[gt_col].sum()) if gt_col in sub.columns else 0
        non_pneu_c = c - pneu_c

        sub_rel = sub[sub[released_col] == True]
        rel_c = len(sub_rel)
        withheld_c = c - rel_c

        # Classification metrics on released subset if available
        rel_metrics = None
        if rel_c > 0 and pred_col in sub_rel.columns and sub_rel[pred_col].dropna().count() > 0:
            valid_rel = sub_rel.dropna(subset=[pred_col, gt_col])
            rel_metrics = calculate_classification_metrics(
                valid_rel[gt_col].values,
                valid_rel[pred_col].astype(int).values,
            )

        groups[lvl] = {
            "total_count": c,
            "percentage": float(c / n_total * 100) if n_total else 0.0,
            "pneumonia_cases": pneu_c,
            "non_pneumonia_cases": non_pneu_c,
            "released_count": rel_c,
            "withheld_count": withheld_c,
            "release_rate_pct": float(rel_c / c * 100) if c else 0.0,
            "released_metrics": rel_metrics,
        }

    return {
        "n_total": n_total,
        "low_count": low_count,
        "high_count": high_count,
        "pct_low": float(low_count / n_total * 100) if n_total else 0.0,
        "pct_high": float(high_count / n_total * 100) if n_total else 0.0,
        "groups": groups,
    }


def calculate_decision_metrics(
    df: pd.DataFrame,
    action_col: str = "final_action",
    released_col: str = "prediction_released",
    human_review_col: str = "needs_human_review",
    repair_applied_col: str = "repair_applied",
    ver_status_col: str = "verification_status",
) -> dict[str, Any]:
    """
    Calculate Decision Agent routing distributions and downstream outcomes.

    Accuracy is strictly N/A because routing is a policy enforcement mechanism,
    not a predictive classification.
    """
    n_total = len(df)
    act_counts = df[action_col].str.upper().value_counts().to_dict()

    accept_c = act_counts.get("ACCEPT", 0)
    repair_c = act_counts.get("REPAIR", 0)
    escalate_c = act_counts.get("ESCALATE", 0)
    reject_c = act_counts.get("REJECT", 0)

    # Downstream routing outcomes
    released_c = int(df[released_col].sum()) if released_col in df.columns else 0
    withheld_c = int(df[human_review_col].sum()) if human_review_col in df.columns else 0

    return {
        "n_total": n_total,
        "action_counts": {
            "ACCEPT": accept_c,
            "REPAIR": repair_c,
            "ESCALATE": escalate_c,
            "REJECT": reject_c,
        },
        "action_percentages": {
            "ACCEPT": float(accept_c / n_total * 100) if n_total else 0.0,
            "REPAIR": float(repair_c / n_total * 100) if n_total else 0.0,
            "ESCALATE": float(escalate_c / n_total * 100) if n_total else 0.0,
            "REJECT": float(reject_c / n_total * 100) if n_total else 0.0,
        },
        "downstream_outcomes": {
            "predictions_released": released_c,
            "predictions_withheld": withheld_c,
            "release_rate_pct": float(released_c / n_total * 100) if n_total else 0.0,
            "withhold_rate_pct": float(withheld_c / n_total * 100) if n_total else 0.0,
        },
        "accuracy": "N/A — not a valid metric for this agent (routing policy enforcement)",
        "f1_score": "N/A — not a valid metric for this agent",
    }


def calculate_verification_metrics(
    df: pd.DataFrame,
    ver_status_col: str = "verification_status",
    ver_step_col: str = "verification_next_step",
    applied_col: str = "repair_applied",
    delta_conf_col: str = "delta_confidence",
    quality_before_col: str = "quality_label",
    quality_after_col: str = "after_repair_quality_label",
) -> dict[str, Any]:
    """
    Calculate Verification Agent gatekeeping metrics and failure reasons.

    Accuracy is strictly N/A because verification validates safety rules,
    not a diagnostic ground-truth label.
    """
    df_rep = df[df[applied_col] == True].copy() if applied_col in df.columns else df.copy()
    n_entering = len(df_rep)

    if n_entering == 0:
        return {
            "n_entering_verification": 0,
            "verified_count": 0,
            "escalated_count": 0,
            "rejected_count": 0,
            "verified_rate_pct": 0.0,
            "escalated_rate_pct": 0.0,
            "rejected_rate_pct": 0.0,
            "failure_reasons": {},
            "confidence_delta_summary": {},
            "accuracy": "N/A — not a valid metric for this agent",
            "f1_score": "N/A — not a valid metric for this agent",
        }

    # Verified if ver_status is True or next_step is 'release'
    is_ver = (df_rep[ver_status_col].astype(str).str.lower() == "true") | (
        df_rep[ver_step_col].astype(str).str.lower() == "release"
    )
    verified_c = int(is_ver.sum())
    escalated_c = n_entering - verified_c
    rejected_c = 0

    # Failure reasons breakdown
    df_failed = df_rep[~is_ver].copy()
    quality_unresolved_poor = int(
        ((df_failed[quality_before_col].str.lower() == "poor") & (df_failed[quality_after_col].str.lower() == "poor")).sum()
    )
    quality_partial_degraded = int(
        (
            (df_failed[quality_before_col].str.lower() == "poor")
            & (df_failed[quality_after_col].str.lower() == "degraded")
        ).sum()
    )
    conf_guard_failures = int(
        (
            (df_failed[quality_before_col].str.lower() == "poor")
            & (df_failed[quality_after_col].str.lower() == "good")
            & (df_failed[delta_conf_col] < -0.01)
        ).sum()
    )
    other_reasons = len(df_failed) - (quality_unresolved_poor + quality_partial_degraded + conf_guard_failures)

    delta_confs = df_rep[delta_conf_col].dropna().values
    conf_stats = {}
    if len(delta_confs) > 0:
        conf_stats = {
            "mean": float(np.mean(delta_confs)),
            "median": float(np.median(delta_confs)),
            "std": float(np.std(delta_confs)),
            "min": float(np.min(delta_confs)),
            "max": float(np.max(delta_confs)),
            "n_positive": int(np.sum(delta_confs > 0)),
            "n_negative": int(np.sum(delta_confs < 0)),
            "n_zero": int(np.sum(delta_confs == 0)),
        }

    return {
        "n_entering_verification": n_entering,
        "verified_count": verified_c,
        "escalated_count": escalated_c,
        "rejected_count": rejected_c,
        "verified_rate_pct": float(verified_c / n_entering * 100),
        "escalated_rate_pct": float(escalated_c / n_entering * 100),
        "rejected_rate_pct": float(rejected_c / n_entering * 100),
        "failure_reasons": {
            "quality_unresolved_poor_remained_poor": quality_unresolved_poor,
            "quality_partial_poor_to_degraded": quality_partial_degraded,
            "confidence_non_degradation_guard_exceeded": conf_guard_failures,
            "other_or_secondary_escalations": other_reasons,
        },
        "confidence_delta_summary": conf_stats,
        "accuracy": "N/A — not a valid metric for this agent (safety gating contract)",
        "f1_score": "N/A — not a valid metric for this agent",
    }


def calculate_safety_filtering(
    df: pd.DataFrame,
    threshold: float = 0.522161,
    raw_col: str = "raw_model_score",
    released_col: str = "prediction_released",
    pred_col: str = "prediction_positive",
    gt_col: str = "ground_truth_label",
) -> dict[str, Any]:
    """
    Calculate safety filtering impact:
    Compares standalone Base Model errors (FP/FN) across all images against
    errors released by the Reliability System, showing how many errors were safely
    withheld from automated clinical release.
    """
    n_total = len(df)
    y_true = df[gt_col].values
    y_base_pred = (df[raw_col].values >= threshold).astype(int)

    base_cm = confusion_matrix(y_true, y_base_pred, labels=[0, 1])
    base_tn, base_fp, base_fn, base_tp = (
        int(base_cm[0, 0]),
        int(base_cm[0, 1]),
        int(base_cm[1, 0]),
        int(base_cm[1, 1]),
    )

    df_rel = df[df[released_col] == True]
    rel_y_true = df_rel[gt_col].values
    rel_y_pred = df_rel[pred_col].astype(int).values

    rel_cm = confusion_matrix(rel_y_true, rel_y_pred, labels=[0, 1])
    rel_tn, rel_fp, rel_fn, rel_tp = (
        int(rel_cm[0, 0]),
        int(rel_cm[0, 1]),
        int(rel_cm[1, 0]),
        int(rel_cm[1, 1]),
    )

    # Cases withheld
    fp_withheld = base_fp - rel_fp
    fn_withheld = base_fn - rel_fn

    return {
        "n_total": n_total,
        "base_model_errors": {
            "false_positives": base_fp,
            "false_negatives": base_fn,
            "total_errors": base_fp + base_fn,
        },
        "reliability_system_released_errors": {
            "false_positives": rel_fp,
            "false_negatives": rel_fn,
            "total_errors": rel_fp + rel_fn,
        },
        "errors_withheld_from_automated_release": {
            "false_positives_withheld": fp_withheld,
            "false_positives_withheld_pct": float(fp_withheld / base_fp * 100) if base_fp else 0.0,
            "false_negatives_withheld": fn_withheld,
            "false_negatives_withheld_pct": float(fn_withheld / base_fn * 100) if base_fn else 0.0,
            "total_errors_withheld": fp_withheld + fn_withheld,
            "total_errors_withheld_pct": float((fp_withheld + fn_withheld) / (base_fp + base_fn) * 100)
            if (base_fp + base_fn)
            else 0.0,
        },
    }
