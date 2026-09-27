"""Pneumonia classification threshold analysis (Phase 4.5).

Responsibility:
    Evaluate a fine grid of classification thresholds against the validation
    set and select a project threshold using a configurable strategy.

    IMPORTANT — NO DATA LEAKAGE:
        All fitting uses ONLY validation.csv predictions.
        test.csv is NEVER touched here.

THRESHOLD SELECTION STRATEGIES:
    1. f1_optimal         — argmax F1; best for imbalanced datasets
    2. youden             — argmax (Sensitivity + Specificity - 1)
    3. balanced_accuracy  — argmax balanced accuracy

    Default: f1_optimal
    Rationale: The validation set is ~1.1% Pneumonia (severely imbalanced).
    Accuracy-based metrics would be dominated by the majority class.
    F1 maximises the harmonic mean of precision and recall, giving equal weight
    to false positives and false negatives — appropriate for a clinical screening
    task where both matter.

    The selected threshold is called 'project_validation_threshold'.
    It is NOT a clinical threshold and has NOT been validated clinically.

Dependencies:
    numpy, scikit-learn, pandas, matplotlib

Implementation phase: P4.5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

StrategyName = Literal["f1_optimal", "youden", "balanced_accuracy"]

# Number of threshold steps to evaluate
N_THRESHOLDS = 200


@dataclass
class ThresholdRow:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float           # = sensitivity
    specificity: float
    f1: float
    balanced_accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    youden_j: float


@dataclass
class ThresholdAnalysisResult:
    thresholds_df: pd.DataFrame          # one row per threshold
    roc_auc: float
    pr_auc: float
    candidates: dict[str, dict]          # strategy → {threshold, metrics}
    selected_strategy: StrategyName
    project_validation_threshold: float  # the chosen threshold
    n_positive: int
    n_negative: int
    n_total: int
    roc_fpr: np.ndarray                  # for ROC plot
    roc_tpr: np.ndarray
    pr_precision: np.ndarray             # for PR plot
    pr_recall: np.ndarray


class ThresholdAnalyzer:
    """
    Evaluate classification performance across a grid of thresholds.

    Parameters
    ----------
    strategy : threshold selection strategy (default: 'f1_optimal')
    n_thresholds : number of threshold values to evaluate
    """

    def __init__(
        self,
        strategy: StrategyName = "f1_optimal",
        n_thresholds: int = N_THRESHOLDS,
    ) -> None:
        self.strategy = strategy
        self.n_thresholds = n_thresholds

    def fit(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> ThresholdAnalysisResult:
        """
        Run threshold analysis on validation-set predictions.

        Parameters
        ----------
        y_true  : int array, shape (n,) — ground-truth binary labels (1=Pneumonia)
        y_score : float array, shape (n,) — raw Pneumonia scores from Base Model

        Returns
        -------
        ThresholdAnalysisResult
        """
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        n_positive = int(y_true.sum())
        n_negative = int(len(y_true) - n_positive)

        # ── Global metrics ────────────────────────────────────────────────
        roc_auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))

        # ROC curve data for plotting
        roc_fpr, roc_tpr, _ = roc_curve(y_true, y_score)
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_score)

        # ── Per-threshold metrics ─────────────────────────────────────────
        thresholds = np.linspace(0.01, 0.99, self.n_thresholds)
        rows: list[ThresholdRow] = []

        for thr in thresholds:
            y_pred = (y_score >= thr).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            bal_acc = (recall + specificity) / 2.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            youden_j = recall + specificity - 1.0
            accuracy = (tp + tn) / len(y_true)

            rows.append(ThresholdRow(
                threshold=round(float(thr), 6),
                tp=tp, fp=fp, tn=tn, fn=fn,
                accuracy=round(accuracy, 6),
                precision=round(precision, 6),
                recall=round(recall, 6),
                specificity=round(specificity, 6),
                f1=round(f1, 6),
                balanced_accuracy=round(bal_acc, 6),
                false_positive_rate=round(fpr, 6),
                false_negative_rate=round(fnr, 6),
                youden_j=round(youden_j, 6),
            ))

        df = pd.DataFrame([vars(r) for r in rows])

        # ── Strategy-based candidate thresholds ───────────────────────────
        candidates: dict[str, dict] = {}

        # 1. F1-optimal
        best_f1_idx = df["f1"].idxmax()
        candidates["f1_optimal"] = {
            "threshold": float(df.loc[best_f1_idx, "threshold"]),
            "f1": float(df.loc[best_f1_idx, "f1"]),
            "precision": float(df.loc[best_f1_idx, "precision"]),
            "recall": float(df.loc[best_f1_idx, "recall"]),
            "specificity": float(df.loc[best_f1_idx, "specificity"]),
            "balanced_accuracy": float(df.loc[best_f1_idx, "balanced_accuracy"]),
            "tp": int(df.loc[best_f1_idx, "tp"]),
            "fp": int(df.loc[best_f1_idx, "fp"]),
            "tn": int(df.loc[best_f1_idx, "tn"]),
            "fn": int(df.loc[best_f1_idx, "fn"]),
        }

        # 2. Youden's J
        best_youden_idx = df["youden_j"].idxmax()
        candidates["youden"] = {
            "threshold": float(df.loc[best_youden_idx, "threshold"]),
            "youden_j": float(df.loc[best_youden_idx, "youden_j"]),
            "recall": float(df.loc[best_youden_idx, "recall"]),
            "specificity": float(df.loc[best_youden_idx, "specificity"]),
            "f1": float(df.loc[best_youden_idx, "f1"]),
            "balanced_accuracy": float(df.loc[best_youden_idx, "balanced_accuracy"]),
            "tp": int(df.loc[best_youden_idx, "tp"]),
            "fp": int(df.loc[best_youden_idx, "fp"]),
            "tn": int(df.loc[best_youden_idx, "tn"]),
            "fn": int(df.loc[best_youden_idx, "fn"]),
        }

        # 3. Balanced accuracy optimal
        best_ba_idx = df["balanced_accuracy"].idxmax()
        candidates["balanced_accuracy"] = {
            "threshold": float(df.loc[best_ba_idx, "threshold"]),
            "balanced_accuracy": float(df.loc[best_ba_idx, "balanced_accuracy"]),
            "recall": float(df.loc[best_ba_idx, "recall"]),
            "specificity": float(df.loc[best_ba_idx, "specificity"]),
            "f1": float(df.loc[best_ba_idx, "f1"]),
            "tp": int(df.loc[best_ba_idx, "tp"]),
            "fp": int(df.loc[best_ba_idx, "fp"]),
            "tn": int(df.loc[best_ba_idx, "tn"]),
            "fn": int(df.loc[best_ba_idx, "fn"]),
        }

        selected_threshold = candidates[self.strategy]["threshold"]

        return ThresholdAnalysisResult(
            thresholds_df=df,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            candidates=candidates,
            selected_strategy=self.strategy,
            project_validation_threshold=selected_threshold,
            n_positive=n_positive,
            n_negative=n_negative,
            n_total=len(y_true),
            roc_fpr=roc_fpr,
            roc_tpr=roc_tpr,
            pr_precision=pr_precision,
            pr_recall=pr_recall,
        )


def plot_threshold_analysis(
    result: ThresholdAnalysisResult,
    output_path: Path,
    selected_threshold: float,
) -> None:
    """
    Generate a 5-panel threshold analysis plot and save it.

    Panels:
        1. Precision vs threshold
        2. Recall (sensitivity) vs threshold
        3. F1 vs threshold
        4. Specificity vs threshold
        5. Balanced accuracy vs threshold
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    df = result.thresholds_df
    thr = df["threshold"].values
    vline_kw = dict(color="red", linestyle="--", linewidth=1.2, alpha=0.7,
                    label=f"Selected ({selected_threshold:.3f})")

    fig, axes = plt.subplots(5, 1, figsize=(10, 18), sharex=True)
    fig.suptitle(
        f"Threshold Analysis — densenet121-res224-nih\n"
        f"Validation set: {result.n_total:,} images  |  "
        f"ROC-AUC: {result.roc_auc:.4f}  |  PR-AUC: {result.pr_auc:.4f}",
        fontsize=13, y=0.995
    )

    panels = [
        ("precision",         "Precision",         "#2196F3"),
        ("recall",            "Recall (Sensitivity)", "#4CAF50"),
        ("f1",                "F1 Score",           "#FF9800"),
        ("specificity",       "Specificity",        "#9C27B0"),
        ("balanced_accuracy", "Balanced Accuracy",  "#F44336"),
    ]

    for ax, (col, label, color) in zip(axes, panels):
        ax.plot(thr, df[col].values, color=color, linewidth=1.8)
        ax.axvline(x=selected_threshold, **vline_kw)
        ax.set_ylabel(label, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")

    axes[-1].set_xlabel("Classification Threshold", fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
