"""Post-hoc probability calibration for Pneumonia raw scores (Phase 4.5).

Responsibility:
    Investigate and correct the calibration of the Base Model's raw Pneumonia
    sigmoid scores for the binary Pneumonia / Non-Pneumonia task.

    The TorchXRayVision densenet121-res224-nih model was trained as a 14-class
    multi-label classifier on the full NIH dataset. Its raw sigmoid outputs are
    not necessarily calibrated for our project's binary task. This module fits
    post-hoc calibration on VALIDATION DATA ONLY.

IMPORTANT — NO DATA LEAKAGE:
    Calibration parameters are fitted EXCLUSIVELY on validation.csv predictions.
    test.csv predictions MUST NOT be used here.

METHODS IMPLEMENTED:
    1. Platt scaling (logistic regression on raw scores)
       - Reliable even with small positive counts
       - Fits a sigmoid σ(a·x + b) to map raw scores to calibrated probabilities
       - Primary recommended method for this dataset

    2. Isotonic regression
       - Non-parametric monotone mapping
       - More flexible but can overfit with small positive class counts
       - Use as a comparison to Platt scaling

METRICS:
    - Brier score (lower is better; 0 = perfect, 1 = worst)
    - ECE (Expected Calibration Error, 15 equal-width bins)
    - Reliability diagram (calibration curve)

Dependencies:
    numpy, scikit-learn, matplotlib, joblib

Implementation phase: P4.5
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

CalibrationMethod = Literal["platt", "isotonic", "raw"]

# Number of ECE bins
_ECE_BINS = 15


@dataclass
class CalibrationResult:
    method: CalibrationMethod
    brier_score: float
    ece: float
    calibrated_scores: np.ndarray


@dataclass
class FullCalibrationResult:
    raw: CalibrationResult
    platt: CalibrationResult
    isotonic: CalibrationResult
    platt_calibrator: LogisticRegression
    isotonic_calibrator: IsotonicRegression
    calibration_curve_data: dict    # fraction_of_positives, mean_predicted_value for each method


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = _ECE_BINS) -> float:
    """
    Compute Expected Calibration Error using equal-width bins.

    ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|
    where B_b is the set of samples in bin b,
    acc(B_b) is mean true label, conf(B_b) is mean predicted probability.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = y_true[mask].mean()
        conf_b = y_prob[mask].mean()
        ece += (n_b / n) * abs(acc_b - conf_b)

    return float(ece)


class ProbabilityCalibrator:
    """
    Fit and evaluate post-hoc probability calibration methods.

    Usage
    -----
        cal = ProbabilityCalibrator()
        result = cal.fit(y_true_val, y_score_val)
        cal.save(output_dir)

        # At inference time:
        cal2 = ProbabilityCalibrator.load(output_dir)
        calibrated = cal2.transform(raw_score, method='platt')
    """

    def __init__(self) -> None:
        self._platt: LogisticRegression | None = None
        self._isotonic: IsotonicRegression | None = None

    def fit(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> FullCalibrationResult:
        """
        Fit Platt scaling and Isotonic regression on validation-set predictions.

        Parameters
        ----------
        y_true  : int array (n,) — binary labels (1=Pneumonia)
        y_score : float array (n,) — raw Pneumonia scores from Base Model

        Returns
        -------
        FullCalibrationResult with metrics for raw, Platt, and isotonic methods
        """
        y_true = np.asarray(y_true, dtype=float)
        y_score = np.asarray(y_score, dtype=float)
        X = y_score.reshape(-1, 1)

        # ── Platt scaling ─────────────────────────────────────────────────
        self._platt = LogisticRegression(
            C=1e10,         # effectively no regularisation on scores
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        self._platt.fit(X, y_true)
        platt_scores = self._platt.predict_proba(X)[:, 1]

        # ── Isotonic regression ───────────────────────────────────────────
        self._isotonic = IsotonicRegression(out_of_bounds="clip")
        self._isotonic.fit(y_score, y_true)
        isotonic_scores = self._isotonic.predict(y_score)

        # ── Brier scores ──────────────────────────────────────────────────
        raw_brier = float(brier_score_loss(y_true, y_score))
        platt_brier = float(brier_score_loss(y_true, platt_scores))
        isotonic_brier = float(brier_score_loss(y_true, isotonic_scores))

        # ── ECE ───────────────────────────────────────────────────────────
        raw_ece = _compute_ece(y_true, y_score)
        platt_ece = _compute_ece(y_true, platt_scores)
        isotonic_ece = _compute_ece(y_true, isotonic_scores)

        # ── Calibration curve data (for reliability diagram) ──────────────
        fop_raw, mpv_raw = calibration_curve(y_true, y_score, n_bins=15, strategy="uniform")
        fop_platt, mpv_platt = calibration_curve(y_true, platt_scores, n_bins=15, strategy="uniform")
        fop_iso, mpv_iso = calibration_curve(y_true, isotonic_scores, n_bins=15, strategy="uniform")

        cal_curve_data = {
            "raw":      {"fraction_of_positives": fop_raw.tolist(),    "mean_predicted": mpv_raw.tolist()},
            "platt":    {"fraction_of_positives": fop_platt.tolist(),  "mean_predicted": mpv_platt.tolist()},
            "isotonic": {"fraction_of_positives": fop_iso.tolist(),    "mean_predicted": mpv_iso.tolist()},
        }

        return FullCalibrationResult(
            raw=CalibrationResult(
                method="raw",
                brier_score=raw_brier,
                ece=raw_ece,
                calibrated_scores=y_score,
            ),
            platt=CalibrationResult(
                method="platt",
                brier_score=platt_brier,
                ece=platt_ece,
                calibrated_scores=platt_scores,
            ),
            isotonic=CalibrationResult(
                method="isotonic",
                brier_score=isotonic_brier,
                ece=isotonic_ece,
                calibrated_scores=isotonic_scores,
            ),
            platt_calibrator=self._platt,
            isotonic_calibrator=self._isotonic,
            calibration_curve_data=cal_curve_data,
        )

    def transform(
        self,
        raw_score: float | np.ndarray,
        method: CalibrationMethod = "platt",
    ) -> float | np.ndarray:
        """
        Apply fitted calibration to a raw score (or array of scores).

        Parameters
        ----------
        raw_score : float or array — raw Pneumonia sigmoid score(s)
        method    : 'platt', 'isotonic', or 'raw' (passthrough)

        Returns
        -------
        Calibrated probability/probabilities in [0, 1]

        NOTE: The returned value is a CALIBRATED PROBABILITY for the binary
        Pneumonia task on the NIH validation distribution. It is still NOT a
        clinical probability and has NOT been validated clinically.
        """
        if method == "raw":
            return raw_score

        scalar = isinstance(raw_score, (int, float))
        arr = np.atleast_1d(np.asarray(raw_score, dtype=float))

        if method == "platt":
            if self._platt is None:
                raise RuntimeError("Platt calibrator not fitted. Call fit() first.")
            result = self._platt.predict_proba(arr.reshape(-1, 1))[:, 1]
        elif method == "isotonic":
            if self._isotonic is None:
                raise RuntimeError("Isotonic calibrator not fitted. Call fit() first.")
            result = self._isotonic.predict(arr)
        else:
            raise ValueError(f"Unknown calibration method '{method}'")

        return float(result[0]) if scalar else result

    def save(self, output_dir: Path) -> None:
        """Persist fitted calibrators to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        if self._platt is not None:
            joblib.dump(self._platt, output_dir / "platt_calibrator.joblib")
        if self._isotonic is not None:
            joblib.dump(self._isotonic, output_dir / "isotonic_calibrator.joblib")

    @classmethod
    def load(cls, output_dir: Path) -> ProbabilityCalibrator:
        """Load fitted calibrators from disk."""
        cal = cls()
        platt_path = output_dir / "platt_calibrator.joblib"
        iso_path = output_dir / "isotonic_calibrator.joblib"
        if platt_path.exists():
            cal._platt = joblib.load(platt_path)
        if iso_path.exists():
            cal._isotonic = joblib.load(iso_path)
        return cal


def plot_calibration_curve(
    result: FullCalibrationResult,
    output_path: Path,
) -> None:
    """
    Generate a reliability diagram comparing raw vs Platt vs Isotonic.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Probability Calibration — densenet121-res224-nih\n"
        "Reliability Diagram: Raw vs Platt Scaling vs Isotonic Regression",
        fontsize=13
    )

    # ── Left: Reliability diagram ─────────────────────────────────────────
    ax = axes[0]
    ccd = result.calibration_curve_data

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax.plot(ccd["raw"]["mean_predicted"],      ccd["raw"]["fraction_of_positives"],
            "s-", color="#F44336", linewidth=1.8, label=f"Raw (ECE={result.raw.ece:.4f})")
    ax.plot(ccd["platt"]["mean_predicted"],    ccd["platt"]["fraction_of_positives"],
            "o-", color="#2196F3", linewidth=1.8, label=f"Platt (ECE={result.platt.ece:.4f})")
    ax.plot(ccd["isotonic"]["mean_predicted"], ccd["isotonic"]["fraction_of_positives"],
            "^-", color="#4CAF50", linewidth=1.8, label=f"Isotonic (ECE={result.isotonic.ece:.4f})")

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Reliability Diagram", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Right: Brier score comparison bar chart ───────────────────────────
    ax2 = axes[1]
    methods = ["Raw", "Platt", "Isotonic"]
    briers = [result.raw.brier_score, result.platt.brier_score, result.isotonic.brier_score]
    eces = [result.raw.ece, result.platt.ece, result.isotonic.ece]
    colors = ["#F44336", "#2196F3", "#4CAF50"]

    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax2.bar(x - width/2, briers, width, label="Brier Score", color=colors, alpha=0.8)
    bars2 = ax2.bar(x + width/2, eces,   width, label="ECE",          color=colors, alpha=0.5,
                    hatch="//")

    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=12)
    ax2.set_ylabel("Score (lower is better)", fontsize=12)
    ax2.set_title("Brier Score & ECE Comparison", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                 f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                 f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
