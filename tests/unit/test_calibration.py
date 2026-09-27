"""Unit tests for Phase 4.5 calibration modules.

Tests ThresholdAnalyzer, ProbabilityCalibrator, and confidence_cal wrappers
using synthetic data. No real model inference is performed.
"""

import numpy as np
import pytest

from cxr_reliability.calibration.confidence_cal import fit_temperature, select_confidence_cuts
from cxr_reliability.calibration.probability_calibration import (
    ProbabilityCalibrator,
    _compute_ece,
)
from cxr_reliability.calibration.threshold_calibration import ThresholdAnalyzer


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def imbalanced_data():
    """Synthetic imbalanced dataset: 900 negatives, 100 positives."""
    np.random.seed(42)
    y_true = np.array([0] * 900 + [1] * 100)
    y_score = np.concatenate([
        np.random.beta(1, 5, 900),   # negatives: low scores
        np.random.beta(5, 1, 100),   # positives: high scores
    ])
    return y_true, y_score


@pytest.fixture
def balanced_data():
    """Synthetic balanced dataset: 500 negatives, 500 positives."""
    np.random.seed(42)
    y_true = np.array([0] * 500 + [1] * 500)
    y_score = np.concatenate([
        np.random.beta(2, 5, 500),   # negatives
        np.random.beta(5, 2, 500),   # positives
    ])
    return y_true, y_score


# ═══════════════════════════════════════════════════════════════════════
#  ThresholdAnalyzer
# ═══════════════════════════════════════════════════════════════════════

class TestThresholdAnalyzer:
    def test_basic_f1_optimal(self, imbalanced_data):
        """F1-optimal strategy produces a threshold in (0, 1) with good AUC."""
        y_true, y_score = imbalanced_data
        analyzer = ThresholdAnalyzer(strategy="f1_optimal", n_thresholds=50)
        result = analyzer.fit(y_true, y_score)

        assert 0.0 < result.project_validation_threshold < 1.0
        assert result.roc_auc > 0.8
        assert result.pr_auc > 0.5
        assert result.n_positive == 100
        assert result.n_negative == 900
        assert result.n_total == 1000
        assert len(result.thresholds_df) == 50

    def test_all_three_strategies_present(self, imbalanced_data):
        """All three candidate strategies are always reported."""
        y_true, y_score = imbalanced_data
        analyzer = ThresholdAnalyzer(strategy="f1_optimal", n_thresholds=50)
        result = analyzer.fit(y_true, y_score)

        assert "f1_optimal" in result.candidates
        assert "youden" in result.candidates
        assert "balanced_accuracy" in result.candidates

    def test_youden_strategy(self, imbalanced_data):
        """Youden strategy selects threshold that maximizes J index."""
        y_true, y_score = imbalanced_data
        analyzer = ThresholdAnalyzer(strategy="youden", n_thresholds=50)
        result = analyzer.fit(y_true, y_score)

        assert result.selected_strategy == "youden"
        youden_thr = result.candidates["youden"]["threshold"]
        assert result.project_validation_threshold == youden_thr

    def test_balanced_accuracy_strategy(self, imbalanced_data):
        """Balanced accuracy strategy selects threshold that maximizes bal. acc."""
        y_true, y_score = imbalanced_data
        analyzer = ThresholdAnalyzer(strategy="balanced_accuracy", n_thresholds=50)
        result = analyzer.fit(y_true, y_score)

        assert result.selected_strategy == "balanced_accuracy"

    def test_threshold_df_columns(self, imbalanced_data):
        """Threshold results DataFrame has all expected columns."""
        y_true, y_score = imbalanced_data
        analyzer = ThresholdAnalyzer(n_thresholds=20)
        result = analyzer.fit(y_true, y_score)

        expected_cols = {
            "threshold", "tp", "fp", "tn", "fn",
            "accuracy", "precision", "recall", "specificity",
            "f1", "balanced_accuracy",
            "false_positive_rate", "false_negative_rate", "youden_j",
        }
        assert expected_cols.issubset(set(result.thresholds_df.columns))

    def test_roc_pr_arrays(self, imbalanced_data):
        """ROC and PR curve arrays are populated for plotting."""
        y_true, y_score = imbalanced_data
        analyzer = ThresholdAnalyzer(n_thresholds=20)
        result = analyzer.fit(y_true, y_score)

        assert len(result.roc_fpr) > 2
        assert len(result.roc_tpr) > 2
        assert len(result.pr_precision) > 2
        assert len(result.pr_recall) > 2


# ═══════════════════════════════════════════════════════════════════════
#  ProbabilityCalibrator
# ═══════════════════════════════════════════════════════════════════════

class TestProbabilityCalibrator:
    def test_platt_reduces_ece(self, balanced_data):
        """Platt scaling should reduce or maintain ECE on separable data."""
        y_true, y_score = balanced_data
        cal = ProbabilityCalibrator()
        result = cal.fit(y_true, y_score)

        # Platt Brier score should not be worse than raw
        assert result.platt.brier_score <= result.raw.brier_score + 0.01

    def test_result_has_all_methods(self, balanced_data):
        """FullCalibrationResult contains raw, platt, and isotonic results."""
        y_true, y_score = balanced_data
        cal = ProbabilityCalibrator()
        result = cal.fit(y_true, y_score)

        assert result.raw.method == "raw"
        assert result.platt.method == "platt"
        assert result.isotonic.method == "isotonic"

    def test_calibration_curve_data(self, balanced_data):
        """Calibration curve data is populated for all methods."""
        y_true, y_score = balanced_data
        cal = ProbabilityCalibrator()
        result = cal.fit(y_true, y_score)

        for method in ("raw", "platt", "isotonic"):
            assert method in result.calibration_curve_data
            assert "fraction_of_positives" in result.calibration_curve_data[method]
            assert "mean_predicted" in result.calibration_curve_data[method]

    def test_transform_platt(self, balanced_data):
        """Platt transform returns values in [0, 1]."""
        y_true, y_score = balanced_data
        cal = ProbabilityCalibrator()
        cal.fit(y_true, y_score)

        # Single value
        calibrated = cal.transform(0.5, method="platt")
        assert 0.0 <= calibrated <= 1.0

        # Array
        calibrated_arr = cal.transform(y_score[:10], method="platt")
        assert all(0.0 <= v <= 1.0 for v in calibrated_arr)

    def test_transform_raw_passthrough(self, balanced_data):
        """Raw method passes scores through unchanged."""
        y_true, y_score = balanced_data
        cal = ProbabilityCalibrator()
        cal.fit(y_true, y_score)

        assert cal.transform(0.42, method="raw") == 0.42

    def test_save_load(self, balanced_data, tmp_path):
        """Save and load preserves calibrator functionality."""
        y_true, y_score = balanced_data
        cal = ProbabilityCalibrator()
        cal.fit(y_true, y_score)
        cal.save(tmp_path)

        cal2 = ProbabilityCalibrator.load(tmp_path)
        orig = cal.transform(0.5, method="platt")
        loaded = cal2.transform(0.5, method="platt")
        assert abs(orig - loaded) < 1e-10


# ═══════════════════════════════════════════════════════════════════════
#  ECE helper
# ═══════════════════════════════════════════════════════════════════════

class TestECE:
    def test_perfect_calibration(self):
        """Perfect calibration has ECE = 0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        ece = _compute_ece(y_true, y_prob, n_bins=5)
        assert ece < 0.01

    def test_ece_bounds(self, balanced_data):
        """ECE is always in [0, 1]."""
        y_true, y_score = balanced_data
        ece = _compute_ece(y_true, y_score)
        assert 0.0 <= ece <= 1.0


# ═══════════════════════════════════════════════════════════════════════
#  confidence_cal wrappers
# ═══════════════════════════════════════════════════════════════════════

class TestConfidenceCal:
    def test_fit_temperature_returns_calibrator(self, imbalanced_data):
        """fit_temperature returns a ProbabilityCalibrator instance."""
        y_true, y_score = imbalanced_data
        cal = fit_temperature(y_score, y_true)
        assert isinstance(cal, ProbabilityCalibrator)

    def test_select_confidence_cuts_returns_thresholds(self, imbalanced_data):
        """select_confidence_cuts returns UncertaintyThresholds with positive_class_threshold set."""
        y_true, y_score = imbalanced_data
        from cxr_reliability.config.thresholds import UncertaintyThresholds
        thresholds = select_confidence_cuts(y_score, y_true, strategy="f1_optimal")
        assert isinstance(thresholds, UncertaintyThresholds)
        assert 0.0 < thresholds.positive_class_threshold < 1.0
