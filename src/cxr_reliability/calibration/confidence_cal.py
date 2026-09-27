"""Confidence calibration and cut selection (Phase 4.5).

Responsibility:
    Wrap ThresholdAnalyzer and ProbabilityCalibrator to provide
    convenience functions for threshold selection and probability
    calibration fitting.

Input:
    Raw Pneumonia sigmoid scores and binary labels from ID-val.

Output:
    Fitted ProbabilityCalibrator and UncertaintyThresholds with
    positive_class_threshold set.

Dependencies:
    numpy, cxr_reliability.calibration.*

Implementation phase: P4.5
"""

from __future__ import annotations

import numpy as np

from cxr_reliability.calibration.probability_calibration import ProbabilityCalibrator
from cxr_reliability.calibration.threshold_calibration import ThresholdAnalyzer
from cxr_reliability.config.thresholds import UncertaintyThresholds


def fit_temperature(
    scores: np.ndarray,
    labels: np.ndarray,
) -> ProbabilityCalibrator:
    """
    Fit post-hoc calibration (Platt scaling + Isotonic) on raw scores.

    Parameters
    ----------
    scores : array-like of shape (n_samples,) — raw sigmoid Pneumonia scores
    labels : array-like of shape (n_samples,) — binary ground truth labels (0 or 1)

    Returns
    -------
    ProbabilityCalibrator
        Fitted probability calibrator instance (both Platt and Isotonic).
    """
    calibrator = ProbabilityCalibrator()
    calibrator.fit(labels, scores)
    return calibrator


def select_confidence_cuts(
    scores: np.ndarray,
    labels: np.ndarray,
    strategy: str = "f1_optimal",
) -> UncertaintyThresholds:
    """
    Select optimal positive class threshold based on validation performance.

    Parameters
    ----------
    scores   : array-like of shape (n_samples,) — raw sigmoid Pneumonia scores
    labels   : array-like of shape (n_samples,) — binary ground truth labels (0 or 1)
    strategy : threshold selection strategy ('f1_optimal', 'youden', 'balanced_accuracy')

    Returns
    -------
    UncertaintyThresholds
        Instance with positive_class_threshold set to the selected threshold.
    """
    analyzer = ThresholdAnalyzer(strategy=strategy)  # type: ignore[arg-type]
    result = analyzer.fit(labels, scores)

    thresholds = UncertaintyThresholds()
    thresholds.positive_class_threshold = result.project_validation_threshold
    return thresholds
