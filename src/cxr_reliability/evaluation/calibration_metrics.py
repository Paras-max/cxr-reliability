"""Calibration error metrics.

Responsibility:
    ECE and reliability-diagram data for raw, temperature-scaled and Accept-bucket
    confidences, reported separately so selection and calibration effects are not mixed.

Input:
    Probabilities and labels (optionally filtered by action).

Output:
    ECE values and binned reliability data.

Dependencies:
    numpy

Implementation phase: P4
"""

from __future__ import annotations


def expected_calibration_error(probabilities, labels, n_bins: int) -> float:
    raise NotImplementedError("Phase 4")


def reliability_bins(probabilities, labels, n_bins: int):
    raise NotImplementedError("Phase 4")
