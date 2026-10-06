"""Calibration error metrics (Phase 4).

Responsibility:
    ECE and reliability-diagram data for raw, temperature-scaled and Accept-bucket
    confidences, reported separately so selection and calibration effects are not mixed.

Input:
    Probabilities and labels (optionally filtered by action).

Output:
    ECE values and binned reliability data.

Dependencies:
    numpy
"""

from __future__ import annotations

import numpy as np


def expected_calibration_error(probabilities, labels, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    """
    probs = np.asarray(probabilities, dtype=float)
    labs = np.asarray(labels, dtype=float)

    if len(probs) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(labs[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def reliability_bins(probabilities, labels, n_bins: int = 10) -> dict[str, np.ndarray]:
    """Compute bin confidences, accuracies, and counts for reliability diagrams."""
    probs = np.asarray(probabilities, dtype=float)
    labs = np.asarray(labels, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        bin_size = int(np.sum(in_bin))
        bin_counts.append(bin_size)
        if bin_size > 0:
            bin_accs.append(float(np.mean(labs[in_bin])))
            bin_confs.append(float(np.mean(probs[in_bin])))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((bin_lower + bin_upper) / 2))

    return {
        "bin_accuracies": np.array(bin_accs),
        "bin_confidences": np.array(bin_confs),
        "bin_counts": np.array(bin_counts),
        "bin_edges": bin_edges,
    }
