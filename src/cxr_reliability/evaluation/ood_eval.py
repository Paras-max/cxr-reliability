"""OOD evaluation (PRD section 7).

Responsibility:
    AUROC of Mahalanobis, energy and combined scores on NIH versus CheXpert/PadChest, and
    the false-positive OOD rate on rare-but-valid cases, reported separately.

Input:
    Scores on in-distribution, natural OOD and rare-but-valid sets.

Output:
    AUROC per detector; false-positive rate at the calibrated threshold.

Dependencies:
    scikit-learn, evaluation.metrics

Implementation phase: P4
"""

from __future__ import annotations


def ood_auroc(id_scores, ood_scores) -> float:
    raise NotImplementedError("Phase 4")


def rare_valid_false_positive_rate(rare_valid_scores, threshold: float) -> float:
    raise NotImplementedError("Phase 4")
