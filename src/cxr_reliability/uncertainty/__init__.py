"""Uncertainty quantification module — Phase 6.

Responsibility
--------------
Estimate model prediction uncertainty using output confidence and predictive entropy.
Designed for multi-label sigmoid outputs (specifically TorchXRayVision DenseNet-121).

Modules:
- confidence: binary confidence max(p, 1-p)
- entropy: binary predictive entropy and normalized entropy with numerical clamping
- estimator: high-level UncertaintyEstimator combining confidence and entropy
"""

from __future__ import annotations

from .confidence import compute_binary_confidence
from .entropy import compute_binary_entropy, compute_normalized_entropy
from .estimator import (
    UncertaintyConfig,
    UncertaintyEstimator,
    UncertaintyEvaluation,
    UncertaintyLevel,
)

__all__ = [
    "compute_binary_confidence",
    "compute_binary_entropy",
    "compute_normalized_entropy",
    "UncertaintyConfig",
    "UncertaintyEstimator",
    "UncertaintyEvaluation",
    "UncertaintyLevel",
]
