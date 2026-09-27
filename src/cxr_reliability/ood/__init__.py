"""OOD (Out-of-Distribution) detection module — Phase 5.

Provides Mahalanobis-distance-based OOD detection for the reliability-aware
pipeline. Energy score is available as an optional secondary signal.

Public API
----------
    OODDetector  — high-level interface: fit / score / predict / analyze
    OODConfig    — configuration dataclass
    OODReferenceStats — fitted in-distribution statistics

Research Disclaimer
-------------------
This module is a research prototype and is NOT a clinically certified system.
OOD detection is a distribution-shift signal, not a medical diagnosis.
Mahalanobis distance does not perfectly detect OOD samples; the results
should be interpreted as a reliability indicator for the decision agent.

Implementation phase: P5
"""

from __future__ import annotations

from cxr_reliability.ood.detector import OODConfig, OODDetector
from cxr_reliability.ood.statistics import OODReferenceStats

__all__ = ["OODDetector", "OODConfig", "OODReferenceStats"]
