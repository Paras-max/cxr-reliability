"""Repair package (PRD FR-6) — Phase 9.

Responsibility:
    Non-destructive image quality repair implementations:
    - Exposure: CLAHE
    - Noise: Non-Local Means
    - Blur: Conservative Unsharp Masking
    - Pipeline: Deterministic sequential repair execution

Implementation phase: P9
"""

from __future__ import annotations

from .blur import repair_blur
from .exposure import repair_exposure
from .noise import repair_noise
from .pipeline import REPAIR_EXECUTION_ORDER, RepairConfig, apply_sequential_repairs

__all__ = [
    "repair_exposure",
    "repair_noise",
    "repair_blur",
    "apply_sequential_repairs",
    "RepairConfig",
    "REPAIR_EXECUTION_ORDER",
]
