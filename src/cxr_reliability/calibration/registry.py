"""Threshold registry: freeze and version.

Responsibility:
    Combine calibrated pieces into one Thresholds object, write it as
    configs/thresholds/v1_calibrated_<hash>.yaml with provenance, and refuse edits to a
    frozen file. Frozen thresholds are required before any test-set evaluation.

Input:
    Calibrated threshold sections and calibration run metadata.

Output:
    Versioned thresholds YAML, its hash, and an entry in docs/calibration_log.md.

Dependencies:
    config.thresholds, hashlib, pyyaml

Implementation phase: P3-P5
"""

from __future__ import annotations

from pathlib import Path

from cxr_reliability.config.thresholds import Thresholds


def freeze_thresholds(thresholds: Thresholds, output_dir: Path) -> Path:
    raise NotImplementedError("Phase 5")
