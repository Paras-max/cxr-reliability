"""Versioned threshold schema and loader.

Responsibility:
    Define every threshold the PRD mentions (quality, OOD, uncertainty, decision, repair,
    verification). Fields the PRD does not quantify are None until calibrated, so the
    system can list exactly which values are still unresolved.

Input:
    A YAML file from configs/thresholds/.

Output:
    Thresholds model; Thresholds.unresolved() lists dotted names of None fields.

Dependencies:
    pydantic, pyyaml

Implementation phase: P0
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QualityThresholds(_Section):
    blur_laplacian_var_min: float | None = None
    snr_db_min: float | None = None
    exposure_mean_min: float | None = None
    exposure_mean_max: float | None = None
    reference_max_laplacian_var: float | None = None


class OODThresholds(_Section):
    in_distribution_percentile: float | None = None
    mahalanobis_borderline: float | None = None
    mahalanobis_severe: float | None = None
    energy_borderline: float | None = None
    energy_severe: float | None = None


class UncertaintyThresholds(_Section):
    confidence_high_min: float | None = None
    confidence_low_max: float | None = None
    positive_class_threshold: float | None = None


class DecisionThresholds(_Section):
    borderline_margin: float | None = None


class RepairThresholds(_Section):
    max_repairable_blur_pct: float | None = None
    min_repairable_snr_db: float | None = None
    repairable_exposure_mean_min: float | None = None
    repairable_exposure_mean_max: float | None = None


class VerificationThresholds(_Section):
    min_confidence_gain: float | None = None
    min_quality_gain: float | None = None


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    provenance: Literal["prd_defaults", "calibrated"]
    description: str = ""
    quality: QualityThresholds
    ood: OODThresholds
    uncertainty: UncertaintyThresholds
    decision: DecisionThresholds
    repair: RepairThresholds
    verification: VerificationThresholds

    def unresolved(self) -> list[str]:
        """Dotted names of thresholds that are still None (must be calibrated)."""
        missing: list[str] = []
        for section_name in ("quality", "ood", "uncertainty", "decision", "repair", "verification"):
            section = getattr(self, section_name)
            for field_name, value in section.model_dump().items():
                if value is None:
                    missing.append(f"{section_name}.{field_name}")
        return missing


def load_thresholds(path: str | Path) -> Thresholds:
    with open(path, encoding="utf-8") as fh:
        return Thresholds.model_validate(yaml.safe_load(fh))
