"""Pipeline wiring configuration schema and loader.

Responsibility:
    Model identifiers, loop limits, execution flags and audit settings that are not
    thresholds. Kept separate so thresholds can be recalibrated without touching wiring.

Input:
    configs/pipeline.yaml.

Output:
    PipelineConfig model.

Dependencies:
    pydantic, pyyaml

Implementation phase: P0
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fast_model_id: str
    escalation_model_id: str
    target_pathology: str = "Pneumonia"
    ood_feature_layer: str | None = None


class LoopLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_repair_attempts: int = Field(default=1, ge=0)
    max_escalation_attempts: int = Field(default=1, ge=0)


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_quality_and_model_in_parallel: bool = True
    borderline_action: Literal["escalate", "reject"] = "escalate"


class AuditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    format: Literal["jsonl"] = "jsonl"
    directory: Path = Path("logs/audit")


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: ModelsConfig
    loop_limits: LoopLimits = LoopLimits()
    execution: ExecutionConfig = ExecutionConfig()
    audit: AuditConfig = AuditConfig()


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    with open(path, encoding="utf-8") as fh:
        return PipelineConfig.model_validate(yaml.safe_load(fh))
