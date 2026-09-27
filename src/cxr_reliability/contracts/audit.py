"""Audit record contract (PRD section 4, Auditability).

Responsibility:
    Schema for one per-inference log entry: signals, action taken, verification deltas,
    and provenance (thresholds version, model ids, weight hashes, input hash).

Input:
    n/a

Output:
    AuditRecord, ExecutionPath.

Dependencies:
    contracts.pipeline

Implementation phase: P0
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from .common import StrictModel
from .pipeline import PipelineOutput


class ExecutionPath(str, Enum):
    FAST = "fast"
    REPAIRED = "repaired"
    ESCALATED = "escalated"
    REPAIRED_THEN_ESCALATED = "repaired_then_escalated"
    REJECTED = "rejected"


class AuditRecord(StrictModel):
    audit_id: str
    timestamp_utc: datetime
    input_id: str | None = None
    input_sha256: str
    thresholds_version: str
    config_hash: str | None = None
    model_ids: dict[str, str]
    weights_sha256: dict[str, str] = {}
    path: ExecutionPath
    output: PipelineOutput
    latency_ms_by_stage: dict[str, float] = {}
    software_versions: dict[str, str] = {}
