"""Per-inference audit logger.

Responsibility:
    Append one AuditRecord per inference to a JSONL file (action taken, driving signals,
    verification deltas, latency, provenance) and read records back for analysis and the
    dashboard. Writes must be atomic per line and never silently dropped.

Input:
    AuditRecord.

Output:
    JSONL file under logs/audit; iterator over stored records.

Dependencies:
    contracts.audit, json, pathlib

Implementation phase: P6
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from cxr_reliability.contracts.audit import AuditRecord


class AuditLogger:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def write(self, record: AuditRecord) -> None:
        raise NotImplementedError("Phase 6")

    def read_all(self) -> Iterator[AuditRecord]:
        raise NotImplementedError("Phase 6")

    def get(self, audit_id: str) -> AuditRecord | None:
        raise NotImplementedError("Phase 6")
