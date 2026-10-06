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
    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.default_log_file = self.directory / "audit.jsonl"

    def write(self, record: AuditRecord) -> None:
        json_line = record.model_dump_json() + "\n"
        with open(self.default_log_file, "a", encoding="utf-8") as f:
            f.write(json_line)
            f.flush()

    def read_all(self) -> Iterator[AuditRecord]:
        if not self.directory.exists():
            return
        for log_file in sorted(self.directory.glob("*.jsonl")):
            try:
                with open(log_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield AuditRecord.model_validate_json(line)
            except Exception:
                continue

    def get(self, audit_id: str) -> AuditRecord | None:
        for record in self.read_all():
            if record.audit_id == audit_id:
                return record
        return None
