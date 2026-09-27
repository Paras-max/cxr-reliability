"""Latency and compute-cost accounting.

Responsibility:
    Time each stage and label the path (fast vs escalated) so the compute-cost metric in
    PRD section 7 can be computed. The budget value itself is set by the project owner
    after baseline measurement.

Input:
    Stage names and callables / timing contexts.

Output:
    Per-stage latency dict (ms) and total.

Dependencies:
    time (stdlib)

Implementation phase: P6
"""

from __future__ import annotations


class StageTimer:
    """Collects per-stage wall-clock timings for one inference."""

    def start(self, stage: str) -> None:
        raise NotImplementedError("Phase 6")

    def stop(self, stage: str) -> float:
        raise NotImplementedError("Phase 6")

    def as_dict(self) -> dict[str, float]:
        raise NotImplementedError("Phase 6")
