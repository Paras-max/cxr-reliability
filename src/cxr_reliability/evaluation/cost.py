"""Compute cost evaluation.

Responsibility:
    Median and p95 latency, fast path versus escalated path, CPU and GPU.

Input:
    Audit records with per-stage latency.

Output:
    Cost table.

Dependencies:
    pandas, pipeline.budget

Implementation phase: P7
"""

from __future__ import annotations


def latency_summary(records):
    raise NotImplementedError("Phase 7")
