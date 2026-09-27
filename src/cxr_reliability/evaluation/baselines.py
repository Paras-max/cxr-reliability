"""Baselines.

Responsibility:
    B0 raw model always answers; B1 confidence-threshold reject only; B2 always escalate.
    The full system (B3, and B4 with Decision v2) is compared against these.

Input:
    Model outputs on the evaluation split.

Output:
    Per-baseline prediction/abstention arrays.

Dependencies:
    models, evaluation.metrics

Implementation phase: P7
"""

from __future__ import annotations


def run_baseline(name: str, model_outputs):
    raise NotImplementedError("Phase 7")
