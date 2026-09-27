"""Ablations.

Responsibility:
    Remove one agent at a time (quality, OOD, uncertainty, verification) and compare
    Decision v1 with v2. Required before any claim about the learned Decision Agent.

Input:
    Evaluation records under each configuration.

Output:
    Ablation table.

Dependencies:
    evaluation.action_breakdown

Implementation phase: P7 / P7b
"""

from __future__ import annotations


def run_ablation(name: str, records, labels):
    raise NotImplementedError("Phase 7")
