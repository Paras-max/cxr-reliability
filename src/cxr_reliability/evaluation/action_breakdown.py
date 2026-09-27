"""Per-action accuracy breakdown (PRD section 7).

Responsibility:
    Accuracy, error rate, sensitivity and specificity within each action bucket (Accept /
    Repair / Escalate / Reject) with counts, and the share of hard cases moved out of
    Accept, so easy-subset accuracy alone is never reported.

Input:
    Per-image PipelineOutput or audit records plus ground-truth labels.

Output:
    Table by action with counts and confidence intervals.

Dependencies:
    pandas, evaluation.metrics

Implementation phase: P7
"""

from __future__ import annotations


def action_breakdown(records, labels):
    raise NotImplementedError("Phase 7")
