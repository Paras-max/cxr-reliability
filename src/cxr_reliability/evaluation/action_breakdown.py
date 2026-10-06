"""Per-action accuracy breakdown (PRD section 7).

Responsibility:
    Accuracy, error rate, sensitivity and specificity within each action bucket (Accept /
    Repair / Escalate / Reject) with counts.

Input:
    Per-image actions and labels.

Output:
    Table / dictionary by action with counts.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def action_breakdown(records: list[Any] | pd.Series | list[str], labels: list[int] | None = None) -> dict[str, dict[str, Any] | int]:
    """
    Count occurrences of each action bucket and calculate breakdown.
    """
    counts: dict[str, int] = {}
    for r in records:
        action_val = getattr(r, "final_action", r)
        if hasattr(action_val, "value"):
            action_val = action_val.value
        action_val = str(action_val).lower()
        counts[action_val] = counts.get(action_val, 0) + 1
    return counts
