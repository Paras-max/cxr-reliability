"""Run full evaluation (Phase 13).

Responsibility:
    Evaluate the frozen configuration on the test split against baselines; write tables
    and figures.

Input:
    Frozen thresholds, manifests, evaluation config.

Output:
    outputs/evaluation_full_test.csv
    outputs/evaluation_full_test_summary.json

Implementation:
    Delegates to scripts/run_full_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root and scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_full_evaluation import run_full_evaluation

if __name__ == "__main__":
    run_full_evaluation()
