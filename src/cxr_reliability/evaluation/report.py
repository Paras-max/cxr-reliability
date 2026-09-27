"""Evaluation report assembly.

Responsibility:
    Compile metrics, figures and tables into outputs/ for the dashboard and docs. Must
    include failure cases and the checked pre-registered targets.

Input:
    Metric tables and figures.

Output:
    Report files under outputs/evaluation.

Dependencies:
    matplotlib, seaborn, pandas

Implementation phase: P7
"""

from __future__ import annotations

from pathlib import Path


def build_report(results_dir: Path, output_dir: Path) -> Path:
    raise NotImplementedError("Phase 7")
