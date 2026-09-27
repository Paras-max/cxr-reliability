"""Experiment tracking wrapper.

Responsibility:
    Thin MLflow wrapper so calibration and evaluation runs log parameters, thresholds
    hash, manifest hash and metrics uniformly.

Input:
    Run name, params, metrics, artifact paths.

Output:
    MLflow run.

Dependencies:
    mlflow, config.settings

Implementation phase: P0
"""

from __future__ import annotations


def start_run(run_name: str, params: dict):
    raise NotImplementedError("Phase 0")


def log_metrics(metrics: dict[str, float]) -> None:
    raise NotImplementedError("Phase 0")
