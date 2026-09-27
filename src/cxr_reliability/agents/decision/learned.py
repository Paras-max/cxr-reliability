"""Decision Agent v2 (stretch): learned weighting.

Responsibility:
    Small scikit-learn model (logistic regression or shallow MLP) over the three scalar
    signals, trained to trade off accuracy, cost and safety. Not to be claimed as a
    contribution unless an ablation shows it beats v1 (PRD section 11).

Input:
    - Scalar signals per sample (quality score, OOD score, confidence)
    - correctness labels
    - cost model (owner-defined)

Output:
    Trained model artifact; DecisionResult at inference.

Dependencies:
    scikit-learn, agents.decision.interface

Implementation phase: P7b (stretch)
"""

from __future__ import annotations

from cxr_reliability.contracts.decision import DecisionResult, DecisionState
from cxr_reliability.contracts.ood import OODResult
from cxr_reliability.contracts.quality import QualityResult
from cxr_reliability.contracts.uncertainty import UncertaintyResult


class LearnedDecisionAgent:
    def __init__(self, model_path, thresholds_version: str) -> None:
        self.model_path = model_path
        self.thresholds_version = thresholds_version

    def fit(self, signals, was_correct, cost_model) -> None:
        raise NotImplementedError("Phase 7b (stretch)")

    def decide(self, quality: QualityResult, ood: OODResult, uncertainty: UncertaintyResult,
               state: DecisionState) -> DecisionResult:
        raise NotImplementedError("Phase 7b (stretch)")
