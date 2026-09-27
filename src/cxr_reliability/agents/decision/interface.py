"""Decision Agent interface.

Responsibility:
    Define the single method both v1 and v2 implement, so they can be swapped and ablated
    without touching the orchestrator.

Input:
    - QualityResult
    - OODResult
    - UncertaintyResult
    - DecisionState (attempt counters)

Output:
    DecisionResult with action, rule id, driving signals and threshold margins.

Dependencies:
    contracts.*

Implementation phase: P6
"""

from __future__ import annotations

from typing import Protocol

from cxr_reliability.contracts.decision import DecisionResult, DecisionState
from cxr_reliability.contracts.ood import OODResult
from cxr_reliability.contracts.quality import QualityResult
from cxr_reliability.contracts.uncertainty import UncertaintyResult


class DecisionAgent(Protocol):
    def decide(
        self,
        quality: QualityResult,
        ood: OODResult,
        uncertainty: UncertaintyResult,
        state: DecisionState,
    ) -> DecisionResult: ...
