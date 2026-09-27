"""Decision Agent contract (PRD FR-5).

Responsibility:
    Schema for the action decision, the rule that fired, the signals that drove it and the
    margins to each threshold (needed for auditability and the borderline safety rule).

Input:
    n/a

Output:
    DecisionResult, DecisionState, Action.

Dependencies:
    contracts.common

Implementation phase: P0
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field

from .common import AgentName, AgentResult, StrictModel


class Action(str, Enum):
    ACCEPT = "accept"
    REPAIR = "repair"
    ESCALATE = "escalate"
    REJECT = "reject"


class DecisionState(StrictModel):
    repair_attempts: int = Field(default=0, ge=0)
    escalation_attempts: int = Field(default=0, ge=0)
    stage: Literal["initial", "post_repair", "post_escalation"] = "initial"


class DecisionResult(AgentResult):
    agent: AgentName = AgentName.DECISION
    action: Action
    rule_id: str
    driving_signals: dict[str, str | float | bool] = Field(default_factory=dict)
    threshold_margins: dict[str, float] = Field(default_factory=dict)
    reliability_label: str
    state: DecisionState = DecisionState()
