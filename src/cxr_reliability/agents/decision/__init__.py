"""Decision Agent package (PRD FR-5).

Responsibility:
    Combine quality, OOD and uncertainty signals into Accept / Repair / Escalate / Reject.
    v1 is a fixed, explainable rule table; v2 (stretch) is a small learned weighting.

Input:
    n/a

Output:
    RuleTableDecisionAgent (v1), DecisionAgent (Protocol)

Dependencies:
    agents.decision.interface, agents.decision.rules

Implementation phase: P8
"""

from cxr_reliability.agents.decision.interface import DecisionAgent
from cxr_reliability.agents.decision.rules import RuleTableDecisionAgent

__all__ = ["DecisionAgent", "RuleTableDecisionAgent"]
