"""Pipeline orchestration package (PRD FR-8).

Responsibility:
    Coordinate the seven agents end to end. Hand-rolled orchestrator per PRD section 5.

Input:
    n/a

Output:
    n/a

Dependencies:
    agents, models, audit

Implementation phase: P11 (extended from P6)
"""

from .orchestrator import ReliabilityPipeline

__all__ = ["ReliabilityPipeline"]
