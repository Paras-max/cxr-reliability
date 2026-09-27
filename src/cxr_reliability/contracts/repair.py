"""Repair Agent contract (PRD FR-6) — Phase 9.

Responsibility:
    Schema describing which non-destructive image-quality repairs were applied.
    The repaired pixel array is held in memory, never in the contract. Every repaired
    output is tagged repaired=True (or False when repair was skipped or refused).

Input:
    n/a

Output:
    RepairResult, RepairStep.

Dependencies:
    contracts.common, contracts.decision, contracts.quality

Implementation phase: P9 (extended from P0)
"""

from __future__ import annotations

from pydantic import Field

from .common import AgentName, AgentResult, StrictModel
from .decision import Action
from .quality import DefectType


class RepairStep(StrictModel):
    """Record of a single non-destructive repair operation."""

    defect: DefectType
    method: str  # e.g. "clahe", "nl_means", "unsharp_mask"
    parameters: dict[str, float | int | str]


class RepairResult(AgentResult):
    """
    Audit and provenance record returned by the Repair Agent.

    NOTE: The Repair Agent does NOT calculate post-repair quality metrics,
    does NOT evaluate verification deltas, and does NOT determine whether repair
    improved reliability. Verification belongs strictly to Phase 10 (Verification Agent).
    """

    agent: AgentName = AgentName.REPAIR
    repaired: bool = True  # Tagged True if repair was applied; False if skipped/refused
    repair_applied: bool = False  # Explicit boolean indicating whether repair was executed
    action: Action | None = None  # Driving decision action (e.g. Action.REPAIR)
    steps: list[RepairStep] = Field(default_factory=list)
    defects_detected: list[DefectType] = Field(default_factory=list)
    original_shape: list[int] | None = None
    repaired_shape: list[int] | None = None
    original_range: list[float] | None = None  # [min_intensity, max_intensity]
    repaired_range: list[float] | None = None  # [min_intensity, max_intensity]
    image_changed: bool = False  # True only if repaired array differs numerically from input
    skipped_reason: str | None = None
    refused_bounds: bool = False
