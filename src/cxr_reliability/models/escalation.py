"""Escalation model (PRD FR-5 'Escalate to larger model').

Responsibility:
    Run resnet50-res512-all on the same image when the Decision Agent escalates. Same
    output contract as the Base Model. Note: this model was trained on NIH + CheXpert +
    PadChest + MIMIC, so it must not be used as the reference for the natural OOD test.

Input:
    Original or repaired image at 512 resolution.

Output:
    ModelForward (same shape as the fast model's output).

Dependencies:
    torch, models.txv_loader, models.base_model

Implementation phase: P2
"""

from __future__ import annotations

from typing import Any

from cxr_reliability.models.base_model import BaseModelAgent, ModelForward


class EscalationModel(BaseModelAgent):
    def run(self, image_tensor: Any) -> ModelForward:
        raise NotImplementedError("Phase 2")
