"""Natural OOD datasets: CheXpert and PadChest (PRD section 6).

Responsibility:
    Load frontal-view samples from CheXpert and PadChest for the natural OOD test of the
    NIH-only fast model. Optional far-OOD (non-chest) set is a proposed addition.

Input:
    Local dataset folders and their metadata files (access-approved downloads).

Output:
    Datasets yielding (image, source_name).

Dependencies:
    pandas, torch, pillow, data.transforms

Implementation phase: P1
"""

from __future__ import annotations

from pathlib import Path


class CheXpertOODSet:
    def __init__(self, root: Path, max_samples: int | None = None, seed: int = 42) -> None:
        self.root, self.max_samples, self.seed = root, max_samples, seed

    def __len__(self) -> int:
        raise NotImplementedError("Phase 1")

    def __getitem__(self, index: int):
        raise NotImplementedError("Phase 1")


class PadChestOODSet(CheXpertOODSet):
    """Same interface; PadChest-specific metadata parsing."""


class FarOODSet(CheXpertOODSet):
    """Optional: non-chest images to test wrong-body-part detection (proposed, not in PRD)."""
