"""Rare-but-valid case selection (PRD section 7).

Responsibility:
    Define and select in-distribution cases with rare label combinations or unusual
    strata, so the OOD false-positive rate can be reported separately from true
    distribution shift.

Input:
    NIH metadata DataFrame, selection criteria (approved by the project owner).

Output:
    List of filenames forming the rare-but-valid evaluation set.

Dependencies:
    pandas

Implementation phase: P1
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def select_rare_valid_cases(metadata: pd.DataFrame, seed: int) -> list[str]:
    raise NotImplementedError("Phase 1: criteria need owner sign-off")
