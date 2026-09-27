"""Provenance hashing.

Responsibility:
    Hash input images, config files and weight files so every audit record and result ties
    back to exact inputs and versions.

Input:
    Bytes, array, or file path.

Output:
    SHA-256 hex strings; software version dict.

Dependencies:
    hashlib

Implementation phase: P6
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def hash_image(image: "np.ndarray") -> str:
    raise NotImplementedError("Phase 6")


def hash_file(path: Path) -> str:
    raise NotImplementedError("Phase 6")


def software_versions() -> dict[str, str]:
    raise NotImplementedError("Phase 6")
