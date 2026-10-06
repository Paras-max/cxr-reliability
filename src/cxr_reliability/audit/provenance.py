"""Provenance hashing.

Responsibility:
    Hash input images, config files and weight files so every audit record and result ties
    back to exact inputs and versions.

Input:
    Bytes, array, or file path.

Output:
    SHA-256 hex strings; software version dict.

Dependencies:
    hashlib, sys, importlib.metadata
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def hash_image(image: np.ndarray) -> str:
    """Compute SHA-256 hash of a numpy image array's contiguous bytes."""
    import numpy as np
    contiguous = np.ascontiguousarray(image)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def hash_file(path: Path | str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def software_versions() -> dict[str, str]:
    """Return dictionary of core dependency software versions."""
    versions = {
        "python": sys.version.split()[0],
    }
    for pkg in ("torch", "torchvision", "numpy", "pandas", "scipy", "pydantic", "sklearn"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions
