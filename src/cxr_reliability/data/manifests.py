"""Split manifest I/O with content hashes (Phase 3).

Responsibility:
    Write and read split manifests (CSV format) and compute a stable SHA-256
    content hash so every result can reference the exact data split it used
    (PRD section 4, Reproducibility).

Input:
    SplitManifest: dict[str, list[str]]  — split name -> image_id list
    output path

Output:
    - One CSV per split (or a combined manifest)
    - SHA-256 hex digest of the serialised manifest content

Dependencies:
    hashlib, json, pandas, pathlib

Implementation phase: P3
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def write_manifest(splits: dict[str, list[str]], path: Path) -> str:
    """
    Write splits to a JSON manifest file and return its SHA-256 hex digest.

    The manifest is a JSON object: {split_name: [image_id, ...], ...}
    Keys and image_id lists are sorted for determinism.

    Parameters
    ----------
    splits : dict mapping split name -> sorted list of image_ids
    path   : destination .json file path

    Returns
    -------
    str — SHA-256 hex digest of the written content
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: sorted(v) for k, v in sorted(splits.items())}
    content = json.dumps(serialisable, indent=2)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def read_manifest(path: Path) -> dict[str, list[str]]:
    """
    Read a manifest JSON file previously written by write_manifest().

    Returns
    -------
    dict mapping split name -> list of image_ids
    """
    content = path.read_text(encoding="utf-8")
    return json.loads(content)


def hash_manifest(path: Path) -> str:
    """
    Compute and return the SHA-256 hex digest of an existing manifest file.
    Used to verify that a manifest has not been modified since it was written.
    """
    content = path.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
