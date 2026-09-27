"""NIH ChestX-ray14 metadata loader and binary label derivation (PRD section 6).

Responsibility:
    Parse Data_Entry_2017.csv, derive the binary Pneumonia/Non-Pneumonia label,
    and resolve image paths relative to the dataset root.

    LABELLING STRATEGY (documented as required by PRD):
        label = 1  ->  "Pneumonia"     — "Pneumonia" appears in Finding Labels
        label = 0  ->  "Non-Pneumonia" — any other finding(s), including "No Finding"

    IMPORTANT: label = 0 does NOT mean "Normal". The NIH ChestX-ray14 labels
    are text-mined from radiology reports and are known to be weakly supervised.
    Many label-0 images contain other pathologies. Do not interpret this binary
    label as a radiologist-confirmed diagnosis.

Input:
    - Path to Data_Entry_2017.csv
    - Path to dataset root (for resolving image_path)

Output:
    - metadata DataFrame with columns:
        image_id, image_path, patient_id, finding_labels,
        label, label_name, age, gender, view_position

Dependencies:
    pandas, pathlib

Implementation phase: P3
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

# Sub-folders inside each images_NNN directory that actually hold the PNGs.
# The NIH download places them under images_NNN/images/*.png
_IMAGE_SUBDIR = "images"

# Ordered list of image batch directories (images_001 .. images_012)
_IMAGE_DIRS = [f"images_{i:03d}" for i in range(1, 13)]


def _find_image_path(image_id: str, dataset_root: Path) -> str:
    """
    Return the relative path (from dataset_root) to image_id, or empty string
    if not found.  Checks images_NNN/images/ then images_NNN/ directly.
    """
    for dir_name in _IMAGE_DIRS:
        candidate = dataset_root / dir_name / _IMAGE_SUBDIR / image_id
        if candidate.exists():
            return str(Path(dir_name) / _IMAGE_SUBDIR / image_id)
        candidate_flat = dataset_root / dir_name / image_id
        if candidate_flat.exists():
            return str(Path(dir_name) / image_id)
    return ""


def load_metadata(csv_path: Path, dataset_root: Path) -> pd.DataFrame:
    """
    Read Data_Entry_2017.csv and return a cleaned metadata DataFrame.

    Columns in returned DataFrame
    ------------------------------
    image_id        : str   — original filename (e.g. 00000001_000.png)
    image_path      : str   — relative path from dataset_root to the file
    patient_id      : int   — Patient ID from CSV
    finding_labels  : str   — raw Finding Labels string (pipe-separated)
    label           : int   — binary: 1=Pneumonia, 0=Non-Pneumonia
    label_name      : str   — "Pneumonia" or "Non-Pneumonia"
    age             : float — Patient Age (NaN if unparseable)
    gender          : str   — "M" or "F" (empty string if missing)
    view_position   : str   — "PA" or "AP" (empty string if missing)
    """
    df = pd.read_csv(csv_path)

    # Rename to canonical internal names
    rename_map = {
        "Image Index": "image_id",
        "Finding Labels": "finding_labels",
        "Patient ID": "patient_id",
        "Patient Age": "age_raw",
        "Patient Gender": "gender",
        "View Position": "view_position",
    }
    df = df.rename(columns=rename_map)

    # Keep only the columns we need
    keep = list(rename_map.values())
    df = df[[c for c in keep if c in df.columns]].copy()

    # Parse age — NIH sometimes stores ages as "058Y"; strip trailing Y
    def _parse_age(val) -> float:
        s = str(val).strip().rstrip("Y").rstrip("y")
        try:
            return float(s)
        except ValueError:
            return float("nan")

    df["age"] = df["age_raw"].apply(_parse_age)
    df.drop(columns=["age_raw"], inplace=True)

    # Normalise string columns
    df["gender"] = df["gender"].fillna("").astype(str).str.strip().str.upper()
    df["view_position"] = (
        df["view_position"].fillna("").astype(str).str.strip().str.upper()
    )
    df["finding_labels"] = df["finding_labels"].fillna("").astype(str).str.strip()
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce")

    # Binary label derivation
    labels, label_names = make_pneumonia_labels(df)
    df["label"] = labels
    df["label_name"] = label_names

    # Resolve image paths relative to dataset_root
    print("  Resolving image paths (building path index)...")
    # Build a flat index once instead of searching per-row
    path_index: dict[str, str] = {}
    for dir_name in _IMAGE_DIRS:
        for sub in [_IMAGE_SUBDIR, ""]:
            search_dir = (
                dataset_root / dir_name / sub if sub else dataset_root / dir_name
            )
            if search_dir.is_dir():
                for p in search_dir.iterdir():
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        rel = (
                            str(Path(dir_name) / sub / p.name)
                            if sub
                            else str(Path(dir_name) / p.name)
                        )
                        path_index[p.name] = rel

    df["image_path"] = df["image_id"].map(path_index).fillna("")

    # Final column order
    col_order = [
        "image_id", "image_path", "patient_id",
        "finding_labels", "label", "label_name",
        "age", "gender", "view_position",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    return df.reset_index(drop=True)


def make_pneumonia_labels(
    metadata: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Derive binary Pneumonia / Non-Pneumonia labels from 'finding_labels' column.

    Returns
    -------
    labels      : pd.Series of int  (1 = Pneumonia, 0 = Non-Pneumonia)
    label_names : pd.Series of str  ("Pneumonia" or "Non-Pneumonia")

    Note: label = 0 means "Non-Pneumonia", NOT "Normal". Many label-0 images
    contain other pathologies. This is a weak text-mined label from the NIH
    radiology reports and should not be treated as a definitive clinical finding.
    """
    def _is_pneumonia(finding_str: str) -> bool:
        parts = [p.strip() for p in str(finding_str).split("|")]
        return "Pneumonia" in parts

    is_pneumonia = metadata["finding_labels"].apply(_is_pneumonia)
    labels = is_pneumonia.astype(int)
    label_names = is_pneumonia.map({True: "Pneumonia", False: "Non-Pneumonia"})
    return labels, label_names


def read_official_split_lists(list_dir: Path) -> dict[str, list[str]]:
    """
    Read the official NIH train_val_list.txt and test_list.txt.
    Returns {'train_val': [...], 'test': [...]} of image filenames.
    These lists are provided for reference; Phase 3 uses a custom patient-level
    split that respects patient boundaries independently.
    """
    result: dict[str, list[str]] = {}
    for key, filename in [("train_val", "train_val_list.txt"), ("test", "test_list.txt")]:
        path = list_dir / filename
        if path.exists():
            result[key] = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            result[key] = []
    return result


class NIHDataset:
    """
    Minimal dataset wrapper over NIH images given a split CSV manifest.
    Full torch.utils.data.Dataset implementation is deferred to Phase 4
    when model training begins.
    """

    def __init__(self, image_root: Path, manifest_path: Path, target_size: int) -> None:
        self.image_root = image_root
        self.manifest_path = manifest_path
        self.target_size = target_size
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.manifest_path)
        return self._df

    def __len__(self) -> int:
        return len(self._load())

    def __getitem__(self, index: int):
        raise NotImplementedError(
            "NIHDataset.__getitem__ will be implemented in Phase 4 "
            "when model training begins."
        )
