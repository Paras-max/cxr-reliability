"""Patient-level dataset splitting for NIH ChestX-ray14 (Phase 3).

Responsibility:
    Build TRAIN / VALIDATION / TEST splits with guaranteed zero patient overlap.
    Use sklearn GroupShuffleSplit to enforce patient-level grouping.

    Split proportions (configurable, PRD default):
        TRAIN      : 70%  of patients
        VALIDATION : 15%  of patients
        TEST       : 15%  of patients

    The split is deterministic given the same seed and metadata order.

Input:
    - cleaned metadata DataFrame (must contain 'patient_id' and 'image_id')
    - seed: int (from CXR_SEED via Settings)
    - val_fraction: float (fraction of non-test patients for validation)
    - test_fraction: float

Output:
    SplitManifest: dict mapping split name -> list of image_id strings
    {'train': [...], 'validation': [...], 'test': [...]}

Dependencies:
    pandas, numpy, scikit-learn

Implementation phase: P3
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SplitManifest = dict[str, list[str]]


def build_patient_level_splits(
    metadata: pd.DataFrame,
    seed: int = 42,
    test_fraction: float = 0.15,
    val_fraction: float = 0.15,
) -> SplitManifest:
    """
    Perform a reproducible, patient-level 70/15/15 split.

    Parameters
    ----------
    metadata      : DataFrame with 'patient_id' and 'image_id' columns
    seed          : random seed (default 42 from CXR_SEED)
    test_fraction : fraction of ALL patients assigned to test set
    val_fraction  : fraction of ALL patients assigned to validation set

    Returns
    -------
    SplitManifest — {'train': [...], 'validation': [...], 'test': [...]}

    Algorithm
    ---------
    1. Extract unique patients and shuffle deterministically.
    2. Use GroupShuffleSplit to carve out test patients first.
    3. Of the remaining patients, use GroupShuffleSplit again to carve
       out validation patients.
    4. The remainder becomes train.
    5. Map patient assignments back to image IDs.
    """
    if "patient_id" not in metadata.columns or "image_id" not in metadata.columns:
        raise ValueError("metadata must contain 'patient_id' and 'image_id' columns")

    groups = metadata["patient_id"].values
    image_ids = metadata["image_id"].values
    indices = np.arange(len(metadata))

    # ── Step 1: Split off TEST ───────────────────────────────────────────
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=seed
    )
    trainval_idx, test_idx = next(gss_test.split(indices, groups=groups))

    # ── Step 2: Split VALIDATION from remaining TRAIN+VAL ────────────────
    # val_fraction is relative to ALL data; recalculate for the remaining subset
    n_all_patients = len(np.unique(groups))
    n_test_patients = len(np.unique(groups[test_idx]))
    n_remaining_patients = n_all_patients - n_test_patients
    val_fraction_of_remaining = (val_fraction * n_all_patients) / n_remaining_patients

    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=min(val_fraction_of_remaining, 0.99),
        random_state=seed,
    )
    trainval_groups = groups[trainval_idx]
    train_rel_idx, val_rel_idx = next(
        gss_val.split(trainval_idx, groups=trainval_groups)
    )
    train_idx = trainval_idx[train_rel_idx]
    val_idx = trainval_idx[val_rel_idx]

    splits: SplitManifest = {
        "train": sorted(image_ids[train_idx].tolist()),
        "validation": sorted(image_ids[val_idx].tolist()),
        "test": sorted(image_ids[test_idx].tolist()),
    }
    return splits


def assert_no_patient_overlap(
    metadata: pd.DataFrame,
    splits: SplitManifest,
) -> None:
    """
    Raise AssertionError if any patient_id appears in more than one split.

    This is a CRITICAL safety check — a patient appearing in both train and
    test would invalidate evaluation results.
    """
    id_to_patient: dict[str, int] = dict(
        zip(metadata["image_id"], metadata["patient_id"], strict=False)
    )

    split_patients: dict[str, set] = {}
    for split_name, image_ids in splits.items():
        split_patients[split_name] = {
            id_to_patient[iid] for iid in image_ids if iid in id_to_patient
        }

    split_names = list(split_patients.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = split_patients[a] & split_patients[b]
            if overlap:
                raise AssertionError(
                    f"CRITICAL: Patient overlap detected between '{a}' and '{b}'! "
                    f"{len(overlap)} patient(s) appear in both splits. "
                    f"Sample patient IDs: {sorted(overlap)[:10]}"
                )


def compute_split_stats(
    metadata: pd.DataFrame, splits: SplitManifest
) -> dict[str, dict]:
    """
    Compute per-split statistics: image counts, patient counts, class distribution.

    Returns
    -------
    dict mapping split name -> stats dict
    """
    id_to_row = metadata.set_index("image_id")
    stats: dict[str, dict] = {}

    for split_name, image_ids in splits.items():
        subset = id_to_row.loc[
            [iid for iid in image_ids if iid in id_to_row.index]
        ]
        total = len(subset)
        pneumonia = int((subset["label"] == 1).sum())
        non_pneumonia = int((subset["label"] == 0).sum())
        unique_patients = int(subset["patient_id"].nunique())

        stats[split_name] = {
            "total_images": total,
            "unique_patients": unique_patients,
            "pneumonia_images": pneumonia,
            "non_pneumonia_images": non_pneumonia,
            "pneumonia_pct": round(100 * pneumonia / total, 2) if total > 0 else 0.0,
            "non_pneumonia_pct": round(100 * non_pneumonia / total, 2) if total > 0 else 0.0,
        }

    return stats
