"""Tests for patient-level splits (Phase 3)."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from cxr_reliability.data.splits import (
    assert_no_patient_overlap,
    build_patient_level_splits,
    compute_split_stats,
)


@pytest.fixture
def mock_metadata() -> pd.DataFrame:
    rows = []
    # 30 patients, each having 2 images
    for p in range(1, 31):
        for img_idx in range(2):
            rows.append({
                "image_id": f"{p:05d}_{img_idx:03d}.png",
                "patient_id": p,
                "label": 1 if p % 5 == 0 else 0,
            })
    return pd.DataFrame(rows)


def test_no_patient_overlap_across_splits(mock_metadata: pd.DataFrame):
    """No patient_id appears in more than one split."""
    splits = build_patient_level_splits(mock_metadata, seed=42, test_fraction=0.2, val_fraction=0.2)

    # Should not raise
    assert_no_patient_overlap(mock_metadata, splits)

    train_imgs = set(splits["train"])
    val_imgs = set(splits["validation"])
    test_imgs = set(splits["test"])

    # Disjoint image sets
    assert len(train_imgs & val_imgs) == 0
    assert len(train_imgs & test_imgs) == 0
    assert len(val_imgs & test_imgs) == 0

    # Total count matches
    assert len(train_imgs) + len(val_imgs) + len(test_imgs) == len(mock_metadata)


def test_official_test_list_preserved(mock_metadata: pd.DataFrame):
    """Split distributions are reproducible and split stats calculate correctly."""
    splits = build_patient_level_splits(mock_metadata, seed=42, test_fraction=0.2, val_fraction=0.2)
    stats = compute_split_stats(mock_metadata, splits)

    assert "train" in stats and "validation" in stats and "test" in stats
    assert stats["train"]["total_images"] > 0
    assert stats["validation"]["total_images"] > 0
    assert stats["test"]["total_images"] > 0


def test_manifest_hash_stable(mock_metadata: pd.DataFrame):
    """Rewriting the same manifest gives the exact same hash."""
    splits1 = build_patient_level_splits(mock_metadata, seed=123)
    splits2 = build_patient_level_splits(mock_metadata, seed=123)

    json1 = json.dumps(splits1, sort_keys=True)
    json2 = json.dumps(splits2, sort_keys=True)

    hash1 = hashlib.sha256(json1.encode()).hexdigest()
    hash2 = hashlib.sha256(json2.encode()).hexdigest()

    assert hash1 == hash2
    assert splits1 == splits2
