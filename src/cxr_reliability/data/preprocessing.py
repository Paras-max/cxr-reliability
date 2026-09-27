"""Data cleaning / preprocessing for NIH ChestX-ray14 metadata (Phase 3).

Responsibility:
    Identify and exclude records that are genuinely unusable for model
    training and evaluation.  Produce a structured exclusion report.

    Policy:
        Records are EXCLUDED only if they are genuinely unusable:
            - Image file cannot be found on disk (image_path == "")
            - Duplicate image ID (keep first occurrence)
            - Missing or NaN label (should never happen after nih.load_metadata)

        Records are FLAGGED but KEPT (documented in report):
            - Duplicate patient ID is fine — patients have multiple visits
            - Age > 120 or Age <= 0  (NIH contains age=414, a known data quirk)
            - Age is NaN
            - Gender not in {"M", "F"}
            - View position not in {"PA", "AP"}

    This module does NOT silently drop anything.  All decisions are recorded
    in the returned exclusion_report dict.

Input:
    Raw metadata DataFrame from nih.load_metadata()

Output:
    (cleaned_df, exclusion_report)
    cleaned_df      : DataFrame with unusable records removed
    exclusion_report: dict with counts and example image IDs for every check

Dependencies:
    pandas

Implementation phase: P3
"""

from __future__ import annotations

import pandas as pd

# Ages above this threshold are flagged as suspicious (known NIH data quirk)
_AGE_UPPER_FLAG = 120
_VALID_GENDERS = {"M", "F"}
_VALID_VIEWS = {"PA", "AP"}


def clean_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply cleaning rules to the raw metadata DataFrame.

    Returns
    -------
    cleaned_df      : DataFrame — unusable records removed, index reset
    exclusion_report: dict      — structured report of all checks
    """
    report: dict = {
        "input_records": int(len(df)),
        "exclusions": {},
        "flags": {},
        "output_records": 0,
    }

    exclude_mask = pd.Series(False, index=df.index)

    # ------------------------------------------------------------------
    # EXCLUSIONS — records that CANNOT be used for training/evaluation
    # ------------------------------------------------------------------

    # 1. Missing image file on disk
    missing_path_mask = df["image_path"].isna() | (df["image_path"].astype(str).str.strip() == "")
    _record_exclusion(
        report, "missing_image_on_disk",
        df[missing_path_mask]["image_id"].tolist(),
        "Image file not found in any images_NNN directory"
    )
    exclude_mask |= missing_path_mask

    # 2. Duplicate image ID — keep first occurrence
    dup_image_mask = df.duplicated(subset=["image_id"], keep="first")
    _record_exclusion(
        report, "duplicate_image_id",
        df[dup_image_mask]["image_id"].tolist(),
        "Duplicate image_id; only first occurrence is kept"
    )
    exclude_mask |= dup_image_mask

    # 3. Missing label (should not happen after load_metadata, but guard anyway)
    missing_label_mask = df["label"].isna()
    _record_exclusion(
        report, "missing_label",
        df[missing_label_mask]["image_id"].tolist(),
        "Binary label could not be derived"
    )
    exclude_mask |= missing_label_mask

    # ------------------------------------------------------------------
    # FLAGS — records kept but documented
    # ------------------------------------------------------------------

    # 4. Suspicious age (> 120 or <= 0) — known NIH data quirk (e.g. age=414)
    age_suspicious_mask = (
        df["age"].notna() & ((df["age"] > _AGE_UPPER_FLAG) | (df["age"] <= 0))
    )
    _record_flag(
        report, "suspicious_age",
        df[age_suspicious_mask]["image_id"].tolist(),
        f"Age > {_AGE_UPPER_FLAG} or <= 0 — NIH data quirk; record kept"
    )

    # 5. Missing age (NaN)
    nan_age_mask = df["age"].isna()
    _record_flag(
        report, "missing_age",
        df[nan_age_mask]["image_id"].tolist(),
        "Patient age could not be parsed; record kept with NaN age"
    )

    # 6. Invalid gender
    invalid_gender_mask = ~df["gender"].isin(_VALID_GENDERS)
    _record_flag(
        report, "invalid_gender",
        df[invalid_gender_mask]["image_id"].tolist(),
        f"Gender not in {_VALID_GENDERS}; record kept"
    )

    # 7. Invalid view position
    invalid_view_mask = ~df["view_position"].isin(_VALID_VIEWS)
    _record_flag(
        report, "invalid_view_position",
        df[invalid_view_mask]["image_id"].tolist(),
        f"View position not in {_VALID_VIEWS}; record kept"
    )

    # ------------------------------------------------------------------
    # Apply exclusions
    # ------------------------------------------------------------------
    cleaned_df = df[~exclude_mask].copy().reset_index(drop=True)

    total_excluded = int(exclude_mask.sum())
    report["total_excluded"] = total_excluded
    report["output_records"] = int(len(cleaned_df))
    report["exclusion_summary"] = (
        f"{total_excluded} records excluded from {report['input_records']} input records. "
        f"{report['output_records']} records remain."
    )

    return cleaned_df, report


def _record_exclusion(
    report: dict, key: str, image_ids: list, reason: str
) -> None:
    report["exclusions"][key] = {
        "count": len(image_ids),
        "reason": reason,
        "sample_image_ids": image_ids[:20],
    }


def _record_flag(
    report: dict, key: str, image_ids: list, reason: str
) -> None:
    report["flags"][key] = {
        "count": len(image_ids),
        "reason": reason,
        "sample_image_ids": image_ids[:20],
    }
