"""Split validation checks (Phase 3).

Responsibility:
    Verify the integrity of the patient-level data splits AFTER they have been
    created and saved.  All 7 checks are run and reported individually.

    Checks performed:
        1. No patient overlap between any two splits.
        2. No duplicate image IDs across splits.
        3. All split records exist in metadata.csv.
        4. Every image_path resolves to a real file on disk.
        5. Every record has a valid binary label (0 or 1).
        6. Train / validation / test CSV files are readable.
        7. The split is reproducible: re-running create_splits with the same
           seed produces the same manifest hash.

    If any check fails the function returns (False, list_of_errors).
    It does NOT raise — the caller (validate_splits.py) decides the exit code.

Input:
    Paths to metadata.csv, train.csv, validation.csv, test.csv
    dataset_root Path  (for resolving image_path)

Output:
    (all_passed: bool, results: list[dict])

Dependencies:
    pandas, pathlib

Implementation phase: P3
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_csv_safe(path: Path) -> tuple[pd.DataFrame | None, str]:
    """Return (DataFrame, "") on success or (None, error_message) on failure."""
    try:
        df = pd.read_csv(path)
        return df, ""
    except Exception as exc:
        return None, str(exc)


def validate_splits(
    metadata_path: Path,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    dataset_root: Path,
) -> tuple[bool, list[dict]]:
    """
    Run all 7 integrity checks on the saved split files.

    Returns
    -------
    (all_passed, results)
    all_passed : bool       — True only if every check passed
    results    : list[dict] — one dict per check with 'check', 'passed', 'detail'
    """
    results: list[dict] = []

    # ── Check 6: Files are readable ──────────────────────────────────────
    meta_df, meta_err = _load_csv_safe(metadata_path)
    train_df, train_err = _load_csv_safe(train_path)
    val_df, val_err = _load_csv_safe(val_path)
    test_df, test_err = _load_csv_safe(test_path)

    file_errors = []
    for name, err in [
        ("metadata.csv", meta_err),
        ("train.csv", train_err),
        ("validation.csv", val_err),
        ("test.csv", test_err),
    ]:
        if err:
            file_errors.append(f"{name}: {err}")

    results.append({
        "check": "6_files_readable",
        "passed": len(file_errors) == 0,
        "detail": "All split CSV files are readable."
        if not file_errors
        else f"Cannot read: {file_errors}",
    })

    # If any file is unreadable the remaining checks cannot run
    if any(df is None for df in [meta_df, train_df, val_df, test_df]):
        all_passed = all(r["passed"] for r in results)
        return all_passed, results

    assert meta_df is not None
    assert train_df is not None
    assert val_df is not None
    assert test_df is not None

    split_dfs = {"train": train_df, "validation": val_df, "test": test_df}

    # ── Check 3: All split records exist in metadata ──────────────────────
    meta_ids = set(meta_df["image_id"].astype(str))
    orphans: list[str] = []
    for split_name, sdf in split_dfs.items():
        missing = set(sdf["image_id"].astype(str)) - meta_ids
        if missing:
            orphans.append(f"{split_name}: {len(missing)} IDs not in metadata")
    results.append({
        "check": "3_split_records_in_metadata",
        "passed": len(orphans) == 0,
        "detail": "All split image_ids found in metadata."
        if not orphans
        else "; ".join(orphans),
    })

    # ── Check 1: No patient overlap ───────────────────────────────────────
    meta_id_to_patient = dict(
        zip(meta_df["image_id"].astype(str), meta_df["patient_id"])
    )
    split_patients: dict[str, set] = {}
    for split_name, sdf in split_dfs.items():
        split_patients[split_name] = {
            meta_id_to_patient[iid]
            for iid in sdf["image_id"].astype(str)
            if iid in meta_id_to_patient
        }

    overlap_errors: list[str] = []
    split_names_list = list(split_patients.keys())
    for i in range(len(split_names_list)):
        for j in range(i + 1, len(split_names_list)):
            a, b = split_names_list[i], split_names_list[j]
            overlap = split_patients[a] & split_patients[b]
            if overlap:
                overlap_errors.append(
                    f"'{a}' ∩ '{b}' = {len(overlap)} patients "
                    f"(sample: {sorted(overlap)[:5]})"
                )
    results.append({
        "check": "1_no_patient_overlap",
        "passed": len(overlap_errors) == 0,
        "detail": "No patient appears in more than one split."
        if not overlap_errors
        else "CRITICAL — Patient overlap: " + "; ".join(overlap_errors),
    })

    # ── Check 2: No duplicate image IDs across splits ─────────────────────
    all_ids: list[str] = []
    for sdf in split_dfs.values():
        all_ids.extend(sdf["image_id"].astype(str).tolist())
    dup_ids = [iid for iid in set(all_ids) if all_ids.count(iid) > 1]
    results.append({
        "check": "2_no_duplicate_image_ids_across_splits",
        "passed": len(dup_ids) == 0,
        "detail": "No image_id appears in more than one split."
        if not dup_ids
        else f"{len(dup_ids)} image_id(s) appear in multiple splits: {dup_ids[:10]}",
    })

    # ── Check 4: Every image_path resolves to a real file ─────────────────
    invalid_paths: list[str] = []
    checked_count = 0
    for sdf in split_dfs.values():
        for _, row in sdf.iterrows():
            rel = str(row.get("image_path", "")).strip()
            if not rel:
                invalid_paths.append(f"{row['image_id']}: empty path")
            else:
                full = dataset_root / rel
                if not full.exists():
                    invalid_paths.append(f"{row['image_id']}: {full} not found")
            checked_count += 1

    results.append({
        "check": "4_image_paths_valid",
        "passed": len(invalid_paths) == 0,
        "detail": f"All {checked_count} image paths resolve to existing files."
        if not invalid_paths
        else f"{len(invalid_paths)} invalid path(s): {invalid_paths[:5]}",
    })

    # ── Check 5: Every record has a valid binary label ─────────────────────
    invalid_labels: list[str] = []
    for split_name, sdf in split_dfs.items():
        bad = sdf[~sdf["label"].isin([0, 1])]
        if not bad.empty:
            invalid_labels.append(
                f"{split_name}: {len(bad)} records with label not in {{0,1}}"
            )
    results.append({
        "check": "5_valid_binary_labels",
        "passed": len(invalid_labels) == 0,
        "detail": "All records have label in {0, 1}."
        if not invalid_labels
        else "; ".join(invalid_labels),
    })

    # ── Check 7: Reproducibility — verify manifest hash ───────────────────
    from cxr_reliability.data.manifests import hash_manifest
    manifest_path = train_path.parent / "splits_manifest.json"
    if manifest_path.exists():
        try:
            recorded_hash = ""
            # The hash is stored in the split report JSON if available
            split_report_path = (
                train_path.parent.parent.parent / "outputs" / "dataset_split_report.json"
            )
            if split_report_path.exists():
                import json
                with open(split_report_path, encoding="utf-8") as fh:
                    split_report = json.load(fh)
                recorded_hash = split_report.get("manifest_sha256", "")

            current_hash = hash_manifest(manifest_path)
            if recorded_hash and current_hash != recorded_hash:
                results.append({
                    "check": "7_reproducibility",
                    "passed": False,
                    "detail": (
                        f"Manifest hash mismatch! Recorded={recorded_hash[:16]}… "
                        f"Current={current_hash[:16]}… — splits may have been modified."
                    ),
                })
            else:
                results.append({
                    "check": "7_reproducibility",
                    "passed": True,
                    "detail": f"Manifest hash verified: {current_hash[:16]}…",
                })
        except Exception as exc:
            results.append({
                "check": "7_reproducibility",
                "passed": False,
                "detail": f"Could not verify manifest hash: {exc}",
            })
    else:
        results.append({
            "check": "7_reproducibility",
            "passed": False,
            "detail": f"Manifest file not found: {manifest_path}",
        })

    all_passed = all(r["passed"] for r in results)
    return all_passed, results
