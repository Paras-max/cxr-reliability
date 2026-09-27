"""
scripts/create_splits.py
=========================
Phase 3 — Step 2: Create patient-level train / validation / test splits.

What this script does
---------------------
1. Loads data/processed/metadata.csv (produced by prepare_metadata.py).
2. Performs a reproducible, patient-level 70/15/15 split using
   sklearn GroupShuffleSplit.
3. Verifies zero patient overlap between all three splits (CRITICAL check).
4. Saves:
     data/processed/train.csv
     data/processed/validation.csv
     data/processed/test.csv
     data/processed/splits_manifest.json   (image_id lists + SHA-256 hash)
     outputs/dataset_split_report.json
     outputs/dataset_split_report.txt

SPLIT STRATEGY:
    - Split is performed at PATIENT level, not image level.
    - The same patient NEVER appears in more than one split.
    - Patient-level separation takes priority over perfect class balance.
    - Actual resulting class distribution is reported honestly.

REPRODUCIBILITY:
    - Seed is read from CXR_SEED in .env / Settings (default: 42).
    - Given the same metadata.csv and seed, the output is always identical.

Usage:
    python scripts/create_splits.py

Configuration:
    CXR_SEED         — random seed (default 42)
    CXR_DATA_DIR     — location of data/processed/
    CXR_OUTPUTS_DIR  — where split reports are written
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Bootstrap ────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from cxr_reliability.config.settings import get_settings
    _s = get_settings()
    _SEED = _s.seed
    _DATA_DIR = (
        _s.data_dir if _s.data_dir.is_absolute()
        else (_PROJECT_ROOT / _s.data_dir).resolve()
    )
    _OUTPUTS_DIR = (
        _s.outputs_dir if _s.outputs_dir.is_absolute()
        else (_PROJECT_ROOT / _s.outputs_dir).resolve()
    )
except Exception as _e:
    import os
    print(f"[WARN] Could not load settings ({_e}). Using defaults.")
    _SEED = int(os.environ.get("CXR_SEED", "42"))
    _DATA_DIR = (_PROJECT_ROOT / "data").resolve()
    _OUTPUTS_DIR = (_PROJECT_ROOT / "outputs").resolve()

_PROCESSED_DIR = _DATA_DIR / "processed"
_METADATA_PATH = _PROCESSED_DIR / "metadata.csv"

# Split proportions
_TEST_FRACTION = 0.15
_VAL_FRACTION = 0.15
# TRAIN fraction = 1 - 0.15 - 0.15 = 0.70  (derived, not hard-coded)


def _write_txt_report(report: dict, path: Path) -> None:
    lines: list[str] = []
    w = 70

    def a(s: str = "") -> None:
        lines.append(s)

    a("=" * w)
    a("  Phase 3 — Patient-Level Dataset Split Report")
    a(f"  Generated: {report.get('generated_at', 'N/A')}")
    a("=" * w)

    a()
    a("CONFIGURATION")
    a("-" * w)
    a(f"  Random seed     : {report.get('random_seed')}")
    a(f"  Test fraction   : {report.get('test_fraction')}")
    a(f"  Val fraction    : {report.get('val_fraction')}")
    a(f"  Train fraction  : {report.get('train_fraction')}")

    a()
    a("OVERALL COUNTS")
    a("-" * w)
    a(f"  Total patients  : {report.get('total_patients'):,}")
    a(f"  Total images    : {report.get('total_images'):,}")

    a()
    a("PER-SPLIT STATISTICS")
    a("-" * w)
    for split_name in ("train", "validation", "test"):
        st = report.get("split_stats", {}).get(split_name, {})
        a(f"  {split_name.upper()}")
        a(f"    Images          : {st.get('total_images', 0):,}")
        a(f"    Unique patients : {st.get('unique_patients', 0):,}")
        a(f"    Pneumonia       : {st.get('pneumonia_images', 0):,} "
          f"({st.get('pneumonia_pct', 0):.2f}%)")
        a(f"    Non-Pneumonia   : {st.get('non_pneumonia_images', 0):,} "
          f"({st.get('non_pneumonia_pct', 0):.2f}%)")

    a()
    a("PATIENT OVERLAP VERIFICATION")
    a("-" * w)
    overlaps = report.get("patient_overlap_verification", {})
    for pair, count in overlaps.items():
        status = "[PASS]" if count == 0 else "[FAIL]"
        a(f"  {status}  {pair}: {count} shared patients")

    a()
    a("EXCLUSION SUMMARY")
    a("-" * w)
    a(f"  {report.get('exclusion_summary', 'N/A')}")

    a()
    a("MANIFEST")
    a("-" * w)
    a(f"  SHA-256 : {report.get('manifest_sha256', 'N/A')}")

    a()
    a("VALIDATION STATUS")
    a("-" * w)
    status = "PASS" if report.get("patient_overlap_ok") else "FAIL"
    a(f"  Overall : {status}")
    a("=" * w)

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import datetime
    import pandas as pd
    from cxr_reliability.data.splits import (
        build_patient_level_splits,
        assert_no_patient_overlap,
        compute_split_stats,
    )
    from cxr_reliability.data.manifests import write_manifest

    print()
    print("=" * 70)
    print("  Phase 3 — Step 2: Create Patient-Level Splits")
    print("=" * 70)
    print(f"  Seed       : {_SEED}")
    print(f"  Test frac  : {_TEST_FRACTION}  Val frac: {_VAL_FRACTION}  "
          f"Train frac: {1 - _TEST_FRACTION - _VAL_FRACTION:.2f}")
    print(f"  Metadata   : {_METADATA_PATH}")
    print()

    # ── Load metadata ────────────────────────────────────────────────────
    if not _METADATA_PATH.exists():
        print(f"[ERROR] metadata.csv not found at {_METADATA_PATH}")
        print("        Run scripts/prepare_metadata.py first.")
        sys.exit(1)

    print("[1/5] Loading metadata.csv...")
    meta = pd.read_csv(_METADATA_PATH)
    print(f"      {len(meta):,} records, {meta['patient_id'].nunique():,} unique patients")

    # Load cleaning report for the split report
    cleaning_report_path = _PROCESSED_DIR / "cleaning_report.json"
    cleaning_summary = "N/A"
    if cleaning_report_path.exists():
        with open(cleaning_report_path, encoding="utf-8") as fh:
            cr = json.load(fh)
        cleaning_summary = cr.get("exclusion_summary", "N/A")

    # ── Build splits ──────────────────────────────────────────────────────
    print("[2/5] Building patient-level splits...")
    splits = build_patient_level_splits(
        meta,
        seed=_SEED,
        test_fraction=_TEST_FRACTION,
        val_fraction=_VAL_FRACTION,
    )
    for split_name, ids in splits.items():
        print(f"      {split_name:<12s}: {len(ids):,} images")

    # ── Verify zero patient overlap (CRITICAL) ────────────────────────────
    print("[3/5] Verifying patient overlap (critical check)...")
    try:
        assert_no_patient_overlap(meta, splits)
        print("      [PASS] No patient overlap detected.")
        overlap_ok = True
    except AssertionError as exc:
        print(f"      [FAIL] {exc}")
        overlap_ok = False
        sys.exit(3)

    # Compute overlap counts for report
    id_to_patient = dict(zip(meta["image_id"].astype(str), meta["patient_id"]))
    split_patients = {
        name: {id_to_patient[iid] for iid in ids if iid in id_to_patient}
        for name, ids in splits.items()
    }
    overlap_verification = {
        "train ∩ validation": len(split_patients["train"] & split_patients["validation"]),
        "train ∩ test": len(split_patients["train"] & split_patients["test"]),
        "validation ∩ test": len(split_patients["validation"] & split_patients["test"]),
    }

    # ── Compute class statistics ──────────────────────────────────────────
    print("[4/5] Computing split statistics...")
    stats = compute_split_stats(meta, splits)
    for split_name, st in stats.items():
        print(f"      {split_name:<12s}: {st['total_images']:>6,} images | "
              f"{st['unique_patients']:>5,} patients | "
              f"Pneumonia {st['pneumonia_pct']:.1f}%")

    # ── Save split CSVs ───────────────────────────────────────────────────
    print("[5/5] Saving split files...")
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    meta_indexed = meta.set_index("image_id")
    saved_paths: dict[str, Path] = {}

    for split_name, image_ids in splits.items():
        subset = meta_indexed.loc[
            [iid for iid in image_ids if iid in meta_indexed.index]
        ].reset_index()
        out_path = _PROCESSED_DIR / f"{split_name}.csv"
        subset.to_csv(out_path, index=False)
        saved_paths[split_name] = out_path
        print(f"      Saved: {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")

    # Save manifest (for reproducibility)
    manifest_path = _PROCESSED_DIR / "splits_manifest.json"
    manifest_hash = write_manifest(splits, manifest_path)
    print(f"      Manifest: {manifest_path}  (SHA-256: {manifest_hash[:16]}...)")

    # ── Build and save split report ───────────────────────────────────────
    now = datetime.datetime.now().isoformat(timespec="seconds")
    report = {
        "generated_at": now,
        "random_seed": _SEED,
        "test_fraction": _TEST_FRACTION,
        "val_fraction": _VAL_FRACTION,
        "train_fraction": round(1 - _TEST_FRACTION - _VAL_FRACTION, 2),
        "total_patients": int(meta["patient_id"].nunique()),
        "total_images": int(len(meta)),
        "split_stats": {
            name: {k: v for k, v in st.items()} for name, st in stats.items()
        },
        "patient_overlap_verification": overlap_verification,
        "patient_overlap_ok": overlap_ok,
        "exclusion_summary": cleaning_summary,
        "manifest_sha256": manifest_hash,
        "output_files": {
            name: str(p) for name, p in saved_paths.items()
        },
    }

    json_path = _OUTPUTS_DIR / "dataset_split_report.json"
    txt_path = _OUTPUTS_DIR / "dataset_split_report.txt"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    _write_txt_report(report, txt_path)

    print(f"      JSON report: {json_path}")
    print(f"      TXT report : {txt_path}")

    # ── Terminal summary ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Split Summary")
    print("=" * 70)
    print(f"  {'Split':<14} {'Images':>8} {'Patients':>10} {'Pneumonia%':>12}")
    print("  " + "-" * 48)
    for split_name, st in stats.items():
        print(f"  {split_name:<14} {st['total_images']:>8,} "
              f"{st['unique_patients']:>10,} {st['pneumonia_pct']:>11.2f}%")
    print()
    print(f"  Patient overlap : {'NONE (PASS)' if overlap_ok else 'DETECTED (FAIL)'}")
    print(f"  Manifest SHA-256: {manifest_hash[:32]}...")
    print()
    print("  Done. Run scripts/validate_splits.py next.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(1)
    except Exception as exc:
        import traceback
        print(f"\n[FATAL ERROR] {exc}")
        traceback.print_exc()
        sys.exit(2)
