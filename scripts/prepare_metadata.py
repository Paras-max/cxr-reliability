"""
scripts/prepare_metadata.py
============================
Phase 3 — Step 1: Build and clean the NIH ChestX-ray14 metadata.

What this script does
---------------------
1. Reads Data_Entry_2017.csv from CXR_DATASET_ROOT.
2. Resolves each image filename to its actual path inside images_001/..images_012/.
3. Derives the binary Pneumonia / Non-Pneumonia label.
4. Cleans the metadata (identifies and excludes genuinely unusable records).
5. Saves data/processed/metadata.csv.
6. Prints a cleaning report summary to the terminal.

LABELLING NOTE:
    label = 1  →  "Pneumonia"     (Finding Labels contains "Pneumonia")
    label = 0  →  "Non-Pneumonia" (all other findings, including "No Finding")
    label = 0 does NOT mean "Normal". The NIH dataset uses weak text-mined
    labels from radiology reports. Many label-0 images contain other pathologies.
    This binary label is NOT a radiologist-confirmed diagnosis.

Usage:
    python scripts/prepare_metadata.py

Configuration:
    CXR_DATASET_ROOT  — path to the dataset folder (in .env or environment)
    CXR_DATA_DIR      — where data/processed/ will be created (default: data/)

Output:
    data/processed/metadata.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Bootstrap: make src/ importable when run as a plain script ────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ── Load project settings ────────────────────────────────────────────────
try:
    from cxr_reliability.config.settings import get_settings
    _s = get_settings()
    _DATASET_ROOT = (
        _s.dataset_root if _s.dataset_root.is_absolute()
        else (_PROJECT_ROOT / _s.dataset_root).resolve()
    )
    _DATA_DIR = (
        _s.data_dir if _s.data_dir.is_absolute()
        else (_PROJECT_ROOT / _s.data_dir).resolve()
    )
except Exception as _e:
    print(f"[WARN] Could not load settings ({_e}). Falling back to defaults.")
    _DATASET_ROOT = (_PROJECT_ROOT / "dataset").resolve()
    _DATA_DIR = (_PROJECT_ROOT / "data").resolve()

_PROCESSED_DIR = _DATA_DIR / "processed"
_CSV_PATH = _DATASET_ROOT / "Data_Entry_2017.csv"


def main() -> None:
    import datetime

    from cxr_reliability.data.nih import load_metadata
    from cxr_reliability.data.preprocessing import clean_metadata

    print()
    print("=" * 70)
    print("  Phase 3 — Step 1: Prepare Metadata")
    print("=" * 70)
    print(f"  Dataset root : {_DATASET_ROOT}")
    print(f"  CSV          : {_CSV_PATH}")
    print(f"  Output dir   : {_PROCESSED_DIR}")
    print()

    # ── Validate inputs ──────────────────────────────────────────────────
    if not _CSV_PATH.exists():
        print(f"[ERROR] Data_Entry_2017.csv not found at {_CSV_PATH}")
        sys.exit(1)
    if not _DATASET_ROOT.exists():
        print(f"[ERROR] Dataset root not found: {_DATASET_ROOT}")
        sys.exit(1)

    # ── Step 1: Load metadata ────────────────────────────────────────────
    print("[1/3] Loading and parsing Data_Entry_2017.csv...")
    df = load_metadata(_CSV_PATH, _DATASET_ROOT)
    print(f"      Loaded {len(df):,} records")

    # ── Label distribution ────────────────────────────────────────────────
    pneumonia_count = int((df["label"] == 1).sum())
    non_pneumonia_count = int((df["label"] == 0).sum())
    total = len(df)
    print()
    print("  Label distribution (raw, before cleaning):")
    print(f"    Pneumonia     : {pneumonia_count:>7,} ({100*pneumonia_count/total:.2f}%)")
    print(f"    Non-Pneumonia : {non_pneumonia_count:>7,} ({100*non_pneumonia_count/total:.2f}%)")
    print(f"    Total         : {total:>7,}")

    # ── Step 2: Clean metadata ────────────────────────────────────────────
    print()
    print("[2/3] Cleaning metadata...")
    cleaned_df, excl_report = clean_metadata(df)

    print(f"      Input records    : {excl_report['input_records']:,}")
    print(f"      Total excluded   : {excl_report['total_excluded']:,}")
    print(f"      Output records   : {excl_report['output_records']:,}")
    print()

    if excl_report["exclusions"]:
        print("  EXCLUSIONS (records removed):")
        for key, info in excl_report["exclusions"].items():
            if info["count"] > 0:
                print(f"    {key:<30s}: {info['count']:>6,}  — {info['reason']}")

    if excl_report["flags"]:
        print()
        print("  FLAGS (records kept but documented):")
        for key, info in excl_report["flags"].items():
            if info["count"] > 0:
                print(f"    {key:<30s}: {info['count']:>6,}  — {info['reason']}")

    # ── Final label distribution after cleaning ───────────────────────────
    pneu_clean = int((cleaned_df["label"] == 1).sum())
    non_pneu_clean = int((cleaned_df["label"] == 0).sum())
    total_clean = len(cleaned_df)
    print()
    print("  Label distribution (after cleaning):")
    print(f"    Pneumonia     : {pneu_clean:>7,} ({100*pneu_clean/total_clean:.2f}%)")
    print(f"    Non-Pneumonia : {non_pneu_clean:>7,} ({100*non_pneu_clean/total_clean:.2f}%)")
    print(f"    Total         : {total_clean:>7,}")
    print(f"    Unique patients: {cleaned_df['patient_id'].nunique():,}")

    # ── Step 3: Save metadata.csv ─────────────────────────────────────────
    print()
    print("[3/3] Saving metadata.csv...")
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _PROCESSED_DIR / "metadata.csv"
    cleaned_df.to_csv(out_path, index=False)
    print(f"      Saved: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")

    # Save cleaning report alongside metadata
    report_out = _PROCESSED_DIR / "cleaning_report.json"
    excl_report["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(report_out, "w", encoding="utf-8") as fh:
        json.dump(excl_report, fh, indent=2, default=str)
    print(f"      Cleaning report: {report_out}")

    print()
    print("  Done. Run scripts/create_splits.py next.")
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
