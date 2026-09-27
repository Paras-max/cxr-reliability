"""
scripts/validate_dataset.py
============================
Phase 2 — NIH ChestX-ray14 Dataset Validation

Responsibility:
    Validate the local copy of the NIH ChestX-ray14 dataset without modifying
    any original files.  Produces two reports:
        outputs/dataset_validation_report.json   – structured results
        outputs/dataset_validation_report.txt    – human-readable summary

Usage:
    python scripts/validate_dataset.py

Configuration:
    Set CXR_DATASET_ROOT in .env (or as an environment variable) to point at
    the folder containing Data_Entry_2017.csv and images_001/ … images_012/.
    If not set, the script falls back to the dataset/ folder next to the
    project root.

Notes:
    • No images are loaded into RAM simultaneously; each is opened, checked for
      basic readability and size, then immediately closed.
    • No dataset files are modified, renamed, moved, or deleted.
    • All paths are resolved with pathlib for Windows compatibility.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap:  make sure src/ is importable when running as a plain script
# (i.e. without `pip install -e .`).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/
_PROJECT_ROOT = _SCRIPT_DIR.parent                     # project root
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# Load settings (reads .env if present)
# ---------------------------------------------------------------------------
try:
    from cxr_reliability.config.settings import get_settings
    _settings = get_settings()
    _DATASET_ROOT: Path = _settings.dataset_root.resolve() \
        if not _settings.dataset_root.is_absolute() \
        else _settings.dataset_root
    # If the resolved path doesn't exist, try resolving relative to project root
    if not _DATASET_ROOT.exists():
        _DATASET_ROOT = (_PROJECT_ROOT / _settings.dataset_root).resolve()
    _OUTPUTS_DIR: Path = _settings.outputs_dir.resolve() \
        if not _settings.outputs_dir.is_absolute() \
        else _settings.outputs_dir
    if not _OUTPUTS_DIR.exists():
        _OUTPUTS_DIR = (_PROJECT_ROOT / _settings.outputs_dir).resolve()
except Exception as _e:
    print(f"[WARN] Could not load project settings ({_e}). "
          "Falling back to environment / defaults.")
    import os
    _raw = os.environ.get("CXR_DATASET_ROOT", "dataset")
    _DATASET_ROOT = Path(_raw).resolve() if Path(_raw).is_absolute() \
        else (_PROJECT_ROOT / _raw).resolve()
    _OUTPUTS_DIR = (_PROJECT_ROOT / "outputs").resolve()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CSV_FILENAME = "Data_Entry_2017.csv"
EXPECTED_IMAGE_DIRS = [f"images_{i:03d}" for i in range(1, 13)]  # images_001..images_012

# All 15 pathology labels defined in the NIH ChestX-ray14 paper
NIH_LABELS = {
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia", "No Finding",
}

REQUIRED_CSV_COLUMNS = {
    "Image Index", "Finding Labels", "Follow-up #", "Patient ID",
    "Patient Age", "Patient Gender", "View Position",
    # NIH CSV splits bracket columns across two column headers, e.g.:
    #   "OriginalImage[Width"  and  "Height]"
    #   "OriginalImagePixelSpacing[x"  and  "y]"
    "OriginalImage[Width", "Height]",
    "OriginalImagePixelSpacing[x", "y]",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "-", width: int = 70) -> str:
    return char * width


def _resolve_image_dirs() -> dict[str, Path]:
    """Return mapping dir_name -> Path for all expected image directories."""
    return {d: _DATASET_ROOT / d for d in EXPECTED_IMAGE_DIRS}


def _build_image_index() -> dict[str, Path]:
    """
    Walk all images_NNN sub-folders and build a flat filename->path mapping.
    Memory-efficient: we only store Path objects, not pixel data.
    """
    index: dict[str, Path] = {}
    for dir_name, dir_path in _resolve_image_dirs().items():
        if not dir_path.exists():
            continue
        images_subdir = dir_path / "images"
        search_dir = images_subdir if images_subdir.is_dir() else dir_path
        for p in search_dir.iterdir():
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                index[p.name] = p
    return index


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_csv_exists(report: dict) -> bool:
    csv_path = _DATASET_ROOT / CSV_FILENAME
    exists = csv_path.is_file()
    report["csv_exists"] = exists
    report["csv_path"] = str(csv_path)
    return exists


def check_image_dirs(report: dict) -> dict[str, bool]:
    dir_status: dict[str, bool] = {}
    for dir_name, dir_path in _resolve_image_dirs().items():
        dir_status[dir_name] = dir_path.is_dir()
    report["image_dirs_status"] = dir_status
    report["image_dirs_all_present"] = all(dir_status.values())
    missing = [d for d, ok in dir_status.items() if not ok]
    report["image_dirs_missing"] = missing
    return dir_status


def load_csv(report: dict):
    """Load CSV with pandas.  Returns DataFrame or None."""
    import pandas as pd
    csv_path = _DATASET_ROOT / CSV_FILENAME
    try:
        df = pd.read_csv(csv_path)
        report["csv_readable"] = True
        return df
    except Exception as exc:
        report["csv_readable"] = False
        report["csv_read_error"] = str(exc)
        return None


def analyse_csv(df, report: dict) -> None:
    """Extract all metadata statistics from the DataFrame."""
    import numpy as np

    # Basic counts
    report["total_csv_records"] = int(len(df))
    report["csv_columns"] = list(df.columns)

    # Check for required columns
    present_cols = set(df.columns)
    missing_cols = [c for c in REQUIRED_CSV_COLUMNS if c not in present_cols]
    report["required_columns_missing"] = missing_cols

    # Patient / image uniqueness
    patient_col = "Patient ID"
    image_col = "Image Index"
    report["unique_patients"] = int(df[patient_col].nunique()) \
        if patient_col in df.columns else None
    report["unique_images_in_csv"] = int(df[image_col].nunique()) \
        if image_col in df.columns else None

    # Duplicate image filenames in CSV
    if image_col in df.columns:
        dup_mask = df.duplicated(subset=[image_col], keep=False)
        dup_images = df.loc[dup_mask, image_col].unique().tolist()
        report["duplicate_image_filenames_in_csv"] = dup_images
        report["duplicate_image_count_in_csv"] = int(dup_mask.sum())
    else:
        report["duplicate_image_filenames_in_csv"] = []
        report["duplicate_image_count_in_csv"] = 0

    # Finding Labels analysis
    if "Finding Labels" in df.columns:
        label_counter: Counter = Counter()
        for entry in df["Finding Labels"].dropna():
            for lbl in str(entry).split("|"):
                label_counter[lbl.strip()] += 1

        report["all_label_counts"] = dict(label_counter.most_common())
        report["pneumonia_count"] = int(label_counter.get("Pneumonia", 0))
        report["no_finding_count"] = int(label_counter.get("No Finding", 0))

        found_labels = set(label_counter.keys())
        report["labels_found"] = sorted(found_labels)
        report["nih_labels_missing_from_data"] = sorted(NIH_LABELS - found_labels)
        report["unexpected_labels_in_data"] = sorted(found_labels - NIH_LABELS)
    else:
        report["all_label_counts"] = {}
        report["pneumonia_count"] = None
        report["no_finding_count"] = None

    # View Position distribution
    if "View Position" in df.columns:
        view_counts = df["View Position"].value_counts(dropna=False)
        report["view_position_distribution"] = {
            str(k): int(v) for k, v in view_counts.items()
        }
    else:
        report["view_position_distribution"] = {}

    # Age statistics
    if "Patient Age" in df.columns:
        ages = df["Patient Age"].apply(
            lambda x: float(str(x).rstrip("Y")) if str(x).endswith("Y") else float(x)
            if str(x).replace(".", "", 1).isdigit() else None
        ).dropna()
        report["age_statistics"] = {
            "count": int(ages.count()),
            "mean": round(float(ages.mean()), 2),
            "std": round(float(ages.std()), 2),
            "min": int(ages.min()),
            "25th_percentile": float(np.percentile(ages, 25)),
            "median": float(np.median(ages)),
            "75th_percentile": float(np.percentile(ages, 75)),
            "max": int(ages.max()),
        }
    else:
        report["age_statistics"] = {}

    # Gender distribution
    if "Patient Gender" in df.columns:
        gender_counts = df["Patient Gender"].value_counts(dropna=False)
        report["gender_distribution"] = {
            str(k): int(v) for k, v in gender_counts.items()
        }
    else:
        report["gender_distribution"] = {}


def check_images_on_disk(df, report: dict) -> None:
    """
    1. Build a flat filename->path index from all image folders.
    2. Compare against CSV to find missing files.
    3. Incrementally open each file to detect corruption.
    4. Collect dimension statistics.

    Images are opened one at a time and immediately closed — no bulk RAM load.
    """
    from PIL import Image as PILImage

    image_col = "Image Index"
    if image_col not in df.columns:
        report["image_disk_check"] = "skipped — 'Image Index' column not found"
        return

    print("  Building image file index (scanning image directories)...")
    t0 = time.time()
    disk_index = _build_image_index()
    elapsed = time.time() - t0
    print(f"  Found {len(disk_index):,} image files on disk in {elapsed:.1f}s")

    csv_filenames: set[str] = set(df[image_col].dropna().astype(str))

    # Missing files
    missing = sorted(csv_filenames - set(disk_index.keys()))
    report["missing_image_count"] = len(missing)
    report["missing_image_files_sample"] = missing[:50]

    # Extra files on disk not in CSV
    extra = sorted(set(disk_index.keys()) - csv_filenames)
    report["extra_image_files_on_disk_count"] = len(extra)
    report["total_image_files_on_disk"] = len(disk_index)

    # Corruption check + dimension collection
    print(f"  Checking {len(disk_index):,} image files for corruption "
          "(this may take a few minutes)...")
    corrupted: list[str] = []
    widths: list[int] = []
    heights: list[int] = []
    checked = 0
    report_every = max(1, len(disk_index) // 20)

    for fname, fpath in disk_index.items():
        try:
            with PILImage.open(fpath) as img:
                img.verify()
            # Re-open to get dimensions (verify() closes the file internally)
            with PILImage.open(fpath) as img:
                w, h = img.size
            widths.append(w)
            heights.append(h)
        except Exception:
            corrupted.append(fname)
        checked += 1
        if checked % report_every == 0:
            pct = 100 * checked / len(disk_index)
            print(f"    ... {checked:,}/{len(disk_index):,} ({pct:.0f}%) checked, "
                  f"{len(corrupted)} corrupted so far")

    report["corrupted_image_count"] = len(corrupted)
    report["corrupted_image_files_sample"] = corrupted[:50]

    # Dimension statistics
    if widths:
        unique_widths = sorted(set(widths))
        unique_heights = sorted(set(heights))
        # Cap unique_values to avoid enormous JSON
        if len(unique_widths) > 30:
            unique_widths = unique_widths[:15] + unique_widths[-15:]
        if len(unique_heights) > 30:
            unique_heights = unique_heights[:15] + unique_heights[-15:]
        report["image_dimension_statistics"] = {
            "count": len(widths),
            "width": {
                "min": int(min(widths)),
                "max": int(max(widths)),
                "mean": round(float(sum(widths) / len(widths)), 1),
                "unique_values_sample": unique_widths,
            },
            "height": {
                "min": int(min(heights)),
                "max": int(max(heights)),
                "mean": round(float(sum(heights) / len(heights)), 1),
                "unique_values_sample": unique_heights,
            },
        }
    else:
        report["image_dimension_statistics"] = {}


def prd_consistency_check(report: dict) -> dict[str, Any]:
    """Verify the dataset structure matches PRD expectations."""
    checks: dict[str, Any] = {}

    checks["csv_present"] = report.get("csv_exists", False)
    checks["all_12_image_dirs_present"] = report.get("image_dirs_all_present", False)

    total = report.get("total_csv_records")
    if total is not None:
        checks["record_count_plausible"] = (100_000 <= total <= 120_000)
        checks["record_count"] = total
    else:
        checks["record_count_plausible"] = False

    pneu = report.get("pneumonia_count")
    checks["pneumonia_cases_present"] = (pneu is not None and pneu > 0)
    checks["pneumonia_count"] = pneu

    checks["no_unexpected_labels"] = (
        len(report.get("unexpected_labels_in_data", [])) == 0
    )
    checks["required_columns_present"] = (
        len(report.get("required_columns_missing", [])) == 0
    )

    missing_count = report.get("missing_image_count", 0)
    checks["no_missing_images"] = (missing_count == 0)
    checks["missing_image_count"] = missing_count

    corrupted_count = report.get("corrupted_image_count", 0)
    checks["no_corrupted_images"] = (corrupted_count == 0)
    checks["corrupted_image_count"] = corrupted_count

    bool_checks = {k: v for k, v in checks.items() if isinstance(v, bool)}
    checks["overall_pass"] = all(bool_checks.values())

    report["prd_consistency_checks"] = checks
    return checks


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _overall_status(report: dict) -> str:
    prd = report.get("prd_consistency_checks", {})
    if prd.get("overall_pass"):
        return "PASS"
    critical = [
        prd.get("csv_present"),
        prd.get("all_12_image_dirs_present"),
        prd.get("record_count_plausible"),
    ]
    if not all(critical):
        return "FAIL"
    return "WARN"


def write_json_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)


def write_txt_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    status = _overall_status(report)
    add("=" * 70)
    add("  NIH ChestX-ray14 Dataset Validation Report")
    add(f"  Generated: {report.get('generated_at', 'N/A')}")
    add(f"  Overall Status: {status}")
    add("=" * 70)

    add()
    add("DATASET PATH")
    add("-" * 70)
    add(f"  Root : {report.get('dataset_root', 'N/A')}")
    add(f"  CSV  : {report.get('csv_path', 'N/A')}")

    add()
    add("CSV STATUS")
    add("-" * 70)
    add(f"  Exists      : {report.get('csv_exists')}")
    add(f"  Readable    : {report.get('csv_readable')}")
    total = report.get('total_csv_records')
    add(f"  Total rows  : {total:,}" if total else "  Total rows  : N/A")
    add(f"  Columns     : {', '.join(report.get('csv_columns', []))}")
    missing_cols = report.get("required_columns_missing", [])
    add(f"  Required cols missing : {missing_cols if missing_cols else 'None'}")

    add()
    add("IMAGE DIRECTORIES (images_001 to images_012)")
    add("-" * 70)
    for d, ok in report.get("image_dirs_status", {}).items():
        tick = "[OK]" if ok else "[MISSING]"
        add(f"  {tick}  {d}")
    missing_dirs = report.get("image_dirs_missing", [])
    if missing_dirs:
        add(f"  MISSING DIRS: {missing_dirs}")

    add()
    add("UNIQUE ENTITY COUNTS")
    add("-" * 70)
    add(f"  Unique patients         : {report.get('unique_patients', 'N/A')}")
    add(f"  Unique images (CSV)     : {report.get('unique_images_in_csv', 'N/A')}")
    add(f"  Image files on disk     : {report.get('total_image_files_on_disk', 'N/A')}")
    add(f"  Missing images          : {report.get('missing_image_count', 'N/A')}")
    add(f"  Extra files on disk     : {report.get('extra_image_files_on_disk_count', 'N/A')}")
    add(f"  Duplicate entries (CSV) : {report.get('duplicate_image_count_in_csv', 'N/A')}")
    add(f"  Corrupted images        : {report.get('corrupted_image_count', 'N/A')}")

    add()
    add("FINDING LABEL DISTRIBUTION")
    add("-" * 70)
    for label, cnt in report.get("all_label_counts", {}).items():
        add(f"  {label:<30s}: {cnt:>7,}")
    unexpected = report.get("unexpected_labels_in_data", [])
    if unexpected:
        add(f"  *** Unexpected labels found: {unexpected}")

    add()
    add("VIEW POSITION DISTRIBUTION")
    add("-" * 70)
    for vp, cnt in report.get("view_position_distribution", {}).items():
        add(f"  {str(vp):<15s}: {cnt:>7,}")

    add()
    add("AGE STATISTICS")
    add("-" * 70)
    age = report.get("age_statistics", {})
    if age:
        add(f"  Count  : {age.get('count')}")
        add(f"  Mean   : {age.get('mean')} +/- {age.get('std')}")
        add(f"  Range  : {age.get('min')} - {age.get('max')}")
        add(f"  Median : {age.get('median')}")
    else:
        add("  No age data available.")

    add()
    add("GENDER DISTRIBUTION")
    add("-" * 70)
    for g, cnt in report.get("gender_distribution", {}).items():
        add(f"  {str(g):<10s}: {cnt:>7,}")

    add()
    add("IMAGE DIMENSION STATISTICS")
    add("-" * 70)
    dim = report.get("image_dimension_statistics", {})
    if dim:
        w = dim.get("width", {})
        h = dim.get("height", {})
        add(f"  Images checked : {dim.get('count', 0):,}")
        add(f"  Width  -- min:{w.get('min')}  max:{w.get('max')}  mean:{w.get('mean')}")
        add(f"  Height -- min:{h.get('min')}  max:{h.get('max')}  mean:{h.get('mean')}")
    else:
        add("  No dimension data collected.")

    add()
    add("PRD CONSISTENCY CHECKS")
    add("-" * 70)
    for check, val in report.get("prd_consistency_checks", {}).items():
        if isinstance(val, bool):
            tick = "[PASS]" if val else "[FAIL]"
            add(f"  {tick}  {check}")
        else:
            add(f"         {check}: {val}")

    if report.get("missing_image_files_sample"):
        add()
        add("SAMPLE MISSING FILES (first 50)")
        add("-" * 70)
        for f in report["missing_image_files_sample"]:
            add(f"  {f}")

    if report.get("corrupted_image_files_sample"):
        add()
        add("SAMPLE CORRUPTED FILES (first 50)")
        add("-" * 70)
        for f in report["corrupted_image_files_sample"]:
            add(f"  {f}")

    add()
    add("=" * 70)
    add(f"  END OF REPORT  --  Status: {status}")
    add("=" * 70)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def print_summary(report: dict) -> None:
    status = _overall_status(report)
    w = 60
    print()
    print("=" * w)
    print("  Dataset Validation Summary")
    print("=" * w)
    print(f"  Dataset root   : {report.get('dataset_root')}")
    print(f"  CSV found      : {report.get('csv_exists')}")
    print(f"  Dirs present   : {report.get('image_dirs_all_present')}")
    total = report.get('total_csv_records')
    print(f"  Total records  : {total:,}" if total else "  Total records  : N/A")
    print(f"  Unique patients: {report.get('unique_patients', 'N/A')}")
    print(f"  Unique images  : {report.get('unique_images_in_csv', 'N/A')}")
    print(f"  Pneumonia cases: {report.get('pneumonia_count', 'N/A')}")
    print(f"  No Finding     : {report.get('no_finding_count', 'N/A')}")
    print(f"  Missing files  : {report.get('missing_image_count', 'N/A')}")
    print(f"  Corrupted files: {report.get('corrupted_image_count', 'N/A')}")
    print("-" * w)
    print(f"  Overall status : {status}")
    print("-" * w)
    json_out = _OUTPUTS_DIR / "dataset_validation_report.json"
    txt_out  = _OUTPUTS_DIR / "dataset_validation_report.txt"
    print(f"  JSON report    : {json_out}")
    print(f"  TXT report     : {txt_out}")
    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import datetime

    report: dict[str, Any] = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(_DATASET_ROOT),
        "script_version": "2.0.0",
    }

    print()
    print("=" * 70)
    print("  NIH ChestX-ray14 -- Phase 2 Dataset Validation")
    print("=" * 70)
    print(f"  Dataset root : {_DATASET_ROOT}")
    print()

    # Step 1: CSV presence
    print("[1/6] Checking CSV presence...")
    csv_ok = check_csv_exists(report)
    print(f"      CSV exists: {csv_ok}")

    # Step 2: Image directories
    print("[2/6] Checking image directories...")
    dir_status = check_image_dirs(report)
    present = sum(1 for v in dir_status.values() if v)
    print(f"      {present}/12 image directories found")
    for d, ok in dir_status.items():
        mark = "[OK]" if ok else "[MISSING]"
        print(f"        {mark} {d}")

    # Step 3: Load & analyse CSV
    df = None
    if csv_ok:
        print("[3/6] Loading and analysing CSV...")
        df = load_csv(report)
        if df is not None:
            analyse_csv(df, report)
            print(f"      {report['total_csv_records']:,} records, "
                  f"{report.get('unique_patients', '?')} patients, "
                  f"{report.get('unique_images_in_csv', '?')} unique images")
            print(f"      Pneumonia: {report.get('pneumonia_count', 'N/A')}  "
                  f"No Finding: {report.get('no_finding_count', 'N/A')}")
        else:
            print("      ERROR: CSV could not be read!")
    else:
        print("[3/6] Skipping CSV analysis -- CSV not found.")
        report.update({
            "csv_readable": False,
            "total_csv_records": None,
            "unique_patients": None,
            "unique_images_in_csv": None,
            "csv_columns": [],
        })

    # Step 4: Image files on disk
    if df is not None and any(dir_status.values()):
        print("[4/6] Checking image files on disk...")
        check_images_on_disk(df, report)
        print(f"      On disk: {report.get('total_image_files_on_disk', 0):,}  "
              f"Missing: {report.get('missing_image_count', 0):,}  "
              f"Corrupted: {report.get('corrupted_image_count', 0):,}")
    else:
        print("[4/6] Skipping image file check (no CSV or no image dirs).")
        report.update({
            "total_image_files_on_disk": 0,
            "missing_image_count": None,
            "corrupted_image_count": None,
        })

    # Step 5: PRD consistency
    print("[5/6] Running PRD consistency checks...")
    prd_consistency_check(report)
    overall = _overall_status(report)
    print(f"      Overall: {overall}")

    # Step 6: Write reports
    print("[6/6] Writing reports...")
    json_path = _OUTPUTS_DIR / "dataset_validation_report.json"
    txt_path  = _OUTPUTS_DIR / "dataset_validation_report.txt"
    write_json_report(report, json_path)
    write_txt_report(report, txt_path)
    print(f"      JSON -> {json_path}")
    print(f"      TXT  -> {txt_path}")

    # Terminal summary
    print_summary(report)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Validation cancelled by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[FATAL ERROR] {exc}")
        traceback.print_exc()
        sys.exit(2)
