"""
scripts/validate_splits.py
===========================
Phase 3 — Step 3: Validate the saved train / validation / test splits.

What this script does
---------------------
Runs all 7 integrity checks defined in src/cxr_reliability/data/validation.py:

    1. No patient overlap between any two splits.
    2. No duplicate image IDs across splits.
    3. All split records exist in metadata.csv.
    4. Every image_path resolves to a real file on disk.
    5. Every record has a valid binary label (0 or 1).
    6. All CSV files (metadata, train, validation, test) are readable.
    7. The split is reproducible (manifest hash verified).

Exit codes:
    0 — All checks PASS
    1 — One or more checks FAILED
    2 — Fatal error (e.g. missing input files)

Usage:
    python scripts/validate_splits.py

Configuration:
    CXR_DATASET_ROOT — path to the NIH dataset
    CXR_DATA_DIR     — location of data/processed/
"""

from __future__ import annotations

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
    _DATASET_ROOT = (
        _s.dataset_root if _s.dataset_root.is_absolute()
        else (_PROJECT_ROOT / _s.dataset_root).resolve()
    )
    _DATA_DIR = (
        _s.data_dir if _s.data_dir.is_absolute()
        else (_PROJECT_ROOT / _s.data_dir).resolve()
    )
except Exception as _e:
    print(f"[WARN] Could not load settings ({_e}). Using defaults.")
    _DATASET_ROOT = (_PROJECT_ROOT / "dataset").resolve()
    _DATA_DIR = (_PROJECT_ROOT / "data").resolve()

_PROCESSED_DIR = _DATA_DIR / "processed"


def main() -> None:
    from cxr_reliability.data.validation import validate_splits

    metadata_path = _PROCESSED_DIR / "metadata.csv"
    train_path = _PROCESSED_DIR / "train.csv"
    val_path = _PROCESSED_DIR / "validation.csv"
    test_path = _PROCESSED_DIR / "test.csv"

    print()
    print("=" * 70)
    print("  Phase 3 — Step 3: Validate Splits")
    print("=" * 70)
    print(f"  Dataset root : {_DATASET_ROOT}")
    print(f"  Processed dir: {_PROCESSED_DIR}")
    print()

    # Pre-flight: check files exist before running validation
    missing_files = []
    for p in [metadata_path, train_path, val_path, test_path]:
        if not p.exists():
            missing_files.append(str(p))
    if missing_files:
        print("[ERROR] Required files are missing. Run prepare_metadata.py and")
        print("        create_splits.py first.")
        for f in missing_files:
            print(f"        Missing: {f}")
        sys.exit(2)

    # Run validation
    print("  Running checks...")
    print()
    all_passed, results = validate_splits(
        metadata_path=metadata_path,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        dataset_root=_DATASET_ROOT,
    )

    # Print results
    for r in results:
        status = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"  {status}  Check {r['check']}")
        print(f"         {r['detail']}")
        print()

    # Summary
    print("=" * 70)
    n_pass = sum(1 for r in results if r["passed"])
    n_fail = sum(1 for r in results if not r["passed"])
    overall = "PASS" if all_passed else "FAIL"
    print(f"  Overall: {overall}  ({n_pass} passed, {n_fail} failed)")
    print("=" * 70)
    print()

    sys.exit(0 if all_passed else 1)


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
