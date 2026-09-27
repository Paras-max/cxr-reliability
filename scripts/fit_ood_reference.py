"""Fit OOD Reference Distribution — Phase 5.

Script: scripts/fit_ood_reference.py

Responsibility
--------------
1. Load the training split manifest (image_ids only).
2. Load the existing Base Model (densenet121-res224-nih) via model_factory.
3. Extract 1024-dim feature vectors for each training image.
4. Compute and validate reference statistics (mean + regularized covariance).
5. Select a provisional threshold from validation distances.
6. Save OOD artifacts to artifacts/ood/.

IMPORTANT Scientific Constraints
---------------------------------
- ONLY training images are used to fit reference statistics.
- ONLY validation images are used to select thresholds.
- TEST images are NEVER loaded during this script.
- NIH validation images are in-distribution NIH data; they are NOT used
  as a true OOD evaluation set.
- No accuracy metrics are fabricated.

Saved Artifacts
---------------
    artifacts/ood/
        reference_stats.npz    — mean, covariance, Cholesky factor
        metadata.json          — provenance, thresholds, timestamps

Usage
-----
    # Full training run (78k images — may take hours on CPU):
    python scripts/fit_ood_reference.py

    # Smoke test with a small subset:
    python scripts/fit_ood_reference.py --max-samples 10 --smoke-test

    # Custom paths:
    python scripts/fit_ood_reference.py \\
        --data-dir data/manifests \\
        --dataset-root dataset \\
        --output-dir artifacts/ood \\
        --device cpu \\
        --lambda-reg 1e-5 \\
        --threshold-percentile 99.0

    # On GPU:
    python scripts/fit_ood_reference.py --device cuda

Implementation phase: P5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Make src/ importable when run directly ───────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cxr_reliability.calibration.ood_fit import calibrate_ood_thresholds, fit_gaussian_stats
from cxr_reliability.models.model_factory import get_model
from cxr_reliability.models.preprocessing import load_image_for_txv
from cxr_reliability.ood.detector import OODConfig, OODDetector

logger = logging.getLogger(__name__)


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── Manifest loading ──────────────────────────────────────────────────────────

def select_deterministic_subset(
    image_ids: list[str],
    n_samples: int | None,
    seed: int = 42,
) -> list[str]:
    """
    Select a deterministic subset of image IDs using a fixed random seed.

    Parameters
    ----------
    image_ids : list of image identifiers from a split
    n_samples : target subset size (if None or >= len, returns sorted/original copy)
    seed      : integer random seed for reproducibility

    Returns
    -------
    list of image identifiers (subset of original, preserving unique IDs)
    """
    if n_samples is None or n_samples >= len(image_ids):
        return list(image_ids)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(image_ids), size=n_samples, replace=False)
    indices.sort()
    return [image_ids[i] for i in indices]


def _load_split_image_ids(data_dir: Path, split: str) -> list[str]:
    """
    Load image IDs from a split manifest.

    Supports two manifest formats:
    1. JSON:  {split_name: [image_id, ...], ...}  (from scripts/create_splits.py)
    2. CSV:   single column 'image_id'

    Returns a list of image_id strings for the requested split.
    """
    if split.lower() == "test":
        raise PermissionError(
            "SCIENTIFIC CONSTRAINT VIOLATION: The 'test' split must NEVER be loaded "
            "for OOD reference distribution fitting or threshold calibration."
        )

    # Try JSON manifest (primary format)
    manifest_json = data_dir / "splits.json"
    if manifest_json.exists():
        with open(manifest_json, encoding="utf-8") as fh:
            manifest = json.load(fh)
        if split in manifest:
            ids = manifest[split]
            logger.info("Loaded %d image IDs for split '%s' from %s", len(ids), split, manifest_json)
            return ids
        else:
            available = list(manifest.keys())
            raise KeyError(
                f"Split '{split}' not found in manifest {manifest_json}. "
                f"Available splits: {available}"
            )

    # Try per-split CSV in data_dir or data_dir/processed
    split_csv = data_dir / f"{split}.csv"
    if not split_csv.exists():
        fallback_csv = data_dir / "processed" / f"{split}.csv"
        if fallback_csv.exists():
            split_csv = fallback_csv

    if split_csv.exists():
        import pandas as pd
        df = pd.read_csv(split_csv)
        if "image_id" not in df.columns:
            raise ValueError(f"CSV {split_csv} has no 'image_id' column. Found: {list(df.columns)}")
        ids = df["image_id"].astype(str).tolist()
        logger.info("Loaded %d image IDs for split '%s' from %s", len(ids), split, split_csv)
        return ids

    raise FileNotFoundError(
        f"No manifest found for split '{split}'.\n"
        f"Searched:\n"
        f"  {manifest_json}\n"
        f"  {split_csv}\n"
        f"Run scripts/create_splits.py first to generate manifests."
    )


def _find_image_path(image_id: str, dataset_root: Path) -> Path | None:
    """
    Locate an image file within the NIH dataset directory structure.
    NIH stores images under images_NNN/images/*.png
    """
    for i in range(1, 13):
        candidate = dataset_root / f"images_{i:03d}" / "images" / image_id
        if candidate.exists():
            return candidate
        candidate_flat = dataset_root / f"images_{i:03d}" / image_id
        if candidate_flat.exists():
            return candidate_flat
    return None


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(
    image_ids: list[str],
    dataset_root: Path,
    model_agent,
    max_samples: int | None = None,
    log_every: int = 500,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract 1024-dim feature vectors for a list of image IDs.

    Parameters
    ----------
    image_ids    : list of image filenames (e.g. '00001234_000.png')
    dataset_root : root directory of the NIH dataset
    model_agent  : loaded BaseModelAgent
    max_samples  : if set, only process the first N images (for smoke tests)
    log_every    : log progress every N images

    Returns
    -------
    (feature_matrix, processed_ids)
        feature_matrix  : np.ndarray (N, 1024)
        processed_ids   : list of image IDs successfully processed
    """
    if max_samples is not None:
        image_ids = image_ids[:max_samples]
        logger.info("Limiting to %d images (--max-samples set)", max_samples)

    features_list: list[np.ndarray] = []
    processed_ids: list[str] = []
    errors: list[str] = []

    n_total = len(image_ids)
    t0 = time.perf_counter()

    logger.info("Extracting features from %d images...", n_total)

    for i, image_id in enumerate(image_ids):
        if i > 0 and i % log_every == 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed
            eta = (n_total - i) / rate if rate > 0 else float("inf")
            logger.info(
                "  Progress: %d/%d (%.1f%%) | %.1f img/s | ETA: %.0fs",
                i, n_total, 100 * i / n_total, rate, eta,
            )

        img_path = _find_image_path(image_id, dataset_root)
        if img_path is None:
            errors.append(f"Image not found: {image_id}")
            if len(errors) <= 5:
                logger.warning("Image not found: %s (skipping)", image_id)
            continue

        try:
            tensor = load_image_for_txv(img_path)
            forward = model_agent.run(tensor)
            feat = forward.features.numpy()  # shape (1024,)
            features_list.append(feat)
            processed_ids.append(image_id)
        except Exception as exc:
            errors.append(f"{image_id}: {exc}")
            if len(errors) <= 5:
                logger.warning("Failed to process %s: %s", image_id, exc)
            continue

    total_elapsed = time.perf_counter() - t0
    n_processed = len(processed_ids)

    if errors:
        logger.warning(
            "%d images could not be processed (showing first 5):",
            len(errors),
        )
        for e in errors[:5]:
            logger.warning("  %s", e)
        if len(errors) > 5:
            logger.warning("  ... and %d more errors", len(errors) - 5)

    logger.info(
        "Feature extraction complete: %d/%d images processed in %.1fs (%.1f img/s)",
        n_processed, n_total, total_elapsed,
        n_processed / total_elapsed if total_elapsed > 0 else 0,
    )

    if n_processed == 0:
        raise RuntimeError(
            "No features extracted. Check that the dataset root is correct and "
            "that image files exist at the expected paths."
        )

    feature_matrix = np.stack(features_list, axis=0)  # (N, 1024)

    # Final validation
    if not np.all(np.isfinite(feature_matrix)):
        n_bad = int(np.sum(~np.isfinite(feature_matrix)))
        raise RuntimeError(
            f"Feature matrix contains {n_bad} NaN/Inf values. "
            "This should not occur with valid images and a correct model."
        )

    return feature_matrix, processed_ids


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> int:
    _setup_logging(verbose=args.verbose)

    logger.info("=" * 60)
    logger.info("OOD Reference Distribution Fitting — Phase 5")
    logger.info("=" * 60)
    logger.info("SCIENTIFIC CONSTRAINT: Using TRAINING split only for fitting.")
    logger.info("TEST data is never used in this script.")
    logger.info("=" * 60)

    # ── Paths ─────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    dataset_root = Path(args.dataset_root)

    # Separate development output directory by default
    if args.dev and args.output_dir == "artifacts/ood":
        output_dir = Path("artifacts/ood_dev")
    else:
        output_dir = Path(args.output_dir)
    val_output_dir = output_dir  # validation stats go in same dir

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Data directory      : %s", data_dir)
    logger.info("Dataset root        : %s", dataset_root)
    logger.info("Output directory    : %s", output_dir)
    logger.info("Mode                : %s", "DEVELOPMENT (Provisional Subset)" if args.dev else "FULL RESEARCH")
    logger.info("Device              : %s", args.device)
    logger.info("Lambda (λ_reg)      : %.2e", args.lambda_reg)
    logger.info("Threshold percentile: %.1f%%", args.threshold_percentile)

    if args.smoke_test:
        logger.info("SMOKE TEST MODE: limiting to %d images per split", args.max_samples or 10)
    elif args.dev:
        logger.info(
            "DEVELOPMENT MODE: Target train=%d, Target val=%d, Seed=%d",
            args.dev_train_samples, args.dev_val_samples, args.seed,
        )

    # ── Load training manifest ────────────────────────────────────────────
    logger.info("\n--- Loading training manifest ---")
    train_ids = _load_split_image_ids(data_dir, "train")
    logger.info("Total available training images: %d", len(train_ids))

    if args.dev:
        logger.info(
            "Selecting deterministic training subset of %d images (seed=%d)...",
            args.dev_train_samples, args.seed,
        )
        train_ids_to_process = select_deterministic_subset(train_ids, args.dev_train_samples, seed=args.seed)
        logger.info("Training subset selected: %d images", len(train_ids_to_process))
    else:
        train_ids_to_process = train_ids

    # ── Load validation manifest (for threshold selection) ────────────────
    logger.info("\n--- Loading validation manifest ---")
    try:
        val_ids = _load_split_image_ids(data_dir, "validation")
        logger.info("Total available validation images: %d", len(val_ids))
        if args.dev and val_ids:
            logger.info(
                "Selecting deterministic validation subset of %d images (seed=%d)...",
                args.dev_val_samples, args.seed + 1,
            )
            val_ids_to_process = select_deterministic_subset(val_ids, args.dev_val_samples, seed=args.seed + 1)
            logger.info("Validation subset selected: %d images", len(val_ids_to_process))
        else:
            val_ids_to_process = val_ids
    except (FileNotFoundError, KeyError) as exc:
        logger.warning("Validation manifest not found (%s). Threshold will not be calibrated.", exc)
        val_ids_to_process = []

    # ── Load Base Model ───────────────────────────────────────────────────
    logger.info("\n--- Loading Base Model ---")
    logger.info("Model: densenet121-res224-nih")
    agent = get_model(
        model_id="densenet121-res224-nih",
        device=args.device,
        load=True,
    )
    logger.info("Model loaded successfully on device: %s", args.device)

    # ── Extract training features ─────────────────────────────────────────
    logger.info("\n--- Extracting Training Features ---")
    logger.info("CONSTRAINT: Only TRAINING images used for fitting.")
    max_train = args.max_samples if (args.smoke_test or args.max_samples) else None

    train_features, train_processed = extract_features(
        image_ids=train_ids_to_process,
        dataset_root=dataset_root,
        model_agent=agent,
        max_samples=max_train,
        log_every=args.log_every,
    )
    logger.info("Training feature matrix shape: %s", train_features.shape)
    logger.info("Feature dimension: %d", train_features.shape[1])
    logger.info("Training samples processed: %d", len(train_processed))

    # ── Fit reference statistics ──────────────────────────────────────────
    logger.info("\n--- Fitting Reference Statistics ---")
    logger.info(
        "Fitting Gaussian (μ, Σ) on %d training samples...", len(train_processed)
    )

    extra_metadata = {
        "is_development": bool(args.dev),
        "mode": "development" if args.dev else "full_research",
        "status": "provisional_development" if args.dev else "final_research",
        "n_train_samples": len(train_processed),
        "n_train_images_total_available": len(train_ids),
        "n_train_images_processed": len(train_processed),
        "n_val_images_processed": 0,
        "sampling_seed": args.seed if args.dev else None,
        "dev_train_target": args.dev_train_samples if args.dev else None,
        "dev_val_target": args.dev_val_samples if args.dev else None,
        "device": args.device,
        "args": vars(args),
        "script": "scripts/fit_ood_reference.py",
        "disclaimer": (
            "DEVELOPMENT / PROVISIONAL reference statistics fitted on a subset of the training split. "
            "NOT for clinical use, NOT validated, and NOT for final research evaluation."
            if args.dev
            else "Reference distribution fitted on full training split. Research prototype only."
        ),
    }

    stats = fit_gaussian_stats(
        features=train_features,
        output_path=output_dir,
        lambda_reg=args.lambda_reg,
        model_id="densenet121-res224-nih",
        split="train",
        extra_metadata=extra_metadata,
    )

    logger.info("Reference statistics saved to: %s", output_dir)
    logger.info("  feature_dim : %d", stats.feature_dim)
    logger.info("  n_samples   : %d", stats.n_samples)
    logger.info("  lambda_reg  : %.2e", stats.lambda_reg)

    # ── Extract validation features and select threshold ──────────────────
    if val_ids_to_process:
        logger.info("\n--- Extracting Validation Features (for threshold selection) ---")
        logger.info("CONSTRAINT: Validation distances used only for threshold selection.")
        logger.info("These are IN-DISTRIBUTION NIH images — NOT true OOD data.")

        max_val = args.max_samples if (args.smoke_test or args.max_samples) else None

        val_features, val_processed = extract_features(
            image_ids=val_ids_to_process,
            dataset_root=dataset_root,
            model_agent=agent,
            max_samples=max_val,
            log_every=args.log_every,
        )

        # Compute validation distances using the fitted stats
        detector = OODDetector(config=OODConfig(
            covariance_regularization=args.lambda_reg,
            threshold_percentile=args.threshold_percentile,
        ))
        detector.fit(train_features, model_id="densenet121-res224-nih", split="train")
        val_distances = detector.batch_score(val_features)

        # Report validation distance statistics
        logger.info("\n--- Validation Distance Statistics (In-Distribution Reference) ---")
        logger.info("NOTE: NIH validation = in-distribution; these are NOT OOD examples.")
        logger.info("  N validation samples    : %d", len(val_distances))
        logger.info("  Min distance            : %.4f", float(np.min(val_distances)))
        logger.info("  Max distance            : %.4f", float(np.max(val_distances)))
        logger.info("  Mean distance           : %.4f", float(np.mean(val_distances)))
        logger.info("  Median distance         : %.4f", float(np.median(val_distances)))
        logger.info("  Std deviation           : %.4f", float(np.std(val_distances)))

        for pct in [90, 95, 97.5, 99, 99.9]:
            logger.info(
                "  %5.1f-th percentile     : %.4f",
                pct, float(np.percentile(val_distances, pct))
            )

        # Select threshold
        thresholds = calibrate_ood_thresholds(
            id_val_scores=val_distances,
            percentile=args.threshold_percentile,
        )

        borderline = thresholds.mahalanobis_borderline
        severe = thresholds.mahalanobis_severe
        n_flagged = int(np.sum(val_distances > borderline))
        pct_flagged = 100.0 * n_flagged / len(val_distances)

        logger.info("\n--- Provisional Thresholds (From Validation In-Distribution Tail) ---")
        logger.info("  Threshold method        : percentile")
        logger.info("  Threshold percentile    : %.1f%%", args.threshold_percentile)
        logger.info("  Mahalanobis borderline  : %.4f", borderline)
        logger.info("  Mahalanobis severe      : %.4f", severe)
        logger.info("  Val images above borderline: %d / %d (%.2f%%)",
                    n_flagged, len(val_distances), pct_flagged)
        logger.info("  (For a perfect calibration, this should equal %.1f%%)",
                    100.0 - args.threshold_percentile)

        # Update metadata with threshold info
        meta_path = output_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as fh:
                metadata = json.load(fh)
        else:
            metadata = {}

        metadata.update({
            "threshold_method": "percentile",
            "threshold_percentile": args.threshold_percentile,
            "mahalanobis_borderline": borderline,
            "mahalanobis_severe": severe,
            "val_n_samples": len(val_distances),
            "val_min": float(np.min(val_distances)),
            "val_max": float(np.max(val_distances)),
            "val_mean": float(np.mean(val_distances)),
            "val_median": float(np.median(val_distances)),
            "val_std": float(np.std(val_distances)),
            "val_n_above_borderline": n_flagged,
            "val_pct_above_borderline": round(pct_flagged, 4),
            "n_val_images_processed": len(val_processed) if val_ids_to_process else 0,
            "threshold_provenance": (
                f"Percentile threshold selected from {len(val_distances)} "
                f"NIH validation images (IN-DISTRIBUTION{' - PROVISIONAL DEV SUBSET' if args.dev else ''}). "
                f"These are NOT true OOD examples."
            ),
        })

        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("Updated metadata.json with threshold information.")

    else:
        logger.warning(
            "No validation manifest available. "
            "Thresholds were NOT calibrated. "
            "Set mahalanobis_borderline manually in configs/thresholds/."
        )

    # ── Verify saved artifacts ────────────────────────────────────────────
    logger.info("\n--- Artifact Verification ---")
    stats_file = output_dir / "reference_stats.npz"
    meta_file = output_dir / "metadata.json"

    if stats_file.exists():
        size_mb = stats_file.stat().st_size / 1_048_576
        logger.info("  reference_stats.npz : %s (%.2f MB)", stats_file, size_mb)
    else:
        logger.error("  reference_stats.npz : NOT FOUND — fitting failed!")
        return 1

    if meta_file.exists():
        logger.info("  metadata.json       : %s", meta_file)
    else:
        logger.warning("  metadata.json       : NOT FOUND")

    logger.info("\n--- Smoke Test Verification ---")
    # Quick round-trip test: load stats and score one feature
    from cxr_reliability.ood.statistics import load_stats as _load
    loaded = _load(output_dir)
    loaded.validate()
    logger.info("  Round-trip load: OK (N=%d, D=%d)", loaded.n_samples, loaded.feature_dim)

    # Score one random vector
    rng = np.random.default_rng(42)
    dummy_feat = rng.standard_normal(stats.feature_dim)
    from cxr_reliability.ood.mahalanobis import mahalanobis_distance as _maha
    test_dist = _maha(dummy_feat, loaded)
    logger.info("  Dummy feature distance: %.4f (sanity check)", test_dist)

    logger.info("\n" + "=" * 60)
    logger.info("OOD fitting complete.")
    logger.info("Artifacts saved to: %s", output_dir)
    logger.info("=" * 60)

    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit OOD reference distribution from training features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-dir",
        default="data/manifests",
        help="Directory containing split manifest (splits.json or train.csv). "
             "Default: data/manifests",
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Root directory of the NIH ChestX-ray14 dataset. "
             "Default: dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/ood",
        help="Directory for output artifacts. Default: artifacts/ood "
             "(defaults to artifacts/ood_dev when --dev is set)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in DEVELOPMENT mode using a deterministic subset of the training split. "
             "Default size: 2000 images. Default output: artifacts/ood_dev.",
    )
    parser.add_argument(
        "--dev-samples",
        "--dev-train-samples",
        type=int,
        default=2000,
        dest="dev_train_samples",
        help="Number of training images to sample in development mode. Default: 2000.",
    )
    parser.add_argument(
        "--dev-val-samples",
        type=int,
        default=500,
        help="Number of validation images to sample for provisional thresholding in dev mode. "
             "Default: 500.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic subset selection. Default: 42.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for model inference. Default: cpu",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=1e-5,
        help="Tikhonov regularization constant λ. Default: 1e-5",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.0,
        help="Percentile of validation distances for borderline threshold. "
             "Default: 99.0 (conservative)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of images per split (for smoke tests). "
             "Default: None (all images)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick smoke test with a small number of images (10). "
             "Equivalent to --max-samples 10.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Log progress every N images. Default: 500",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.smoke_test and args.max_samples is None:
        args.max_samples = 10
    sys.exit(main(args))
