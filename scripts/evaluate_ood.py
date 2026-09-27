"""Evaluate OOD Detector on NIH Validation Set — Phase 5.

Script: scripts/evaluate_ood.py

Responsibility
--------------
Report the OOD detector's behavior on the NIH validation set, which
serves as an in-distribution reference (NOT a true OOD evaluation).

IMPORTANT Scientific Labelling
--------------------------------
NIH ChestX-ray14 validation images are IN-DISTRIBUTION examples (same
dataset, same collection protocol as training data). They are NOT a true
OOD dataset. Therefore:
    - We do NOT report OOD AUROC from NIH validation alone.
    - We do NOT label NIH validation examples as "true OOD."
    - Results are labelled as "validation in-distribution reference behavior."
    - The percentage above threshold reflects the in-distribution tail, NOT
      the true OOD detection rate.

External OOD Evaluation (Optional)
------------------------------------
If an external OOD dataset (e.g., CheXpert, PadChest) is configured via
--external-ood-dir, it will be used for a genuine OOD evaluation.
Otherwise, the script reports:
    "Natural OOD evaluation not run: external OOD dataset not configured."

This script does NOT download any external datasets.

Usage
-----
    # Evaluate on NIH validation set (in-distribution reference behavior):
    python scripts/evaluate_ood.py

    # With external OOD dataset (if available):
    python scripts/evaluate_ood.py --external-ood-dir /path/to/chexpert

    # Custom stats directory:
    python scripts/evaluate_ood.py --stats-dir artifacts/ood

    # Smoke test (small subset):
    python scripts/evaluate_ood.py --max-samples 20 --smoke-test

Implementation phase: P5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# ── Make src/ importable when run directly ───────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_split_image_ids(data_dir: Path, split: str) -> list[str]:
    """Load image IDs from manifest (JSON or CSV)."""
    manifest_json = data_dir / "splits.json"
    if manifest_json.exists():
        with open(manifest_json, encoding="utf-8") as fh:
            manifest = json.load(fh)
        if split in manifest:
            return manifest[split]
        raise KeyError(f"Split '{split}' not in {manifest_json}. Keys: {list(manifest.keys())}")

    split_csv = data_dir / f"{split}.csv"
    if split_csv.exists():
        import pandas as pd
        df = pd.read_csv(split_csv)
        if "image_id" not in df.columns:
            raise ValueError(f"CSV {split_csv} has no 'image_id' column.")
        return df["image_id"].astype(str).tolist()

    raise FileNotFoundError(
        f"No manifest found for split '{split}'. "
        f"Searched: {manifest_json}, {split_csv}"
    )


def _find_image_path(image_id: str, dataset_root: Path) -> Path | None:
    for i in range(1, 13):
        for sub in ["images", ""]:
            if sub:
                candidate = dataset_root / f"images_{i:03d}" / sub / image_id
            else:
                candidate = dataset_root / f"images_{i:03d}" / image_id
            if candidate.exists():
                return candidate
    return None


def _plot_distance_histogram(
    distances: np.ndarray,
    threshold_borderline: float | None,
    threshold_severe: float | None,
    output_path: Path,
    title: str = "Mahalanobis Distance — NIH Validation (In-Distribution Reference)",
) -> None:
    """Save a histogram of distances with threshold markers."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(distances, bins=50, alpha=0.75, color="#2196F3", edgecolor="white",
                label=f"NIH Validation (N={len(distances)})")

        if threshold_borderline is not None:
            ax.axvline(threshold_borderline, color="#FF5722", linewidth=2, linestyle="--",
                       label=f"Borderline threshold: {threshold_borderline:.2f}")
        if threshold_severe is not None:
            ax.axvline(threshold_severe, color="#D32F2F", linewidth=2, linestyle=":",
                       label=f"Severe threshold: {threshold_severe:.2f}")

        ax.set_xlabel("Mahalanobis Distance", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)

        # Scientific note as figure text
        fig.text(
            0.02, 0.02,
            "NOTE: NIH validation images are IN-DISTRIBUTION. "
            "This is NOT a true OOD evaluation.",
            fontsize=8, color="gray", style="italic",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved histogram to %s", output_path)
    except ImportError:
        logger.warning(
            "matplotlib not available — skipping histogram plot. "
            "Install with: pip install matplotlib"
        )
    except Exception as exc:
        logger.warning("Could not generate histogram: %s", exc)


def _evaluate_split(
    image_ids: list[str],
    dataset_root: Path,
    model_agent,
    detector,
    max_samples: int | None,
    log_every: int,
    split_label: str,
) -> np.ndarray:
    """
    Extract features and compute Mahalanobis distances for a split.

    Returns array of distances.
    """
    from cxr_reliability.models.preprocessing import load_image_for_txv
    from cxr_reliability.ood.mahalanobis import mahalanobis_distance

    if max_samples is not None:
        image_ids = image_ids[:max_samples]

    distances = []
    errors = []
    n_total = len(image_ids)

    logger.info("Evaluating %d images from split '%s'...", n_total, split_label)

    for i, image_id in enumerate(image_ids):
        if i > 0 and i % log_every == 0:
            logger.info("  %d/%d processed...", i, n_total)

        img_path = _find_image_path(image_id, dataset_root)
        if img_path is None:
            errors.append(image_id)
            continue

        try:
            tensor = load_image_for_txv(img_path)
            forward = model_agent.run(tensor)
            dist = mahalanobis_distance(forward.features, detector.stats)
            distances.append(dist)
        except Exception as exc:
            errors.append(f"{image_id}: {exc}")
            continue

    if errors:
        logger.warning("%d images failed (first 3): %s", len(errors), errors[:3])

    return np.array(distances, dtype=np.float64)


def _report_distance_stats(distances: np.ndarray, split_label: str, is_ood: bool) -> None:
    """Print distance statistics with appropriate scientific labels."""

    if is_ood:
        header = f"\n=== External OOD Evaluation: {split_label} ==="
        note = "(These are EXTERNAL OOD examples — true OOD evaluation)"
    else:
        header = f"\n=== In-Distribution Reference Evaluation: {split_label} ==="
        note = "(These are IN-DISTRIBUTION examples — NOT true OOD evaluation)"

    logger.info(header)
    logger.info(note)
    logger.info("  N samples   : %d", len(distances))
    logger.info("  Min         : %.4f", float(np.min(distances)))
    logger.info("  Max         : %.4f", float(np.max(distances)))
    logger.info("  Mean        : %.4f", float(np.mean(distances)))
    logger.info("  Median      : %.4f", float(np.median(distances)))
    logger.info("  Std dev     : %.4f", float(np.std(distances)))

    for pct in [90, 95, 97.5, 99, 99.9]:
        logger.info("  %5.1f-pctile: %.4f", pct, float(np.percentile(distances, pct)))


def main(args: argparse.Namespace) -> int:
    _setup_logging(verbose=args.verbose)

    logger.info("=" * 65)
    logger.info("OOD Detector Evaluation — Phase 5")
    logger.info("=" * 65)

    # ── Load stats ────────────────────────────────────────────────────────
    stats_dir = Path(args.stats_dir)
    if not stats_dir.exists() or not (stats_dir / "reference_stats.npz").exists():
        logger.error(
            "OOD reference statistics not found at: %s\n"
            "Run scripts/fit_ood_reference.py first.",
            stats_dir,
        )
        return 1

    from cxr_reliability.ood.detector import OODConfig, OODDetector
    from cxr_reliability.ood.statistics import load_stats

    stats = load_stats(stats_dir)
    logger.info(
        "Loaded OOD stats: N_ref=%d, D=%d, model=%s",
        stats.n_samples, stats.feature_dim, stats.model_id,
    )

    # Load thresholds from metadata
    meta_path = stats_dir / "metadata.json"
    borderline: float | None = None
    severe: float | None = None
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        borderline = meta.get("mahalanobis_borderline")
        severe = meta.get("mahalanobis_severe")

    if borderline is None:
        logger.warning(
            "mahalanobis_borderline not found in metadata. "
            "Run fit_ood_reference.py with a validation set to calibrate."
        )

    config = OODConfig(
        mahalanobis_borderline=borderline,
        mahalanobis_severe=severe,
    )
    detector = OODDetector(config=config)
    detector.load_stats(stats_dir)

    # ── Load model ─────────────────────────────────────────────────────────
    logger.info("\n--- Loading Base Model ---")
    from cxr_reliability.models.model_factory import get_model
    agent = get_model(
        model_id="densenet121-res224-nih",
        device=args.device,
        load=True,
    )

    data_dir = Path(args.data_dir)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"

    max_samples = args.max_samples
    log_every = args.log_every

    # ── Evaluate NIH validation (in-distribution reference) ───────────────
    logger.info("\n--- Evaluating NIH Validation Set (In-Distribution Reference) ---")
    logger.info("SCIENTIFIC NOTE: NIH validation = in-distribution NIH data.")
    logger.info("This is NOT a true OOD evaluation.")

    try:
        val_ids = _load_split_image_ids(data_dir, "validation")
    except Exception as exc:
        logger.error("Could not load validation manifest: %s", exc)
        return 1

    val_distances = _evaluate_split(
        image_ids=val_ids,
        dataset_root=dataset_root,
        model_agent=agent,
        detector=detector,
        max_samples=max_samples,
        log_every=log_every,
        split_label="NIH Validation (In-Distribution)",
    )

    if len(val_distances) == 0:
        logger.error("No distances computed. Check dataset paths.")
        return 1

    _report_distance_stats(val_distances, "NIH Validation (In-Distribution)", is_ood=False)

    # Flagging statistics
    if borderline is not None:
        n_above = int(np.sum(val_distances > borderline))
        pct_above = 100.0 * n_above / len(val_distances)
        logger.info("\n--- Flagging Statistics (In-Distribution Reference) ---")
        logger.info("  Borderline threshold: %.4f", borderline)
        logger.info("  Images above threshold: %d / %d (%.2f%%)",
                    n_above, len(val_distances), pct_above)
        logger.info(
            "  INTERPRETATION: ~%.1f%% of IN-DISTRIBUTION images are flagged "
            "(expected ~%.1f%% for a %.1f-pctile threshold).",
            pct_above, 100.0 - (meta.get("threshold_percentile", 99.0) if meta_path.exists() else 99.0),
            meta.get("threshold_percentile", 99.0) if meta_path.exists() else 99.0,
        )

    # ── Histogram ─────────────────────────────────────────────────────────
    if not args.no_plot:
        _plot_distance_histogram(
            distances=val_distances,
            threshold_borderline=borderline,
            threshold_severe=severe,
            output_path=plots_dir / "val_distance_histogram.png",
            title="Mahalanobis Distance — NIH Validation (In-Distribution Reference)",
        )

    # ── External OOD evaluation ────────────────────────────────────────────
    if args.external_ood_dir:
        ext_dir = Path(args.external_ood_dir)
        if not ext_dir.exists():
            logger.warning(
                "External OOD directory not found: %s\n"
                "Natural OOD evaluation not run.", ext_dir
            )
        else:
            logger.info("\n--- External OOD Evaluation (Optional) ---")
            logger.info("External OOD dir: %s", ext_dir)
            logger.info(
                "NOTE: Full CheXpert/PadChest adapter not yet implemented. "
                "This feature will be added in a future phase when the external "
                "dataset structure has been inspected."
            )
            logger.info(
                "When implemented, this section will report:\n"
                "  - ID vs OOD distance distributions\n"
                "  - ROC curve\n"
                "  - AUROC\n"
                "  - AUPR\n"
                "using genuine OOD positive/negative labels."
            )
    else:
        logger.info(
            "\nNatural OOD evaluation not run: "
            "external OOD dataset not configured. "
            "Use --external-ood-dir to provide a CheXpert or PadChest path."
        )

    logger.info("\n" + "=" * 65)
    logger.info("Evaluation complete.")
    if not args.no_plot:
        logger.info("Plots saved to: %s", plots_dir)
    logger.info("=" * 65)

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OOD detector on NIH validation set.",
        epilog=__doc__,
    )
    parser.add_argument(
        "--stats-dir", default="artifacts/ood",
        help="Directory with reference_stats.npz. Default: artifacts/ood",
    )
    parser.add_argument(
        "--data-dir", default="data/manifests",
        help="Split manifest directory. Default: data/manifests",
    )
    parser.add_argument(
        "--dataset-root", default="dataset",
        help="NIH dataset root. Default: dataset",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/ood",
        help="Output for plots. Default: artifacts/ood",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Inference device. Default: cpu",
    )
    parser.add_argument(
        "--external-ood-dir", default=None,
        help="Optional: path to CheXpert or PadChest for natural OOD eval.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max images per split (for smoke tests).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick smoke test with 20 images.",
    )
    parser.add_argument(
        "--log-every", type=int, default=200,
        help="Log progress every N images. Default: 200",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip histogram plots.")
    parser.add_argument("--verbose", action="store_true", help="Debug logging.")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.smoke_test and args.max_samples is None:
        args.max_samples = 20
    sys.exit(main(args))
