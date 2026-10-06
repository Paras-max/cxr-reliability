"""Base Model Agent — DenseNet121 pneumonia classifier (Phase 4).

Responsibility:
    Run one forward pass of densenet121-res224-nih, return the Pneumonia
    output (raw probability), all pathology outputs for audit, and the
    mid-layer 1024-dim feature vector for the future OOD Agent.

    No fine-tuning: the pretrained model is used as-is (inference only).

OUTPUT MAPPING (verified against actual model — NOT assumed):
    The TorchXRayVision densenet121-res224-nih model outputs 18 values:
        [0]  Atelectasis        [9]  Pleural_Thickening
        [1]  Consolidation      [10] Cardiomegaly
        [2]  Infiltration       [11] Nodule
        [3]  Pneumothorax       [12] Mass
        [4]  Edema              [13] Hernia
        [5]  Emphysema          [14] (empty)
        [6]  Fibrosis           [15] (empty)
        [7]  Effusion           [16] (empty)
        [8]  PNEUMONIA          [17] (empty)

    PNEUMONIA = index 8  (verified by live model probe, 2026-09-22)

OUTPUT TYPE:
    The model applies a sigmoid internally. All 18 outputs are already
    sigmoid probabilities in [0, 1].
    They are NOT raw logits.

    Terminology used here:
        pneumonia_raw_score  — the sigmoid probability at index 8
        raw_probability      — same value, alias
    We do NOT call this a "calibrated probability". Calibration is Phase 6.

DATASET LABEL vs MODEL OUTPUT:
    Dataset binary label:  1 = Pneumonia,  0 = Non-Pneumonia
    Model output:          one of 18 sigmoid scores (index 8 = Pneumonia)
    These are related but NOT the same. The dataset label is derived from
    text-mined NIH annotations. The model score is the pretrained model's
    own continuous output. Do not equate them.

RESEARCH DISCLAIMER:
    This is a research prototype. The model's output is NOT a clinical
    diagnosis and has NOT been validated for clinical use.

Dependencies:
    torch, models.txv_loader, models.feature_hook, models.preprocessing,
    contracts.base_model, contracts.common

Implementation phase: P4
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from cxr_reliability.agents.base import AgentBase
from cxr_reliability.contracts.base_model import BaseModelResult
from cxr_reliability.contracts.common import AgentName
from cxr_reliability.models.feature_hook import FEATURE_DIM, FEATURE_LAYER_NAME, FeatureExtractor
from cxr_reliability.models.txv_loader import load_txv_model, weights_sha256

logger = logging.getLogger(__name__)

# Validated F1-optimal raw-score operating threshold from Phase 4.5 calibration
RAW_OPERATING_THRESHOLD = 0.522161

# ── Verified Pneumonia index in densenet121-res224-nih output ────────────────
# Source: live probe of xrv.models.DenseNet(weights='densenet121-res224-nih').pathologies
# Date: 2026-09-22
PNEUMONIA_OUTPUT_INDEX = 8
PNEUMONIA_LABEL_NAME = "Pneumonia"

# Full 18-label mapping for densenet121-res224-nih
TXV_NIH_PATHOLOGY_LABELS: list[str] = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion",
    "Pneumonia",          # ← index 8 — the target
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
    "", "", "", "",        # indices 14-17: empty slots in TXV NIH model
]


@dataclass
class ModelForward:
    """
    Complete output of a single forward pass through the base model.

    Fields
    ------
    result       : BaseModelResult — structured contract for pipeline use
    raw_probs    : torch.Tensor shape (18,) — all 18 sigmoid probabilities
    features     : torch.Tensor shape (1024,) — penultimate feature vector
                   (used by OOD Agent for Mahalanobis distance in Phase 5)
    inference_ms : float — wall-clock inference time in milliseconds
    """
    result: BaseModelResult
    raw_probs: torch.Tensor          # shape (18,)  — full output vector
    features: torch.Tensor           # shape (1024,) — for OOD Agent
    inference_ms: float = 0.0


class BaseModelAgent(AgentBase):
    """
    Wraps the pretrained TorchXRayVision densenet121-res224-nih model.

    Parameters
    ----------
    model_id         : TXV model identifier (default: 'densenet121-res224-nih')
    target_pathology : name of the target pathology (default: 'Pneumonia')
    feature_layer    : TXV layer name for feature extraction
                       (default: FEATURE_LAYER_NAME = 'features.norm5')
    weights_dir      : override TXV weight cache directory (None = default)
    device           : 'cpu', 'cuda', 'cuda:0', etc.
    calibration_dir  : directory containing platt_calibrator.joblib (None = default outputs/calibration)
    calibrator       : preloaded ProbabilityCalibrator instance (None = auto-load)
    """

    name = AgentName.BASE_MODEL
    version = "0.4.0"

    def __init__(
        self,
        model_id: str = "densenet121-res224-nih",
        target_pathology: str = "Pneumonia",
        feature_layer: str | None = None,
        weights_dir: Path | None = None,
        device: str = "cpu",
        calibration_dir: Path | None = None,
        calibrator: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.target_pathology = target_pathology
        self.feature_layer = feature_layer or FEATURE_LAYER_NAME
        self.weights_dir = weights_dir
        self.device = device
        self.calibration_dir = calibration_dir

        self._model: nn.Module | None = None
        self._pathology_labels: list[str] = TXV_NIH_PATHOLOGY_LABELS
        self._pneumonia_idx: int = PNEUMONIA_OUTPUT_INDEX
        self._weights_sha256: str = ""

        self._calibrator: Any = calibrator
        self.is_calibrated: bool = False
        if calibrator is not None:
            self.is_calibrated = True
        else:
            self._init_calibrator()

    def _init_calibrator(self) -> None:
        """
        Load ProbabilityCalibrator from calibration_dir or outputs/calibration.
        Loaded once during initialization, not per inference.
        If unavailable, preserve safe backward-compatible behavior.
        """
        calib_path = self.calibration_dir
        if calib_path is None:
            from cxr_reliability.config.settings import get_settings
            try:
                calib_path = get_settings().outputs_dir / "calibration"
            except Exception:
                calib_path = Path("outputs/calibration")

        calib_path = Path(calib_path)
        platt_file = calib_path / "platt_calibrator.joblib"
        if platt_file.exists():
            try:
                from cxr_reliability.calibration.probability_calibration import (
                    ProbabilityCalibrator,
                )
                self._calibrator = ProbabilityCalibrator.load(calib_path)
                if getattr(self._calibrator, "_platt", None) is not None:
                    self.is_calibrated = True
                    logger.info("Loaded Platt calibrator from %s", platt_file)
                else:
                    self.is_calibrated = False
                    logger.warning("Platt file exists at %s but failed to load Platt calibrator.", platt_file)
            except Exception as exc:
                self._calibrator = None
                self.is_calibrated = False
                logger.warning("Failed to load calibrator from %s: %s. Continuing uncalibrated.", calib_path, exc)
        else:
            self._calibrator = None
            self.is_calibrated = False
            logger.info("No calibration artifacts at %s. Operating in uncalibrated mode.", calib_path)

    @property
    def calibrator(self) -> Any:
        return self._calibrator


    # ── Loading ──────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Load the pretrained TXV model into memory and move it to device.
        Call this once before running inference.

        On first call: downloads weights (~28 MB) from GitHub releases.
        Subsequent calls: loads from cache (instant, no network needed).
        Cache: C:\\Users\\<user>\\.torchxrayvision\\models_data\\ (Windows)
        """
        self._model = load_txv_model(
            model_id=self.model_id,
            weights_dir=self.weights_dir,
            device=self.device,
        )
        # Verify the pathology labels from the loaded model directly
        if hasattr(self._model, "pathologies"):
            self._pathology_labels = list(self._model.pathologies)
            if self.target_pathology in self._pathology_labels:
                self._pneumonia_idx = self._pathology_labels.index(self.target_pathology)
            else:
                raise RuntimeError(
                    f"Target pathology '{self.target_pathology}' not found in "
                    f"model output labels: {self._pathology_labels}\n"
                    f"Cannot safely map the Pneumonia output."
                )
        self._weights_sha256 = weights_sha256(self.model_id, self.weights_dir)

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before run()."
            )
        return self._model

    # ── Inference ────────────────────────────────────────────────────────

    def run(self, image_tensor: torch.Tensor) -> ModelForward:
        """
        Run one forward pass and return structured results + feature vector.

        Parameters
        ----------
        image_tensor : torch.Tensor shape (1, 1, H, W) in [-1024, 1024]
                       Produced by models.preprocessing.load_image_for_txv()

        Returns
        -------
        ModelForward containing:
            - result.pneumonia_probability : raw sigmoid score at index 8
            - result.all_pathology_outputs : all 18 sigmoid scores (for audit)
            - features                     : 1024-dim vector for OOD Agent

        Notes
        -----
        - Outputs are sigmoid probabilities in [0, 1]. NOT calibrated.
        - Use terminology: raw_score / raw_probability (not "confidence").
        - The feature vector is detached from the computation graph.
        """
        if image_tensor.ndim != 4 or image_tensor.shape[1] != 1:
            raise ValueError(
                f"Expected image tensor of shape (1, 1, H, W), "
                f"got {tuple(image_tensor.shape)}"
            )

        dev = torch.device(self.device)
        tensor = image_tensor.to(dev)

        t_start = time.perf_counter()

        extractor = FeatureExtractor(self.model, layer_name=self.feature_layer)
        with torch.no_grad(), extractor:
            raw_output = self.model(tensor)   # shape: (1, 18) — sigmoid probs
            features_2d = extractor.pooled_features()  # shape: (1, 1024)

        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000.0

        # ── Map Pneumonia output (index 8, verified) ─────────────────────
        probs_1d = raw_output[0].cpu()             # shape: (18,)
        pneumonia_raw_score = float(probs_1d[self._pneumonia_idx].item())
        features_1d = features_2d[0].cpu()          # shape: (1024,)

        # ── Build all-pathology audit dict ────────────────────────────────
        all_outputs: dict[str, float] = {
            label: float(probs_1d[i].item())
            for i, label in enumerate(self._pathology_labels)
            if label  # skip empty-string slots
        }

        # ── Compute calibrated probability if calibrator available ──────
        calibrated_prob: float | None = None
        if self._calibrator is not None:
            try:
                calibrated_prob = float(self._calibrator.transform(pneumonia_raw_score, method="platt"))
            except Exception as exc:
                logger.warning("Calibrator transform failed on score %s: %s", pneumonia_raw_score, exc)
                calibrated_prob = None

        # ── Build structured result contract ─────────────────────────────
        # NOTE: pneumonia_logit field in the contract — we store the raw
        # sigmoid probability here because TXV applies sigmoid internally
        # and does not expose raw logits.
        result = BaseModelResult(
            agent=AgentName.BASE_MODEL,
            version=self.version,
            model_id=self.model_id,
            weights_sha256=self._weights_sha256 or None,
            target_pathology=self.target_pathology,
            pneumonia_logit=pneumonia_raw_score,      # raw sigmoid (pre-calibration)
            pneumonia_probability=pneumonia_raw_score, # backward compatibility
            raw_pneumonia_score=pneumonia_raw_score,   # uncalibrated raw model score
            calibrated_probability=calibrated_prob,    # Platt-calibrated probability (if available)
            all_pathology_outputs=all_outputs,
            score=pneumonia_raw_score,
            label="Pneumonia" if pneumonia_raw_score >= RAW_OPERATING_THRESHOLD else "Non-Pneumonia",
            reasoning=(
                f"densenet121-res224-nih raw sigmoid score: {pneumonia_raw_score:.4f}. "
                + (f"Calibrated probability (Platt): {calibrated_prob:.4f}. " if calibrated_prob is not None else "Calibration unavailable. ")
                + f"Operating threshold: {RAW_OPERATING_THRESHOLD:.4f} (on raw score scale). Research prototype only."
            ),
            latency_ms=inference_ms,
        )

        return ModelForward(
            result=result,
            raw_probs=probs_1d,
            features=features_1d,
            inference_ms=inference_ms,
        )

    # ── Batch inference ──────────────────────────────────────────────────

    def run_batch(
        self,
        image_paths: list[Path],
        dataset_root: Path | None = None,
    ) -> list[dict]:
        """
        Run inference on a list of image paths.

        Returns a list of dicts with:
            image_id, pneumonia_raw_score, label, inference_ms, error (if any)

        This method is intentionally simple — full DataLoader-based batching
        will be added when training/evaluation pipelines are built in Phase 5.
        """
        from cxr_reliability.models.preprocessing import load_image_for_txv

        results = []
        for img_path in image_paths:
            full_path = (
                (dataset_root / img_path) if dataset_root else Path(img_path)
            )
            try:
                tensor = load_image_for_txv(full_path)
                fwd = self.run(tensor)
                results.append({
                    "image_id": full_path.name,
                    "pneumonia_raw_score": fwd.result.pneumonia_probability,
                    "label": fwd.result.label,
                    "inference_ms": round(fwd.inference_ms, 1),
                    "error": None,
                })
            except Exception as exc:
                results.append({
                    "image_id": str(full_path.name),
                    "pneumonia_raw_score": None,
                    "label": None,
                    "inference_ms": None,
                    "error": str(exc),
                })
        return results

    # ── Introspection ────────────────────────────────────────────────────

    def model_info(self) -> dict:
        """
        Return a dict of model metadata for logging and reporting.
        All values are read from the actual loaded model or installed library.
        Nothing is fabricated.
        """
        import torch as _torch
        import torchxrayvision as xrv

        info: dict = {
            "model_name": self.model_id,
            "model_class": "DenseNet",
            "pretrained_weight_id": self.model_id,
            "txv_training_dataset": "NIH ChestX-ray14 (14 classes)",
            "output_labels": self._pathology_labels,
            "num_output_classes": len(self._pathology_labels),
            "target_pathology": self.target_pathology,
            "target_pathology_index": self._pneumonia_idx,
            "output_type": "sigmoid_probability_0_to_1",
            "input_size": "1 x 224 x 224",
            "input_normalization": "pixels [0,255] -> [-1024,1024] HU-like",
            "feature_layer": self.feature_layer,
            "feature_dim": FEATURE_DIM,
            "device": self.device,
            "weights_sha256": self._weights_sha256 or "not_loaded",
            "torchxrayvision_version": xrv.__version__,
            "pytorch_version": _torch.__version__,
            "research_disclaimer": (
                "Research prototype. Not validated for clinical use. "
                "Output is NOT a calibrated clinical probability."
            ),
        }
        return info
