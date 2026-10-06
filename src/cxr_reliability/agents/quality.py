"""Quality Agent (PRD FR-1) — Phase 7 Implementation.

Responsibility
--------------
Assess whether a chest X-ray has image-quality degradations (blur, noise, exposure)
that may affect downstream model reliability.

Input
-----
    - image: 2D/3D np.ndarray, torch.Tensor, PIL.Image, or file path.
    - image_id: optional str for audit logging.

Output
------
    - QualityResult (Pydantic contract).

Constraints
-----------
- Does NOT run the Base Model or OOD Agent.
- Does NOT modify the input image.
- Does NOT make diagnostic claims.
"""

from __future__ import annotations

import time
from typing import Any

from cxr_reliability.agents.base import AgentBase
from cxr_reliability.config.thresholds import QualityThresholds, RepairThresholds
from cxr_reliability.contracts.common import AgentName
from cxr_reliability.contracts.quality import QualityFlags, QualityLevel, QualityResult
from cxr_reliability.quality.evaluator import QualityConfig, QualityEvaluator


class QualityAgent(AgentBase):
    """
    Quality Agent evaluating blur, noise, and exposure.

    Parameters
    ----------
    thresholds : QualityThresholds, optional
        Loaded from YAML config (e.g., configs/thresholds/v0_prd_defaults.yaml).
    repair_bounds : RepairThresholds, optional
        Repair reversibility bounds (if calibrated).
    thresholds_version : str, optional
        Identifier for the threshold set (default: 'v0_prd_defaults').
    config : QualityConfig, optional
        Direct configuration override.
    """

    name = AgentName.QUALITY
    version = "0.7.0"

    def __init__(
        self,
        thresholds: QualityThresholds | None = None,
        repair_bounds: RepairThresholds | None = None,
        thresholds_version: str = "v0_prd_defaults",
        config: QualityConfig | None = None,
    ) -> None:
        self.thresholds = thresholds
        self.repair_bounds = repair_bounds
        self.thresholds_version = thresholds_version

        if config is not None:
            self._config = config
        else:
            self._config = self._build_config_from_thresholds(thresholds)

        self._evaluator = QualityEvaluator(self._config)

    def _build_config_from_thresholds(
        self,
        thresholds: QualityThresholds | None,
    ) -> QualityConfig:
        """Merge QualityThresholds from YAML or use safe defaults."""
        cfg = QualityConfig()
        if thresholds is not None:
            if thresholds.blur_laplacian_var_min is not None:
                cfg.blur_laplacian_var_min = float(thresholds.blur_laplacian_var_min)
            if thresholds.snr_db_min is not None:
                cfg.snr_db_min = float(thresholds.snr_db_min)
            if thresholds.exposure_mean_min is not None:
                cfg.exposure_mean_min = float(thresholds.exposure_mean_min)
            if thresholds.exposure_mean_max is not None:
                cfg.exposure_mean_max = float(thresholds.exposure_mean_max)
            if thresholds.reference_max_laplacian_var is not None:
                cfg.reference_max_variance = float(thresholds.reference_max_laplacian_var)
        return cfg

    @property
    def config(self) -> QualityConfig:
        return self._config

    def run(
        self,
        image: Any,
        image_id: str | None = None,
    ) -> QualityResult:
        """
        Run quality evaluation on an input image.

        Parameters
        ----------
        image : np.ndarray, torch.Tensor, PIL.Image, Path, or str
        image_id : str, optional

        Returns
        -------
        QualityResult
        """
        t_start = time.perf_counter()

        blur_res, noise_res, exp_res, overall_status, reasoning, is_near = self._evaluator.evaluate(
            image=image,
            image_id=image_id,
        )

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        flags = QualityFlags(
            blur=blur_res.is_blurred,
            noise=noise_res.is_noisy,
            exposure=exp_res.is_underexposed or exp_res.is_overexposed,
        )

        repairable = self._check_repairable(
            blur_pct=blur_res.blur_pct,
            snr_db=noise_res.snr_db,
            mean_intensity=exp_res.mean_intensity,
            flags=flags,
        )

        q_level = QualityLevel(overall_status)

        return QualityResult(
            agent=self.name,
            version=self.version,
            score=blur_res.blur_pct,
            label=q_level.value,
            reasoning=reasoning,
            latency_ms=latency_ms,
            thresholds_version=self.thresholds_version,
            laplacian_variance=blur_res.laplacian_variance,
            blur_pct=blur_res.blur_pct,
            snr_db=noise_res.snr_db,
            mean_intensity=exp_res.mean_intensity,
            histogram_std=exp_res.histogram_std,
            flags=flags,
            overall=q_level,
            repairable=repairable,
            near_threshold=is_near,
            image_id=image_id,
            component_statuses={
                "blur": blur_res.status,
                "noise": noise_res.status,
                "exposure": exp_res.status,
            },
            thresholds=self._config.to_dict(),
        )

    def _check_repairable(
        self,
        blur_pct: float,
        snr_db: float,
        mean_intensity: float,
        flags: QualityFlags,
    ) -> bool | None:
        """Check if defects fall within calibrated repair bounds (None if uncalibrated)."""
        if not (flags.blur or flags.noise or flags.exposure):
            return True  # No defects to repair

        if self.repair_bounds is None:
            return None

        # Check if all relevant bounds are calibrated
        b = self.repair_bounds
        if (
            b.max_repairable_blur_pct is None
            or b.min_repairable_snr_db is None
            or b.repairable_exposure_mean_min is None
            or b.repairable_exposure_mean_max is None
        ):
            return None

        if flags.blur and blur_pct > b.max_repairable_blur_pct:
            return False
        if flags.noise and snr_db < b.min_repairable_snr_db:
            return False
        if flags.exposure:
            if (
                mean_intensity < b.repairable_exposure_mean_min
                or mean_intensity > b.repairable_exposure_mean_max
            ):
                return False

        return True
