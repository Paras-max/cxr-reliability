"""Repair Agent (PRD FR-6) — Phase 9 Implementation.

Responsibility:
    Apply targeted, non-destructive image-quality repairs when the Decision Agent
    recommends Action.REPAIR. Handles exposure (CLAHE), noise (NL-means), and
    blur (conservative unsharp masking) in a deterministic execution order.

Constraints:
    - Non-destructive processing: never modifies the original image array or tensor.
    - Does NOT make clinical or diagnostic claims.
    - Does NOT calculate post-repair quality metrics (no quality_after).
    - Does NOT determine whether repair improved reliability (Verification belongs to Phase 10).
    - image_changed=True indicates only numerical alteration, NOT repair success.
    - All repair bounds are configured provisional development defaults, NOT calibrated values.
    - Does NOT invent anatomy, hallucinate structures, or use generative inpainting.

Dependencies:
    opencv-python-headless, scikit-image, numpy, torch, contracts.repair,
    contracts.decision, contracts.quality, config.thresholds

Implementation phase: P9
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from cxr_reliability.agents.base import AgentBase
from cxr_reliability.config.thresholds import RepairThresholds
from cxr_reliability.contracts.common import AgentName
from cxr_reliability.contracts.decision import Action, DecisionResult
from cxr_reliability.contracts.quality import DefectType, QualityResult
from cxr_reliability.contracts.repair import RepairResult, RepairStep
from cxr_reliability.repair.pipeline import (
    REPAIR_EXECUTION_ORDER,
    RepairConfig,
    apply_sequential_repairs,
)


class _ImageContainer:
    """Helper to safely manage non-destructive conversions across image representations."""

    def __init__(self, image: Any) -> None:
        if image is None:
            raise ValueError("Input image cannot be None.")

        self.is_torch = isinstance(image, torch.Tensor)
        self.torch_device = image.device if self.is_torch else None
        self.torch_dtype = image.dtype if self.is_torch else None

        if self.is_torch:
            if torch.isnan(image).any():
                raise ValueError("Input image tensor contains NaN.")
            if torch.isinf(image).any():
                raise ValueError("Input image tensor contains Inf.")
            if image.numel() == 0:
                raise ValueError("Input image tensor cannot be empty.")
            # Non-destructive: detach and clone to ensure input tensor is never modified
            self.np_orig = image.detach().cpu().numpy().copy()
        elif isinstance(image, np.ndarray):
            if np.isnan(image).any():
                raise ValueError("Input image array contains NaN.")
            if np.isinf(image).any():
                raise ValueError("Input image array contains Inf.")
            if image.size == 0:
                raise ValueError("Input image array cannot be empty.")
            # Non-destructive: deep copy array
            self.np_orig = image.copy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image).__name__}")

        self.orig_shape = list(self.np_orig.shape)
        self.orig_dtype = self.np_orig.dtype
        self.ndim = self.np_orig.ndim

        if self.ndim not in (2, 3, 4):
            raise ValueError(f"Unsupported image dimension: ndim={self.ndim}, shape={self.orig_shape}")

        self.orig_min = float(np.min(self.np_orig))
        self.orig_max = float(np.max(self.np_orig))

        # Determine value range mode
        if self.np_orig.dtype == np.uint8:
            self.range_mode = "uint8_255"
        elif np.issubdtype(self.np_orig.dtype, np.floating):
            if 0.0 <= self.orig_min and self.orig_max <= 1.0:
                self.range_mode = "float_1"
            else:
                self.range_mode = "float_255"
        elif np.issubdtype(self.np_orig.dtype, np.integer):
            self.range_mode = "int_255"
        else:
            raise ValueError(f"Unsupported image dtype: {self.orig_dtype}")

    def to_uint8_2d_slices(self) -> tuple[list[np.ndarray], Any]:
        """Convert image to list of 2D uint8 slices for filtering."""
        arr = self.np_orig

        # Standardize range to [0, 255] uint8
        if self.range_mode == "float_1":
            arr_u8 = np.clip(np.round(arr * 255.0), 0.0, 255.0).astype(np.uint8)
        elif self.range_mode in ("float_255", "int_255"):
            arr_u8 = np.clip(np.round(arr), 0.0, 255.0).astype(np.uint8)
        else:
            arr_u8 = arr.copy()

        # Handle 2D, 3D, 4D slicing
        if self.ndim == 2:
            return [arr_u8], ("2d", None)
        elif self.ndim == 3:
            # (H, W, C) or (C, H, W)
            if arr_u8.shape[2] in (1, 3, 4) and arr_u8.shape[0] not in (1, 3, 4):
                slices = [arr_u8[:, :, c] for c in range(arr_u8.shape[2])]
                return slices, ("3d_hwc", arr_u8.shape[2])
            elif arr_u8.shape[0] in (1, 3, 4):
                slices = [arr_u8[c, :, :] for c in range(arr_u8.shape[0])]
                return slices, ("3d_chw", arr_u8.shape[0])
            else:
                # Default assume trailing channel
                slices = [arr_u8[:, :, c] for c in range(arr_u8.shape[2])]
                return slices, ("3d_hwc", arr_u8.shape[2])
        elif self.ndim == 4:
            # e.g. (1, 1, H, W) or (1, C, H, W)
            if arr_u8.shape[0] == 1:
                c_dim = arr_u8.shape[1]
                slices = [arr_u8[0, c, :, :] for c in range(c_dim)]
                return slices, ("4d_1chw", c_dim)
            else:
                raise ValueError(f"Batch dimension > 1 not supported in single image repair: {self.orig_shape}")
        raise ValueError(f"Cannot slice image with shape {self.orig_shape}")

    def reconstruct(self, repaired_slices: list[np.ndarray], metadata: Any) -> Any:
        """Reconstruct the original format, dtype, shape, device from repaired uint8 slices."""
        mode, extra = metadata

        if mode == "2d":
            reassembled = repaired_slices[0]
        elif mode == "3d_hwc":
            reassembled = np.stack(repaired_slices, axis=2)
        elif mode == "3d_chw":
            reassembled = np.stack(repaired_slices, axis=0)
        elif mode == "4d_1chw":
            stacked = np.stack(repaired_slices, axis=0)  # (C, H, W)
            reassembled = np.expand_dims(stacked, axis=0)  # (1, C, H, W)
        else:
            raise ValueError(f"Unknown reassembly mode: {mode}")

        # Restore numerical value range and dtype
        if self.range_mode == "float_1":
            reassembled = (reassembled.astype(np.float32) / 255.0).astype(self.orig_dtype)
            reassembled = np.clip(reassembled, 0.0, 1.0)
        elif self.range_mode in ("float_255", "int_255"):
            reassembled = reassembled.astype(self.orig_dtype)
        else:
            reassembled = reassembled.astype(np.uint8)

        # Restore torch tensor if input was torch
        if self.is_torch:
            out_tensor = torch.from_numpy(reassembled).to(
                device=self.torch_device,
                dtype=self.torch_dtype,
            )
            return out_tensor
        return reassembled


class RepairAgent(AgentBase):
    """
    Repair Agent applying quality repairs when directed by Decision Agent.

    Parameters
    ----------
    bounds : RepairThresholds, optional
        Configured provisional repair limits (development defaults).
    thresholds_version : str, optional
        Threshold version tag.
    config : RepairConfig, optional
        Direct operational parameters for filtering methods.
    """

    name = AgentName.REPAIR
    version = "0.9.0"

    def __init__(
        self,
        bounds: RepairThresholds | None = None,
        thresholds_version: str = "v0_prd_defaults",
        config: RepairConfig | None = None,
    ) -> None:
        self.bounds = bounds or RepairThresholds()
        self.thresholds_version = thresholds_version
        self.config = config or RepairConfig()

    def run(
        self,
        image: Any,
        quality: QualityResult | None = None,
        decision: DecisionResult | Action | str | None = None,
        image_id: str | None = None,
    ) -> tuple[Any, RepairResult]:
        """
        Run repair evaluation and execution.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input chest radiograph.
        quality : QualityResult, optional
            Output of Quality Agent detailing defects.
        decision : DecisionResult, Action, or str, optional
            Action decision from Decision Agent.
        image_id : str, optional
            Image identifier for tracking.

        Returns
        -------
        tuple[Any, RepairResult]
            Repaired image (same representation as input) and structured RepairResult contract.
        """
        t_start = time.perf_counter()

        # 1. Parse and validate image non-destructively
        container = _ImageContainer(image)
        orig_shape = container.orig_shape
        orig_range = [container.orig_min, container.orig_max]

        # 2. Parse Decision action
        resolved_action = self._resolve_action(decision)

        # 3. Handle non-REPAIR actions: strictly do NOT modify image
        if resolved_action != Action.REPAIR:
            t_ms = (time.perf_counter() - t_start) * 1000.0
            act_str = resolved_action.value.upper() if resolved_action else "NONE"
            return self._return_unmodified(
                image=image,
                container=container,
                action=resolved_action,
                label="skipped",
                reasoning=f"No repair performed: decision action is {act_str}.",
                skipped_reason=f"Decision action is {act_str}; repair not requested.",
                latency_ms=t_ms,
            )

        # 4. Handle missing QualityResult
        if quality is None:
            t_ms = (time.perf_counter() - t_start) * 1000.0
            return self._return_unmodified(
                image=image,
                container=container,
                action=resolved_action,
                label="skipped",
                reasoning="No repair performed: QualityResult is missing; cannot identify defects.",
                skipped_reason="QualityResult is missing; cannot determine defects.",
                latency_ms=t_ms,
            )

        # 5. Check configured provisional repair bounds
        refused, refusal_reason = self._check_provisional_bounds(quality)
        if refused:
            t_ms = (time.perf_counter() - t_start) * 1000.0
            return self._return_unmodified(
                image=image,
                container=container,
                action=resolved_action,
                label="refused",
                reasoning=f"Repair refused: {refusal_reason}",
                skipped_reason=refusal_reason,
                refused_bounds=True,
                latency_ms=t_ms,
            )

        # 6. Extract detected defect flags
        defects = self._extract_defects(quality)
        if not defects:
            t_ms = (time.perf_counter() - t_start) * 1000.0
            return self._return_unmodified(
                image=image,
                container=container,
                action=resolved_action,
                label="skipped",
                reasoning="No repair performed: no quality defect flags detected.",
                skipped_reason="No quality defect flags detected.",
                latency_ms=t_ms,
            )

        # 7. Apply sequential repair operations in deterministic order
        slices, meta = container.to_uint8_2d_slices()
        repaired_slices = []
        all_steps: list[RepairStep] = []

        for i, s in enumerate(slices):
            repaired_s, steps = apply_sequential_repairs(
                image=s,
                defects=defects,
                config=self.config,
            )
            repaired_slices.append(repaired_s)
            if i == 0:
                # Record steps from the primary channel
                all_steps = steps

        # 8. Reconstruct repaired image with original representation
        repaired_output = container.reconstruct(repaired_slices, meta)

        # 9. Compute property ranges and numerical difference
        repaired_shape = list(repaired_output.shape) if hasattr(repaired_output, "shape") else orig_shape

        if container.is_torch:
            rep_min = float(torch.min(repaired_output).item())
            rep_max = float(torch.max(repaired_output).item())
            # image_changed indicates strictly numerical difference
            image_changed = bool(not torch.equal(image, repaired_output))
        else:
            rep_min = float(np.min(repaired_output))
            rep_max = float(np.max(repaired_output))
            image_changed = bool(not np.array_equal(container.np_orig, repaired_output))

        t_ms = (time.perf_counter() - t_start) * 1000.0

        # 10. Construct non-clinical audit reasoning
        op_names = [s.method for s in all_steps]
        reasoning = (
            f"REPAIR applied: executed operations ({', '.join(op_names)}) "
            f"in deterministic order for detected defect(s) "
            f"({', '.join(d.value for d in defects)})."
        )

        result = RepairResult(
            agent=self.name,
            version=self.version,
            score=None,
            label="repaired",
            reasoning=reasoning,
            latency_ms=t_ms,
            thresholds_version=self.thresholds_version,
            repaired=True,
            repair_applied=True,
            action=resolved_action,
            steps=all_steps,
            defects_detected=defects,
            original_shape=orig_shape,
            repaired_shape=repaired_shape,
            original_range=orig_range,
            repaired_range=[rep_min, rep_max],
            image_changed=image_changed,
            skipped_reason=None,
            refused_bounds=False,
        )

        return repaired_output, result

    def _resolve_action(self, decision: Any) -> Action | None:
        """Extract Action enum from DecisionResult, Action, or str."""
        if decision is None:
            return None
        if isinstance(decision, Action):
            return decision
        if isinstance(decision, DecisionResult):
            return decision.action
        if isinstance(decision, str):
            try:
                return Action(decision.lower())
            except ValueError:
                return None
        return None

    def _check_provisional_bounds(self, quality: QualityResult) -> tuple[bool, str]:
        """Check if defects exceed configured provisional repair limits."""
        # QualityResult may already have repairable=False evaluated
        if quality.repairable is False:
            return True, "Defect severity exceeds configured provisional repair limits."

        # Check against agent's own provisional repair bounds if populated
        b = self.bounds
        flags = quality.flags

        if flags.blur and b.max_repairable_blur_pct is not None:
            if quality.blur_pct > b.max_repairable_blur_pct:
                return True, (
                    f"Blur percentage ({quality.blur_pct:.1f}%) exceeds "
                    f"provisional limit ({b.max_repairable_blur_pct:.1f}%)."
                )

        if flags.noise and b.min_repairable_snr_db is not None:
            if quality.snr_db < b.min_repairable_snr_db:
                return True, (
                    f"SNR ({quality.snr_db:.1f} dB) falls below "
                    f"provisional limit ({b.min_repairable_snr_db:.1f} dB)."
                )

        if flags.exposure:
            if b.repairable_exposure_mean_min is not None and quality.mean_intensity < b.repairable_exposure_mean_min:
                return True, (
                    f"Mean intensity ({quality.mean_intensity:.1f}) is below "
                    f"provisional limit ({b.repairable_exposure_mean_min:.1f})."
                )
            if b.repairable_exposure_mean_max is not None and quality.mean_intensity > b.repairable_exposure_mean_max:
                return True, (
                    f"Mean intensity ({quality.mean_intensity:.1f}) exceeds "
                    f"provisional limit ({b.repairable_exposure_mean_max:.1f})."
                )

        return False, ""

    def _extract_defects(self, quality: QualityResult) -> list[DefectType]:
        """Extract active defect flags in deterministic order."""
        defects = []
        flags = quality.flags
        for defect in REPAIR_EXECUTION_ORDER:
            if defect == DefectType.EXPOSURE and flags.exposure:
                defects.append(defect)
            elif defect == DefectType.NOISE and flags.noise:
                defects.append(defect)
            elif defect == DefectType.BLUR and flags.blur:
                defects.append(defect)
        return defects

    def _return_unmodified(
        self,
        image: Any,
        container: _ImageContainer,
        action: Action | None,
        label: str,
        reasoning: str,
        skipped_reason: str,
        refused_bounds: bool = False,
        latency_ms: float = 0.0,
    ) -> tuple[Any, RepairResult]:
        """Return non-destructive copy of original image with audit record."""
        if container.is_torch:
            out_img = image.clone()
        else:
            out_img = container.np_orig.copy()

        orig_shape = container.orig_shape
        orig_range = [container.orig_min, container.orig_max]

        res = RepairResult(
            agent=self.name,
            version=self.version,
            score=None,
            label=label,
            reasoning=reasoning,
            latency_ms=latency_ms,
            thresholds_version=self.thresholds_version,
            repaired=False,
            repair_applied=False,
            action=action,
            steps=[],
            defects_detected=[],
            original_shape=orig_shape,
            repaired_shape=orig_shape,
            original_range=orig_range,
            repaired_range=orig_range,
            image_changed=False,
            skipped_reason=skipped_reason,
            refused_bounds=refused_bounds,
        )
        return out_img, res
