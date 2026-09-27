"""Verification Agent (PRD FR-7) — Phase 10 Implementation.

Responsibility:
    Confirm that an image repair produced a sufficiently useful change across
    reliability signals by comparing before and after states. Accepts/releases
    only if confidence improves by at least the configured provisional threshold
    and quality/OOD signals do not degrade; otherwise escalates further or rejects.

Constraints:
    - Does NOT modify images.
    - Does NOT perform repair or image enhancement.
    - Does NOT orchestrate the pipeline.
    - Does NOT make clinical or diagnostic claims.
    - Uses existing QualityResult, OODResult, BaseModelResult, and UncertaintyResult
      contracts as the authoritative source of before/after data.
    - min_confidence_gain is strictly a configurable PROVISIONAL DEVELOPMENT THRESHOLD.
    - Ambiguous, contradictory, or insufficient evidence defaults safely to ESCALATE/REJECT.

Implementation phase: P10
"""

from __future__ import annotations

import time
from typing import Any

from cxr_reliability.agents.base import AgentBase
from cxr_reliability.config.thresholds import VerificationThresholds
from cxr_reliability.contracts.common import AgentName
from cxr_reliability.contracts.decision import Action
from cxr_reliability.contracts.ood import OODLevel, OODResult
from cxr_reliability.contracts.quality import QualityLevel, QualityResult
from cxr_reliability.contracts.repair import RepairResult
from cxr_reliability.contracts.base_model import BaseModelResult
from cxr_reliability.contracts.uncertainty import UncertaintyLevel, UncertaintyResult
from cxr_reliability.contracts.verification import (
    NextStep,
    SignalBundle,
    VerificationResult,
    VerificationStatus,
)


class VerificationAgent(AgentBase):
    """
    Verification Agent evaluating post-repair signal changes against provisional thresholds.

    Parameters
    ----------
    thresholds : VerificationThresholds, optional
        Verification thresholds containing provisional min_confidence_gain.
    thresholds_version : str, optional
        Threshold set version identifier (default: "v0_prd_defaults").
    """

    name = AgentName.VERIFICATION
    version = "0.10.0"

    DEFAULT_MIN_CONFIDENCE_GAIN: float = -0.01

    def __init__(
        self,
        thresholds: VerificationThresholds | None = None,
        thresholds_version: str = "v0_prd_defaults",
    ) -> None:
        self.thresholds = thresholds or VerificationThresholds(min_confidence_gain=self.DEFAULT_MIN_CONFIDENCE_GAIN)
        self.thresholds_version = thresholds_version

        # Starting threshold is provisional non-degradation guard (default: -0.01)
        if thresholds is not None and thresholds.min_confidence_gain is not None:
            self._min_confidence_gain = float(thresholds.min_confidence_gain)
        else:
            self._min_confidence_gain = self.DEFAULT_MIN_CONFIDENCE_GAIN

    @property
    def min_confidence_gain(self) -> float:
        """Provisional development threshold for confidence gain / non-degradation guard."""
        return self._min_confidence_gain

    def run(
        self,
        before: SignalBundle | None = None,
        after: SignalBundle | None = None,
        can_escalate_further: bool = True,
        action: Action | str | None = Action.REPAIR,
        repair: RepairResult | None = None,
        image_before: Any | None = None,
        image_after: Any | None = None,
        # Direct keyword arguments
        quality_before: QualityResult | None = None,
        quality_after: QualityResult | None = None,
        ood_before: OODResult | None = None,
        ood_after: OODResult | None = None,
        base_model_before: BaseModelResult | None = None,
        base_model_after: BaseModelResult | None = None,
        uncertainty_before: UncertaintyResult | None = None,
        uncertainty_after: UncertaintyResult | None = None,
    ) -> VerificationResult:
        """
        Run verification evaluation comparing signals before and after repair.

        Parameters
        ----------
        before : SignalBundle, optional
        after : SignalBundle, optional
        can_escalate_further : bool
            Whether another escalation step is available if verification fails.
        action : Action or str, optional
            Driving action from Decision Agent.
        repair : RepairResult, optional
            Result of the Repair Agent operation.
        image_before : Any, optional (read-only)
        image_after : Any, optional (read-only)
        """
        t_start = time.perf_counter()

        # 1. Resolve Action
        resolved_action = self._resolve_action(action, repair)

        # 2. Action Gating: Skip verification if action is not REPAIR
        if resolved_action != Action.REPAIR:
            t_ms = (time.perf_counter() - t_start) * 1000.0
            act_str = resolved_action.value.upper() if resolved_action else "NONE"
            return VerificationResult(
                agent=self.name,
                version=self.version,
                score=None,
                label="not_applicable",
                reasoning=(
                    f"Verification not applicable: Decision Agent action is {act_str}, "
                    "not REPAIR. No repair verification performed."
                ),
                latency_ms=t_ms,
                thresholds_version=self.thresholds_version,
                verified=False,
                status=VerificationStatus.NOT_APPLICABLE,
                next_step=NextStep.NOT_APPLICABLE,
                delta_confidence=0.0,
                delta_quality=0.0,
                delta_ood=0.0,
                delta_entropy=None,
                label_flipped=False,
                min_confidence_gain_used=self._min_confidence_gain,
                threshold_is_provisional=True,
                action=resolved_action,
                repair_applied=False,
            )

        # 3. Assemble bundles if direct keyword arguments passed
        resolved_before = self._resolve_bundle(
            bundle=before,
            quality=quality_before,
            ood=ood_before,
            base_model=base_model_before,
            uncertainty=uncertainty_before,
        )
        resolved_after = self._resolve_bundle(
            bundle=after,
            quality=quality_after,
            ood=ood_after,
            base_model=base_model_after,
            uncertainty=uncertainty_after,
        )

        # 4. Handle Missing Evidence (Safe default: Escalate/Reject)
        if resolved_before is None or resolved_after is None:
            t_ms = (time.perf_counter() - t_start) * 1000.0
            next_step = NextStep.ESCALATE if can_escalate_further else NextStep.REJECT
            status = VerificationStatus.ESCALATE if can_escalate_further else VerificationStatus.REJECT
            action_desc = "escalate" if can_escalate_further else "reject (no escalations left)"
            return VerificationResult(
                agent=self.name,
                version=self.version,
                score=None,
                label=status.value,
                reasoning=f"ESCALATE: Missing before or after signal bundle; insufficient evidence for verification. Next step: {action_desc}.",
                latency_ms=t_ms,
                thresholds_version=self.thresholds_version,
                verified=False,
                status=status,
                next_step=next_step,
                delta_confidence=0.0,
                delta_quality=0.0,
                delta_ood=0.0,
                delta_entropy=None,
                label_flipped=False,
                min_confidence_gain_used=self._min_confidence_gain,
                threshold_is_provisional=True,
                action=resolved_action,
                repair_applied=repair.repair_applied if repair else False,
            )

        # 5. Extract Authoritative Signals
        prob_before = resolved_before.get_probability()
        prob_after = resolved_after.get_probability()

        conf_before = resolved_before.get_confidence()
        conf_after = resolved_after.get_confidence()

        ood_dist_before = resolved_before.get_ood_distance()
        ood_dist_after = resolved_after.get_ood_distance()

        ood_lvl_before = resolved_before.get_ood_level()
        ood_lvl_after = resolved_after.get_ood_level()

        q_lvl_before = resolved_before.get_quality_level()
        q_lvl_after = resolved_after.get_quality_level()

        u_lvl_before = resolved_before.get_uncertainty_level()
        u_lvl_after = resolved_after.get_uncertainty_level()

        # Quality score & metrics
        q_score_before = resolved_before.quality_score
        q_score_after = resolved_after.quality_score

        # 6. Calculate Deltas
        # Confidence delta
        if conf_before is not None and conf_after is not None:
            delta_conf = float(conf_after - conf_before)
        else:
            delta_conf = 0.0

        # OOD delta
        if ood_dist_before is not None and ood_dist_after is not None:
            delta_ood = float(ood_dist_after - ood_dist_before)
        else:
            delta_ood = 0.0

        # Quality score delta
        if q_score_before is not None and q_score_after is not None:
            delta_qual = float(q_score_after - q_score_before)
        else:
            delta_qual = 0.0

        # Metric deltas dictionary
        quality_metric_deltas: dict[str, float] = {}
        if resolved_before.quality is not None and resolved_after.quality is not None:
            qb = resolved_before.quality
            qa = resolved_after.quality
            quality_metric_deltas["delta_laplacian_variance"] = float(qa.laplacian_variance - qb.laplacian_variance)
            quality_metric_deltas["delta_blur_pct"] = float(qa.blur_pct - qb.blur_pct)
            quality_metric_deltas["delta_snr_db"] = float(qa.snr_db - qb.snr_db)
            quality_metric_deltas["delta_mean_intensity"] = float(qa.mean_intensity - qb.mean_intensity)

        # Entropy delta
        delta_entropy = None
        if resolved_before.uncertainty is not None and resolved_after.uncertainty is not None:
            delta_entropy = float(resolved_after.uncertainty.entropy - resolved_before.uncertainty.entropy)

        # Label flip detection
        label_flipped = False
        if prob_before is not None and prob_after is not None:
            label_flipped = bool((prob_before >= 0.5) != (prob_after >= 0.5))
        elif resolved_before.base_model is not None and resolved_after.base_model is not None:
            label_flipped = bool(resolved_before.base_model.label != resolved_after.base_model.label)

        # 7. Evaluate Verification Rules
        verified, reason_detail = self._evaluate_rules(
            conf_before=conf_before,
            conf_after=conf_after,
            delta_conf=delta_conf,
            ood_lvl_before=ood_lvl_before,
            ood_lvl_after=ood_lvl_after,
            q_lvl_before=q_lvl_before,
            q_lvl_after=q_lvl_after,
            quality_metric_deltas=quality_metric_deltas,
            repair=repair,
            label_flipped=label_flipped,
            q_score_before=q_score_before,
            q_score_after=q_score_after,
            ood_dist_before=ood_dist_before,
            ood_dist_after=ood_dist_after,
        )

        t_ms = (time.perf_counter() - t_start) * 1000.0

        # 8. Route Next Step
        if verified:
            status = VerificationStatus.VERIFIED
            next_step = NextStep.RELEASE
            label = "verified"
            q_after_str = q_lvl_after.value.upper() if q_lvl_after else "ACCEPTABLE"
            reasoning = (
                f"VERIFIED: quality improved to {q_after_str} and confidence non-degradation guard "
                f"met ({delta_conf:+.4f} >= {self._min_confidence_gain:.2f}). "
                f"OOD status is {ood_lvl_after.value if ood_lvl_after else 'acceptable'}."
            )
        else:
            status = VerificationStatus.ESCALATE if can_escalate_further else VerificationStatus.REJECT
            next_step = NextStep.ESCALATE if can_escalate_further else NextStep.REJECT
            label = status.value
            act_word = "ESCALATE" if can_escalate_further else "REJECT"
            reasoning = (
                f"{act_word}: {reason_detail}. "
                f"Confidence delta: {delta_conf:+.4f} (provisional threshold: {self._min_confidence_gain:.2f})."
            )

        repair_applied_flag = repair.repair_applied if repair is not None else True

        return VerificationResult(
            agent=self.name,
            version=self.version,
            score=delta_conf,
            label=label,
            reasoning=reasoning,
            latency_ms=t_ms,
            thresholds_version=self.thresholds_version,
            verified=verified,
            status=status,
            next_step=next_step,
            delta_confidence=delta_conf,
            delta_quality=delta_qual,
            delta_ood=delta_ood,
            delta_entropy=delta_entropy,
            label_flipped=label_flipped,
            min_confidence_gain_used=self._min_confidence_gain,
            threshold_is_provisional=True,
            action=resolved_action,
            repair_applied=repair_applied_flag,
            quality_before=resolved_before.quality,
            quality_after=resolved_after.quality,
            quality_level_before=q_lvl_before,
            quality_level_after=q_lvl_after,
            ood_level_before=ood_lvl_before,
            ood_level_after=ood_lvl_after,
            uncertainty_level_before=u_lvl_before,
            uncertainty_level_after=u_lvl_after,
            probability_before=prob_before,
            probability_after=prob_after,
            confidence_before=conf_before,
            confidence_after=conf_after,
            quality_metric_deltas=quality_metric_deltas,
        )

    def _evaluate_rules(
        self,
        conf_before: float | None,
        conf_after: float | None,
        delta_conf: float,
        ood_lvl_before: OODLevel | None,
        ood_lvl_after: OODLevel | None,
        q_lvl_before: QualityLevel | None,
        q_lvl_after: QualityLevel | None,
        quality_metric_deltas: dict[str, float],
        repair: RepairResult | None,
        label_flipped: bool = False,
        q_score_before: float | None = None,
        q_score_after: float | None = None,
        ood_dist_before: float | None = None,
        ood_dist_after: float | None = None,
    ) -> tuple[bool, str]:
        """Evaluate deterministic verification conditions."""
        # Condition 1: Check if repair was skipped/refused
        if repair is not None and not repair.repair_applied:
            return False, f"Repair was not applied ({repair.skipped_reason or 'repair skipped'})"

        # Condition 2: Ambiguous or missing confidence data
        if conf_before is None or conf_after is None:
            return False, "Confidence signals are missing or incomplete"

        # Condition 3: Prediction label flip check (diagnostic prediction label must not change)
        if label_flipped:
            return False, "Prediction label flipped after repair; cannot verify"

        # Condition 4: OOD evaluation: must remain IN_DISTRIBUTION and must not materially worsen
        if ood_lvl_after is not None:
            if ood_lvl_after != OODLevel.IN_DISTRIBUTION:
                return False, f"Post-repair OOD level is {ood_lvl_after.value.upper()}; must remain IN_DISTRIBUTION"
            if ood_lvl_before is not None and ood_lvl_before == OODLevel.IN_DISTRIBUTION and ood_lvl_after != OODLevel.IN_DISTRIBUTION:
                return False, f"OOD level degraded from IN_DISTRIBUTION to {ood_lvl_after.value.upper()}"
        elif ood_dist_after is not None:
            if ood_dist_before is not None and ood_dist_after > ood_dist_before + 5.0:
                return False, f"OOD distance degraded significantly ({ood_dist_after:.2f} vs {ood_dist_before:.2f})"
        else:
            return False, "OOD signals are missing or incomplete"

        # Condition 5: Quality evaluation: repair must produce objective quality improvement
        # Poor -> Good = quality resolved
        # Poor -> Degraded = partial improvement; do not automatically treat this as fully verified
        if q_lvl_before is not None and q_lvl_after is not None:
            if q_lvl_before == QualityLevel.POOR:
                if q_lvl_after == QualityLevel.POOR:
                    return False, "Quality remained POOR after repair; defect not resolved"
                elif q_lvl_after == QualityLevel.DEGRADED:
                    return False, "Quality only partially improved from POOR to DEGRADED; not fully verified"
                elif q_lvl_after != QualityLevel.GOOD:
                    return False, f"Quality did not resolve to GOOD (current: {q_lvl_after.value.upper()})"
            elif q_lvl_before == QualityLevel.DEGRADED:
                if q_lvl_after != QualityLevel.GOOD:
                    return False, f"Quality remained {q_lvl_after.value.upper()} after repair; defect not resolved to GOOD"
            elif q_lvl_before == QualityLevel.GOOD:
                if q_lvl_after != QualityLevel.GOOD:
                    return False, f"Overall quality degraded from GOOD to {q_lvl_after.value.upper()}"
        elif q_score_before is not None and q_score_after is not None:
            if q_score_after < q_score_before:
                return False, f"Quality score degraded ({q_score_after:.2f} < {q_score_before:.2f})"
        else:
            return False, "Quality signals are missing or incomplete"

        # Condition 6: Severe noise increase check (SNR drop > 5 dB)
        if "delta_snr_db" in quality_metric_deltas and quality_metric_deltas["delta_snr_db"] < -5.0:
            return False, f"Noise increased significantly (SNR dropped {abs(quality_metric_deltas['delta_snr_db']):.1f} dB)"

        # Condition 7: Confidence non-degradation guard check (provisional threshold >= -0.01)
        # Numerical tolerance (1e-7) protects against IEEE-754 float precision artifacts (e.g., 0.69 - 0.70 = -0.010000000000000009)
        if delta_conf < (self._min_confidence_gain - 1e-7):
            return False, (
                f"Confidence degraded beyond tolerance ({delta_conf:+.4f} < {self._min_confidence_gain:.2f})"
            )

        return True, ""

    def _resolve_action(self, action: Any, repair: RepairResult | None) -> Action | None:
        if action is not None:
            if isinstance(action, Action):
                return action
            if isinstance(action, str):
                try:
                    return Action(action.lower())
                except ValueError:
                    return None
        if repair is not None and repair.action is not None:
            return repair.action
        return None

    def _resolve_bundle(
        self,
        bundle: SignalBundle | None,
        quality: QualityResult | None,
        ood: OODResult | None,
        base_model: BaseModelResult | None,
        uncertainty: UncertaintyResult | None,
    ) -> SignalBundle | None:
        if bundle is not None:
            # If explicit bundle passed, fill any missing individual contracts if provided
            data = bundle.model_dump()
            if data.get("quality") is None and quality is not None:
                data["quality"] = quality
            if data.get("ood") is None and ood is not None:
                data["ood"] = ood
            if data.get("base_model") is None and base_model is not None:
                data["base_model"] = base_model
            if data.get("uncertainty") is None and uncertainty is not None:
                data["uncertainty"] = uncertainty
            return SignalBundle(**data)

        # Construct new bundle if any individual contracts provided
        if any(c is not None for c in (quality, ood, base_model, uncertainty)):
            return SignalBundle(
                quality=quality,
                ood=ood,
                base_model=base_model,
                uncertainty=uncertainty,
            )

        return None
