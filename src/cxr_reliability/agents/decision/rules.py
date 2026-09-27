"""Decision Agent v1: deterministic precedence-ordered rule table.

Responsibility
--------------
Combine Quality, OOD and Uncertainty signals into a single reliability-aware
action (ACCEPT / REPAIR / ESCALATE / REJECT) using the fixed rule table
described in PRD FR-5 and the proposed decision precedence in docs/architecture.md.

Precedence order (architecture.md §Decision precedence):
  1. SEVERE OOD                         → REJECT
  2. Any signal within borderline_margin of its threshold
     OR OOD is BORDERLINE               → ESCALATE  (or REJECT per config)
  3. POOR/DEGRADED quality + IN_DISTRIBUTION
                                        → REPAIR
  4. GOOD quality + IN_DISTRIBUTION + HIGH uncertainty
                                        → ESCALATE
  5. GOOD quality + IN_DISTRIBUTION + LOW uncertainty
                                        → ACCEPT
  6. Any other combination             → ESCALATE  (safe default)

Two-Level Uncertainty Rationale
-------------------------------
Uncertainty has two categorical levels: LOW and HIGH.
- LOW uncertainty (confidence >= 0.85 AND normalized entropy <= 0.25) routes to ACCEPT
  when image quality is GOOD and OOD status is IN_DISTRIBUTION.
- HIGH uncertainty (all other samples, including borderline or previously intermediate
  confidence) routes to ESCALATE for human review.

Input
-----
    quality     : QualityResult
    ood         : OODResult
    uncertainty : UncertaintyResult
    state       : DecisionState

Output
------
    DecisionResult

Constraints
-----------
- Does NOT modify images.
- Does NOT run neural network inference.
- Does NOT make clinical claims.
- Never silently chooses ACCEPT for ambiguous / missing / invalid inputs.

Dependencies
------------
    config.thresholds.DecisionThresholds
    config.pipeline_config.ExecutionConfig
    contracts.decision, contracts.ood, contracts.quality, contracts.uncertainty

Implementation phase: P8
"""

from __future__ import annotations

import time
from typing import Literal

from cxr_reliability.agents.base import AgentBase
from cxr_reliability.config.pipeline_config import ExecutionConfig
from cxr_reliability.config.thresholds import DecisionThresholds
from cxr_reliability.contracts.common import AgentName
from cxr_reliability.contracts.decision import Action, DecisionResult, DecisionState
from cxr_reliability.contracts.ood import OODLevel, OODResult
from cxr_reliability.contracts.quality import QualityLevel, QualityResult
from cxr_reliability.contracts.uncertainty import UncertaintyLevel, UncertaintyResult

# ---------------------------------------------------------------------------
# Rule IDs — stable string identifiers used in audit records and tests
# ---------------------------------------------------------------------------

_RULE_SEVERE_OOD = "R1_SEVERE_OOD_REJECT"
_RULE_BORDERLINE = "R2_BORDERLINE_SIGNAL"
_RULE_REPAIR = "R3_QUALITY_REPAIR"
_RULE_ESCALATE_HIGH = "R4_HIGH_UNCERTAINTY_ESCALATE"
_RULE_ESCALATE_MEDIUM = "R5_MEDIUM_UNCERTAINTY_ESCALATE"  # Deprecated legacy identifier; no longer actively routed
_RULE_ACCEPT = "R6_GOOD_IN_LOW_ACCEPT"
_RULE_SAFE_DEFAULT = "R7_SAFE_DEFAULT_ESCALATE"
_RULE_INVALID_INPUT = "R0_INVALID_INPUT_ESCALATE"


class RuleTableDecisionAgent(AgentBase):
    """
    Decision Agent v1: fixed, explainable, precedence-ordered rule table.

    Parameters
    ----------
    thresholds : DecisionThresholds
        Decision-specific thresholds (borderline_margin).
    execution : ExecutionConfig
        Pipeline wiring, including borderline_action ("escalate" | "reject").
    thresholds_version : str
        Identifier for the active threshold set (for audit records).
    """

    name = AgentName.DECISION
    version = "v1-0.8.0"

    def __init__(
        self,
        thresholds: DecisionThresholds,
        execution: ExecutionConfig,
        thresholds_version: str,
    ) -> None:
        self.thresholds = thresholds
        self.execution = execution
        self.thresholds_version = thresholds_version

    # ------------------------------------------------------------------
    # Public interface (matches DecisionAgent Protocol in interface.py)
    # ------------------------------------------------------------------

    def decide(
        self,
        quality: QualityResult,
        ood: OODResult,
        uncertainty: UncertaintyResult,
        state: DecisionState,
    ) -> DecisionResult:
        """
        Apply the deterministic rule table and return a structured decision.

        Parameters
        ----------
        quality     : QualityResult      — output of the Quality Agent
        ood         : OODResult          — output of the OOD Agent
        uncertainty : UncertaintyResult  — output of the Uncertainty Agent
        state       : DecisionState      — repair/escalation attempt counters

        Returns
        -------
        DecisionResult with action, rule_id, driving_signals, reasoning, etc.
        """
        t_start = time.perf_counter()

        # ── Step 1: Validate inputs ───────────────────────────────────────
        error = self._validate_inputs(quality, ood, uncertainty, state)
        if error is not None:
            t_ms = (time.perf_counter() - t_start) * 1000.0
            return self._make_result(
                action=Action.ESCALATE,
                rule_id=_RULE_INVALID_INPUT,
                reasoning=f"ESCALATE: input validation failed — {error}. Safe default applied.",
                reliability_label="escalated_invalid_input",
                driving_signals={
                    "quality_overall": "unknown",
                    "ood_level": "unknown",
                    "uncertainty_level": "unknown",
                    "validation_error": error,
                },
                threshold_margins={},
                state=state,
                latency_ms=t_ms,
            )

        # ── Step 2: Extract and normalise signal labels ───────────────────
        q_level: QualityLevel = quality.overall
        ood_level: OODLevel = ood.level
        u_level: UncertaintyLevel = uncertainty.uncertainty_level

        # Build driving signals dict for audit / traceability
        driving_signals: dict[str, str | float | bool] = {
            "quality_overall": q_level.value,
            "ood_level": ood_level.value,
            "uncertainty_level": u_level.value,
            "confidence": uncertainty.confidence,
            "mahalanobis_distance": ood.mahalanobis_distance,
            "near_threshold": quality.near_threshold,
        }

        # Compute threshold margins for audit
        threshold_margins = self._compute_margins(quality, ood, uncertainty)

        # ── Step 3: Apply precedence-ordered rules ────────────────────────

        # Rule 1 — SEVERE OOD → REJECT regardless of all other signals
        if ood_level == OODLevel.SEVERE:
            t_ms = (time.perf_counter() - t_start) * 1000.0
            return self._make_result(
                action=Action.REJECT,
                rule_id=_RULE_SEVERE_OOD,
                reasoning=(
                    f"REJECT: OOD level is SEVERE (Mahalanobis distance "
                    f"{ood.mahalanobis_distance:.4f}). "
                    "Image is likely out-of-distribution; human review is required. "
                    "No clinical claim is made."
                ),
                reliability_label="rejected_severe_ood",
                driving_signals=driving_signals,
                threshold_margins=threshold_margins,
                state=state,
                latency_ms=t_ms,
            )

        # Rule 2 — BORDERLINE OOD or borderline_margin breach → ESCALATE / REJECT
        borderline_fired, borderline_reason = self._check_borderline(
            quality, ood, uncertainty
        )
        if borderline_fired:
            borderline_action = self._borderline_action()
            t_ms = (time.perf_counter() - t_start) * 1000.0
            return self._make_result(
                action=borderline_action,
                rule_id=_RULE_BORDERLINE,
                reasoning=(
                    f"{borderline_action.value.upper()}: {borderline_reason}. "
                    "PRD safety default: signals near decision boundaries "
                    "never silently produce ACCEPT."
                ),
                reliability_label=f"borderline_{borderline_action.value}",
                driving_signals=driving_signals,
                threshold_margins=threshold_margins,
                state=state,
                latency_ms=t_ms,
            )

        # Rule 3 — POOR or DEGRADED quality + IN_DISTRIBUTION → REPAIR
        # (Safety: if OOD status is unknown / unexpected, this rule does not fire)
        if (
            q_level in (QualityLevel.POOR, QualityLevel.DEGRADED)
            and ood_level == OODLevel.IN_DISTRIBUTION
        ):
            t_ms = (time.perf_counter() - t_start) * 1000.0
            q_word = q_level.value.upper()
            return self._make_result(
                action=Action.REPAIR,
                rule_id=_RULE_REPAIR,
                reasoning=(
                    f"REPAIR: image quality is {q_word} while OOD status is "
                    "IN_DISTRIBUTION; image repair is recommended before re-evaluation. "
                    "The Decision Agent does not perform repair — a downstream Repair Agent "
                    "will process the image."
                ),
                reliability_label="repair_recommended",
                driving_signals=driving_signals,
                threshold_margins=threshold_margins,
                state=state,
                latency_ms=t_ms,
            )

        # Rules 4-6 apply only when quality is GOOD and OOD is IN_DISTRIBUTION.
        # Any other combination falls through to the safe default (Rule 7).

        if q_level == QualityLevel.GOOD and ood_level == OODLevel.IN_DISTRIBUTION:

            # Rule 4 — HIGH uncertainty → ESCALATE
            if u_level == UncertaintyLevel.HIGH:
                t_ms = (time.perf_counter() - t_start) * 1000.0
                return self._make_result(
                    action=Action.ESCALATE,
                    rule_id=_RULE_ESCALATE_HIGH,
                    reasoning=(
                        f"ESCALATE: image quality is GOOD, OOD status is IN_DISTRIBUTION, "
                        f"but uncertainty is HIGH (confidence {uncertainty.confidence:.3f}). "
                        "Human or escalation-model review is recommended."
                    ),
                    reliability_label="escalated_high_uncertainty",
                    driving_signals=driving_signals,
                    threshold_margins=threshold_margins,
                    state=state,
                    latency_ms=t_ms,
                )


            # Rule 6 — LOW uncertainty → ACCEPT
            if u_level == UncertaintyLevel.LOW:
                t_ms = (time.perf_counter() - t_start) * 1000.0
                return self._make_result(
                    action=Action.ACCEPT,
                    rule_id=_RULE_ACCEPT,
                    reasoning=(
                        f"ACCEPT: image quality is GOOD, OOD status is IN_DISTRIBUTION, "
                        f"and uncertainty is LOW (confidence {uncertainty.confidence:.3f}). "
                        "This reflects model reliability signals; it does not guarantee "
                        "diagnostic correctness."
                    ),
                    reliability_label="accepted",
                    driving_signals=driving_signals,
                    threshold_margins=threshold_margins,
                    state=state,
                    latency_ms=t_ms,
                )

        # Rule 7 — Safe default: any ambiguous / unhandled combination → ESCALATE
        t_ms = (time.perf_counter() - t_start) * 1000.0
        return self._make_result(
            action=Action.ESCALATE,
            rule_id=_RULE_SAFE_DEFAULT,
            reasoning=(
                f"ESCALATE: signal combination is ambiguous or unhandled "
                f"(quality={q_level.value}, OOD={ood_level.value}, "
                f"uncertainty={u_level.value}). "
                "PRD safety default applied: never silent ACCEPT for unsupported states."
            ),
            reliability_label="escalated_safe_default",
            driving_signals=driving_signals,
            threshold_margins=threshold_margins,
            state=state,
            latency_ms=t_ms,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        quality: QualityResult,
        ood: OODResult,
        uncertainty: UncertaintyResult,
        state: DecisionState,
    ) -> str | None:
        """
        Validate all inputs.  Return an error string if invalid, else None.
        Strictly typed checks; does not raise — error propagates to safe default.
        """
        if not isinstance(quality, QualityResult):
            return f"quality must be QualityResult, got {type(quality).__name__}"
        if not isinstance(ood, OODResult):
            return f"ood must be OODResult, got {type(ood).__name__}"
        if not isinstance(uncertainty, UncertaintyResult):
            return f"uncertainty must be UncertaintyResult, got {type(uncertainty).__name__}"
        if not isinstance(state, DecisionState):
            return f"state must be DecisionState, got {type(state).__name__}"

        # Check that quality.overall is a valid QualityLevel
        if not isinstance(quality.overall, QualityLevel):
            return f"quality.overall is not a valid QualityLevel: {quality.overall!r}"

        # Check that ood.level is a valid OODLevel
        if not isinstance(ood.level, OODLevel):
            return f"ood.level is not a valid OODLevel: {ood.level!r}"

        # Check that uncertainty.uncertainty_level is a valid UncertaintyLevel
        if not isinstance(uncertainty.uncertainty_level, UncertaintyLevel):
            return (
                "uncertainty.uncertainty_level is not a valid UncertaintyLevel: "
                f"{uncertainty.uncertainty_level!r}"
            )

        # Sanity-check confidence range
        if not (0.5 <= uncertainty.confidence <= 1.0):
            return (
                f"uncertainty.confidence out of expected range [0.5, 1.0]: "
                f"{uncertainty.confidence}"
            )

        return None

    def _check_borderline(
        self,
        quality: QualityResult,
        ood: OODResult,
        uncertainty: UncertaintyResult,
    ) -> tuple[bool, str]:
        """
        Check whether any signal is in the borderline zone or whether OOD is BORDERLINE.

        Returns (fired, reason_string).
        """
        reasons: list[str] = []

        # OOD BORDERLINE is always treated as borderline (no margin needed)
        if ood.level == OODLevel.BORDERLINE:
            reasons.append(
                f"OOD level is BORDERLINE (Mahalanobis distance "
                f"{ood.mahalanobis_distance:.4f})"
            )

        # Apply borderline_margin if calibrated
        margin = self.thresholds.borderline_margin
        if margin is not None and margin > 0:
            # Quality near-threshold flag (already computed by Quality Agent)
            if quality.near_threshold:
                reasons.append("quality signal is near its threshold")

        return (len(reasons) > 0), "; ".join(reasons) if reasons else ""

    def _borderline_action(self) -> Action:
        """Return ESCALATE or REJECT based on execution config."""
        if self.execution.borderline_action == "reject":
            return Action.REJECT
        return Action.ESCALATE

    def _compute_margins(
        self,
        quality: QualityResult,
        ood: OODResult,
        uncertainty: UncertaintyResult,
    ) -> dict[str, float]:
        """
        Compute signed margin of each signal to its decision threshold.
        Positive margin = safely within threshold.  Negative = breached.
        Returns an empty dict if thresholds are not yet calibrated.
        """
        margins: dict[str, float] = {}

        # OOD Mahalanobis margin
        if ood.mahalanobis_threshold is not None:
            margins["mahalanobis_to_borderline"] = (
                ood.mahalanobis_threshold - ood.mahalanobis_distance
            )

        # Uncertainty confidence margin
        # We report distance from the HIGH boundary (confidence_low_threshold)
        # A positive value means confidence is above the "low" threshold.
        # We don't have direct access to the estimator thresholds here, so we
        # record what we know from the result.
        margins["confidence_score"] = float(uncertainty.confidence)
        margins["normalized_entropy"] = float(uncertainty.normalized_entropy)

        return margins

    def _make_result(
        self,
        action: Action,
        rule_id: str,
        reasoning: str,
        reliability_label: str,
        driving_signals: dict[str, str | float | bool],
        threshold_margins: dict[str, float],
        state: DecisionState,
        latency_ms: float,
    ) -> DecisionResult:
        """Construct a fully validated DecisionResult.

        If ``state`` is None (which can happen on the invalid-input path), a
        default DecisionState() is substituted so that the safe-default
        ESCALATE result can always be constructed without a secondary exception.
        """
        safe_state = state if isinstance(state, DecisionState) else DecisionState()
        return DecisionResult(
            agent=self.name,
            version=self.version,
            score=None,           # No scalar score for the decision action
            label=action.value,
            reasoning=reasoning,
            latency_ms=latency_ms,
            thresholds_version=self.thresholds_version,
            action=action,
            rule_id=rule_id,
            driving_signals=driving_signals,
            threshold_margins=threshold_margins,
            reliability_label=reliability_label,
            state=safe_state,
        )
