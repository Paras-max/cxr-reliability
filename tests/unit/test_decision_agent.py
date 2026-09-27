"""tests/unit/test_decision_agent.py

Comprehensive unit tests for Decision Agent v1 (Phase 8).

Coverage:
  1.  GOOD + IN_DISTRIBUTION + LOW uncertainty  → ACCEPT
  2.  GOOD + IN_DISTRIBUTION + HIGH uncertainty (including old intermediate/medium range) → ESCALATE
  4.  DEGRADED + IN_DISTRIBUTION               → REPAIR
  5.  POOR + IN_DISTRIBUTION                   → REPAIR
  6.  SEVERE OOD                               → REJECT (regardless of quality/uncertainty)
  7.  OOD precedence over poor quality         → REJECT beats REPAIR
  8.  Missing / invalid inputs                 → ESCALATE (safe default)
  9.  Contradictory / ambiguous signals        → ESCALATE (safe default)
  10. Human-readable reasoning completeness
  11. Determinism across repeated calls
  12. DecisionResult contract validation
  13. No image processing inside Decision Agent
  14. Borderline OOD (BORDERLINE level)        → ESCALATE
  15. borderline_action=reject config          → REJECT on borderline
  16. Exhaustive signal-grid returns valid Action
  17. POOR quality + SEVERE OOD               → REJECT (OOD rule 1 takes precedence)
  18. Rule IDs are stable strings
  19. Driving signals are fully populated
  20. Latency field is populated and non-negative

Two-Level uncertainty rationale:
    Uncertainty is evaluated in 2 levels: LOW and HIGH.
    LOW uncertainty (confidence >= 0.85 and normalized entropy <= 0.25) routes to ACCEPT.
    HIGH uncertainty (everything else, including borderline and intermediate confidence)
    routes to ESCALATE. There is no active MEDIUM state.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cxr_reliability.agents.decision.rules import (
    RuleTableDecisionAgent,
    _RULE_ACCEPT,
    _RULE_BORDERLINE,
    _RULE_ESCALATE_HIGH,
    _RULE_ESCALATE_MEDIUM,
    _RULE_INVALID_INPUT,
    _RULE_REPAIR,
    _RULE_SAFE_DEFAULT,
    _RULE_SEVERE_OOD,
)
from cxr_reliability.config.pipeline_config import ExecutionConfig
from cxr_reliability.config.thresholds import DecisionThresholds
from cxr_reliability.contracts.common import AgentName
from cxr_reliability.contracts.decision import Action, DecisionResult, DecisionState
from cxr_reliability.contracts.ood import OODLevel, OODResult
from cxr_reliability.contracts.quality import QualityFlags, QualityLevel, QualityResult
from cxr_reliability.contracts.uncertainty import UncertaintyLevel, UncertaintyResult


# ── Helpers / factories ────────────────────────────────────────────────────────


def make_agent(
    borderline_margin: float | None = None,
    borderline_action: str = "escalate",
) -> RuleTableDecisionAgent:
    """Construct an agent with minimal, reproducible configuration."""
    thresholds = DecisionThresholds(borderline_margin=borderline_margin)
    execution = ExecutionConfig(borderline_action=borderline_action)  # type: ignore[arg-type]
    return RuleTableDecisionAgent(
        thresholds=thresholds,
        execution=execution,
        thresholds_version="v0_test",
    )


def make_quality(
    overall: QualityLevel = QualityLevel.GOOD,
    near_threshold: bool = False,
    blur: bool = False,
    noise: bool = False,
    exposure: bool = False,
) -> QualityResult:
    """Construct a minimal QualityResult for testing."""
    return QualityResult(
        version="0.7.0",
        label=overall.value,
        reasoning=f"Quality is {overall.value}.",
        laplacian_variance=200.0,
        blur_pct=5.0,
        snr_db=20.0,
        mean_intensity=128.0,
        histogram_std=40.0,
        flags=QualityFlags(blur=blur, noise=noise, exposure=exposure),
        overall=overall,
        near_threshold=near_threshold,
    )


def make_ood(
    level: OODLevel = OODLevel.IN_DISTRIBUTION,
    mahalanobis_distance: float = 10.0,
    mahalanobis_threshold: float | None = 50.0,
) -> OODResult:
    """Construct a minimal OODResult for testing."""
    return OODResult(
        version="0.5.0",
        label=level.value,
        reasoning=f"OOD level is {level.value}.",
        mahalanobis_distance=mahalanobis_distance,
        mahalanobis_threshold=mahalanobis_threshold,
        level=level,
    )


def make_uncertainty(
    uncertainty_level: UncertaintyLevel = UncertaintyLevel.LOW,
    confidence: float = 0.90,
    normalized_entropy: float = 0.10,
) -> UncertaintyResult:
    """Construct a minimal UncertaintyResult for testing."""
    return UncertaintyResult(
        version="0.6.0",
        label=uncertainty_level.value,
        reasoning=f"Uncertainty is {uncertainty_level.value}.",
        confidence=confidence,
        normalized_entropy=normalized_entropy,
        uncertainty_level=uncertainty_level,
    )


_DEFAULT_STATE = DecisionState()


# ── Test 1: GOOD + IN_DISTRIBUTION + LOW uncertainty → ACCEPT ─────────────────


class TestAcceptRule:
    def test_good_in_distribution_low_uncertainty_accept(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW, confidence=0.92),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.ACCEPT
        assert result.rule_id == _RULE_ACCEPT

    def test_accept_result_is_decision_result_instance(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW, confidence=0.95),
            state=_DEFAULT_STATE,
        )
        assert isinstance(result, DecisionResult)
        assert result.agent == AgentName.DECISION

    def test_accept_label_matches_action(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW),
            state=_DEFAULT_STATE,
        )
        assert result.label == Action.ACCEPT.value

    def test_accept_reasoning_mentions_signals(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW),
            state=_DEFAULT_STATE,
        )
        r = result.reasoning.upper()
        assert "ACCEPT" in r
        assert "GOOD" in r
        assert "IN_DISTRIBUTION" in r
        assert "LOW" in r

    def test_accept_reasoning_no_clinical_claims(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW),
            state=_DEFAULT_STATE,
        )
        forbidden = ["diagnosis is reliable", "patient has pneumonia", "guarantees correctness"]
        r = result.reasoning.lower()
        for phrase in forbidden:
            assert phrase not in r, f"Forbidden clinical phrase found: {phrase!r}"


# ── Test 2: GOOD + IN_DISTRIBUTION + HIGH uncertainty (incl. old MEDIUM range) → ESCALATE ─


class TestHighUncertaintyDecisionRouting:
    """
    HIGH uncertainty maps to ESCALATE when Quality is GOOD and OOD is IN_DISTRIBUTION.
    This includes samples in the old MEDIUM band (e.g. confidence = 0.72), which now
    evaluate to HIGH uncertainty and route to ESCALATE via _RULE_ESCALATE_HIGH.
    """

    def test_good_in_distribution_intermediate_confidence_escalates_via_high_rule(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.HIGH, confidence=0.72),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.ESCALATE
        assert result.rule_id == _RULE_ESCALATE_HIGH

    def test_good_in_distribution_low_uncertainty_accepts(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW, confidence=0.92),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.ACCEPT
        assert result.rule_id == _RULE_ACCEPT

    def test_high_uncertainty_never_produces_accept(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.HIGH, confidence=0.75),
            state=_DEFAULT_STATE,
        )
        assert result.action != Action.ACCEPT


# ── Test 3: GOOD + IN_DISTRIBUTION + HIGH uncertainty → ESCALATE ─────────────


class TestHighUncertaintyEscalate:
    def test_good_in_distribution_high_uncertainty_escalate(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.HIGH, confidence=0.55),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.ESCALATE
        assert result.rule_id == _RULE_ESCALATE_HIGH

    def test_high_uncertainty_reasoning_contains_high(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.HIGH, confidence=0.55),
            state=_DEFAULT_STATE,
        )
        assert "HIGH" in result.reasoning.upper()
        assert "ESCALATE" in result.reasoning.upper()

    def test_high_uncertainty_never_accept(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.HIGH, confidence=0.51),
            state=_DEFAULT_STATE,
        )
        assert result.action != Action.ACCEPT


# ── Test 4: DEGRADED + IN_DISTRIBUTION → REPAIR ──────────────────────────────


class TestDegradedQualityRepair:
    def test_degraded_in_distribution_repair(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.DEGRADED, blur=True),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.HIGH, confidence=0.55),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.REPAIR
        assert result.rule_id == _RULE_REPAIR

    def test_degraded_repair_reasoning_mentions_degraded(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.DEGRADED),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW),
            state=_DEFAULT_STATE,
        )
        r = result.reasoning.upper()
        assert "DEGRADED" in r
        assert "REPAIR" in r

    def test_degraded_repair_reasoning_no_image_modification_claim(self):
        """Decision Agent must not claim to perform repair itself."""
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.DEGRADED),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.HIGH),
            state=_DEFAULT_STATE,
        )
        r = result.reasoning.lower()
        assert "does not perform repair" in r or "downstream" in r or "repair agent" in r


# ── Test 5: POOR + IN_DISTRIBUTION → REPAIR ──────────────────────────────────


class TestPoorQualityRepair:
    def test_poor_in_distribution_repair(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.POOR, blur=True, noise=True),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.HIGH),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.REPAIR
        assert result.rule_id == _RULE_REPAIR

    def test_poor_repair_reasoning_mentions_poor(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.POOR),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW),
            state=_DEFAULT_STATE,
        )
        r = result.reasoning.upper()
        assert "POOR" in r
        assert "REPAIR" in r


# ── Test 6: SEVERE OOD → REJECT ───────────────────────────────────────────────


class TestSevereOODReject:
    def test_severe_ood_good_quality_low_uncertainty_reject(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.SEVERE, mahalanobis_distance=999.9),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW, confidence=0.95),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.REJECT
        assert result.rule_id == _RULE_SEVERE_OOD

    def test_severe_ood_any_quality_rejects(self):
        """SEVERE OOD rejects regardless of quality level."""
        agent = make_agent()
        for q_level in QualityLevel:
            result = agent.decide(
                quality=make_quality(q_level),
                ood=make_ood(OODLevel.SEVERE, mahalanobis_distance=500.0),
                uncertainty=make_uncertainty(UncertaintyLevel.LOW, confidence=0.99),
                state=_DEFAULT_STATE,
            )
            assert result.action == Action.REJECT, (
                f"Expected REJECT for SEVERE OOD with quality={q_level.value}, "
                f"got {result.action}"
            )

    def test_severe_ood_any_uncertainty_rejects(self):
        """SEVERE OOD rejects regardless of uncertainty level."""
        agent = make_agent()
        for u_level in UncertaintyLevel:
            conf = 0.55 if u_level == UncertaintyLevel.HIGH else 0.90
            result = agent.decide(
                quality=make_quality(QualityLevel.GOOD),
                ood=make_ood(OODLevel.SEVERE, mahalanobis_distance=500.0),
                uncertainty=make_uncertainty(u_level, confidence=conf),
                state=_DEFAULT_STATE,
            )
            assert result.action == Action.REJECT, (
                f"Expected REJECT for SEVERE OOD with uncertainty={u_level.value}, "
                f"got {result.action}"
            )

    def test_severe_ood_reasoning_mentions_severe(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.SEVERE),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW),
            state=_DEFAULT_STATE,
        )
        r = result.reasoning.upper()
        assert "SEVERE" in r
        assert "REJECT" in r


# ── Test 7: OOD precedence over poor quality ─────────────────────────────────


class TestOODPrecedence:
    def test_poor_quality_plus_severe_ood_reject_not_repair(self):
        """
        Safety precedence: SEVERE OOD (Rule 1) fires before quality-based REPAIR (Rule 3).
        """
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.POOR, blur=True),
            ood=make_ood(OODLevel.SEVERE, mahalanobis_distance=800.0),
            uncertainty=make_uncertainty(UncertaintyLevel.HIGH, confidence=0.51),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.REJECT
        assert result.rule_id == _RULE_SEVERE_OOD

    def test_degraded_quality_plus_severe_ood_reject_not_repair(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.DEGRADED),
            ood=make_ood(OODLevel.SEVERE, mahalanobis_distance=800.0),
            uncertainty=make_uncertainty(UncertaintyLevel.HIGH),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.REJECT

    def test_borderline_ood_plus_good_quality_escalates_not_accepts(self):
        """
        BORDERLINE OOD (Rule 2) fires before the ACCEPT rule (Rule 6).
        """
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.BORDERLINE, mahalanobis_distance=48.0),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW, confidence=0.95),
            state=_DEFAULT_STATE,
        )
        # Borderline OOD must not produce ACCEPT
        assert result.action != Action.ACCEPT
        assert result.rule_id == _RULE_BORDERLINE


# ── Test 8: Missing / invalid inputs → safe fallback ─────────────────────────


class TestInvalidInputsFallback:
    def test_non_quality_result_escalates(self):
        agent = make_agent()
        result = agent.decide(
            quality="not_a_quality_result",  # type: ignore[arg-type]
            ood=make_ood(),
            uncertainty=make_uncertainty(),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.ESCALATE
        assert result.rule_id == _RULE_INVALID_INPUT

    def test_none_ood_escalates(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(),
            ood=None,  # type: ignore[arg-type]
            uncertainty=make_uncertainty(),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.ESCALATE
        assert result.rule_id == _RULE_INVALID_INPUT

    def test_none_uncertainty_escalates(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(),
            ood=make_ood(),
            uncertainty=None,  # type: ignore[arg-type]
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.ESCALATE
        assert result.rule_id == _RULE_INVALID_INPUT

    def test_none_state_escalates(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(),
            ood=make_ood(),
            uncertainty=make_uncertainty(),
            state=None,  # type: ignore[arg-type]
        )
        assert result.action == Action.ESCALATE
        assert result.rule_id == _RULE_INVALID_INPUT

    def test_invalid_input_result_has_non_empty_reasoning(self):
        agent = make_agent()
        result = agent.decide(
            quality=42,  # type: ignore[arg-type]
            ood=make_ood(),
            uncertainty=make_uncertainty(),
            state=_DEFAULT_STATE,
        )
        assert len(result.reasoning) > 0

    def test_invalid_input_never_accepts(self):
        agent = make_agent()
        result = agent.decide(
            quality=None,  # type: ignore[arg-type]
            ood=None,  # type: ignore[arg-type]
            uncertainty=None,  # type: ignore[arg-type]
            state=None,  # type: ignore[arg-type]
        )
        assert result.action != Action.ACCEPT


# ── Test 9: Contradictory / ambiguous signals → safe fallback ────────────────


class TestAmbiguousSignals:
    def test_good_quality_borderline_ood_low_uncertainty_does_not_accept(self):
        """
        Even with GOOD quality and LOW uncertainty, BORDERLINE OOD must not ACCEPT.
        """
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.BORDERLINE, mahalanobis_distance=48.0),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW, confidence=0.98),
            state=_DEFAULT_STATE,
        )
        assert result.action != Action.ACCEPT

    def test_poor_quality_in_distribution_high_uncertainty_repair_not_accept(self):
        """
        POOR quality + IN_DISTRIBUTION triggers REPAIR even with HIGH uncertainty.
        """
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.POOR),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.HIGH, confidence=0.51),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.REPAIR

    def test_degraded_quality_borderline_ood_does_not_accept(self):
        """
        BORDERLINE OOD (Rule 2) fires before quality-based REPAIR (Rule 3).
        """
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.DEGRADED),
            ood=make_ood(OODLevel.BORDERLINE),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW),
            state=_DEFAULT_STATE,
        )
        # Must be ESCALATE (or REJECT), never REPAIR or ACCEPT
        assert result.action in (Action.ESCALATE, Action.REJECT)
        assert result.action != Action.ACCEPT


# ── Test 10: Human-readable reasoning ─────────────────────────────────────────


class TestReasoning:
    def test_all_paths_have_non_empty_reasoning(self):
        agent = make_agent()
        scenarios = [
            # (quality, ood, uncertainty)
            (make_quality(QualityLevel.GOOD), make_ood(OODLevel.IN_DISTRIBUTION),
             make_uncertainty(UncertaintyLevel.LOW)),
            (make_quality(QualityLevel.GOOD), make_ood(OODLevel.IN_DISTRIBUTION),
             make_uncertainty(UncertaintyLevel.HIGH, confidence=0.72)),
            (make_quality(QualityLevel.GOOD), make_ood(OODLevel.IN_DISTRIBUTION),
             make_uncertainty(UncertaintyLevel.HIGH, confidence=0.55)),
            (make_quality(QualityLevel.DEGRADED), make_ood(OODLevel.IN_DISTRIBUTION),
             make_uncertainty(UncertaintyLevel.HIGH)),
            (make_quality(QualityLevel.POOR), make_ood(OODLevel.IN_DISTRIBUTION),
             make_uncertainty(UncertaintyLevel.LOW)),
            (make_quality(QualityLevel.GOOD), make_ood(OODLevel.SEVERE, 999.0),
             make_uncertainty(UncertaintyLevel.LOW)),
            (make_quality(QualityLevel.GOOD), make_ood(OODLevel.BORDERLINE),
             make_uncertainty(UncertaintyLevel.LOW)),
        ]
        for q, o, u in scenarios:
            result = agent.decide(quality=q, ood=o, uncertainty=u, state=_DEFAULT_STATE)
            assert len(result.reasoning) >= 10, (
                f"Reasoning too short for action={result.action}: {result.reasoning!r}"
            )

    def test_reasoning_starts_with_action_word(self):
        """
        By convention, each reasoning string begins with the action name.
        """
        agent = make_agent()
        cases = [
            (QualityLevel.GOOD, OODLevel.IN_DISTRIBUTION, UncertaintyLevel.LOW, "ACCEPT"),
            (QualityLevel.GOOD, OODLevel.IN_DISTRIBUTION, UncertaintyLevel.HIGH, "ESCALATE"),
            (QualityLevel.GOOD, OODLevel.SEVERE, OODLevel.SEVERE, None),  # just check REJECT
        ]
        result_accept = agent.decide(
            make_quality(QualityLevel.GOOD),
            make_ood(OODLevel.IN_DISTRIBUTION),
            make_uncertainty(UncertaintyLevel.LOW),
            _DEFAULT_STATE,
        )
        assert result_accept.reasoning.upper().startswith("ACCEPT")

        result_reject = agent.decide(
            make_quality(QualityLevel.GOOD),
            make_ood(OODLevel.SEVERE, 999.0),
            make_uncertainty(UncertaintyLevel.LOW),
            _DEFAULT_STATE,
        )
        assert result_reject.reasoning.upper().startswith("REJECT")

        result_repair = agent.decide(
            make_quality(QualityLevel.POOR),
            make_ood(OODLevel.IN_DISTRIBUTION),
            make_uncertainty(UncertaintyLevel.LOW),
            _DEFAULT_STATE,
        )
        assert result_repair.reasoning.upper().startswith("REPAIR")


# ── Test 11: Determinism ───────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_inputs_same_output(self):
        agent = make_agent()
        q = make_quality(QualityLevel.GOOD)
        o = make_ood(OODLevel.IN_DISTRIBUTION)
        u = make_uncertainty(UncertaintyLevel.LOW)
        s = DecisionState()

        r1 = agent.decide(q, o, u, s)
        r2 = agent.decide(q, o, u, s)

        assert r1.action == r2.action
        assert r1.rule_id == r2.rule_id
        assert r1.reasoning == r2.reasoning
        assert r1.reliability_label == r2.reliability_label

    def test_repeated_calls_same_rule_id(self):
        agent = make_agent()
        q = make_quality(QualityLevel.DEGRADED)
        o = make_ood(OODLevel.IN_DISTRIBUTION)
        u = make_uncertainty(UncertaintyLevel.HIGH)
        s = DecisionState()

        rule_ids = {agent.decide(q, o, u, s).rule_id for _ in range(5)}
        assert len(rule_ids) == 1  # Always the same rule

    @pytest.mark.parametrize("u_level,conf", [
        (UncertaintyLevel.LOW, 0.92),
        (UncertaintyLevel.HIGH, 0.55),
    ])
    def test_determinism_parametrized(self, u_level, conf):
        agent = make_agent()
        q = make_quality(QualityLevel.GOOD)
        o = make_ood(OODLevel.IN_DISTRIBUTION)
        u = make_uncertainty(u_level, confidence=conf)
        s = DecisionState()

        actions = {agent.decide(q, o, u, s).action for _ in range(3)}
        assert len(actions) == 1


# ── Test 12: Contract validation ──────────────────────────────────────────────


class TestContractValidation:
    def test_result_is_pydantic_validated(self):
        agent = make_agent()
        result = agent.decide(
            make_quality(QualityLevel.GOOD),
            make_ood(OODLevel.IN_DISTRIBUTION),
            make_uncertainty(UncertaintyLevel.LOW),
            _DEFAULT_STATE,
        )
        # Verify all required DecisionResult fields are present
        assert result.agent == AgentName.DECISION
        assert isinstance(result.version, str) and len(result.version) > 0
        assert isinstance(result.action, Action)
        assert isinstance(result.rule_id, str) and len(result.rule_id) > 0
        assert isinstance(result.driving_signals, dict)
        assert isinstance(result.threshold_margins, dict)
        assert isinstance(result.reliability_label, str) and len(result.reliability_label) > 0
        assert isinstance(result.state, DecisionState)
        assert result.label == result.action.value

    def test_result_extra_fields_rejected(self):
        """DecisionResult inherits from StrictModel; extra fields must be rejected."""
        with pytest.raises((ValidationError, TypeError)):
            DecisionResult(
                agent=AgentName.DECISION,
                version="v1-0.8.0",
                label="accept",
                reasoning="test",
                action=Action.ACCEPT,
                rule_id="R6",
                reliability_label="accepted",
                state=DecisionState(),
                _surprise_field="oops",  # type: ignore[call-arg]
            )

    def test_decision_result_requires_reasoning(self):
        with pytest.raises(ValidationError):
            DecisionResult(
                agent=AgentName.DECISION,
                version="v1-0.8.0",
                label="accept",
                reasoning="",  # empty — must fail
                action=Action.ACCEPT,
                rule_id="R6",
                reliability_label="accepted",
                state=DecisionState(),
            )

    def test_state_is_preserved(self):
        state = DecisionState(repair_attempts=1, escalation_attempts=0, stage="post_repair")
        agent = make_agent()
        result = agent.decide(
            make_quality(QualityLevel.GOOD),
            make_ood(OODLevel.IN_DISTRIBUTION),
            make_uncertainty(UncertaintyLevel.LOW),
            state,
        )
        assert result.state.repair_attempts == 1
        assert result.state.stage == "post_repair"


# ── Test 13: No image processing inside Decision Agent ───────────────────────


class TestNoImageProcessing:
    def test_agent_has_no_image_modifying_methods(self):
        """
        Decision Agent must not expose image-modifying methods.
        """
        agent = make_agent()
        forbidden_attrs = [
            "modify_image", "repair_image", "process_image",
            "apply_filter", "sharpen", "denoise", "clahe",
        ]
        for attr in forbidden_attrs:
            assert not hasattr(agent, attr), (
                f"DecisionAgent must not have attribute: {attr}"
            )

    def test_decide_returns_without_touching_image_arrays(self):
        """
        Calling decide() on a result with no image data works fine.
        The Decision Agent never requires or uses pixel arrays.
        """
        agent = make_agent()
        result = agent.decide(
            make_quality(QualityLevel.POOR),
            make_ood(OODLevel.IN_DISTRIBUTION),
            make_uncertainty(UncertaintyLevel.HIGH, confidence=0.55),
            _DEFAULT_STATE,
        )
        # If the agent tried to access image data, it would fail.
        # Reaching here confirms it does not.
        assert result.action == Action.REPAIR

    def test_reasoning_does_not_claim_image_was_modified(self):
        agent = make_agent()
        result = agent.decide(
            make_quality(QualityLevel.POOR),
            make_ood(OODLevel.IN_DISTRIBUTION),
            make_uncertainty(UncertaintyLevel.LOW),
            _DEFAULT_STATE,
        )
        r = result.reasoning.lower()
        # The Decision Agent recommends repair but must not claim it was done
        assert "image was repaired" not in r
        assert "repaired the image" not in r


# ── Test 14: Borderline OOD → ESCALATE ───────────────────────────────────────


class TestBorderlineOOD:
    def test_borderline_ood_escalates_by_default(self):
        agent = make_agent(borderline_action="escalate")
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.BORDERLINE, mahalanobis_distance=48.0),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW, confidence=0.95),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.ESCALATE
        assert result.rule_id == _RULE_BORDERLINE

    def test_borderline_ood_reasoning_mentions_borderline(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.BORDERLINE, mahalanobis_distance=48.0),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW),
            state=_DEFAULT_STATE,
        )
        assert "BORDERLINE" in result.reasoning.upper()


# ── Test 15: borderline_action=reject ─────────────────────────────────────────


class TestBorderlineActionReject:
    def test_borderline_ood_rejects_when_configured(self):
        agent = make_agent(borderline_action="reject")
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.BORDERLINE, mahalanobis_distance=48.0),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW, confidence=0.95),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.REJECT
        assert result.rule_id == _RULE_BORDERLINE


# ── Test 16: Exhaustive signal grid returns valid Action ─────────────────────


class TestExhaustiveSignalGrid:
    """Every combination of (QualityLevel × OODLevel × UncertaintyLevel) returns
    exactly one Action with a non-empty rule_id."""

    _CONFIDENCE = {
        UncertaintyLevel.LOW: 0.90,
        UncertaintyLevel.HIGH: 0.55,
    }

    @pytest.mark.parametrize("q_level", list(QualityLevel))
    @pytest.mark.parametrize("ood_level", list(OODLevel))
    @pytest.mark.parametrize("u_level", list(UncertaintyLevel))
    def test_grid_returns_valid_action(self, q_level, ood_level, u_level):
        agent = make_agent()
        conf = self._CONFIDENCE[u_level]
        result = agent.decide(
            quality=make_quality(q_level),
            ood=make_ood(ood_level),
            uncertainty=make_uncertainty(u_level, confidence=conf),
            state=_DEFAULT_STATE,
        )
        assert isinstance(result.action, Action)
        assert result.rule_id and len(result.rule_id) > 0
        assert result.reasoning and len(result.reasoning) > 0

    @pytest.mark.parametrize("q_level", list(QualityLevel))
    @pytest.mark.parametrize("ood_level", list(OODLevel))
    @pytest.mark.parametrize("u_level", list(UncertaintyLevel))
    def test_grid_never_silent_accept_for_ood_not_in_distribution(
        self, q_level, ood_level, u_level
    ):
        """
        No combination with OOD ≠ IN_DISTRIBUTION should produce ACCEPT.
        """
        if ood_level == OODLevel.IN_DISTRIBUTION:
            pytest.skip("Only testing non-in-distribution OOD")
        agent = make_agent()
        conf = self._CONFIDENCE[u_level]
        result = agent.decide(
            quality=make_quality(q_level),
            ood=make_ood(ood_level),
            uncertainty=make_uncertainty(u_level, confidence=conf),
            state=_DEFAULT_STATE,
        )
        assert result.action != Action.ACCEPT, (
            f"ACCEPT must never fire when OOD level is {ood_level.value}. "
            f"Got action={result.action} with quality={q_level.value}, "
            f"uncertainty={u_level.value}."
        )


# ── Test 17: POOR quality + SEVERE OOD → REJECT ──────────────────────────────


class TestPoorQualitySevereOOD:
    def test_poor_quality_severe_ood_reject(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.POOR, blur=True, noise=True),
            ood=make_ood(OODLevel.SEVERE, mahalanobis_distance=999.0),
            uncertainty=make_uncertainty(UncertaintyLevel.HIGH, confidence=0.51),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.REJECT
        assert result.rule_id == _RULE_SEVERE_OOD

    def test_degraded_quality_borderline_ood_not_repair(self):
        """
        When quality is DEGRADED but OOD is BORDERLINE, Rule 2 (borderline) fires
        before Rule 3 (repair); result must not be REPAIR.
        """
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.DEGRADED),
            ood=make_ood(OODLevel.BORDERLINE),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW),
            state=_DEFAULT_STATE,
        )
        assert result.action != Action.REPAIR


# ── Test 18: Rule IDs are stable strings ──────────────────────────────────────


class TestRuleIDs:
    def test_rule_ids_are_strings_and_non_empty(self):
        rule_ids = [
            _RULE_SEVERE_OOD,
            _RULE_BORDERLINE,
            _RULE_REPAIR,
            _RULE_ESCALATE_HIGH,
            _RULE_ESCALATE_MEDIUM,
            _RULE_ACCEPT,
            _RULE_SAFE_DEFAULT,
            _RULE_INVALID_INPUT,
        ]
        for r in rule_ids:
            assert isinstance(r, str) and len(r) > 0

    def test_rule_ids_are_unique(self):
        rule_ids = [
            _RULE_SEVERE_OOD,
            _RULE_BORDERLINE,
            _RULE_REPAIR,
            _RULE_ESCALATE_HIGH,
            _RULE_ESCALATE_MEDIUM,
            _RULE_ACCEPT,
            _RULE_SAFE_DEFAULT,
            _RULE_INVALID_INPUT,
        ]
        assert len(rule_ids) == len(set(rule_ids))


# ── Test 19: Driving signals are fully populated ──────────────────────────────


class TestDrivingSignals:
    def test_driving_signals_include_all_three_signals(self):
        agent = make_agent()
        result = agent.decide(
            make_quality(QualityLevel.GOOD),
            make_ood(OODLevel.IN_DISTRIBUTION),
            make_uncertainty(UncertaintyLevel.LOW),
            _DEFAULT_STATE,
        )
        ds = result.driving_signals
        assert "quality_overall" in ds
        assert "ood_level" in ds
        assert "uncertainty_level" in ds

    def test_driving_signals_values_match_inputs(self):
        agent = make_agent()
        result = agent.decide(
            make_quality(QualityLevel.DEGRADED),
            make_ood(OODLevel.IN_DISTRIBUTION),
            make_uncertainty(UncertaintyLevel.HIGH, confidence=0.72),
            _DEFAULT_STATE,
        )
        ds = result.driving_signals
        assert ds["quality_overall"] == QualityLevel.DEGRADED.value
        assert ds["ood_level"] == OODLevel.IN_DISTRIBUTION.value
        assert ds["uncertainty_level"] == UncertaintyLevel.HIGH.value


# ── Test 20: Latency field ────────────────────────────────────────────────────


class TestLatency:
    def test_latency_is_non_negative(self):
        agent = make_agent()
        result = agent.decide(
            make_quality(QualityLevel.GOOD),
            make_ood(OODLevel.IN_DISTRIBUTION),
            make_uncertainty(UncertaintyLevel.LOW),
            _DEFAULT_STATE,
        )
        assert result.latency_ms is not None
        assert result.latency_ms >= 0.0

    def test_latency_is_finite_and_reasonable(self):
        """Decision is deterministic and CPU-only; should complete in < 1 second."""
        agent = make_agent()
        result = agent.decide(
            make_quality(QualityLevel.POOR),
            make_ood(OODLevel.SEVERE, 999.0),
            make_uncertainty(UncertaintyLevel.HIGH, confidence=0.51),
            _DEFAULT_STATE,
        )
        assert result.latency_ms is not None
        assert result.latency_ms < 1000.0  # < 1 second


# ── Test: attempt limits are carried through ──────────────────────────────────


class TestAttemptLimits:
    def test_state_fields_propagate_to_result(self):
        """
        The Decision Agent does not enforce attempt limits itself (the orchestrator does),
        but it must faithfully pass DecisionState through to the result.
        """
        state = DecisionState(repair_attempts=1, escalation_attempts=1, stage="post_repair")
        agent = make_agent()
        result = agent.decide(
            make_quality(QualityLevel.GOOD),
            make_ood(OODLevel.IN_DISTRIBUTION),
            make_uncertainty(UncertaintyLevel.LOW),
            state,
        )
        assert result.state.repair_attempts == 1
        assert result.state.escalation_attempts == 1
        assert result.state.stage == "post_repair"


# ── Requirement 11: Two-Level Decision Routing Verification ───────────────────

class TestTwoLevelDecisionRoutingRequirements:
    """Explicit tests verifying Requirement 11 for Decision Agent:
    - Decision Agent routes LOW to ACCEPT when quality is GOOD and OOD is IN_DISTRIBUTION
    - Decision Agent routes HIGH to ESCALATE when quality is GOOD and OOD is IN_DISTRIBUTION
    - Old intermediate confidence samples (e.g. 0.72) evaluated as HIGH route to ESCALATE
    """

    def test_low_uncertainty_routes_to_accept_when_other_conditions_satisfied(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.LOW, confidence=0.95),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.ACCEPT
        assert result.rule_id == _RULE_ACCEPT
        assert result.reliability_label == "accepted"

    def test_high_uncertainty_routes_to_escalate(self):
        agent = make_agent()
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.HIGH, confidence=0.55),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.ESCALATE
        assert result.rule_id == _RULE_ESCALATE_HIGH
        assert result.reliability_label == "escalated_high_uncertainty"

    def test_old_medium_intermediate_confidence_routes_to_escalate_via_high_rule(self):
        agent = make_agent()
        # In the old system, confidence=0.72 was MEDIUM and routed via R5_MEDIUM_UNCERTAINTY_ESCALATE.
        # Now, intermediate confidence is HIGH and routes via R4_HIGH_UNCERTAINTY_ESCALATE.
        result = agent.decide(
            quality=make_quality(QualityLevel.GOOD),
            ood=make_ood(OODLevel.IN_DISTRIBUTION),
            uncertainty=make_uncertainty(UncertaintyLevel.HIGH, confidence=0.72),
            state=_DEFAULT_STATE,
        )
        assert result.action == Action.ESCALATE
        assert result.rule_id == _RULE_ESCALATE_HIGH
        assert result.reliability_label == "escalated_high_uncertainty"

