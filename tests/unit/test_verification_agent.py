"""tests/unit/test_verification_agent.py

Comprehensive unit tests for the Verification Agent (Phase 10 — PRD FR-7).

Test Coverage:
    1.  Confidence gain at or above margin (>= 0.15) verifies repair (next_step=RELEASE).
    2.  Confidence gain below margin (< 0.15) does not verify (next_step=ESCALATE).
    3.  Label flip reported when predicted binary label changes.
    4.  Label flip reported as False when predicted label remains unchanged.
    5.  Unverified repair with no escalation left rejects (next_step=REJECT).
    6.  Action gating: ACCEPT action results in NOT_APPLICABLE.
    7.  Action gating: ESCALATE action results in NOT_APPLICABLE.
    8.  Action gating: REJECT action results in NOT_APPLICABLE.
    9.  Severe OOD after repair fails verification (ESCALATE / REJECT).
    10. OOD level worsening (IN_DISTRIBUTION -> BORDERLINE) fails verification.
    11. OOD level worsening (BORDERLINE -> SEVERE) fails verification.
    12. Overall quality worsening (GOOD -> POOR) fails verification.
    13. Severe noise increase (SNR drop > 5 dB) fails verification.
    14. Quality metric deltas accurately computed with directionality.
    15. Uncertainty comparison (entropy delta and uncertainty level recorded).
    16. Authoritative contracts as primary source in SignalBundle.
    17. Legacy scalar SignalBundle backwards compatibility.
    18. Direct keyword arguments input support.
    19. Missing before bundle safely defaults to ESCALATE / REJECT.
    20. Missing after bundle safely defaults to ESCALATE / REJECT.
    21. Contradictory evidence: high confidence gain but severe OOD fails.
    22. Contradictory evidence: high confidence gain but quality drops to POOR fails.
    23. Configurable provisional development threshold behavior (custom min_confidence_gain).
    24. Threshold marked as provisional in contract audit.
    25. Determinism across repeated runs.
    26. Strict contract validation (unknown fields rejected, reasoning required).
    27. Human-readable reasoning without clinical claims.
    28. Read-only image handling: input images remain bit-identical and unmutated.
    29. Verification Agent does not perform image repair or enhancement.
    30. Edge case: identical before and after states.
    31. Edge case: exact margin boundary (gain == 0.15).
    32. Edge case: repair was not applied (repair.repair_applied=False) fails verification.
    33. Separate quality_before and quality_after objects accessible in result.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from cxr_reliability.agents.verification import VerificationAgent
from cxr_reliability.config.thresholds import VerificationThresholds
from cxr_reliability.contracts.base_model import BaseModelResult
from cxr_reliability.contracts.common import AgentName
from cxr_reliability.contracts.decision import Action
from cxr_reliability.contracts.ood import OODLevel, OODResult
from cxr_reliability.contracts.quality import DefectType, QualityFlags, QualityLevel, QualityResult
from cxr_reliability.contracts.repair import RepairResult, RepairStep
from cxr_reliability.contracts.uncertainty import UncertaintyLevel, UncertaintyResult
from cxr_reliability.contracts.verification import (
    NextStep,
    SignalBundle,
    VerificationResult,
    VerificationStatus,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_quality_before() -> QualityResult:
    return QualityResult(
        agent=AgentName.QUALITY,
        version="0.7.0",
        label="poor",
        reasoning="Degraded blur and noise.",
        laplacian_variance=50.0,
        blur_pct=70.0,
        snr_db=12.0,
        mean_intensity=100.0,
        histogram_std=30.0,
        flags=QualityFlags(blur=True, noise=True, exposure=False),
        overall=QualityLevel.POOR,
    )


@pytest.fixture
def base_quality_after_good() -> QualityResult:
    return QualityResult(
        agent=AgentName.QUALITY,
        version="0.7.0",
        label="good",
        reasoning="Acceptable quality after filtering.",
        laplacian_variance=120.0,
        blur_pct=25.0,
        snr_db=18.0,
        mean_intensity=105.0,
        histogram_std=32.0,
        flags=QualityFlags(blur=False, noise=False, exposure=False),
        overall=QualityLevel.GOOD,
    )


@pytest.fixture
def base_ood_in_distribution() -> OODResult:
    return OODResult(
        version="0.5.0",
        label="in_distribution",
        reasoning="Feature vector within reference distribution.",
        mahalanobis_distance=8.5,
        level=OODLevel.IN_DISTRIBUTION,
    )


@pytest.fixture
def base_ood_severe() -> OODResult:
    return OODResult(
        version="0.5.0",
        label="severe",
        reasoning="Severe out-of-distribution feature distance.",
        mahalanobis_distance=35.0,
        level=OODLevel.SEVERE,
    )


@pytest.fixture
def base_model_before() -> BaseModelResult:
    return BaseModelResult(
        version="0.3.0",
        label="negative",
        reasoning="Model inference.",
        model_id="densenet121",
        pneumonia_logit=-0.4,
        pneumonia_probability=0.40,
    )


@pytest.fixture
def base_model_before_positive() -> BaseModelResult:
    return BaseModelResult(
        version="0.3.0",
        label="positive",
        reasoning="Model inference.",
        model_id="densenet121",
        pneumonia_logit=0.4,
        pneumonia_probability=0.60,
    )


@pytest.fixture
def base_model_after_gain() -> BaseModelResult:
    return BaseModelResult(
        version="0.3.0",
        label="positive",
        reasoning="Model inference post repair.",
        model_id="densenet121",
        pneumonia_logit=1.4,
        pneumonia_probability=0.80,
    )


@pytest.fixture
def uncertainty_before() -> UncertaintyResult:
    return UncertaintyResult(
        version="0.6.0",
        label="HIGH",
        reasoning="Moderate predictive entropy.",
        raw_model_score=0.40,
        confidence=0.60,
        entropy=0.67,
        uncertainty_level=UncertaintyLevel.HIGH,
    )


@pytest.fixture
def uncertainty_after_gain() -> UncertaintyResult:
    return UncertaintyResult(
        version="0.6.0",
        label="LOW",
        reasoning="Low predictive entropy; high confidence.",
        raw_model_score=0.80,
        confidence=0.80,
        entropy=0.50,
        uncertainty_level=UncertaintyLevel.LOW,
    )


@pytest.fixture
def applied_repair_result() -> RepairResult:
    return RepairResult(
        version="0.9.0",
        label="repaired",
        reasoning="Applied CLAHE and NL-means.",
        repaired=True,
        repair_applied=True,
        action=Action.REPAIR,
        steps=[
            RepairStep(defect=DefectType.NOISE, method="nl_means", parameters={"h": 3.0})
        ],
        image_changed=True,
    )


# ── PRD Specification & Verification Policy Tests ─────────────────────────────

class TestPRDSpecification:
    def test_gain_at_margin_verified(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        base_model_before_positive,
        base_model_after_gain,
        uncertainty_before,
        uncertainty_after_gain,
        applied_repair_result,
    ):
        """Quality improved (Poor->Good), no label flip, confidence non-degraded verifies."""
        agent = VerificationAgent()
        before_bundle = SignalBundle(
            quality=base_quality_before,
            ood=base_ood_in_distribution,
            base_model=base_model_before_positive,
            uncertainty=uncertainty_before,
        )
        after_bundle = SignalBundle(
            quality=base_quality_after_good,
            ood=base_ood_in_distribution,
            base_model=base_model_after_gain,
            uncertainty=uncertainty_after_gain,
        )

        res = agent.run(
            before=before_bundle,
            after=after_bundle,
            can_escalate_further=True,
            action=Action.REPAIR,
            repair=applied_repair_result,
        )

        assert res.verified is True
        assert res.status == VerificationStatus.VERIFIED
        assert res.next_step == NextStep.RELEASE
        assert pytest.approx(res.delta_confidence, 0.001) == 0.20
        assert res.min_confidence_gain_used == -0.01
        assert res.threshold_is_provisional is True
        assert "VERIFIED" in res.reasoning

    def test_confidence_drop_exceeding_guard_escalates(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Confidence drop exceeding provisional guard (0.65 - 0.70 = -0.05 < -0.01) escalates."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.70)
        u_after = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.65)

        before_bundle = SignalBundle(
            quality=base_quality_before,
            ood=base_ood_in_distribution,
            uncertainty=u_before,
        )
        after_bundle = SignalBundle(
            quality=base_quality_after_good,
            ood=base_ood_in_distribution,
            uncertainty=u_after,
        )

        res = agent.run(
            before=before_bundle,
            after=after_bundle,
            can_escalate_further=True,
            action=Action.REPAIR,
            repair=applied_repair_result,
        )

        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE
        assert res.next_step == NextStep.ESCALATE
        assert pytest.approx(res.delta_confidence, 0.001) == -0.05
        assert "ESCALATE" in res.reasoning
        assert "Confidence degraded beyond tolerance" in res.reasoning

    def test_label_flip_reported(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        base_model_before,
        base_model_after_gain,
        uncertainty_before,
        uncertainty_after_gain,
        applied_repair_result,
    ):
        """A predicted-label change (prob 0.40 -> 0.80 flips binary label) is surfaced and fails verification."""
        agent = VerificationAgent()
        before_bundle = SignalBundle(
            quality=base_quality_before,
            ood=base_ood_in_distribution,
            base_model=base_model_before,  # prob = 0.40 (< 0.50)
            uncertainty=uncertainty_before,
        )
        after_bundle = SignalBundle(
            quality=base_quality_after_good,
            ood=base_ood_in_distribution,
            base_model=base_model_after_gain,  # prob = 0.80 (>= 0.50)
            uncertainty=uncertainty_after_gain,
        )

        res = agent.run(
            before=before_bundle,
            after=after_bundle,
            action=Action.REPAIR,
            repair=applied_repair_result,
        )
        assert res.label_flipped is True
        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE
        assert "flipped" in res.reasoning

    def test_label_flip_false_when_label_same(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Label flip is False when prediction stays on the same side of 0.50."""
        agent = VerificationAgent()
        bm_before = BaseModelResult(
            version="0.3.0", label="positive", reasoning="Positive pred.", model_id="d121", pneumonia_logit=0.4, pneumonia_probability=0.60
        )
        bm_after = BaseModelResult(
            version="0.3.0", label="positive", reasoning="Positive pred.", model_id="d121", pneumonia_logit=1.5, pneumonia_probability=0.85
        )
        u_before = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="Low uncertainty.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, base_model=bm_before, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, base_model=bm_after, uncertainty=u_after)

        res = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)
        assert res.label_flipped is False
        assert res.verified is True

    def test_unverified_with_no_escalation_left_rejects(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Unverified repair when can_escalate_further=False results in next_step=REJECT."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.70)
        u_after = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.65)  # drop -0.05 < -0.01

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(
            before=before_bundle,
            after=after_bundle,
            can_escalate_further=False,
            action=Action.REPAIR,
            repair=applied_repair_result,
        )

        assert res.verified is False
        assert res.status == VerificationStatus.REJECT
        assert res.next_step == NextStep.REJECT
        assert "REJECT" in res.reasoning


# ── Action Gating Tests ───────────────────────────────────────────────────────

class TestActionGating:
    @pytest.mark.parametrize("action", [Action.ACCEPT, Action.ESCALATE, Action.REJECT])
    def test_non_repair_action_returns_not_applicable(
        self,
        action,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
    ):
        """Non-repair actions skip verification and return NOT_APPLICABLE."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="LOW", reasoning="High confidence.", confidence=0.85)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="High confidence.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(
            before=before_bundle,
            after=after_bundle,
            action=action,
        )

        assert res.verified is False
        assert res.status == VerificationStatus.NOT_APPLICABLE
        assert res.next_step == NextStep.NOT_APPLICABLE
        assert res.label == "not_applicable"
        assert action.value.upper() in res.reasoning


# ── OOD Degradation Tests ─────────────────────────────────────────────────────

class TestOODDegradation:
    def test_severe_ood_after_repair_fails(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        base_ood_severe,
        applied_repair_result,
    ):
        """If post-repair OOD status is SEVERE, verification strictly fails."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="MEDIUM", reasoning="Moderate entropy.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="High confidence.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_severe, uncertainty=u_after)

        res = agent.run(
            before=before_bundle,
            after=after_bundle,
            can_escalate_further=True,
            action=Action.REPAIR,
            repair=applied_repair_result,
        )

        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE
        assert "SEVERE" in res.reasoning

    def test_ood_degradation_in_distribution_to_borderline_fails(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """OOD degrading from IN_DISTRIBUTION to BORDERLINE fails verification."""
        agent = VerificationAgent()
        ood_borderline = OODResult(
            version="0.5.0", label="borderline", reasoning="Borderline distance.", mahalanobis_distance=15.0, level=OODLevel.BORDERLINE
        )
        u_before = UncertaintyResult(version="0.6.0", label="MEDIUM", reasoning="Moderate entropy.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="High confidence.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=ood_borderline, uncertainty=u_after)

        res = agent.run(
            before=before_bundle,
            after=after_bundle,
            action=Action.REPAIR,
            repair=applied_repair_result,
        )

        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE
        assert "BORDERLINE" in res.reasoning


# ── Quality Degradation & Deltas Tests ─────────────────────────────────────────

class TestQualityDegradationAndDeltas:
    def test_quality_degrades_good_to_poor_fails(
        self,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Quality degrading from GOOD to POOR fails verification despite confidence gain."""
        agent = VerificationAgent()
        quality_poor_after = QualityResult(
            agent=AgentName.QUALITY,
            version="0.7.0",
            label="poor",
            reasoning="Image severely degraded.",
            laplacian_variance=20.0,
            blur_pct=95.0,
            snr_db=5.0,
            mean_intensity=10.0,
            histogram_std=5.0,
            flags=QualityFlags(blur=True, noise=True, exposure=True),
            overall=QualityLevel.POOR,
        )
        u_before = UncertaintyResult(version="0.6.0", label="MEDIUM", reasoning="Moderate entropy.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="High confidence.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=quality_poor_after, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(
            before=before_bundle,
            after=after_bundle,
            action=Action.REPAIR,
            repair=applied_repair_result,
        )

        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE
        assert "POOR" in res.reasoning

    def test_severe_noise_increase_fails(
        self,
        base_quality_before,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Severe noise increase (SNR drop > 5 dB) fails verification."""
        agent = VerificationAgent()
        noisy_after = QualityResult(
            agent=AgentName.QUALITY,
            version="0.7.0",
            label="poor",
            reasoning="Severe noise.",
            laplacian_variance=60.0,
            blur_pct=60.0,
            snr_db=5.0,
            mean_intensity=100.0,
            histogram_std=30.0,
            flags=QualityFlags(blur=False, noise=True, exposure=False),
            overall=QualityLevel.POOR,
        )
        u_before = UncertaintyResult(version="0.6.0", label="MEDIUM", reasoning="Moderate entropy.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="High confidence.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=noisy_after, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(
            before=before_bundle,
            after=after_bundle,
            action=Action.REPAIR,
            repair=applied_repair_result,
        )

        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE

    def test_quality_metric_deltas_directionality(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Quality deltas accurately capture metric differences and direction."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="MEDIUM", reasoning="Moderate entropy.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="High confidence.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)

        # laplacian: 120.0 - 50.0 = 70.0
        assert pytest.approx(res.quality_metric_deltas["delta_laplacian_variance"], 0.1) == 70.0
        # blur_pct: 25.0 - 70.0 = -45.0 (negative indicates improvement)
        assert pytest.approx(res.quality_metric_deltas["delta_blur_pct"], 0.1) == -45.0
        # snr_db: 18.0 - 12.0 = 6.0 (positive indicates improvement)
        assert pytest.approx(res.quality_metric_deltas["delta_snr_db"], 0.1) == 6.0


# ── Uncertainty & Entropy Comparison Tests ────────────────────────────────────

class TestUncertaintyComparison:
    def test_entropy_delta_calculation(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Entropy delta correctly calculated (after - before)."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="MEDIUM", reasoning="Moderate entropy.", confidence=0.60, entropy=0.67)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="Low entropy.", confidence=0.85, entropy=0.45)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)
        # 0.45 - 0.67 = -0.22 (reduction in uncertainty)
        assert res.delta_entropy is not None
        assert pytest.approx(res.delta_entropy, 0.01) == -0.22


# ── Authoritative Contract and Backward Compatibility Tests ───────────────────

class TestAuthoritativeContractsAndCompatibility:
    def test_direct_keyword_arguments_supported(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        base_model_before_positive,
        base_model_after_gain,
        uncertainty_before,
        uncertainty_after_gain,
        applied_repair_result,
    ):
        """VerificationAgent can be called with direct keyword arguments."""
        agent = VerificationAgent()
        res = agent.run(
            quality_before=base_quality_before,
            quality_after=base_quality_after_good,
            ood_before=base_ood_in_distribution,
            ood_after=base_ood_in_distribution,
            base_model_before=base_model_before_positive,
            base_model_after=base_model_after_gain,
            uncertainty_before=uncertainty_before,
            uncertainty_after=uncertainty_after_gain,
            action=Action.REPAIR,
            repair=applied_repair_result,
        )
        assert res.verified is True
        assert res.status == VerificationStatus.VERIFIED

    def test_legacy_scalar_signal_bundle(self, applied_repair_result):
        """SignalBundle constructed with scalar fields works seamlessly without label flip."""
        agent = VerificationAgent()
        before_bundle = SignalBundle(probability=0.60, confidence=0.60, quality_score=12.0, ood_score=8.5)
        after_bundle = SignalBundle(probability=0.80, confidence=0.80, quality_score=18.0, ood_score=8.5)

        res = agent.run(
            before=before_bundle,
            after=after_bundle,
            action=Action.REPAIR,
            repair=applied_repair_result,
        )
        assert res.verified is True
        assert pytest.approx(res.delta_confidence, 0.01) == 0.20
        assert pytest.approx(res.delta_quality, 0.1) == 6.0

    def test_quality_after_stored_in_result(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """VerificationResult retains quality_after and quality_before contracts."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="MEDIUM", reasoning="Moderate entropy.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="Low entropy.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)
        assert res.quality_before is not None
        assert res.quality_after is not None
        assert res.quality_level_before == QualityLevel.POOR
        assert res.quality_level_after == QualityLevel.GOOD


# ── Missing Evidence & Safety Default Tests ───────────────────────────────────

class TestMissingEvidenceAndSafetyDefaults:
    def test_missing_before_bundle_escalates(self, base_quality_after_good, base_ood_in_distribution):
        """Missing before bundle safely defaults to ESCALATE."""
        agent = VerificationAgent()
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution)

        res = agent.run(before=None, after=after_bundle, can_escalate_further=True)
        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE
        assert res.next_step == NextStep.ESCALATE

    def test_missing_after_bundle_escalates(self, base_quality_before, base_ood_in_distribution):
        """Missing after bundle safely defaults to ESCALATE."""
        agent = VerificationAgent()
        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution)

        res = agent.run(before=before_bundle, after=None, can_escalate_further=True)
        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE

    def test_repair_not_applied_fails_verification(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
    ):
        """If repair.repair_applied is False, verification fails."""
        agent = VerificationAgent()
        unapplied_repair = RepairResult(
            version="0.9.0",
            label="skipped",
            reasoning="Skipped repair.",
            repaired=False,
            repair_applied=False,
            action=Action.REPAIR,
            skipped_reason="No defects flagged",
        )
        u_before = UncertaintyResult(version="0.6.0", label="MEDIUM", reasoning="Moderate entropy.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="Low entropy.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(
            before=before_bundle,
            after=after_bundle,
            action=Action.REPAIR,
            repair=unapplied_repair,
        )
        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE
        assert "not applied" in res.reasoning


# ── Configurable Provisional Threshold Tests ──────────────────────────────────

class TestConfigurableProvisionalThreshold:
    def test_custom_min_confidence_gain_threshold(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Custom provisional threshold (0.25) correctly gates a gain of 0.20."""
        custom_thresholds = VerificationThresholds(min_confidence_gain=0.25)
        agent = VerificationAgent(thresholds=custom_thresholds)

        u_before = UncertaintyResult(version="0.6.0", label="MEDIUM", reasoning="Moderate entropy.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="Low entropy.", confidence=0.80)  # gain 0.20 < 0.25

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)
        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE
        assert res.min_confidence_gain_used == 0.25

    def test_exact_margin_boundary_verifies(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Confidence delta exactly equal to guard (-0.01) verifies; delta < -0.01 escalates."""
        agent = VerificationAgent()  # default guard -0.01
        u_before = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.70)
        u_after_exact = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.69)  # delta = -0.01

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle_exact = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after_exact)

        res_exact = agent.run(before=before_bundle, after=after_bundle_exact, action=Action.REPAIR, repair=applied_repair_result)
        assert res_exact.verified is True
        assert res_exact.status == VerificationStatus.VERIFIED

        # Drop just below guard: delta -0.011 < -0.01
        u_after_below = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.689)
        after_bundle_below = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after_below)
        res_below = agent.run(before=before_bundle, after=after_bundle_below, action=Action.REPAIR, repair=applied_repair_result)
        assert res_below.verified is False
        assert res_below.status == VerificationStatus.ESCALATE


# ── Read-Only Image Safety Tests ──────────────────────────────────────────────

class TestReadOnlyImageSafety:
    def test_input_images_remain_unmodified(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Images passed to run() are completely untouched (read-only)."""
        agent = VerificationAgent()
        img_b = np.full((32, 32), 100, dtype=np.uint8)
        img_a = np.full((32, 32), 120, dtype=np.uint8)

        img_b_copy = img_b.copy()
        img_a_copy = img_a.copy()

        u_before = UncertaintyResult(version="0.6.0", label="MEDIUM", reasoning="Moderate entropy.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="Low entropy.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after)

        agent.run(
            before=before_bundle,
            after=after_bundle,
            action=Action.REPAIR,
            repair=applied_repair_result,
            image_before=img_b,
            image_after=img_a,
        )

        assert np.array_equal(img_b, img_b_copy)
        assert np.array_equal(img_a, img_a_copy)


# ── Non-Clinical Safety & Contract Tests ───────────────────────────────────────

class TestNonClinicalSafetyAndContracts:
    def test_no_clinical_claims_in_reasoning(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Reasoning contains strictly non-clinical engineering explanations."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="MEDIUM", reasoning="Moderate entropy.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="Low entropy.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)

        forbidden_terms = ["pneumonia", "patient", "diagnos", "cure", "treatment", "guarantee"]
        reasoning_lower = res.reasoning.lower()
        for term in forbidden_terms:
            assert term not in reasoning_lower, f"Forbidden clinical term '{term}' found: {res.reasoning}"

    def test_determinism_across_repeated_runs(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Repeated evaluations on identical inputs yield bit-identical outputs."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="MEDIUM", reasoning="Moderate entropy.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="Low entropy.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after)

        res1 = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)
        res2 = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)

        assert res1.verified == res2.verified
        assert res1.status == res2.status
        assert res1.next_step == res2.next_step
        assert res1.delta_confidence == res2.delta_confidence
        assert res1.reasoning == res2.reasoning

    def test_contract_rejects_unknown_fields(self):
        """VerificationResult rejects unknown fields."""
        with pytest.raises(ValidationError):
            VerificationResult(
                agent=AgentName.VERIFICATION,
                version="0.10.0",
                label="verified",
                reasoning="Test",
                verified=True,
                delta_confidence=0.2,
                delta_quality=5.0,
                delta_ood=0.0,
                label_flipped=False,
                min_confidence_gain_used=0.15,
                next_step=NextStep.RELEASE,
                unexpected_field="invalid",
            )

    def test_contract_requires_reasoning(self):
        """VerificationResult rejects empty reasoning string."""
        with pytest.raises(ValidationError):
            VerificationResult(
                agent=AgentName.VERIFICATION,
                version="0.10.0",
                label="verified",
                reasoning="",
                verified=True,
                delta_confidence=0.2,
                delta_quality=5.0,
                delta_ood=0.0,
                label_flipped=False,
                min_confidence_gain_used=-0.01,
                next_step=NextStep.RELEASE,
            )


# ── New Verification Policy Comprehensive Tests ───────────────────────────────

class TestVerificationPolicy:
    """
    Tests for the updated Verification Policy:
    1. Poor -> Good = quality resolved (verified).
    2. Poor -> Degraded = partial improvement; not automatically fully verified (escalates).
    3. Poor -> Poor = defect not resolved (escalates).
    4. OOD must remain IN_DISTRIBUTION and must not materially worsen.
    5. label_flipped must be false.
    6. confidence_after - confidence_before >= -0.01 non-degradation guard.
    7. Raw-score uncertainty architecture preserved.
    8. Threshold explicitly marked provisional.
    """

    @pytest.fixture
    def quality_degraded(self) -> QualityResult:
        return QualityResult(
            agent=AgentName.QUALITY,
            version="0.7.0",
            label="degraded",
            reasoning="Partial blur reduction; still degraded.",
            laplacian_variance=75.0,
            blur_pct=50.0,
            snr_db=14.0,
            mean_intensity=102.0,
            histogram_std=31.0,
            flags=QualityFlags(blur=True, noise=False, exposure=False),
            overall=QualityLevel.DEGRADED,
        )

    def test_poor_to_good_quality_resolved_verifies(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Poor -> Good is quality resolved; verifies when confidence is non-degraded."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.72)
        u_after = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.72)  # delta = 0.00

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)
        assert res.verified is True
        assert res.status == VerificationStatus.VERIFIED
        assert res.next_step == NextStep.RELEASE
        assert res.threshold_is_provisional is True
        assert res.min_confidence_gain_used == -0.01

    def test_poor_to_degraded_partial_improvement_escalates(
        self,
        base_quality_before,
        quality_degraded,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Poor -> Degraded is partial improvement only; does not fully verify (escalates)."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="Low uncertainty.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=quality_degraded, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)
        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE
        assert res.next_step == NextStep.ESCALATE
        assert "partially improved from POOR to DEGRADED" in res.reasoning

    def test_poor_to_poor_defect_unresolved_escalates(
        self,
        base_quality_before,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Poor -> Poor defect not resolved; fails verification despite confidence gain."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="Low uncertainty.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)
        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE
        assert "remained POOR after repair" in res.reasoning

    def test_ood_must_remain_in_distribution_borderline_escalates(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Post-repair OOD level BORDERLINE fails verification; must remain IN_DISTRIBUTION."""
        agent = VerificationAgent()
        ood_borderline = OODResult(
            version="0.5.0", label="borderline", reasoning="Borderline distance.", mahalanobis_distance=15.0, level=OODLevel.BORDERLINE
        )
        u = UncertaintyResult(version="0.6.0", label="LOW", reasoning="Confidence high.", confidence=0.85)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=ood_borderline, uncertainty=u)

        res = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)
        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE
        assert "must remain IN_DISTRIBUTION" in res.reasoning

    def test_label_flip_fails_verification_even_with_high_confidence(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Diagnostic prediction label flip strictly fails verification."""
        agent = VerificationAgent()
        bm_before = BaseModelResult(
            version="0.3.0", label="negative", reasoning="Negative.", model_id="d121", pneumonia_logit=-0.4, pneumonia_probability=0.40
        )
        bm_after = BaseModelResult(
            version="0.3.0", label="positive", reasoning="Positive.", model_id="d121", pneumonia_logit=1.4, pneumonia_probability=0.80
        )
        u_before = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.60)
        u_after = UncertaintyResult(version="0.6.0", label="LOW", reasoning="Low uncertainty.", confidence=0.80)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, base_model=bm_before, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, base_model=bm_after, uncertainty=u_after)

        res = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)
        assert res.label_flipped is True
        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE
        assert "Prediction label flipped after repair" in res.reasoning

    def test_confidence_small_gain_passes(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Small confidence gain (+0.005) passes under non-degradation guard (previously failed under 0.15)."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.700)
        u_after = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.705)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)
        assert res.verified is True
        assert res.status == VerificationStatus.VERIFIED
        assert pytest.approx(res.delta_confidence, 0.001) == 0.005

    def test_confidence_mild_drop_within_tolerance_passes(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Mild confidence drop of -0.005 (>= -0.01 tolerance) passes verification."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.800)
        u_after = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.795)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)
        assert res.verified is True
        assert res.status == VerificationStatus.VERIFIED
        assert pytest.approx(res.delta_confidence, 0.001) == -0.005

    def test_confidence_drop_exceeding_tolerance_fails(
        self,
        base_quality_before,
        base_quality_after_good,
        base_ood_in_distribution,
        applied_repair_result,
    ):
        """Confidence drop of -0.02 (< -0.01 tolerance) fails verification."""
        agent = VerificationAgent()
        u_before = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.80)
        u_after = UncertaintyResult(version="0.6.0", label="HIGH", reasoning="High uncertainty.", confidence=0.78)

        before_bundle = SignalBundle(quality=base_quality_before, ood=base_ood_in_distribution, uncertainty=u_before)
        after_bundle = SignalBundle(quality=base_quality_after_good, ood=base_ood_in_distribution, uncertainty=u_after)

        res = agent.run(before=before_bundle, after=after_bundle, action=Action.REPAIR, repair=applied_repair_result)
        assert res.verified is False
        assert res.status == VerificationStatus.ESCALATE
        assert "Confidence degraded beyond tolerance" in res.reasoning
