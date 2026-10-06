"""tests/unit/test_uncertainty_agent.py

Comprehensive unit tests and integration tests for the Uncertainty Agent (Phase 6).

Test Coverage:
    1.  p = 0 entropy handling (finite, safe, 0.0)
    2.  p = 1 entropy handling (finite, safe, 0.0)
    3.  p = 0.5 maximum binary entropy (ln(2) ≈ 0.6931, normalized = 1.0)
    4.  Entropy symmetry: H(p) == H(1 - p)
    5.  Confidence calculation: max(p, 1 - p) in [0.5, 1.0]
    6.  Normalized entropy range: [0.0, 1.0]
    7.  LOW uncertainty classification (high confidence + low entropy)
    8.  Old MEDIUM region now evaluates to HIGH (2-level uncertainty)
    9.  HIGH uncertainty classification (everything not satisfying LOW)
    10. Threshold boundary behavior (exact threshold crossings)
    11. NaN handling (raises ValueError)
    12. Inf handling (raises ValueError)
    13. Invalid dimensions / shapes (raises ValueError)
    14. NumPy array input support
    15. PyTorch tensor input support
    16. Structured output contract validation (Pydantic schema compliance)
    17. Human-readable reasoning compliance (no diagnostic/clinical claims)
    18. Deterministic output across repeated calls
    19. Integration tests using synthetic BaseModelResult and ModelForward (Cases 1, 2, 3)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from cxr_reliability.agents.uncertainty import UncertaintyAgent
from cxr_reliability.contracts.base_model import BaseModelResult
from cxr_reliability.contracts.common import AgentName
from cxr_reliability.contracts.uncertainty import UncertaintyLevel, UncertaintyResult
from cxr_reliability.models.base_model import ModelForward
from cxr_reliability.uncertainty import (
    UncertaintyConfig,
    UncertaintyEstimator,
    compute_binary_confidence,
    compute_binary_entropy,
    compute_normalized_entropy,
)

# ── Test 1 & 2: p = 0 and p = 1 entropy handling ──────────────────────────────

class TestBoundaryEntropy:
    def test_p_zero_entropy_is_zero(self):
        h = compute_binary_entropy(0.0)
        assert h == 0.0
        h_norm = compute_normalized_entropy(0.0)
        assert h_norm == 0.0

    def test_p_one_entropy_is_zero(self):
        h = compute_binary_entropy(1.0)
        assert h == 0.0
        h_norm = compute_normalized_entropy(1.0)
        assert h_norm == 0.0

    def test_near_zero_is_finite_and_small(self):
        h = compute_binary_entropy(1e-6)
        assert np.isfinite(h)
        assert 0.0 < h < 0.001

    def test_near_one_is_finite_and_small(self):
        h = compute_binary_entropy(1.0 - 1e-6)
        assert np.isfinite(h)
        assert 0.0 < h < 0.001


# ── Test 3: p = 0.5 maximum binary entropy ────────────────────────────────────

class TestMaxEntropy:
    def test_half_peaks_at_ln2(self):
        h = compute_binary_entropy(0.5)
        expected = math.log(2.0)
        assert abs(h - expected) < 1e-6

    def test_half_normalized_is_one(self):
        h_norm = compute_normalized_entropy(0.5)
        assert abs(h_norm - 1.0) < 1e-6


# ── Test 4: Entropy symmetry ──────────────────────────────────────────────────

class TestEntropySymmetry:
    @pytest.mark.parametrize("p", [0.05, 0.1, 0.25, 0.33, 0.7, 0.85, 0.99])
    def test_h_symmetric(self, p):
        h_p = compute_binary_entropy(p)
        h_comp = compute_binary_entropy(1.0 - p)
        assert abs(h_p - h_comp) < 1e-7

    @pytest.mark.parametrize("p", [0.1, 0.4, 0.6, 0.9])
    def test_normalized_symmetric(self, p):
        hn_p = compute_normalized_entropy(p)
        hn_comp = compute_normalized_entropy(1.0 - p)
        assert abs(hn_p - hn_comp) < 1e-7


# ── Test 5: Confidence calculation ───────────────────────────────────────────

class TestConfidenceCalculation:
    def test_confidence_above_half(self):
        assert compute_binary_confidence(0.9) == 0.9
        assert compute_binary_confidence(0.1) == 0.9
        assert compute_binary_confidence(0.5) == 0.5
        assert compute_binary_confidence(0.0) == 1.0
        assert compute_binary_confidence(1.0) == 1.0

    @pytest.mark.parametrize("p", [0.0, 0.2, 0.4, 0.5, 0.7, 0.95, 1.0])
    def test_confidence_range(self, p):
        conf = compute_binary_confidence(p)
        assert 0.5 <= conf <= 1.0


# ── Test 6: Normalized entropy range ──────────────────────────────────────────

class TestNormalizedEntropyRange:
    @pytest.mark.parametrize("p", np.linspace(0.0, 1.0, 21))
    def test_normalized_entropy_in_zero_one(self, p):
        hn = compute_normalized_entropy(float(p))
        assert 0.0 <= hn <= 1.0


# ── Test 7, 8, 9, 10: Uncertainty Levels & Boundary Behavior ─────────────────

class TestUncertaintyLevels:
    @pytest.fixture
    def estimator(self):
        # Default config:
        # confidence_high >= 0.85, entropy_low <= 0.25 -> LOW
        # otherwise -> HIGH
        return UncertaintyEstimator(UncertaintyConfig())

    def test_low_uncertainty_extreme_positive(self, estimator):
        # p = 0.95 -> conf = 0.95 (>= 0.85), norm_ent ≈ 0.286? Let's check:
        # At p = 0.97 -> conf = 0.97, norm_ent ≈ 0.194 (<= 0.25) -> LOW
        eval_res = estimator.evaluate(0.97)
        assert eval_res.uncertainty_level == UncertaintyLevel.LOW
        assert eval_res.confidence >= 0.85
        assert eval_res.normalized_entropy <= 0.25

    def test_low_uncertainty_extreme_negative(self, estimator):
        # p = 0.03 -> conf = 0.97 (>= 0.85), norm_ent <= 0.25 -> LOW
        eval_res = estimator.evaluate(0.03)
        assert eval_res.uncertainty_level == UncertaintyLevel.LOW

    def test_high_uncertainty_near_half(self, estimator):
        # p = 0.50 -> conf = 0.50 (<= 0.60), norm_ent = 1.0 (>= 0.60) -> HIGH
        eval_res = estimator.evaluate(0.50)
        assert eval_res.uncertainty_level == UncertaintyLevel.HIGH

    def test_high_uncertainty_at_058(self, estimator):
        # p = 0.58 -> conf = 0.58 (<= 0.60) -> HIGH
        eval_res = estimator.evaluate(0.58)
        assert eval_res.uncertainty_level == UncertaintyLevel.HIGH

    def test_old_medium_region_now_becomes_high(self, estimator):
        # In the old 3-level system, p = 0.80 fell into MEDIUM (conf=0.80 < 0.85).
        # In the 2-level system, it must evaluate to HIGH.
        res = estimator.evaluate(0.80)
        assert res.uncertainty_level == UncertaintyLevel.HIGH
        assert "HIGH" in res.reasoning

    def test_threshold_boundary_behavior(self):
        cfg = UncertaintyConfig(
            confidence_low_threshold=0.60,
            confidence_high_threshold=0.85,
            entropy_low_threshold=0.25,
            entropy_high_threshold=0.60,
        )
        est = UncertaintyEstimator(cfg)
        # Exactly at confidence = 0.60 -> HIGH (conf <= 0.60)
        res_60 = est.evaluate(0.60)
        assert res_60.confidence == 0.60
        assert res_60.uncertainty_level == UncertaintyLevel.HIGH


# ── Test 11 & 12: NaN and Inf Handling ────────────────────────────────────────

class TestNaNAndInfHandling:
    def test_nan_float_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            compute_binary_confidence(float("nan"))
        with pytest.raises(ValueError, match="NaN"):
            compute_binary_entropy(float("nan"))

    def test_inf_float_raises(self):
        with pytest.raises(ValueError, match="Inf"):
            compute_binary_confidence(float("inf"))
        with pytest.raises(ValueError, match="Inf"):
            compute_binary_entropy(float("inf"))

    def test_nan_numpy_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            compute_binary_confidence(np.array([0.5, np.nan]))
        with pytest.raises(ValueError, match="NaN"):
            compute_binary_entropy(np.array([np.nan]))

    def test_inf_torch_raises(self):
        with pytest.raises(ValueError, match="Inf"):
            compute_binary_confidence(torch.tensor([float("inf")]))
        with pytest.raises(ValueError, match="Inf"):
            compute_binary_entropy(torch.tensor([float("inf")]))


# ── Test 13: Invalid Dimensions & Out of Bounds ───────────────────────────────

class TestInputValidation:
    def test_out_of_bounds_negative(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            compute_binary_confidence(-0.1)
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            compute_binary_entropy(-0.01)

    def test_out_of_bounds_greater_than_one(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            compute_binary_confidence(1.05)
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            compute_binary_entropy(1.5)

    def test_invalid_dimension_2d_raises(self):
        with pytest.raises(ValueError, match="shape"):
            compute_binary_confidence(np.ones((2, 2)))
        with pytest.raises(ValueError, match="shape"):
            compute_binary_entropy(torch.ones((2, 2)))

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            compute_binary_confidence("not_a_number")  # type: ignore


# ── Test 14 & 15: NumPy and PyTorch Input Support ─────────────────────────────

class TestDataStructuresSupport:
    def test_numpy_scalar(self):
        conf = compute_binary_confidence(np.float64(0.8))
        assert isinstance(conf, float)
        assert abs(conf - 0.8) < 1e-6

    def test_numpy_1d_array(self):
        probs = np.array([0.1, 0.5, 0.9])
        confs = compute_binary_confidence(probs)
        assert isinstance(confs, np.ndarray)
        np.testing.assert_allclose(confs, [0.9, 0.5, 0.9])

        ents = compute_binary_entropy(probs)
        assert isinstance(ents, np.ndarray)
        assert ents[1] > ents[0]

    def test_torch_scalar(self):
        t = torch.tensor(0.85)
        conf = compute_binary_confidence(t)
        assert isinstance(conf, float)
        assert abs(conf - 0.85) < 1e-6

    def test_torch_1d_tensor(self):
        t = torch.tensor([0.2, 0.5, 0.8])
        confs = compute_binary_confidence(t)
        assert isinstance(confs, np.ndarray)
        np.testing.assert_allclose(confs, [0.8, 0.5, 0.8])


# ── Test 16: Structured Output Contract ───────────────────────────────────────

class TestStructuredOutputContract:
    def test_agent_returns_valid_contract(self):
        agent = UncertaintyAgent()
        res = agent.run(0.75, image_id="test_001.png")

        assert isinstance(res, UncertaintyResult)
        assert res.agent == AgentName.UNCERTAINTY
        assert res.version == "0.6.0"
        assert res.image_id == "test_001.png"
        assert res.raw_model_score == 0.75
        assert res.confidence == 0.75
        assert res.entropy > 0.0
        assert 0.0 <= res.normalized_entropy <= 1.0
        assert res.uncertainty_level in (UncertaintyLevel.LOW, UncertaintyLevel.HIGH)
        assert res.method == "binary_confidence_entropy"
        assert isinstance(res.thresholds, dict)
        assert len(res.reasoning) > 0
        assert res.latency_ms is not None and res.latency_ms >= 0.0


# ── Test 17: Human-Readable Reasoning & Safety Checks ─────────────────────────

class TestReasoningSafety:
    @pytest.mark.parametrize("score", [0.02, 0.50, 0.80, 0.98])
    def test_no_clinical_claims_in_reasoning(self, score):
        agent = UncertaintyAgent()
        res = agent.run(score)
        text = res.reasoning.lower()

        # Must not claim clinical diagnosis or patient status
        assert "patient" not in text
        assert "definitely wrong" not in text
        assert "definitely has" not in text
        assert "clinical diagnosis" not in text
        # Must describe uncertainty or confidence
        assert "uncertainty" in text or "confidence" in text or "entropy" in text


# ── Test 18: Deterministic Output ─────────────────────────────────────────────

class TestDeterminism:
    def test_repeated_runs_identical(self):
        agent = UncertaintyAgent()
        res1 = agent.run(0.67)
        res2 = agent.run(0.67)

        assert res1.raw_model_score == res2.raw_model_score
        assert res1.confidence == res2.confidence
        assert res1.entropy == res2.entropy
        assert res1.normalized_entropy == res2.normalized_entropy
        assert res1.uncertainty_level == res2.uncertainty_level
        assert res1.reasoning == res2.reasoning


# ── Test 19: Integration Test with Synthetic BaseModelOutput ──────────────────

class TestBaseModelIntegration:
    @pytest.fixture
    def agent(self):
        return UncertaintyAgent()

    def _make_dummy_base_result(self, p: float) -> BaseModelResult:
        return BaseModelResult(
            agent=AgentName.BASE_MODEL,
            version="0.4.0",
            model_id="densenet121-res224-nih",
            target_pathology="Pneumonia",
            pneumonia_logit=p,
            pneumonia_probability=p,
            all_pathology_outputs={"Pneumonia": p},
            score=p,
            label="Pneumonia" if p >= 0.5 else "Non-Pneumonia",
            reasoning=f"Synthetic base model output with p={p}",
            latency_ms=12.5,
        )

    def test_case_1_high_confidence_low_entropy(self, agent):
        """Case 1: p = 0.98 -> High confidence (0.98), very low entropy -> LOW uncertainty."""
        base_res = self._make_dummy_base_result(0.98)
        unc_res = agent.run(base_res, image_id="img_case_1.png")

        assert unc_res.uncertainty_level == UncertaintyLevel.LOW
        assert unc_res.confidence == 0.98
        assert unc_res.normalized_entropy < 0.20
        assert "LOW" in unc_res.reasoning

    def test_case_2_old_medium_intermediate_now_high(self, agent):
        """Case 2: Intermediate score (p = 0.80) in old MEDIUM band now evaluates to HIGH."""
        base_res = self._make_dummy_base_result(0.80)
        unc_res = agent.run(base_res, image_id="img_case_2.png")

        assert unc_res.uncertainty_level == UncertaintyLevel.HIGH
        assert unc_res.confidence == 0.80
        assert "HIGH" in unc_res.reasoning

    def test_case_3_near_half_probability_high_entropy(self, agent):
        """Case 3: p = 0.51 -> Near 0.5, high entropy (≈ 1.0) -> HIGH uncertainty."""
        base_res = self._make_dummy_base_result(0.51)
        unc_res = agent.run(base_res, image_id="img_case_3.png")

        assert unc_res.uncertainty_level == UncertaintyLevel.HIGH
        assert unc_res.confidence == 0.51
        assert unc_res.normalized_entropy > 0.95
        assert "HIGH" in unc_res.reasoning

    def test_modelforward_integration(self, agent):
        """Test accepting a ModelForward dataclass containing BaseModelResult."""
        base_res = self._make_dummy_base_result(0.99)
        forward = ModelForward(
            result=base_res,
            raw_probs=torch.tensor([0.1] * 8 + [0.99] + [0.1] * 9),
            features=torch.randn(1024),
            inference_ms=15.0,
        )
        unc_res = agent.run(forward, image_id="forward_001.png")
        assert unc_res.uncertainty_level == UncertaintyLevel.LOW
        assert unc_res.raw_model_score == 0.99

    def test_uncertainty_strictly_uses_raw_score_even_with_calibrated_probability(self, agent):
        """
        Verify UncertaintyAgent strictly consumes raw_pneumonia_score.
        When raw_pneumonia_score = 0.50 (maximum ambiguity) and calibrated_probability = 0.0163,
        uncertainty must evaluate to HIGH (from raw 0.50), NEVER to LOW (from calibrated 0.0163).
        """
        base_res = BaseModelResult(
            agent=AgentName.BASE_MODEL,
            version="0.4.0",
            model_id="densenet121-res224-nih",
            target_pathology="Pneumonia",
            pneumonia_logit=0.50,
            pneumonia_probability=0.50,
            raw_pneumonia_score=0.50,
            calibrated_probability=0.01628,
            score=0.50,
            label="Non-Pneumonia",
            reasoning="Test calibration separation",
        )
        unc_res = agent.run(base_res, image_id="calib_sep.png")
        assert unc_res.raw_model_score == 0.50
        assert unc_res.confidence == 0.50
        assert unc_res.uncertainty_level == UncertaintyLevel.HIGH


# ── Requirement 11: Two-Level Uncertainty Dedicated Test Suite ────────────────

class TestTwoLevelUncertaintyRequirements:
    """Explicit tests for all criteria specified in the 2-level uncertainty refactoring:
    - clearly confident sample becomes LOW
    - borderline/uncertain sample becomes HIGH
    - old MEDIUM region now becomes HIGH
    - calibrated probability does not affect uncertainty
    """

    @pytest.fixture
    def agent(self):
        return UncertaintyAgent()

    @pytest.fixture
    def estimator(self):
        return UncertaintyEstimator()

    def test_clearly_confident_sample_becomes_low(self, estimator):
        """Clearly confident sample (e.g. p=0.98 or p=0.02) becomes LOW."""
        res_pos = estimator.evaluate(0.98)
        assert res_pos.uncertainty_level == UncertaintyLevel.LOW
        assert res_pos.confidence >= 0.85
        assert res_pos.normalized_entropy <= 0.25
        assert "LOW" in res_pos.reasoning

        res_neg = estimator.evaluate(0.02)
        assert res_neg.uncertainty_level == UncertaintyLevel.LOW
        assert res_neg.confidence >= 0.85
        assert res_neg.normalized_entropy <= 0.25
        assert "LOW" in res_neg.reasoning

    def test_borderline_uncertain_sample_becomes_high(self, estimator):
        """Borderline/uncertain sample (near 0.50, including 0.522161) becomes HIGH."""
        for p in [0.50, 0.52, 0.522161, 0.55]:
            res = estimator.evaluate(p)
            assert res.uncertainty_level == UncertaintyLevel.HIGH
            assert "HIGH" in res.reasoning

    def test_old_medium_region_becomes_high(self, estimator):
        """Old MEDIUM region (e.g. confidence in (0.60, 0.85)) now evaluates to HIGH."""
        # Check several scores that previously fell in the MEDIUM band
        for p in [0.65, 0.70, 0.75, 0.80, 0.82]:
            res = estimator.evaluate(p)
            assert res.uncertainty_level == UncertaintyLevel.HIGH
            assert "HIGH" in res.reasoning

    def test_no_medium_state_exists_in_enum(self):
        """Verify UncertaintyLevel enum has exactly LOW and HIGH, no MEDIUM."""
        levels = [m.value for m in UncertaintyLevel]
        assert levels == ["LOW", "HIGH"]
        assert "MEDIUM" not in levels

    def test_calibrated_probability_does_not_affect_uncertainty(self, agent):
        """
        Calibrated probability does not affect uncertainty.
        Even if calibrated_probability is extremely low (~0.01, which would imply
        confidence ~0.99 under calibrated scale), the raw score (0.522161) dominates
        and forces HIGH uncertainty.
        """
        base_res = BaseModelResult(
            agent=AgentName.BASE_MODEL,
            version="0.4.0",
            model_id="densenet121-res224-nih",
            target_pathology="Pneumonia",
            pneumonia_logit=0.088,
            pneumonia_probability=0.522161,
            raw_pneumonia_score=0.522161,
            calibrated_probability=0.0112,  # Val set base rate ~1.12%
            score=0.522161,
            label="Pneumonia",
            reasoning="Test raw score dominance",
        )
        unc_res = agent.run(base_res)
        assert unc_res.raw_model_score == 0.522161
        assert unc_res.uncertainty_level == UncertaintyLevel.HIGH
        assert "HIGH" in unc_res.reasoning

