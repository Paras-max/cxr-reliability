"""
tests/unit/test_ood_detector.py

Unit tests for the OODDetector high-level interface.

Tests:
    1. OODDetector.fit() → valid stats
    2. OODDetector.predict() → correct OODPrediction structure
    3. Level assignment: in_distribution / borderline / severe
    4. OODConfig validation (invalid values raise ValueError)
    5. Energy score enabled/disabled
    6. Detector not fitted → clear RuntimeError
    7. Save/load round-trip (temp directory)
    8. analyze() returns serializable dict
    9. Threshold from validation percentile
    10. Deterministic prediction (same input → same output)

No model weights required. All tests use synthetic Gaussian data.

Usage:
    python -m pytest tests/unit/test_ood_detector.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cxr_reliability.ood.detector import OODConfig, OODDetector, OODPrediction
from cxr_reliability.ood.statistics import fit_reference_stats

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_detector(dim: int = 10, n_train: int = 300, seed: int = 42) -> OODDetector:
    """Create a fitted OODDetector with identity-like covariance."""
    rng = np.random.default_rng(seed)
    train_data = rng.standard_normal((n_train, dim))
    detector = OODDetector(config=OODConfig(
        covariance_regularization=1e-4,
        threshold_percentile=95.0,
        mahalanobis_borderline=None,
        mahalanobis_severe=None,
    ))
    detector.fit(train_data, split="train")
    return detector


# ── Test 1: fit() produces valid stats ────────────────────────────────────────

class TestFit:
    def test_fit_produces_valid_stats(self):
        detector = _make_detector(dim=8, n_train=200)
        assert detector.is_fitted
        assert detector.stats is not None
        assert detector.stats.feature_dim == 8
        assert detector.stats.n_samples == 200

    def test_fit_with_large_dim(self):
        detector = _make_detector(dim=1024, n_train=1100)
        assert detector.is_fitted
        assert detector.stats.feature_dim == 1024

    def test_fit_returns_self(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 5))
        detector = OODDetector()
        result = detector.fit(data)
        assert result is detector  # method chaining

    def test_fit_rejects_nan_data(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((100, 5))
        data[0, 0] = float("nan")
        detector = OODDetector()
        with pytest.raises(ValueError, match="[Nn]aN|[Ii]nf|finite"):
            detector.fit(data)


# ── Test 2: predict() output structure ────────────────────────────────────────

class TestPredict:
    def test_predict_returns_ood_prediction(self):
        detector = _make_detector(dim=6)
        # Set a threshold
        detector.config.mahalanobis_borderline = 1000.0

        rng = np.random.default_rng(5)
        query = rng.standard_normal(6)
        pred = detector.predict(query)
        assert isinstance(pred, OODPrediction)
        assert isinstance(pred.mahalanobis_distance, float)
        assert pred.mahalanobis_distance >= 0
        assert isinstance(pred.reasoning, str) and len(pred.reasoning) > 0
        assert pred.level in ("in_distribution", "borderline", "severe")
        assert pred.method == "mahalanobis"

    def test_predict_is_deterministic(self):
        detector = _make_detector(dim=6, seed=42)
        detector.config.mahalanobis_borderline = 100.0
        rng = np.random.default_rng(99)
        query = rng.standard_normal(6)
        pred1 = detector.predict(query)
        pred2 = detector.predict(query)
        assert pred1.mahalanobis_distance == pred2.mahalanobis_distance
        assert pred1.level == pred2.level

    def test_in_distribution_vector_low_distance(self):
        """A vector sampled from the fitted distribution should be near the mean."""
        dim = 8
        n = 500
        rng = np.random.default_rng(7)
        data = rng.standard_normal((n, dim))
        detector = OODDetector(config=OODConfig(
            covariance_regularization=1e-4,
            mahalanobis_borderline=1000.0,  # high threshold so we can measure
        ))
        detector.fit(data)

        # A vector very close to the mean should have a small distance
        near_mean = np.mean(data, axis=0) + 0.01 * np.ones(dim)
        pred = detector.predict(near_mean)
        # Near mean → should have lower distance than a far outlier
        assert pred.mahalanobis_distance < 50.0, (
            f"Distance to near-mean vector is too high: {pred.mahalanobis_distance}"
        )


# ── Test 3: Level assignment ──────────────────────────────────────────────────

class TestLevelAssignment:
    def _make_with_thresholds(
        self,
        borderline: float,
        severe: float | None = None,
        dim: int = 4,
    ) -> OODDetector:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((200, dim))
        config = OODConfig(
            covariance_regularization=1e-4,
            mahalanobis_borderline=borderline,
            mahalanobis_severe=severe,
        )
        detector = OODDetector(config=config)
        detector.fit(data)
        return detector

    def test_in_distribution_level(self):
        """A vector very close to the mean → in_distribution."""
        detector = self._make_with_thresholds(borderline=100.0)
        # Near-zero vector (close to mean=0) → distance should be small
        vec = np.zeros(4)
        pred = detector.predict(vec)
        assert pred.level == "in_distribution"
        assert not pred.is_ood

    def test_severe_level_when_above_severe_threshold(self):
        """A vector with very large distance → severe."""
        detector = self._make_with_thresholds(borderline=1.0, severe=2.0)
        # A far vector: mean is ~0, so [100, 100, 100, 100] is very far
        far_vec = np.array([100.0, 100.0, 100.0, 100.0])
        pred = detector.predict(far_vec)
        assert pred.level == "severe"
        assert pred.is_ood

    def test_borderline_level_between_thresholds(self):
        """Distance between borderline and severe → borderline."""
        rng = np.random.default_rng(5)
        dim = 4
        data = rng.standard_normal((500, dim))
        detector = OODDetector(config=OODConfig(
            covariance_regularization=1e-4,
        ))
        detector.fit(data)

        # Compute distance to get a concrete value
        query = rng.standard_normal(dim) * 2.0
        dist = detector.score(query)["mahalanobis_distance"]

        # Set thresholds to bracket this distance
        detector.config.mahalanobis_borderline = dist - 1.0
        detector.config.mahalanobis_severe = dist + 10.0
        pred = detector.predict(query)
        assert pred.level == "borderline"
        assert pred.is_ood


# ── Test 4: Config validation ─────────────────────────────────────────────────

class TestConfigValidation:
    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError, match="covariance_regularization"):
            OODConfig(covariance_regularization=-1.0).validate()

    def test_zero_lambda_raises(self):
        with pytest.raises(ValueError, match="covariance_regularization"):
            OODConfig(covariance_regularization=0.0).validate()

    def test_invalid_percentile_raises(self):
        with pytest.raises(ValueError, match="threshold_percentile"):
            OODConfig(threshold_percentile=101.0).validate()

    def test_zero_percentile_raises(self):
        with pytest.raises(ValueError, match="threshold_percentile"):
            OODConfig(threshold_percentile=0.0).validate()

    def test_invalid_threshold_method_raises(self):
        with pytest.raises(ValueError, match="threshold_method"):
            OODConfig(threshold_method="unknown").validate()

    def test_zero_energy_temperature_raises(self):
        with pytest.raises(ValueError, match="energy_temperature"):
            OODConfig(energy_temperature=0.0).validate()


# ── Test 5: Energy score ──────────────────────────────────────────────────────

class TestEnergyScore:
    def test_energy_disabled_by_default(self):
        detector = _make_detector(dim=6)
        detector.config.mahalanobis_borderline = 100.0
        rng = np.random.default_rng(20)
        query = rng.standard_normal(6)
        pred = detector.predict(query)
        # energy_enabled=False → energy_score should be None
        assert pred.energy_score is None

    def test_energy_enabled_with_probs(self):
        dim = 6
        n = 200
        rng = np.random.default_rng(21)
        data = rng.standard_normal((n, dim))
        config = OODConfig(
            covariance_regularization=1e-4,
            mahalanobis_borderline=100.0,
            energy_enabled=True,
            energy_temperature=1.0,
        )
        detector = OODDetector(config=config)
        detector.fit(data)

        query = rng.standard_normal(dim)
        fake_probs = np.array([0.1, 0.3, 0.5, 0.2, 0.4, 0.7])
        pred = detector.predict(query, raw_probs=fake_probs)
        # energy should be computed
        assert pred.energy_score is not None
        assert isinstance(pred.energy_score, float)
        assert np.isfinite(pred.energy_score)


# ── Test 6: Unfitted detector raises RuntimeError ─────────────────────────────

class TestUnfittedDetector:
    def test_predict_without_fit_raises(self):
        detector = OODDetector()
        vec = np.zeros(10)
        with pytest.raises(RuntimeError, match="fit|load|fitted"):
            detector.predict(vec)

    def test_score_without_fit_raises(self):
        detector = OODDetector()
        with pytest.raises(RuntimeError, match="fit|load|fitted"):
            detector.score(np.zeros(10))

    def test_batch_score_without_fit_raises(self):
        detector = OODDetector()
        with pytest.raises(RuntimeError, match="fit|load|fitted"):
            detector.batch_score(np.zeros((5, 10)))


# ── Test 7: Save/load round-trip ─────────────────────────────────────────────

class TestSaveLoad:
    def test_save_load_round_trip(self, tmp_path: Path):
        detector = _make_detector(dim=8, n_train=200)
        detector.config.mahalanobis_borderline = 12.5
        detector.config.mahalanobis_severe = 20.0

        paths = detector.save(tmp_path)
        assert paths["stats_path"].exists()
        assert paths["metadata_path"].exists()

        # Load into a new detector
        detector2 = OODDetector()
        detector2.load_stats(tmp_path)

        assert detector2.is_fitted
        assert detector2.stats.feature_dim == 8
        assert detector2.stats.n_samples == 200

        # Distances should be identical
        rng = np.random.default_rng(9)
        query = rng.standard_normal(8)
        d1 = detector.score(query)["mahalanobis_distance"]
        d2 = detector2.score(query)["mahalanobis_distance"]
        assert abs(d1 - d2) < 1e-9, f"Original={d1}, Loaded={d2}"

    def test_thresholds_persisted_in_metadata(self, tmp_path: Path):
        detector = _make_detector(dim=6)
        detector.config.mahalanobis_borderline = 7.77
        detector.config.mahalanobis_severe = 15.55
        detector.save(tmp_path)

        detector2 = OODDetector()
        detector2.load_stats(tmp_path)
        assert abs(detector2.config.mahalanobis_borderline - 7.77) < 1e-6
        assert abs(detector2.config.mahalanobis_severe - 15.55) < 1e-6

    def test_missing_stats_dir_raises_file_not_found(self, tmp_path: Path):
        nonexistent = tmp_path / "does_not_exist"
        detector = OODDetector()
        with pytest.raises(FileNotFoundError):
            detector.load_stats(nonexistent)


# ── Test 8: analyze() returns serializable dict ───────────────────────────────

class TestAnalyze:
    def test_analyze_returns_dict_with_expected_keys(self):
        detector = _make_detector(dim=6)
        detector.config.mahalanobis_borderline = 100.0
        rng = np.random.default_rng(30)
        query = rng.standard_normal(6)

        result = detector.analyze(query, image_id="test_001.png")

        required_keys = [
            "image_id", "is_ood", "level", "mahalanobis_distance",
            "mahalanobis_threshold", "method", "reasoning",
            "latency_ms", "disclaimer",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: '{key}'"

        assert result["image_id"] == "test_001.png"
        assert isinstance(result["is_ood"], bool)
        assert isinstance(result["reasoning"], str)
        assert "disclaimer" in result

    def test_analyze_result_is_json_serializable(self):
        import json
        detector = _make_detector(dim=6)
        detector.config.mahalanobis_borderline = 100.0
        rng = np.random.default_rng(31)
        query = rng.standard_normal(6)
        result = detector.analyze(query)
        # Should not raise
        json.dumps(result)


# ── Test 9: Threshold from validation percentile ──────────────────────────────

class TestValidationThreshold:
    def test_set_thresholds_from_validation(self):
        detector = _make_detector(dim=6, n_train=300)
        rng = np.random.default_rng(50)
        # Simulate validation distances
        val_dists = np.abs(rng.standard_normal(200)) * 5 + 10
        borderline, severe = detector.set_thresholds_from_validation(val_dists, percentile=99.0)
        assert borderline > 0
        assert severe >= borderline
        assert detector.config.mahalanobis_borderline == borderline
        assert detector.config.mahalanobis_severe == severe

    def test_threshold_at_99th_percentile(self):
        detector = _make_detector(dim=4)
        val_dists = np.array([float(i) for i in range(1, 101)])  # 1..100
        borderline, _ = detector.set_thresholds_from_validation(val_dists, percentile=99.0)
        # 99th percentile of [1..100] = 99.0 (approximately)
        assert abs(borderline - 99.0) < 1.0


# ── Test 10: Covariance regularization preserves PD ──────────────────────────

class TestCovarianceRegularization:
    def test_small_lambda_gives_positive_definite_matrix(self):
        """Regularized covariance should always be positive definite."""
        import scipy.linalg
        dim = 5
        rng = np.random.default_rng(60)
        data = rng.standard_normal((200, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-6)

        # Cholesky succeeded during fitting → matrix is PD
        # Verify by attempting Cholesky on the loaded matrix
        try:
            scipy.linalg.cholesky(stats.regularized_cov, lower=True)
        except scipy.linalg.LinAlgError:
            pytest.fail("Regularized covariance is not positive definite")

    def test_near_singular_cov_fixed_by_regularization(self):
        """N < D → singular raw covariance → regularization should rescue it."""
        dim = 50
        n = 20  # N < D → raw covariance is rank-deficient
        rng = np.random.default_rng(61)
        data = rng.standard_normal((n, dim))
        # With lambda=0.1, the matrix should become PD
        stats = fit_reference_stats(data, lambda_reg=0.1)
        assert stats is not None
        assert stats.feature_dim == dim
