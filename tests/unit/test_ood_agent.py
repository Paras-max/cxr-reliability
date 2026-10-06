"""
tests/unit/test_ood_agent.py

Unit tests for the OOD Agent (Phase 5).

Tests:
    1. In-distribution features score low (below threshold)
    2. Shifted features score severe (far OOD)
    3. Borderline band reported correctly
    4. Detector agreement flag
    5. OODResult schema is valid (Pydantic)
    6. OODAgent.is_stats_available() reflects disk state
    7. OODAgent.run() returns OODResult with required fields
    8. OODResult.level matches OODLevel enum
    9. Reasoning string is human-readable and non-empty
    10. OOD result does NOT use clinical diagnostic language

No model weights required for most tests.
Tests marked @pytest.mark.slow require model weights.

Usage:
    python -m pytest tests/unit/test_ood_agent.py -v
    python -m pytest tests/unit/test_ood_agent.py -v -m slow  # includes model tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cxr_reliability.agents.ood import OODAgent
from cxr_reliability.config.thresholds import OODThresholds
from cxr_reliability.contracts.ood import OODLevel, OODResult
from cxr_reliability.ood.detector import OODConfig, OODDetector
from cxr_reliability.ood.statistics import fit_reference_stats, save_stats

# ── Helpers ───────────────────────────────────────────────────────────────────

def _create_stats_in_tmp(
    tmp_path: Path,
    dim: int = 1024,
    n_train: int = 300,
    seed: int = 42,
    borderline: float = 10.0,
    severe: float = 20.0,
) -> Path:
    """Fit reference stats and save to tmp_path. Returns stats_dir."""
    rng = np.random.default_rng(seed)
    train_data = rng.standard_normal((n_train, dim))
    stats = fit_reference_stats(train_data, lambda_reg=1e-4)
    save_stats(stats, output_dir=tmp_path, extra_metadata={
        "mahalanobis_borderline": borderline,
        "mahalanobis_severe": severe,
        "threshold_percentile": 99.0,
    })
    return tmp_path


def _make_agent(
    tmp_path: Path,
    dim: int = 1024,
    borderline: float = 10.0,
    severe: float = 20.0,
) -> OODAgent:
    """Create a ready-to-use OODAgent with saved stats."""
    stats_dir = _create_stats_in_tmp(tmp_path, dim=dim, borderline=borderline, severe=severe)
    thresholds = OODThresholds(
        in_distribution_percentile=99.0,
        mahalanobis_borderline=borderline,
        mahalanobis_severe=severe,
    )
    config = OODConfig(
        covariance_regularization=1e-4,
        mahalanobis_borderline=borderline,
        mahalanobis_severe=severe,
    )
    agent = OODAgent(
        stats_dir=stats_dir,
        thresholds=thresholds,
        thresholds_version="v_test",
        config=config,
    )
    return agent


# ── Test 1: In-distribution features score low ────────────────────────────────

def test_in_distribution_features_score_low(tmp_path: Path):
    """Features drawn from the fitted Gaussian should yield level=in_distribution."""
    dim = 8  # small dim for speed
    rng = np.random.default_rng(0)
    train_data = rng.standard_normal((500, dim))
    stats = fit_reference_stats(train_data, lambda_reg=1e-4)
    stats_dir = tmp_path / "stats_low"
    save_stats(stats, output_dir=stats_dir, extra_metadata={
        "mahalanobis_borderline": 1000.0,  # very high threshold
        "mahalanobis_severe": 2000.0,
        "threshold_percentile": 99.0,
    })

    thresholds = OODThresholds(
        in_distribution_percentile=99.0,
        mahalanobis_borderline=1000.0,
        mahalanobis_severe=2000.0,
    )
    config = OODConfig(
        covariance_regularization=1e-4,
        mahalanobis_borderline=1000.0,
        mahalanobis_severe=2000.0,
    )
    agent = OODAgent(stats_dir=stats_dir, thresholds=thresholds,
                     thresholds_version="v_test", config=config)

    # A vector near the mean of the training data
    in_dist_vec = torch.tensor(np.mean(train_data, axis=0).astype(np.float32))
    result = agent.run(features=in_dist_vec)

    assert isinstance(result, OODResult)
    assert result.level == OODLevel.IN_DISTRIBUTION
    assert not result.is_ood if hasattr(result, "is_ood") else True


# ── Test 2: Shifted features score severe ────────────────────────────────────

def test_shifted_features_score_severe(tmp_path: Path):
    """Features far from the fitted mean should be severe."""
    dim = 8
    rng = np.random.default_rng(1)
    train_data = rng.standard_normal((300, dim))  # mean ≈ 0

    stats = fit_reference_stats(train_data, lambda_reg=1e-4)
    stats_dir = tmp_path / "stats_severe"
    save_stats(stats, output_dir=stats_dir, extra_metadata={
        "mahalanobis_borderline": 5.0,
        "mahalanobis_severe": 10.0,
        "threshold_percentile": 99.0,
    })

    thresholds = OODThresholds(
        mahalanobis_borderline=5.0,
        mahalanobis_severe=10.0,
    )
    config = OODConfig(
        covariance_regularization=1e-4,
        mahalanobis_borderline=5.0,
        mahalanobis_severe=10.0,
    )
    agent = OODAgent(stats_dir=stats_dir, thresholds=thresholds,
                     thresholds_version="v_test", config=config)

    # A vector very far from the reference mean
    far_vec = torch.tensor(np.array([100.0] * dim, dtype=np.float32))
    result = agent.run(features=far_vec)

    assert result.level == OODLevel.SEVERE
    assert result.mahalanobis_distance > 10.0


# ── Test 3: Borderline band reported correctly ────────────────────────────────

def test_borderline_band_reported(tmp_path: Path):
    """Distances between borderline and severe → borderline level."""
    dim = 6
    rng = np.random.default_rng(2)
    train_data = rng.standard_normal((300, dim))

    from cxr_reliability.ood.statistics import fit_reference_stats as _fit
    stats = _fit(train_data, lambda_reg=1e-4)

    # Find a vector with known distance in the borderline zone
    from cxr_reliability.ood.mahalanobis import mahalanobis_distance
    # Scale a vector until it's in (borderline, severe)
    direction = rng.standard_normal(dim)
    direction /= np.linalg.norm(direction)

    detector = OODDetector(config=OODConfig(covariance_regularization=1e-4))
    detector.fit(train_data)

    # Find a scale that gives a distance between the thresholds
    for scale in np.linspace(1.0, 20.0, 100):
        vec = direction * scale
        dist = mahalanobis_distance(vec, stats)
        if 5.0 < dist < 10.0:
            break

    stats_dir = tmp_path / "stats_borderline"
    save_stats(stats, output_dir=stats_dir, extra_metadata={
        "mahalanobis_borderline": 5.0,
        "mahalanobis_severe": 10.0,
        "threshold_percentile": 99.0,
    })

    thresholds = OODThresholds(mahalanobis_borderline=5.0, mahalanobis_severe=10.0)
    config = OODConfig(covariance_regularization=1e-4,
                       mahalanobis_borderline=5.0, mahalanobis_severe=10.0)
    agent = OODAgent(stats_dir=stats_dir, thresholds=thresholds,
                     thresholds_version="v_test", config=config)

    result = agent.run(features=torch.tensor(vec.astype(np.float32)))
    assert result.level == OODLevel.BORDERLINE, (
        f"Distance={result.mahalanobis_distance:.4f} should be borderline (5-10)"
    )


# ── Test 4: Detector agreement flag ──────────────────────────────────────────

def test_detector_agreement_flag(tmp_path: Path):
    """When energy is enabled, detectors_agree should be a bool."""
    dim = 8
    rng = np.random.default_rng(3)
    train_data = rng.standard_normal((200, dim))
    stats = fit_reference_stats(train_data, lambda_reg=1e-4)
    stats_dir = tmp_path / "stats_energy"
    save_stats(stats, output_dir=stats_dir, extra_metadata={
        "mahalanobis_borderline": 100.0,
        "mahalanobis_severe": 200.0,
        "threshold_percentile": 99.0,
    })

    thresholds = OODThresholds(mahalanobis_borderline=100.0, mahalanobis_severe=200.0)
    config = OODConfig(
        covariance_regularization=1e-4,
        mahalanobis_borderline=100.0,
        mahalanobis_severe=200.0,
        energy_enabled=True,
        energy_temperature=1.0,
    )
    agent = OODAgent(stats_dir=stats_dir, thresholds=thresholds,
                     thresholds_version="v_test", config=config)

    vec = torch.tensor(rng.standard_normal(dim).astype(np.float32))
    probs = torch.tensor(np.abs(rng.standard_normal(18)).clip(0.01, 0.99).astype(np.float32))

    result = agent.run(features=vec, raw_probs=probs)

    # When energy is enabled and probs provided, energy_score should be a float
    assert result.energy_score is not None
    assert isinstance(result.energy_score, float)
    # detectors_agree should be a bool
    assert result.detectors_agree is not None
    assert isinstance(result.detectors_agree, bool)


# ── Test 5: OODResult schema validation ──────────────────────────────────────

def test_ood_result_schema_valid(tmp_path: Path):
    """OODAgent.run() should return a valid OODResult (Pydantic contract)."""
    agent = _make_agent(tmp_path, dim=8)
    rng = np.random.default_rng(4)
    vec = torch.tensor(rng.standard_normal(8).astype(np.float32))
    result = agent.run(features=vec, image_id="test.png")

    # Pydantic model — validate all required fields
    assert isinstance(result, OODResult)
    assert result.agent.value == "ood"
    assert isinstance(result.mahalanobis_distance, float)
    assert result.mahalanobis_distance >= 0
    assert result.level in OODLevel.__members__.values()
    assert len(result.reasoning) > 0
    assert result.version == "0.5.0"
    assert result.thresholds_version == "v_test"


# ── Test 6: is_stats_available() ─────────────────────────────────────────────

def test_is_stats_available_true_when_files_exist(tmp_path: Path):
    stats_dir = _create_stats_in_tmp(tmp_path, dim=4)
    thresholds = OODThresholds()
    agent = OODAgent(stats_dir=stats_dir, thresholds=thresholds, thresholds_version="v")
    assert agent.is_stats_available


def test_is_stats_available_false_when_missing(tmp_path: Path):
    missing_dir = tmp_path / "nonexistent"
    thresholds = OODThresholds()
    agent = OODAgent(stats_dir=missing_dir, thresholds=thresholds, thresholds_version="v")
    assert not agent.is_stats_available


# ── Test 7: OODAgent.run() fields ────────────────────────────────────────────

def test_run_returns_ood_result_with_all_required_fields(tmp_path: Path):
    agent = _make_agent(tmp_path, dim=8)
    rng = np.random.default_rng(10)
    vec = torch.tensor(rng.standard_normal(8).astype(np.float32))
    result = agent.run(features=vec)

    assert hasattr(result, "mahalanobis_distance")
    assert hasattr(result, "level")
    assert hasattr(result, "reasoning")
    assert hasattr(result, "energy_score")
    assert hasattr(result, "detectors_agree")
    assert hasattr(result, "feature_layer")
    assert hasattr(result, "version")


# ── Test 8: OODLevel enum values ─────────────────────────────────────────────

def test_ood_level_enum_values():
    assert OODLevel.IN_DISTRIBUTION.value == "in_distribution"
    assert OODLevel.BORDERLINE.value == "borderline"
    assert OODLevel.SEVERE.value == "severe"


# ── Test 9: Reasoning is non-clinical and non-empty ──────────────────────────

def test_reasoning_string_is_non_clinical(tmp_path: Path):
    """OOD reasoning should describe distribution shift, NOT clinical diagnosis."""
    agent = _make_agent(tmp_path, dim=8)
    rng = np.random.default_rng(20)
    vec = torch.tensor(rng.standard_normal(8).astype(np.float32))
    result = agent.run(features=vec)

    reasoning = result.reasoning.lower()
    assert len(reasoning) > 10

    # Must NOT contain clinical diagnostic language
    forbidden_terms = ["diagnosis", "pathology", "disease", "abnormal", "definitely"]
    for term in forbidden_terms:
        assert term not in reasoning, (
            f"Reasoning contains forbidden clinical term '{term}': {result.reasoning}"
        )


# ── Test 10: Save/load deterministic ─────────────────────────────────────────

def test_save_load_gives_identical_distances(tmp_path: Path):
    """Saving and reloading stats should give identical Mahalanobis distances."""
    dim = 8
    rng = np.random.default_rng(99)
    train_data = rng.standard_normal((300, dim))

    detector = OODDetector(config=OODConfig(covariance_regularization=1e-4,
                                             mahalanobis_borderline=100.0))
    detector.fit(train_data)
    detector.save(tmp_path / "stats_roundtrip")

    detector2 = OODDetector()
    detector2.load_stats(tmp_path / "stats_roundtrip")

    query = rng.standard_normal(dim)
    d1 = detector.score(query)["mahalanobis_distance"]
    d2 = detector2.score(query)["mahalanobis_distance"]
    assert abs(d1 - d2) < 1e-9


# ── Test: Integration (slow — requires model) ─────────────────────────────────

@pytest.mark.slow
def test_integration_with_real_model(tmp_path: Path):
    """
    End-to-end integration test:
    1. Load real Base Model
    2. Run inference on a synthetic image to get feature vector
    3. Fit OOD stats from synthetic features
    4. Run OOD detector → valid OODResult

    Skipped unless -m slow is specified (requires model weights).
    """
    import numpy as np
    from PIL import Image

    # Create a synthetic chest-like image
    rng = np.random.default_rng(42)
    arr = rng.integers(50, 200, (224, 224), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")

    # Save to temp
    img_path = tmp_path / "synthetic.png"
    img.save(img_path)

    # Load model
    from cxr_reliability.models.model_factory import get_model
    from cxr_reliability.models.preprocessing import load_image_for_txv

    agent = get_model(model_id="densenet121-res224-nih", device="cpu", load=True)
    tensor = load_image_for_txv(img_path)
    forward = agent.run(tensor)

    features_np = forward.features.numpy()
    assert features_np.shape == (1024,)
    assert np.all(np.isfinite(features_np))

    # Fit OOD stats from 50 synthetic near-identical feature vectors
    fake_train = np.stack([features_np + np.random.randn(1024) * 0.01 for _ in range(50)])
    detector = OODDetector(config=OODConfig(
        covariance_regularization=1e-3,
        mahalanobis_borderline=1000.0,
    ))
    detector.fit(fake_train)
    detector.save(tmp_path / "ood_stats")

    # Create OODAgent and run
    from cxr_reliability.config.thresholds import OODThresholds
    thresholds = OODThresholds(mahalanobis_borderline=1000.0, mahalanobis_severe=2000.0)
    ood_agent = OODAgent(
        stats_dir=tmp_path / "ood_stats",
        thresholds=thresholds,
        thresholds_version="integration_test",
    )
    result = ood_agent.run(features=forward.features, raw_probs=forward.raw_probs)

    assert isinstance(result, OODResult)
    assert result.mahalanobis_distance >= 0
    assert result.level in (OODLevel.IN_DISTRIBUTION, OODLevel.BORDERLINE, OODLevel.SEVERE)
    assert len(result.reasoning) > 0
    print(f"\n[Integration] mahalanobis_distance={result.mahalanobis_distance:.4f}, "
          f"level={result.level.value}")
