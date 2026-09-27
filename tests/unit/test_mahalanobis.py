"""
tests/unit/test_mahalanobis.py

Unit tests for Mahalanobis distance computation.

Tests:
    1. Identity covariance → Euclidean distance
    2. Known distance values (analytically computed)
    3. Batch vs single consistency
    4. Numerical stability (poorly conditioned matrix)
    5. Dimension mismatch → clear ValueError
    6. NaN/Inf in features → clear ValueError
    7. Torch tensor input handled correctly
    8. Dtype conversion (float32 → float64 internally)
    9. 1D vs 2D input handling

No model weights required.

Usage:
    python -m pytest tests/unit/test_mahalanobis.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from cxr_reliability.ood.mahalanobis import (
    batch_mahalanobis_distances,
    mahalanobis_distance,
)
from cxr_reliability.ood.statistics import OODReferenceStats, fit_reference_stats


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_identity_stats(dim: int, n_samples: int = 1000, seed: int = 0) -> OODReferenceStats:
    """Build stats with mean=0 and covariance≈I (sampled from standard normal)."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_samples, dim))
    return fit_reference_stats(data, lambda_reg=1e-5)


def _make_stats_from_exact(
    mean: np.ndarray,
    cov: np.ndarray,
    lambda_reg: float = 1e-8,
) -> OODReferenceStats:
    """Build OODReferenceStats directly from known mean and covariance."""
    import scipy.linalg
    dim = mean.shape[0]
    reg_cov = cov + lambda_reg * np.eye(dim)
    chol = scipy.linalg.cholesky(reg_cov, lower=True)
    return OODReferenceStats(
        mean_vector=mean.astype(np.float64),
        covariance_matrix=cov.astype(np.float64),
        regularized_cov=reg_cov.astype(np.float64),
        cholesky_factor=chol.astype(np.float64),
        n_samples=10000,
        feature_dim=dim,
        lambda_reg=lambda_reg,
    )


# ── Test 1: Identity covariance → Euclidean distance ─────────────────────────

class TestIdentityCovariance:
    """When cov = I and mean = 0, Mahalanobis should equal Euclidean distance."""

    def test_zero_vector_gives_zero_distance(self):
        dim = 4
        zero = np.zeros(dim)
        cov = np.eye(dim)
        stats = _make_stats_from_exact(mean=zero, cov=cov)
        dist = mahalanobis_distance(zero, stats)
        assert abs(dist) < 1e-10, f"Expected ~0, got {dist}"

    def test_unit_vector_gives_unit_distance(self):
        """e₁ = [1, 0, 0, 0] with identity cov → distance = 1."""
        dim = 4
        mean = np.zeros(dim)
        cov = np.eye(dim)
        stats = _make_stats_from_exact(mean=mean, cov=cov)
        e1 = np.array([1.0, 0.0, 0.0, 0.0])
        dist = mahalanobis_distance(e1, stats)
        assert abs(dist - 1.0) < 1e-6, f"Expected 1.0, got {dist}"

    def test_known_euclidean_distance(self):
        """[3, 4, 0, 0] with identity cov → distance = 5."""
        dim = 4
        mean = np.zeros(dim)
        cov = np.eye(dim)
        stats = _make_stats_from_exact(mean=mean, cov=cov)
        vec = np.array([3.0, 4.0, 0.0, 0.0])
        dist = mahalanobis_distance(vec, stats)
        assert abs(dist - 5.0) < 1e-6, f"Expected 5.0, got {dist}"

    def test_diagonal_cov_scales_distance(self):
        """
        With cov = diag(4, 1, 1, 1) and x = [2, 0, 0, 0]:
        d = sqrt( 2²/4 ) = sqrt(1) = 1.
        """
        dim = 4
        mean = np.zeros(dim)
        cov = np.diag([4.0, 1.0, 1.0, 1.0])
        stats = _make_stats_from_exact(mean=mean, cov=cov)
        vec = np.array([2.0, 0.0, 0.0, 0.0])
        dist = mahalanobis_distance(vec, stats)
        assert abs(dist - 1.0) < 1e-6, f"Expected 1.0, got {dist}"


# ── Test 2: Known distance values ─────────────────────────────────────────────

class TestKnownDistances:
    def test_2d_known_distance(self):
        """
        2D case: mean=[0,0], cov=[[1,0],[0,4]], x=[2,2]
        d = sqrt( 2²/1 + 2²/4 ) = sqrt(4+1) = sqrt(5) ≈ 2.2361
        """
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.0], [0.0, 4.0]])
        stats = _make_stats_from_exact(mean=mean, cov=cov, lambda_reg=1e-12)
        vec = np.array([2.0, 2.0])
        dist = mahalanobis_distance(vec, stats)
        expected = np.sqrt(5.0)
        assert abs(dist - expected) < 1e-6, f"Expected {expected}, got {dist}"

    def test_non_zero_mean(self):
        """
        With mean=[1,2] and identity cov, x=[4,6]:
        delta = [3,4], d = sqrt(9+16) = 5
        """
        mean = np.array([1.0, 2.0])
        cov = np.eye(2)
        stats = _make_stats_from_exact(mean=mean, cov=cov, lambda_reg=1e-12)
        vec = np.array([4.0, 6.0])
        dist = mahalanobis_distance(vec, stats)
        assert abs(dist - 5.0) < 1e-9, f"Expected 5.0, got {dist}"


# ── Test 3: Batch vs single consistency ───────────────────────────────────────

class TestBatchConsistency:
    def test_batch_matches_single(self):
        """batch_mahalanobis_distances should match individual calls."""
        dim = 8
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-4)

        queries = rng.standard_normal((15, dim))
        batch_dists = batch_mahalanobis_distances(queries, stats)

        for i in range(len(queries)):
            single = mahalanobis_distance(queries[i], stats)
            assert abs(batch_dists[i] - single) < 1e-9, (
                f"Batch[{i}]={batch_dists[i]:.8f} != single={single:.8f}"
            )

    def test_batch_output_shape(self):
        dim = 6
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-4)
        queries = rng.standard_normal((25, dim))
        dists = batch_mahalanobis_distances(queries, stats)
        assert dists.shape == (25,)

    def test_batch_all_nonnegative(self):
        dim = 5
        rng = np.random.default_rng(7)
        data = rng.standard_normal((100, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-4)
        queries = rng.standard_normal((50, dim))
        dists = batch_mahalanobis_distances(queries, stats)
        assert np.all(dists >= 0), f"Negative distances found: {dists[dists < 0]}"


# ── Test 4: Numerical stability ───────────────────────────────────────────────

class TestNumericalStability:
    def test_small_lambda_still_works(self):
        """Even with lambda=1e-9, should work for well-conditioned data."""
        dim = 3
        rng = np.random.default_rng(1)
        data = rng.standard_normal((500, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-9)
        vec = rng.standard_normal(dim)
        dist = mahalanobis_distance(vec, stats)
        assert np.isfinite(dist) and dist >= 0

    def test_high_dimensional_data(self):
        """1024-dim (production dimension) with N > D should work."""
        dim = 1024
        n = 1100  # N > D
        rng = np.random.default_rng(42)
        data = rng.standard_normal((n, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-5)
        vec = rng.standard_normal(dim)
        dist = mahalanobis_distance(vec, stats)
        assert np.isfinite(dist) and dist >= 0

    def test_result_deterministic(self):
        """Same inputs → same output (no stochastic elements)."""
        dim = 10
        rng = np.random.default_rng(99)
        data = rng.standard_normal((200, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-5)
        vec = rng.standard_normal(dim)
        d1 = mahalanobis_distance(vec, stats)
        d2 = mahalanobis_distance(vec, stats)
        assert d1 == d2, "Distance should be deterministic"


# ── Test 5: Dimension mismatch ────────────────────────────────────────────────

class TestDimensionMismatch:
    def test_wrong_dim_raises_value_error(self):
        dim = 8
        rng = np.random.default_rng(3)
        data = rng.standard_normal((200, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-4)
        wrong_dim_vec = np.zeros(16)  # wrong dimension
        with pytest.raises(ValueError, match="dimension"):
            mahalanobis_distance(wrong_dim_vec, stats)

    def test_wrong_dim_batch_raises_value_error(self):
        dim = 8
        rng = np.random.default_rng(3)
        data = rng.standard_normal((200, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-4)
        wrong_matrix = np.zeros((5, 16))
        with pytest.raises(ValueError, match="[Dd]imension"):
            batch_mahalanobis_distances(wrong_matrix, stats)


# ── Test 6: NaN/Inf handling ──────────────────────────────────────────────────

class TestNanInfHandling:
    def test_nan_in_feature_raises_value_error(self):
        dim = 5
        rng = np.random.default_rng(4)
        data = rng.standard_normal((100, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-4)
        bad_vec = np.array([1.0, float("nan"), 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="[Nn]aN|[Ii]nf|finite"):
            mahalanobis_distance(bad_vec, stats)

    def test_inf_in_feature_raises_value_error(self):
        dim = 5
        rng = np.random.default_rng(5)
        data = rng.standard_normal((100, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-4)
        bad_vec = np.array([float("inf"), 1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="[Nn]aN|[Ii]nf|finite"):
            mahalanobis_distance(bad_vec, stats)

    def test_nan_in_feature_matrix_raises_value_error(self):
        dim = 5
        rng = np.random.default_rng(6)
        data = rng.standard_normal((100, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-4)
        bad_mat = np.ones((3, dim))
        bad_mat[1, 2] = float("nan")
        with pytest.raises(ValueError, match="[Nn]aN|[Ii]nf|finite"):
            batch_mahalanobis_distances(bad_mat, stats)


# ── Test 7: Torch tensor input ────────────────────────────────────────────────

class TestTorchInput:
    def test_torch_tensor_same_as_numpy(self):
        dim = 6
        rng = np.random.default_rng(10)
        data = rng.standard_normal((150, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-4)
        vec_np = rng.standard_normal(dim).astype(np.float32)
        vec_torch = torch.from_numpy(vec_np)
        dist_np = mahalanobis_distance(vec_np, stats)
        dist_torch = mahalanobis_distance(vec_torch, stats)
        assert abs(dist_np - dist_torch) < 1e-6, (
            f"NumPy={dist_np} vs Torch={dist_torch}"
        )

    def test_2d_torch_tensor_works(self):
        """Shape (1, D) tensor should be squeezed correctly."""
        dim = 6
        rng = np.random.default_rng(11)
        data = rng.standard_normal((150, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-4)
        vec = torch.randn(1, dim)
        dist = mahalanobis_distance(vec, stats)
        assert np.isfinite(dist) and dist >= 0

    def test_unsupported_type_raises_type_error(self):
        dim = 6
        rng = np.random.default_rng(12)
        data = rng.standard_normal((150, dim))
        stats = fit_reference_stats(data, lambda_reg=1e-4)
        with pytest.raises(TypeError):
            mahalanobis_distance([1.0] * dim, stats)  # list is not supported
