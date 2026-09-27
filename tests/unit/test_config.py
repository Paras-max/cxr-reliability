"""Tests for configuration schemas and the v0 PRD-default thresholds file."""
from cxr_reliability.config.pipeline_config import load_pipeline_config
from cxr_reliability.config.thresholds import load_thresholds


def test_v0_thresholds_hold_prd_starting_points(v0_thresholds_path):
    t = load_thresholds(v0_thresholds_path)
    assert t.provenance == "prd_defaults"
    assert t.quality.blur_laplacian_var_min == 100.0
    assert t.quality.snr_db_min == 15.0
    assert t.quality.exposure_mean_min == 20.0
    assert t.quality.exposure_mean_max == 235.0
    assert t.ood.in_distribution_percentile == 95.0
    assert t.verification.min_confidence_gain == 0.15


def test_values_the_prd_does_not_specify_are_unresolved(v0_thresholds_path):
    unresolved = set(load_thresholds(v0_thresholds_path).unresolved())
    for name in [
        "ood.mahalanobis_severe",
        "uncertainty.confidence_high_min",
        "uncertainty.confidence_low_max",
        "uncertainty.positive_class_threshold",
        "decision.borderline_margin",
        "repair.max_repairable_blur_pct",
        "quality.reference_max_laplacian_var",
    ]:
        assert name in unresolved
    assert "quality.blur_laplacian_var_min" not in unresolved


def test_pipeline_config_uses_prd_models(pipeline_config_path):
    cfg = load_pipeline_config(pipeline_config_path)
    assert cfg.models.fast_model_id == "densenet121-res224-nih"
    assert cfg.models.escalation_model_id == "resnet50-res512-all"
    assert cfg.audit.enabled is True
    assert cfg.execution.borderline_action in {"escalate", "reject"}
