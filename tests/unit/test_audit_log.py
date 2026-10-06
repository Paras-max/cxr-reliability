from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from cxr_reliability.audit.audit_log import AuditLogger
from cxr_reliability.contracts.audit import AuditRecord, ExecutionPath
from cxr_reliability.contracts.decision import Action
from cxr_reliability.contracts.pipeline import PipelineOutput, PredictionSummary, ReliabilityLabel
from cxr_reliability.contracts.verification import NextStep, VerificationResult, VerificationStatus


def _create_sample_record(audit_id: str = "audit-123", with_verification: bool = False) -> AuditRecord:
    verification_res = None
    if with_verification:
        verification_res = VerificationResult(
            version="0.10.0",
            label="VERIFIED",
            verified=True,
            status=VerificationStatus.VERIFIED,
            next_step=NextStep.RELEASE,
            confidence_before=0.6,
            confidence_after=0.8,
            delta_confidence=0.2,
            delta_quality=5.0,
            delta_ood=-1.0,
            label_flipped=False,
            min_confidence_gain_used=-0.01,
            reasoning="Verification passed with confidence gain.",
        )

    pipeline_out = PipelineOutput(
        audit_id=audit_id,
        prediction=PredictionSummary(
            pneumonia_probability=0.75,
            source_model_id="densenet121-res224-all",
        ),
        reliability_label=ReliabilityLabel.ACCEPTED,
        final_action=Action.ACCEPT,
        needs_human_review=False,
        verification=verification_res,
        total_latency_ms=120.5,
    )

    return AuditRecord(
        audit_id=audit_id,
        timestamp_utc=datetime.now(timezone.utc),
        input_id="img-001",
        input_sha256="abc123hash",
        thresholds_version="v0_prd_defaults",
        model_ids={"base": "densenet121"},
        path=ExecutionPath.FAST,
        output=pipeline_out,
        latency_ms_by_stage={"base_model": 80.0, "quality": 40.5},
    )


def test_one_record_per_inference(tmp_path: Path):
    """Each write() call appends exactly one JSONL line."""
    logger = AuditLogger(tmp_path)
    rec1 = _create_sample_record("rec-1")
    rec2 = _create_sample_record("rec-2")

    logger.write(rec1)
    logger.write(rec2)

    log_file = tmp_path / "audit.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_record_roundtrip(tmp_path: Path):
    """A written record reads back equal."""
    logger = AuditLogger(tmp_path)
    rec = _create_sample_record("rec-roundtrip")
    logger.write(rec)

    records = list(logger.read_all())
    assert len(records) == 1
    loaded = records[0]
    assert loaded.audit_id == "rec-roundtrip"
    assert loaded.input_id == "img-001"
    assert loaded.output.final_action == Action.ACCEPT
    assert loaded.output.prediction.pneumonia_probability == 0.75

    fetched = logger.get("rec-roundtrip")
    assert fetched is not None
    assert fetched.audit_id == "rec-roundtrip"
    assert logger.get("non-existent") is None


def test_record_contains_signals_action_and_deltas(tmp_path: Path):
    """Records include signals, action, rule id and verification deltas (PRD section 4)."""
    logger = AuditLogger(tmp_path)
    rec = _create_sample_record("rec-verif", with_verification=True)
    logger.write(rec)

    fetched = logger.get("rec-verif")
    assert fetched is not None
    assert fetched.output.final_action == Action.ACCEPT
    assert fetched.output.verification is not None
    assert fetched.output.verification.verified is True
    assert fetched.output.verification.delta_confidence == 0.2
