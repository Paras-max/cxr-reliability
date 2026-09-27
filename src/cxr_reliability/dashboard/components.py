"""Shared dashboard components (Phase 12).

Responsibility:
    Modular Streamlit UI rendering components for the reliability dashboard.
    Consumes PipelineResult contracts to render research-prototype metrics,
    agent signals, before/after repair visual comparisons, and audit traces.

Safety & Governance:
    - Never generates clinical claims or diagnostic statements.
    - Explicitly labels all model scores as uncalibrated model scores.
    - OOD levels are presented strictly as distribution-shift categories.
    - Uncertainty is documented as binary entropy derived from sigmoid output.
    - Prominently displays the mandatory research-prototype disclaimer.

Dependencies:
    streamlit, PIL, contracts.*
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

from cxr_reliability.contracts.base_model import BaseModelResult
from cxr_reliability.contracts.common import DISCLAIMER
from cxr_reliability.contracts.decision import DecisionResult
from cxr_reliability.contracts.ood import OODLevel, OODResult
from cxr_reliability.contracts.pipeline import (
    PipelineOutput,
    PipelineResult,
    PipelineState,
    PredictionSummary,
    ReliabilityLabel,
)
from cxr_reliability.contracts.quality import QualityLevel, QualityResult
from cxr_reliability.contracts.repair import RepairResult
from cxr_reliability.contracts.uncertainty import UncertaintyLevel, UncertaintyResult
from cxr_reliability.contracts.verification import NextStep, VerificationResult, VerificationStatus

RESEARCH_DISCLAIMER_TEXT = (
    "Research prototype only — not clinically validated and not intended for medical diagnosis."
)


def render_disclaimer_banner() -> None:
    """Render the persistent research prototype disclaimer banner."""
    st.warning(f"🔬 **{RESEARCH_DISCLAIMER_TEXT}**")


def render_header() -> None:
    """Render page title, subtitle, and primary disclaimer."""
    st.title("Reliability-Aware Chest X-Ray Analysis")
    st.caption("Multi-Agent Reliability Assessment — Research Prototype")
    render_disclaimer_banner()
    st.divider()


def render_sidebar(
    device: str = "CPU",
    thresholds_version: str = "v0_prd_defaults",
    ood_info: dict[str, Any] | None = None,
) -> None:
    """Render sidebar with project description, safety notices, and device info."""
    with st.sidebar:
        st.header("About System")
        st.info(
            "This research prototype demonstrates a reliability-aware multi-agent architecture "
            "for chest radiograph analysis. It evaluates image quality, out-of-distribution shift, "
            "and predictive uncertainty before allowing model inference or initiating non-destructive "
            "repair and delta verification."
        )

        st.subheader("Safety & Scope")
        st.caption(
            "**Non-Clinical Research System**: The outputs produced by this tool are experimental "
            "model scores and reliability flags. They do not constitute medical advice or diagnosis."
        )

        st.divider()
        st.subheader("System Status")
        st.write(f"**Execution Device:** `{device}`")
        st.write(f"**Thresholds:** `{thresholds_version}`")
        st.write("**Calibration Status:** `Provisional (Uncalibrated)`")

        if ood_info:
            is_dev = ood_info.get("is_dev", False)
            path_str = ood_info.get("path", "")
            n_samples = ood_info.get("n_samples")
            if is_dev:
                st.write(f"**OOD Stats:** `🟡 Development (N={n_samples or 'subset'})`")
                if path_str:
                    st.caption(f"Artifact path: `{path_str}`")
                st.caption(
                    "_Fitted on a development training subset for software testing. "
                    "Not for clinical or final research evaluation._"
                )
            else:
                st.write("**OOD Stats:** `🟢 Full Research Reference`")
                if path_str:
                    st.caption(f"Artifact path: `{path_str}`")
        else:
            st.write("**OOD Stats:** `Reference baseline (Phase 13 pending)`")


def render_human_review_banner(result: PipelineResult) -> None:
    """Render prominent banner when human review is required."""
    if result.needs_human_review:
        st.error(
            "⚠️ **HUMAN REVIEW REQUIRED**\n\n"
            f"**Reason:** {result.reason}\n\n"
            "_Model prediction is withheld. A human clinical expert must inspect this radiograph._"
        )


def render_final_result(result: PipelineResult) -> None:
    """Render the top-level pipeline outcome cards."""
    st.subheader("System Reliability Assessment")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        state_val = result.pipeline_state.value.upper()
        if result.pipeline_state in (PipelineState.ACCEPT, PipelineState.VERIFIED):
            st.metric("Pipeline State", state_val, delta="Released")
        elif result.pipeline_state == PipelineState.ERROR:
            st.metric("Pipeline State", state_val, delta="-Error")
        else:
            st.metric("Pipeline State", state_val, delta="-Withheld")

    with c2:
        st.metric("Final Action", result.final_action.value.upper())

    with c3:
        st.metric("Reliability Label", result.reliability_label.value)

    with c4:
        hr_status = "REQUIRED" if result.needs_human_review else "NOT REQUIRED"
        st.metric("Human Review", hr_status)

    if result.reason:
        with st.expander("System Assessment Rationale", expanded=True):
            st.write(result.reason)


def render_base_model_section(
    base_model: BaseModelResult | None,
    prediction: PredictionSummary | None,
) -> None:
    """Render Base Model agent score and metadata."""
    st.subheader("Base Model Output & Probability Calibration")
    if base_model is None:
        st.info("Base model was not executed or output is unavailable.")
        return

    raw_score = getattr(base_model, "raw_pneumonia_score", base_model.pneumonia_probability)
    calibrated_prob = getattr(base_model, "calibrated_probability", None)
    if calibrated_prob is None and prediction is not None:
        calibrated_prob = prediction.calibrated_probability

    raw_thresh = 0.522161
    if prediction is not None and prediction.raw_decision_threshold is not None:
        raw_thresh = prediction.raw_decision_threshold
    elif prediction is not None and prediction.decision_threshold is not None:
        raw_thresh = prediction.decision_threshold

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Raw Model Sigmoid Score", f"{raw_score:.4f}")
        st.caption("Uncalibrated DenseNet-121 sigmoid output in [0, 1]. Reflects model margin.")

    with c2:
        if calibrated_prob is not None:
            st.metric("Calibrated Probability", f"{calibrated_prob:.4f}")
            st.caption("Platt-calibrated probability (NIH val base rate ~1.12%). Research prototype only.")
        else:
            st.metric("Calibrated Probability", "Unavailable")
            st.caption("Calibration artifact not loaded; raw score used.")

    with c3:
        st.metric("Raw-Score Decision Threshold", f"{raw_thresh:.4f}")
        st.caption("Validated F1-optimal operating point on raw score scale.")

    if prediction is not None and prediction.positive is not None:
        pred_text = f"Positive (Raw Score >= {raw_thresh:.4f})" if prediction.positive else f"Negative (Raw Score < {raw_thresh:.4f})"
        st.write(f"**Classification Status:** `{pred_text}`")
        if prediction.calibrated_decision_threshold is not None:
            st.caption(f"Equivalent calibrated operating threshold: `{prediction.calibrated_decision_threshold:.4f}`")
    else:
        st.write("**Prediction Output:** `Withheld pending review`")

    st.caption(
        "_Note: Probability calibration maps raw model activations to empirical event frequencies on the NIH validation set "
        "(Platt scaling ECE: 0.00004). It does NOT represent a clinical diagnosis or clinical patient risk._"
    )


    if base_model.all_pathology_outputs:
        with st.expander("All Pathology Scores (TorchXRayVision Output)"):
            df = pd.DataFrame(
                list(base_model.all_pathology_outputs.items()),
                columns=["Pathology", "Sigmoid Score"],
            )
            st.dataframe(df, use_container_width=True)


def render_quality_section(quality: QualityResult | None) -> None:
    """Render image quality evaluation metrics and defect flags."""
    st.subheader("Image Quality Screening")
    if quality is None:
        st.info("Quality assessment was not executed.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Quality Level", quality.overall.value.upper())
    with c2:
        st.metric("Laplacian Var (Sharpness)", f"{quality.laplacian_variance:.1f}")
    with c3:
        st.metric("SNR", f"{quality.snr_db:.1f} dB")
    with c4:
        st.metric("Mean Intensity", f"{quality.mean_intensity:.1f}")

    # Defect flags
    flags = []
    if quality.flags.blur:
        flags.append("Blur detected")
    if quality.flags.noise:
        flags.append("Excessive noise detected")
    if quality.flags.exposure:
        flags.append("Improper exposure detected")

    if flags:
        st.warning(f"**Identified Quality Defects:** {', '.join(flags)}")
    else:
        st.success("No critical quality defects detected.")

    st.write(f"**Reasoning:** {quality.reasoning}")


def render_ood_section(ood: OODResult | None, ood_info: dict[str, Any] | None = None) -> None:
    """Render out-of-distribution shift detection."""
    st.subheader("Distribution Shift (OOD)")
    if ood is None:
        st.info("OOD assessment was not executed.")
        return

    if ood_info and ood_info.get("is_dev"):
        n_samples = ood_info.get("n_samples", "subset")
        st.warning(
            f"🟡 **PROVISIONAL DEVELOPMENT REFERENCE (N={n_samples} training images)**: "
            "This OOD score is evaluated against a development subset of the training distribution. "
            "It is intended solely for local software verification and UI testing. "
            "It is NOT clinically validated and NOT for final research evaluation."
        )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("OOD Category", ood.level.value.upper())
        st.caption("Distribution shift category (not a clinical severity rating).")
    with c2:
        st.metric("Mahalanobis Distance", f"{ood.mahalanobis_distance:.2f}")
        st.caption(f"Reference threshold: {ood.mahalanobis_threshold:.2f}")

    if ood.energy_score is not None:
        st.write(f"**Approximate Energy Score:** `{ood.energy_score:.2f}`")

    if not (ood_info and ood_info.get("is_dev")):
        st.caption(
            "_Note: OOD reference statistics and distance cuts are provisional baseline values. "
            "Empirical calibration across external cohorts is planned for Phase 13._"
        )
    st.write(f"**Reasoning:** {ood.reasoning}")


def render_uncertainty_section(uncertainty: UncertaintyResult | None) -> None:
    """Render predictive uncertainty signals."""
    st.subheader("Predictive Uncertainty")
    if uncertainty is None:
        st.info("Uncertainty assessment was not executed.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Uncertainty Level", uncertainty.uncertainty_level.value.upper())
    with c2:
        st.metric("Model Confidence", f"{uncertainty.confidence:.4f}")
    with c3:
        st.metric("Normalized Entropy", f"{uncertainty.normalized_entropy:.4f}")

    st.caption(
        "_Note: Uncertainty uses a 2-level classification (LOW: confidence >= 0.85 and normalized entropy <= 0.25; HIGH: otherwise). "
        "It is computed from binary entropy and confidence of the uncalibrated raw score. This is not calibrated predictive uncertainty._"
    )
    st.write(f"**Reasoning:** {uncertainty.reasoning}")


def render_decision_section(decision_history: list[DecisionResult]) -> None:
    """Render decision routing records."""
    st.subheader("Decision Agent Routing")
    if not decision_history:
        st.info("No decision records available.")
        return

    for idx, dec in enumerate(decision_history, start=1):
        st.write(f"**Decision #{idx}:** `{dec.action.value.upper()}` (Rule: `{dec.rule_id}`)")
        st.write(f"**Reasoning:** {dec.reasoning}")
        if dec.driving_signals:
            st.caption(f"Driving signals: {dec.driving_signals}")


def render_repair_section(
    repair: RepairResult | None,
    original_img: Any,
    repaired_png: bytes | None,
) -> None:
    """Render image repair details and side-by-side comparison."""
    if repair is None:
        return

    st.subheader("Image Restoration / Repair")
    if not repair.repair_applied:
        st.info(f"Repair was evaluated but not applied: {repair.skipped_reason or 'Skipped'}")
        return

    st.success("Targeted non-destructive repair was applied to address detected defects.")
    st.write(f"**Defects Addressed:** {', '.join(str(d) for d in repair.defects_detected)}")

    if repair.steps:
        st.write("**Applied Repair Steps:**")
        for step in repair.steps:
            st.write(f"- Method: `{step.method}` (Defect: `{step.defect}`, Params: `{step.parameters}`)")

    st.caption(
        "_Note: Repair operations are non-destructive image restorations (e.g. CLAHE, bilateral filter). "
        "They do not alter diagnostic anatomical features and require post-repair verification._"
    )

    # Side-by-side comparison
    st.write("### Before / After Image Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption="Original Image (Unaltered)", use_container_width=True)
    with col2:
        if repaired_png:
            st.image(repaired_png, caption="Repaired Image (Restored)", use_container_width=True)
        else:
            st.caption("Repaired image preview unavailable.")


def render_verification_section(verification: VerificationResult | None) -> None:
    """Render delta verification outcome."""
    if verification is None:
        return

    st.subheader("Post-Repair Verification")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Verification Status", verification.status.value.upper())
    with c2:
        st.metric("Confidence Gain", f"{verification.delta_confidence:+.4f}")
    with c3:
        st.metric("Quality Gain", f"{verification.delta_quality:+.2f}")
    with c4:
        st.metric("Next Step", verification.next_step.value.upper())

    st.caption(
        "_Note: Verification thresholds are provisional research parameters and not clinically calibrated._"
    )
    st.write(f"**Verification Rationale:** {verification.reasoning}")


def render_agent_trace(output: Any) -> None:
    """Render comprehensive audit and latency trace (supports PipelineResult or PipelineOutput)."""
    with st.expander("Pipeline Audit / Trace"):
        if output is None:
            st.info("No audit trace available.")
            return

        st.write(f"**Audit ID:** `{getattr(output, 'audit_id', 'N/A')}`")
        st.write(f"**Total Latency:** `{getattr(output, 'total_latency_ms', 0.0):.1f} ms`")

        latencies = getattr(output, "intermediate_latencies", None)
        if latencies:
            st.write("**Stage Latencies (ms):**")
            df = pd.DataFrame(
                list(latencies.items()),
                columns=["Pipeline Stage", "Latency (ms)"],
            )
            st.dataframe(df, use_container_width=True)

        err = getattr(output, "error_message", None)
        if err:
            st.error(f"**Error Details:** {err}")

        st.caption(f"**Disclaimer:** {getattr(output, 'disclaimer', DISCLAIMER)}")
