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

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from cxr_reliability.contracts.base_model import BaseModelResult
from cxr_reliability.contracts.common import DISCLAIMER
from cxr_reliability.contracts.decision import DecisionResult
from cxr_reliability.contracts.ood import OODResult
from cxr_reliability.contracts.pipeline import (
    FinalClassification,
    PipelineResult,
    PipelineState,
    PredictionSummary,
)
from cxr_reliability.contracts.quality import QualityResult
from cxr_reliability.contracts.repair import RepairResult
from cxr_reliability.contracts.uncertainty import UncertaintyResult
from cxr_reliability.contracts.verification import VerificationResult

RESEARCH_DISCLAIMER_TEXT = (
    "Research prototype only — not clinically validated and not intended for medical diagnosis."
)


def render_pneumonia_classification(result: PipelineResult) -> None:
    """
    Render top-level Pneumonia Classification section (Phase additive requirement).

    Displays one of:
      - PNEUMONIA DETECTED (Reliability: ACCEPTED)
      - PNEUMONIA NOT DETECTED (Reliability: ACCEPTED)
      - HUMAN REVIEW REQUIRED (Reliable classification was not released.)
    """
    st.subheader("PNEUMONIA CLASSIFICATION")

    classification = getattr(result, "final_classification", None)
    if classification is None:
        cls_val = "HUMAN_REVIEW_REQUIRED" if result.needs_human_review else (
            "PNEUMONIA" if (result.prediction and result.prediction.positive) else "NO_PNEUMONIA"
        )
    else:
        cls_val = classification.value if hasattr(classification, "value") else str(classification)

    rel_label = (
        result.reliability_label.value.upper()
        if hasattr(result.reliability_label, "value")
        else str(result.reliability_label).upper()
    )

    if cls_val == "PNEUMONIA":
        st.markdown(
            f"""
            <div style="background: rgba(255, 75, 75, 0.08); border: 2px solid #ff4b4b; border-radius: 10px; padding: 22px; text-align: center; margin: 10px 0 20px 0;">
                <h1 style="color: #ff4b4b; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: 1.5px;">PNEUMONIA DETECTED</h1>
                <p style="font-size: 1.15rem; margin-top: 12px; margin-bottom: 0; font-weight: 600;">
                    Reliability: <span style="color: #00c04b; font-weight: 700;">{rel_label}</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif cls_val == "NO_PNEUMONIA":
        st.markdown(
            f"""
            <div style="background: rgba(0, 192, 75, 0.08); border: 2px solid #00c04b; border-radius: 10px; padding: 22px; text-align: center; margin: 10px 0 20px 0;">
                <h1 style="color: #00c04b; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: 1.5px;">PNEUMONIA NOT DETECTED</h1>
                <p style="font-size: 1.15rem; margin-top: 12px; margin-bottom: 0; font-weight: 600;">
                    Reliability: <span style="color: #00c04b; font-weight: 700;">{rel_label}</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(255, 170, 0, 0.08); border: 2px solid #ffaa00; border-radius: 10px; padding: 22px; text-align: center; margin: 10px 0 20px 0;">
                <h1 style="color: #ffaa00; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: 1.5px;">HUMAN REVIEW REQUIRED</h1>
                <p style="font-size: 1.15rem; margin-top: 12px; margin-bottom: 0; font-weight: 600; color: #ffaa00;">
                    Reliable classification was not released.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_probability_before_after(result: PipelineResult) -> None:
    """Render Initial vs Final Pneumonia Probability section and threshold visualization (additive)."""
    st.subheader("Pneumonia Probability — Before vs After")

    init_prob = getattr(result, "initial_pneumonia_probability", None)
    if init_prob is None and result.base_model is not None:
        init_prob = float(getattr(result.base_model, "raw_pneumonia_score", result.base_model.pneumonia_probability))

    final_prob = getattr(result, "final_pneumonia_probability", None)
    if final_prob is None:
        if result.after_repair_base_model is not None:
            final_prob = float(getattr(result.after_repair_base_model, "raw_pneumonia_score", result.after_repair_base_model.pneumonia_probability))
        elif init_prob is not None:
            final_prob = init_prob

    delta = getattr(result, "probability_delta", None)
    if delta is None and final_prob is not None and init_prob is not None:
        delta = final_prob - init_prob

    thresh = 0.522161

    init_class = getattr(result, "initial_classification", None)
    if init_class is None and init_prob is not None:
        init_class = "Pneumonia" if init_prob >= thresh else "No Pneumonia"

    final_class_raw = getattr(result, "final_classification", None)
    if final_class_raw is not None:
        final_class = final_class_raw.value if hasattr(final_class_raw, "value") else str(final_class_raw)
    else:
        final_class = "HUMAN_REVIEW_REQUIRED" if result.needs_human_review else (
            "PNEUMONIA" if (result.prediction and result.prediction.positive) else "NO_PNEUMONIA"
        )

    init_str = f"{init_prob * 100:.2f}%" if init_prob is not None else "N/A"
    final_str = f"{final_prob * 100:.2f}%" if final_prob is not None else "N/A"
    if delta is not None:
        delta_pp = delta * 100.0
        delta_str = f"{delta_pp:+.2f} percentage points"
    else:
        delta_str = "0.00 percentage points"

    col_tbl, col_meta = st.columns([3, 2])
    with col_tbl:
        table_md = f"""
| Prediction Stage | Pneumonia Probability |
|:---|---:|
| **Initial DenseNet** | `{init_str}` |
| **Final Pipeline Output** | `{final_str}` |
| **Change** | `{delta_str}` |
"""
        st.markdown(table_md)

    with col_meta:
        st.markdown(f"**Initial Classification:** `{init_class or 'N/A'}`")
        st.markdown(f"**Final Classification:** `{final_class}`")
        st.markdown(f"**Operating Threshold:** `52.2161%` (0.522161)")

    # ── Threshold Visualization ──────────────────────────────────────────────
    if init_prob is not None and final_prob is not None:
        init_pct = max(0.0, min(100.0, init_prob * 100.0))
        final_pct = max(0.0, min(100.0, final_prob * 100.0))
        thresh_pct = 52.2161

        viz_html = f"""
        <div style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 8px; padding: 16px 20px; margin: 12px 0 20px 0;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600; margin-bottom: 8px;">
                <span>0%</span>
                <span style="color: #f59e0b;">Operating Threshold: 52.2161%</span>
                <span>100%</span>
            </div>
            <div style="position: relative; height: 16px; background: #262730; border-radius: 8px; margin: 12px 0 24px 0;">
                <div style="position: absolute; left: {thresh_pct}%; top: -6px; bottom: -6px; width: 3px; background: #f59e0b; border-radius: 2px; z-index: 2;"></div>
                <div style="position: absolute; left: {init_pct}%; top: 50%; transform: translate(-50%, -50%); width: 14px; height: 14px; border-radius: 50%; background: #3b82f6; border: 2px solid #ffffff; z-index: 3;" title="Initial: {init_str}"></div>
                <div style="position: absolute; left: {final_pct}%; top: 50%; transform: translate(-50%, -50%); width: 14px; height: 14px; border-radius: 50%; background: #10b981; border: 2px solid #ffffff; z-index: 4;" title="Final: {final_str}"></div>
            </div>
            <div style="display: flex; gap: 20px; font-size: 0.82rem; justify-content: center; flex-wrap: wrap;">
                <div><span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #3b82f6; margin-right: 5px;"></span>Initial: <strong>{init_str}</strong></div>
                <div><span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #10b981; margin-right: 5px;"></span>Final: <strong>{final_str}</strong></div>
                <div><span style="display: inline-block; width: 3px; height: 10px; background: #f59e0b; margin-right: 5px;"></span>Threshold: <strong>52.2161%</strong></div>
            </div>
        </div>
        """
        st.markdown(viz_html, unsafe_allow_html=True)

    # ── Final Output Table ───────────────────────────────────────────────────
    init_quality = result.quality.overall.value.upper() if result.quality else "N/A"
    final_quality = result.after_repair_quality.overall.value.upper() if result.after_repair_quality else init_quality
    init_decision = result.decision_history[0].action.value.upper() if result.decision_history else "N/A"
    final_decision = result.decision_history[-1].action.value.upper() if result.decision_history else "N/A"
    final_action = result.final_action.value.upper() if result.final_action else "N/A"

    output_table_md = f"""
| Output | Before | After |
|:---|:---|:---|
| **Pneumonia Probability** | {init_str} | {final_str} |
| **Classification** | {init_class or 'N/A'} | {final_class} |
| **Image Quality** | {init_quality} | {final_quality} |
| **Decision** | {init_decision} | {final_decision} |
| **Final Action** | — | {final_action} |
"""
    with st.expander("Comprehensive Before vs After Summary Table", expanded=True):
        st.markdown(output_table_md)


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

        st.divider()
        st.subheader("Presentation & Viva")
        import pathlib
        presentation_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "SYSTEM_PRESENTATION.html"
        if presentation_path.exists():
            with open(presentation_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📥 Download Presentation (HTML)",
                    data=f.read().encode("utf-8"),
                    file_name="CXR_System_Executive_Presentation.html",
                    mime="text/html",
                    use_container_width=True,
                )


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


def _extract_quality_label(obj: Any) -> str | None:
    """Extract uppercase categorical quality label (e.g. POOR, DEGRADED, GOOD)."""
    if obj is None:
        return None
    if hasattr(obj, "value"):
        return str(obj.value).upper()
    if hasattr(obj, "overall"):
        val = getattr(obj.overall, "value", obj.overall)
        return str(val).upper()
    if hasattr(obj, "label") and isinstance(obj.label, str):
        val = obj.label.strip().upper()
        if val in ("POOR", "DEGRADED", "GOOD"):
            return val
    if isinstance(obj, str):
        val = obj.strip().upper()
        if val in ("POOR", "DEGRADED", "GOOD"):
            return val
    return None


def render_verification_section(
    verification: VerificationResult | None,
    quality_before: QualityResult | None = None,
    quality_after: QualityResult | None = None,
) -> None:
    """Render delta verification outcome with categorical quality transition."""
    if verification is None:
        return

    st.subheader("Post-Repair Verification")

    # 1. Resolve categorical quality before and after
    q_before = _extract_quality_label(quality_before)
    q_after = _extract_quality_label(quality_after)

    if q_before is None:
        q_before = _extract_quality_label(getattr(verification, "quality_level_before", None))
    if q_after is None:
        q_after = _extract_quality_label(getattr(verification, "quality_level_after", None))

    if q_before is None:
        q_before = _extract_quality_label(getattr(verification, "quality_before", None))
    if q_after is None:
        q_after = _extract_quality_label(getattr(verification, "quality_after", None))

    # Fallback to parsing verification reasoning if not directly populated on objects
    reasoning = getattr(verification, "reasoning", "") or ""
    if q_before is None or q_after is None:
        import re

        m = re.search(r"\b(POOR|DEGRADED|GOOD)\s*(?:->|→|to)\s*(POOR|DEGRADED|GOOD)\b", reasoning, re.IGNORECASE)
        if m:
            if q_before is None:
                q_before = m.group(1).upper()
            if q_after is None:
                q_after = m.group(2).upper()
        else:
            m_to = re.search(r"improved to\s+(POOR|DEGRADED|GOOD)", reasoning, re.IGNORECASE)
            if m_to and q_after is None:
                q_after = m_to.group(1).upper()
                if q_before is None:
                    q_before = "POOR"
            m_rem = re.search(r"remained\s+(POOR|DEGRADED|GOOD)", reasoning, re.IGNORECASE)
            if m_rem:
                lvl = m_rem.group(1).upper()
                if q_before is None:
                    q_before = lvl
                if q_after is None:
                    q_after = lvl

    # Safe defaults if neither contracts nor reasoning explicitly specify levels
    if q_before is None:
        q_before = "POOR" if getattr(verification, "delta_quality", 0.0) != 0 else "UNKNOWN"
    if q_after is None:
        q_after = "GOOD" if getattr(verification, "verified", False) else q_before

    transition_str = f"{q_before} → {q_after}"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Verification Status", verification.status.value.upper())
    with c2:
        st.metric("Confidence Gain", f"{verification.delta_confidence:+.4f}")
    with c3:
        st.metric("Quality Transition", transition_str)
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
