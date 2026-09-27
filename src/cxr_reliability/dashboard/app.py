"""Streamlit entry point — Reliability-Aware Chest X-Ray Analysis (Phase 12).

Responsibility:
    Main dashboard application for the reliability-aware multi-agent system.
    Invokes the Phase 11 ReliabilityPipeline as the single source of truth
    and renders multi-agent signals, reliability assessments, before/after
    repair comparisons, and audit traces.

Governance & Safety:
    - Research prototype only; not intended for clinical diagnosis.
    - Zero duplication of agent decision or repair logic.
    - Local inference only; no external network transmission.

Launch Command:
    streamlit run src/cxr_reliability/dashboard/app.py
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Ensure project src is in sys.path
_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cxr_reliability.agents.decision.rules import RuleTableDecisionAgent
from cxr_reliability.agents.ood import OODAgent
from cxr_reliability.agents.quality import QualityAgent
from cxr_reliability.agents.repair import RepairAgent
from cxr_reliability.agents.uncertainty import UncertaintyAgent
from cxr_reliability.agents.verification import VerificationAgent
from cxr_reliability.config.pipeline_config import ExecutionConfig
from cxr_reliability.config.thresholds import Thresholds, load_thresholds
from cxr_reliability.contracts.common import DISCLAIMER
from cxr_reliability.contracts.pipeline import PipelineResult, PipelineState
from cxr_reliability.dashboard.components import (
    render_agent_trace,
    render_base_model_section,
    render_decision_section,
    render_disclaimer_banner,
    render_final_result,
    render_header,
    render_human_review_banner,
    render_ood_section,
    render_quality_section,
    render_repair_section,
    render_sidebar,
    render_uncertainty_section,
    render_verification_section,
)
from cxr_reliability.models.base_model import BaseModelAgent
from cxr_reliability.pipeline.orchestrator import ReliabilityPipeline

logger = logging.getLogger(__name__)

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "thresholds" / "v0_prd_defaults.yaml"
DEV_OOD_STATS_PATH = PROJECT_ROOT / "artifacts" / "ood_dev"
FULL_OOD_STATS_PATH = PROJECT_ROOT / "artifacts" / "ood"
LEGACY_OOD_STATS_PATH = PROJECT_ROOT / "outputs" / "ood"


def discover_available_ood_stats() -> dict[str, Path]:
    """Discover available OOD reference directories in order of preference."""
    import os
    available: dict[str, Path] = {}

    env_dir = os.environ.get("CXR_OOD_STATS_DIR")
    if env_dir:
        p = Path(env_dir)
        if (p / "reference_stats.npz").exists():
            available[f"Environment Override ({p})"] = p

    if (FULL_OOD_STATS_PATH / "reference_stats.npz").exists():
        available["Full Research Reference (artifacts/ood)"] = FULL_OOD_STATS_PATH
    if (DEV_OOD_STATS_PATH / "reference_stats.npz").exists():
        available["Development Reference (artifacts/ood_dev)"] = DEV_OOD_STATS_PATH
    if (LEGACY_OOD_STATS_PATH / "reference_stats.npz").exists():
        available["Legacy Reference (outputs/ood)"] = LEGACY_OOD_STATS_PATH

    return available


def resolve_ood_stats_path(requested_dir: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    """
    Resolve active OOD reference path and metadata.
    """
    if requested_dir is not None:
        p = Path(requested_dir)
    else:
        available = discover_available_ood_stats()
        if available:
            p = next(iter(available.values()))
        else:
            p = DEV_OOD_STATS_PATH

    # Read metadata if exists
    meta_file = p / "metadata.json"
    meta: dict[str, Any] = {}
    if meta_file.exists():
        try:
            import json
            with open(meta_file, encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}

    is_dev = meta.get("is_development", False) or "ood_dev" in str(p).lower()
    n_samples = meta.get("n_train_samples") or meta.get("n_train_images_processed")

    info = {
        "path": str(p),
        "is_dev": is_dev,
        "n_samples": n_samples,
        "metadata": meta,
        "exists": (p / "reference_stats.npz").exists(),
    }
    return p, info


@st.cache_resource(show_spinner="Initializing Reliability Pipeline and loading base model...")
def load_pipeline(ood_stats_path_str: str | None = None) -> tuple[ReliabilityPipeline, Thresholds, str, dict[str, Any]]:
    """
    Initialize all agents and build the ReliabilityPipeline once.
    Cached across browser interactions using st.cache_resource.
    """
    # 1. Detect compute device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load thresholds
    if CONFIG_PATH.exists():
        thresholds = load_thresholds(CONFIG_PATH)
    else:
        from cxr_reliability.config.thresholds import (
            DecisionThresholds,
            OODThresholds,
            QualityThresholds,
            RepairThresholds,
            UncertaintyThresholds,
            VerificationThresholds,
        )
        thresholds = Thresholds(
            version="v0_fallback",
            provenance="prd_defaults",
            quality=QualityThresholds(blur_laplacian_var_min=100.0, snr_db_min=15.0),
            ood=OODThresholds(in_distribution_percentile=95.0),
            uncertainty=UncertaintyThresholds(),
            decision=DecisionThresholds(borderline_margin=0.05),
            repair=RepairThresholds(),
            verification=VerificationThresholds(min_confidence_gain=0.15),
        )

    # 3. Instantiate agents
    quality_agent = QualityAgent()

    base_model = BaseModelAgent(device=device)
    base_model.load_model()

    uncertainty_agent = UncertaintyAgent()

    ood_path, ood_info = resolve_ood_stats_path(ood_stats_path_str)

    ood_agent = OODAgent(
        stats_dir=ood_path,
        thresholds=thresholds.ood,
        thresholds_version=thresholds.version,
    )

    decision_agent = RuleTableDecisionAgent(
        thresholds=thresholds.decision,
        execution=ExecutionConfig(),
        thresholds_version=thresholds.version,
    )

    repair_agent = RepairAgent()
    verification_agent = VerificationAgent()

    # 4. Build orchestrator
    pipeline = ReliabilityPipeline(
        thresholds=thresholds,
        quality_agent=quality_agent,
        base_model=base_model,
        ood_agent=ood_agent,
        uncertainty_agent=uncertainty_agent,
        decision_agent=decision_agent,
        repair_agent=repair_agent,
        verification_agent=verification_agent,
    )

    return pipeline, thresholds, device, ood_info


def main() -> None:
    """Streamlit dashboard main application."""
    st.set_page_config(
        page_title="Reliability-Aware CXR Analysis",
        page_icon="🩻",
        layout="wide",
    )

    # Render header and persistent disclaimer
    render_header()

    # Discover available OOD reference directories
    available_ood = discover_available_ood_stats()
    selected_path_str = None
    if len(available_ood) > 1:
        st.sidebar.subheader("OOD Reference Selection")
        chosen_label = st.sidebar.selectbox("Active OOD Reference", list(available_ood.keys()))
        selected_path_str = str(available_ood[chosen_label])
    elif len(available_ood) == 1:
        selected_path_str = str(next(iter(available_ood.values())))

    # Load cached pipeline
    try:
        pipeline, thresholds, device, ood_info = load_pipeline(selected_path_str)
    except Exception as exc:
        st.error(f"Failed to initialize Reliability Pipeline: {exc}")
        st.info("Ensure model weights and required dependencies are configured.")
        return

    # Render sidebar
    render_sidebar(device=device.upper(), thresholds_version=thresholds.version, ood_info=ood_info)

    # Image upload section
    st.subheader("1. Image Ingestion")
    uploaded_file = st.file_uploader(
        "Upload Chest Radiograph (Supported formats: PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        help="All processing is performed locally on your machine. Images are not transmitted externally.",
    )

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to read uploaded image: {exc}")
            return

        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(pil_image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        with col_info:
            st.write(f"**Filename:** `{uploaded_file.name}`")
            st.write(f"**Image Dimensions:** `{pil_image.size[0]} x {pil_image.size[1]}` px")
            st.write(f"**Color Mode:** `{pil_image.mode}`")
            st.caption(
                "_Upload verified. Click 'Analyze X-Ray' below to initiate the multi-agent reliability assessment._"
            )

        st.divider()

        # Explicit analysis trigger
        st.subheader("2. Reliability Assessment Execution")
        if st.button("Analyze X-Ray", type="primary"):
            with st.spinner("Executing Multi-Agent Reliability Pipeline..."):
                try:
                    # Convert to 2D grayscale uint8 array for pipeline
                    grayscale_arr = np.array(pil_image.convert("L"))
                    input_id = f"upload_{uploaded_file.name}_{uuid.uuid4().hex[:6]}"

                    # The pipeline is the single source of truth
                    result = pipeline.predict(grayscale_arr, input_id=input_id)

                    # Store in session state (per-session, never global across users)
                    st.session_state["pipeline_result"] = result
                    st.session_state["uploaded_image"] = pil_image
                except Exception as exc:
                    st.error(f"Pipeline execution encountered an unexpected error: {exc}")
                    logger.exception("Dashboard analysis error")
                    return

    # Render results if available in session state
    result: PipelineResult | None = st.session_state.get("pipeline_result")
    saved_img: Image.Image | None = st.session_state.get("uploaded_image")

    if result is not None:
        st.divider()
        st.header("3. Assessment Results")

        # Human Review Alert (if required)
        render_human_review_banner(result)

        # Top-level result summary
        render_final_result(result)

        # Before / After Repair section (if repair occurred)
        if result.repair is not None and result.repair.repair_applied:
            st.divider()
            render_repair_section(result.repair, saved_img, result.repaired_image_png)
            render_verification_section(result.verification)

        # Detailed agent inspection tabs
        st.divider()
        st.subheader("Detailed Agent Signal Inspection")
        tab_model, tab_quality, tab_decision, tab_audit = st.tabs(
            [
                "Base Model & Uncertainty",
                "Quality & Distribution (OOD)",
                "Decision Routing",
                "Full Audit / Latency Trace",
            ]
        )

        with tab_model:
            render_base_model_section(result.base_model, result.prediction)
            st.divider()
            render_uncertainty_section(result.uncertainty)

        with tab_quality:
            render_quality_section(result.quality)
            st.divider()
            render_ood_section(result.ood, ood_info=ood_info)

        with tab_decision:
            render_decision_section(result.decision_history)

        with tab_audit:
            render_agent_trace(result)


if __name__ == "__main__":
    main()
