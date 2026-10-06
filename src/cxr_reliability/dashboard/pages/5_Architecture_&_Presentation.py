"""Streamlit page — Architecture, Flowchart & Presentation Handouts.

Allows examiners, professors, and reviewers to view interactive architecture diagrams
and download presentation assets, flowcharts, block diagrams, and master cheatsheets.
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

st.set_page_config(
    page_title="Architecture & Presentation | CXR Reliability",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ System Architecture, Flowchart & Presentation")
st.caption("Executive overview and downloadable presentation materials for project viva and evaluations.")

st.divider()

# Download Bar
st.subheader("📥 Download Project Handouts & Presentation")
col1, col2, col3, col4 = st.columns(4)

html_path = _PROJECT_ROOT / "SYSTEM_PRESENTATION.html"
flowchart_path = _PROJECT_ROOT / "FLOWCHART.md"
block_path = _PROJECT_ROOT / "BLOCK_DIAGRAM.md"
cheatsheet_path = _PROJECT_ROOT / "PROJECT_MASTER_CHEATSHEET.md"

if html_path.exists():
    with open(html_path, "r", encoding="utf-8") as f:
        html_bytes = f.read().encode("utf-8")
    col1.download_button(
        label="📄 Download HTML Presentation",
        data=html_bytes,
        file_name="CXR_System_Executive_Presentation.html",
        mime="text/html",
        help="Self-contained interactive HTML presentation with live diagrams and printable PDF layout.",
        use_container_width=True,
    )

if flowchart_path.exists():
    with open(flowchart_path, "r", encoding="utf-8") as f:
        fc_bytes = f.read().encode("utf-8")
    col2.download_button(
        label="🔄 Download Flowchart (.md)",
        data=fc_bytes,
        file_name="FLOWCHART.md",
        mime="text/markdown",
        use_container_width=True,
    )

if block_path.exists():
    with open(block_path, "r", encoding="utf-8") as f:
        bd_bytes = f.read().encode("utf-8")
    col3.download_button(
        label="🏛️ Download Block Diagram (.md)",
        data=bd_bytes,
        file_name="BLOCK_DIAGRAM.md",
        mime="text/markdown",
        use_container_width=True,
    )

if cheatsheet_path.exists():
    with open(cheatsheet_path, "r", encoding="utf-8") as f:
        cs_bytes = f.read().encode("utf-8")
    col4.download_button(
        label="📑 Download Master Cheatsheet (.md)",
        data=cs_bytes,
        file_name="PROJECT_MASTER_CHEATSHEET.md",
        mime="text/markdown",
        use_container_width=True,
    )

st.divider()

# Tabs for visual inspection
tab_block, tab_flow, tab_metrics, tab_viva = st.tabs([
    "🏛️ System Block Diagram",
    "🔄 Pipeline Flowchart",
    "📊 Benchmark Results",
    "💬 1-Minute Viva Speech & Q&A"
])

with tab_block:
    st.markdown("### 🏛️ Multi-Agent System Architecture")
    st.markdown("""
    The architecture separates model inference into **5 distinct operational layers**:
    1. **Presentation Layer:** Streamlit Dashboard, CLI scripts, and 448 automated test harnesses.
    2. **Orchestration Layer:** `ReliabilityPipeline` coordinator with immutable Pydantic type-checks.
    3. **Multi-Agent Specialist Layer:** 7 specialized agents (Quality, BaseModel, OOD, Uncertainty, Decision, Repair, Verification).
    4. **Reference Artifacts Layer:** 1024-D reference statistics from 78,299 patient-strict training images, Platt calibration models.
    5. **Clinical Routing Layer:** Final triage into **ACCEPTED (Released)**, **ESCALATE (Doctor Review)**, or **REJECT (Corrupted/OOD)**.
    """)
    if block_path.exists():
        with open(block_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())

with tab_flow:
    st.markdown("### 🔄 End-to-End Decision Flowchart")
    st.markdown("""
    Every X-ray undergoes a standardized pipeline where defects trigger bounded non-destructive repair, and every repair is subjected to a **5-gate anti-hallucination verification check**.
    """)
    if flowchart_path.exists():
        with open(flowchart_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())

with tab_metrics:
    st.markdown("### 📊 Scientific Benchmark Performance (16,724 Test Images)")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Negative Predictive Value", "99.02%", "Triage Safe")
    mcol2.metric("Released Accuracy", "92.31%", "+8.2% vs Raw Model")
    mcol3.metric("Safely Automated", "43.94%", "7,349 CXRs")
    mcol4.metric("Withheld for Review", "56.06%", "Zero Silent Failures")

    st.markdown("""
    | Evaluation Dimension | Benchmark Result | Clinical Significance |
    |---|---|---|
    | **OOD Rejection** | 100% on synthetic/invalid images | Blocks wrong anatomy or non-medical images |
    | **Quality Defect Catch Rate** | 100% sensitivity on noise/blur | Catches corrupted scans before inference |
    | **Repair Integrity** | 78.28% improved, 0.00% worsened | Guarantees non-destructive image restoration |
    | **Anti-Hallucination Guard** | 0.00% label flips allowed | Predictions never change disease class post-repair |
    | **ECE Calibration** | 31.15% &rarr; 0.004% (Platt Scaling) | Confidence probabilities accurately reflect true error |
    """)

with tab_viva:
    st.markdown("### 💬 1-Minute Speech & Viva Talking Points")
    st.info(
        "\"Respected Sir, conventional medical AI models fail silently — they produce confident, incorrect outputs "
        "when fed blurry or unfamiliar X-rays. Our multi-agent system wraps a DenseNet-121 pneumonia classifier "
        "with 7 reliability agents. We evaluate image quality, check for out-of-distribution shift via Mahalanobis distance, "
        "quantify predictive entropy, and repair degraded images using a 5-gate verified loop. Across 16,724 test images, "
        "we safely released 44% with 99.02% NPV while escalating all uncertain cases to human clinicians.\""
    )
    st.markdown("""
    - **Why Multi-Agent?** Each agent has a single, testable responsibility. When a prediction is withheld, clinicians get an exact audit explanation (e.g. *'Low SNR / High Entropy'*) rather than a silent failure.
    - **Why Zero Patient Overlap?** Splitting by patient ID prevents the network from memorizing individual thoracic structures, ensuring honest generalization.
    - **Why Platt Scaling?** Raw deep learning models output overconfident logits. Platt scaling aligns model confidence with true clinical frequencies.
    """)
