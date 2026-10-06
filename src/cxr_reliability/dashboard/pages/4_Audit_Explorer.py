"""Audit Log Explorer page (Phase 12).

Responsibility:
    Browse, filter, and inspect immutable per-inference audit records
    stored in logs/audit/ using AuditLogger.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cxr_reliability.audit.audit_log import AuditLogger
from cxr_reliability.dashboard.components import render_disclaimer_banner

PROJECT_ROOT = _SRC.parent


def main() -> None:
    st.set_page_config(
        page_title="Audit Explorer - CXR Reliability",
        page_icon="📜",
        layout="wide",
    )
    st.title("Audit Trail & Provenance Explorer")
    st.caption("Immutable per-inference records: signals, action taken, verification deltas, and model provenance")
    render_disclaimer_banner()
    st.divider()

    audit_dir = PROJECT_ROOT / "logs" / "audit"
    logger = AuditLogger(audit_dir)

    records = list(logger.read_all())

    st.subheader(f"Stored Audit Records ({len(records):,})")

    if not records:
        st.info(
            "No audit records found in `logs/audit/`. "
            "When inferences are processed with audit logging enabled, each run appends an immutable JSONL record here."
        )
        return

    # Convert records to table format
    rows = []
    for r in records:
        out = r.output
        rows.append({
            "Audit ID": r.audit_id,
            "Timestamp": str(r.timestamp_utc),
            "Input ID": r.input_id or "N/A",
            "Action": out.final_action.value if out.final_action else "N/A",
            "Human Review": "Yes" if out.needs_human_review else "No",
            "Reliability Label": out.reliability_label.value if out.reliability_label else "N/A",
            "Execution Path": r.path.value if r.path else "N/A",
            "Pneumonia Prob": f"{out.prediction.pneumonia_probability:.4f}" if out.prediction else "Withheld",
            "Latency (ms)": f"{out.total_latency_ms:.1f}" if out.total_latency_ms else "N/A",
        })

    df = pd.DataFrame(rows)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        actions_list = ["All"] + sorted(df["Action"].unique().tolist())
        selected_act = st.selectbox("Filter by Final Action:", actions_list)
    with col2:
        search_id = st.text_input("Search by Audit ID or Input ID:")

    filtered_df = df
    if selected_act != "All":
        filtered_df = filtered_df[filtered_df["Action"] == selected_act]
    if search_id.strip():
        q = search_id.strip().lower()
        filtered_df = filtered_df[
            filtered_df["Audit ID"].str.lower().str.contains(q)
            | filtered_df["Input ID"].str.lower().str.contains(q)
        ]

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    # Detail inspector
    st.markdown("---")
    st.subheader("Record Detail Inspector")
    selected_audit_id = st.selectbox(
        "Select Audit ID to inspect complete trace:",
        [r.audit_id for r in records],
    )

    selected_record = logger.get(selected_audit_id)
    if selected_record:
        st.json(selected_record.model_dump_json())


if __name__ == "__main__":
    main()
