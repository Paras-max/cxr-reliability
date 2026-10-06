"""Calibration and Threshold Analysis page (Phase 12).

Responsibility:
    Visualize probability calibration results (Platt / Isotonic scaling)
    and threshold optimization metrics from outputs/calibration/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cxr_reliability.dashboard.components import render_disclaimer_banner

PROJECT_ROOT = _SRC.parent


def main() -> None:
    st.set_page_config(
        page_title="Calibration & Thresholds - CXR Reliability",
        page_icon="⚖️",
        layout="wide",
    )
    st.title("Probability Calibration & Threshold Optimization")
    st.caption("Post-Hoc Platt Scaling, Reliability Curves, and Validation Set Operating Points")
    render_disclaimer_banner()
    st.divider()

    calib_dir = PROJECT_ROOT / "outputs" / "calibration"
    calib_json = calib_dir / "calibration_results.json"
    thresh_csv = calib_dir / "threshold_results.csv"
    calib_curve_img = calib_dir / "calibration_curve.png"
    thresh_analysis_img = calib_dir / "threshold_analysis.png"

    if not calib_json.exists():
        st.warning("No calibration results found under `outputs/calibration/`.")
        return

    try:
        with open(calib_json, encoding="utf-8") as f:
            calib_data = json.load(f)
    except Exception as e:
        st.error(f"Error loading calibration results: {e}")
        return

    # Top KPI summary
    st.subheader("Calibration Quality Metrics")
    raw_ece = calib_data.get("raw_ECE", 0.0)
    calib_ece = calib_data.get("calibrated_ECE", 0.0)
    raw_brier = calib_data.get("raw_Brier", 0.0)
    calib_brier = calib_data.get("calibrated_Brier", 0.0)
    roc_auc = calib_data.get("roc_auc", 0.0)
    opt_thresh = calib_data.get("selected_threshold", 0.5)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Selected Threshold", f"{opt_thresh:.4f}")
    k2.metric("ROC AUC", f"{roc_auc:.4f}")
    k3.metric("Raw ECE", f"{raw_ece:.4f}")
    k4.metric("Calibrated ECE", f"{calib_ece:.6f}", delta=f"{-(raw_ece - calib_ece):.4f}")
    k5.metric("Calibrated Brier", f"{calib_brier:.4f}", delta=f"{-(raw_brier - calib_brier):.4f}")

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reliability Diagram (Calibration Curve)")
        if calib_curve_img.exists():
            st.image(str(calib_curve_img), caption="Pre vs Post-Hoc Platt Reliability Curves", use_container_width=True)
        else:
            st.info("Calibration curve image not found.")

    with col2:
        st.subheader("Threshold Operating Analysis")
        if thresh_analysis_img.exists():
            st.image(str(thresh_analysis_img), caption="F1, Sensitivity, Specificity across Thresholds", use_container_width=True)
        else:
            st.info("Threshold analysis image not found.")

    st.markdown("---")

    # Strategy comparison table
    st.subheader("Candidate Decision Threshold Strategies")
    candidates = calib_data.get("threshold_candidates", {})
    if candidates:
        rows = []
        for name, m in candidates.items():
            rows.append({
                "Strategy": name,
                "Threshold": m.get("threshold"),
                "F1 Score": m.get("f1"),
                "Recall (Sens.)": m.get("recall"),
                "Specificity": m.get("specificity"),
                "Balanced Acc": m.get("balanced_accuracy"),
                "TP": m.get("tp"),
                "FP": m.get("fp"),
                "TN": m.get("tn"),
                "FN": m.get("fn"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Threshold results csv preview
    if thresh_csv.exists():
        st.markdown("---")
        st.subheader("Threshold Grid Search Table")
        try:
            df_th = pd.read_csv(thresh_csv)
            st.dataframe(df_th.head(100), use_container_width=True, hide_index=True)
            st.caption(f"Showing sample of {len(df_th)} evaluated candidate thresholds on validation set.")
        except Exception as e:
            st.error(f"Error loading threshold CSV: {e}")


if __name__ == "__main__":
    main()
