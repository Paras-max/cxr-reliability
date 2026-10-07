"""Evaluation results page (Phase 12 / Agent-Wise Before vs After Evaluation Framework).

Responsibility:
    Visualize quantitative performance, agent-wise before vs after impact,
    and reliability auditing from outputs/before_after_evaluation_summary.json
    and outputs/evaluation_full_test.csv.
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
        page_title="Evaluation Results - CXR Reliability",
        page_icon="📊",
        layout="wide",
    )
    st.title("System Evaluation & Agent-Wise Benchmark Results")
    st.caption("Comprehensive Before vs After Analysis, Multi-Agent Signal Diagnostics, and Reliability Gating")
    render_disclaimer_banner()
    st.divider()

    # Load agent-wise before-after summary
    ba_summary_path = PROJECT_ROOT / "outputs" / "before_after_evaluation_summary.json"
    full_summary_path = PROJECT_ROOT / "outputs" / "evaluation_full_test_summary.json"
    if not full_summary_path.exists():
        full_summary_path = PROJECT_ROOT / "outputs" / "evaluation_smoke_100_summary.json"

    csv_path = PROJECT_ROOT / "outputs" / "evaluation_full_test.csv"
    if not csv_path.exists():
        csv_path = PROJECT_ROOT / "outputs" / "evaluation_smoke_100.csv"

    ba_data = {}
    if ba_summary_path.exists():
        try:
            with open(ba_summary_path, encoding="utf-8") as f:
                ba_data = json.load(f)
        except Exception as e:
            st.error(f"Error reading before-after summary: {e}")

    summary_data = {}
    if full_summary_path.exists():
        try:
            with open(full_summary_path, encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception as e:
            st.error(f"Error reading summary file: {e}")

    if not ba_data and not summary_data and not csv_path.exists():
        st.warning("No evaluation artifacts found under `outputs/`. Run `python scripts/run_before_after_evaluation.py` to populate data.")
        return

    # Top KPI Row
    st.subheader("Key Performance Indicators")
    total_imgs = ba_data.get("total_test_images", summary_data.get("total_images", "N/A"))
    coverage_pct = ba_data.get("coverage_pct", summary_data.get("human_review", {}).get("coverage_pct", None))
    cov_str = f"{coverage_pct:.1f}%" if coverage_pct is not None else "N/A"

    bm_metrics = ba_data.get("base_model", {})
    bm_acc = bm_metrics.get("accuracy")
    bm_acc_str = f"{bm_acc * 100:.1f}%" if bm_acc is not None else "N/A"

    cls_metrics = ba_data.get("reliability_system_released", summary_data.get("classification_on_released_predictions", {}))
    acc = cls_metrics.get("accuracy")
    acc_str = f"{acc * 100:.1f}%" if acc is not None else "N/A"
    sens = cls_metrics.get("recall_sensitivity")
    sens_str = f"{sens * 100:.1f}%" if sens is not None else "N/A"
    spec = cls_metrics.get("specificity")
    spec_str = f"{spec * 100:.1f}%" if spec is not None else "N/A"
    f1 = cls_metrics.get("f1_score")
    f1_str = f"{f1:.3f}" if f1 is not None else "N/A"

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Total Test Cohort", f"{total_imgs:,}" if isinstance(total_imgs, int) else total_imgs)
    kpi2.metric("Base Model Accuracy", bm_acc_str)
    kpi3.metric("Release Coverage", cov_str)
    kpi4.metric("Released Accuracy", acc_str, delta=f"+{(acc - bm_acc)*100:.2f}%" if (acc and bm_acc) else None)
    kpi5.metric("Released Specificity", spec_str)
    kpi6.metric("Released F1 Score", f1_str)

    st.markdown("---")

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 System Comparison",
        "🤖 Agent-Wise Performance",
        "🔄 Before vs After Repair",
        "🛡️ Safety & Confusion Matrices",
        "🔍 Data Explorer",
    ])

    # TAB 1: System Comparison
    with tab1:
        st.subheader("Base Model vs Reliability-Aware System")
        st.caption(
            "Comparing standalone DenseNet-121 performance across the entire test set against selective predictions released by the reliability layer."
        )

        if ba_data:
            bm = ba_data["base_model"]
            rel = ba_data["reliability_system_released"]
            sf = ba_data["safety_filtering"]

            col_comp, col_safe = st.columns([3, 2])

            with col_comp:
                df_comparison = pd.DataFrame([
                    {"Metric": "Cohort Population", "Base Model": f"{bm['n_total']:,} (Full Test)", "Reliability System": f"{rel['n_total']:,} (Released)", "Delta / Change": f"-{bm['n_total'] - rel['n_total']:,} (-{ba_data['withhold_pct']:.1f}%)"},
                    {"Metric": "Accuracy", "Base Model": f"{bm['accuracy']*100:.2f}%", "Reliability System": f"{rel['accuracy']*100:.2f}%", "Delta / Change": f"+{(rel['accuracy'] - bm['accuracy'])*100:+.2f}%"},
                    {"Metric": "Specificity", "Base Model": f"{bm['specificity']*100:.2f}%", "Reliability System": f"{rel['specificity']*100:.2f}%", "Delta / Change": f"+{(rel['specificity'] - bm['specificity'])*100:+.2f}%"},
                    {"Metric": "Recall / Sensitivity", "Base Model": f"{bm['recall_sensitivity']*100:.2f}%", "Reliability System": f"{rel['recall_sensitivity']*100:.2f}%", "Delta / Change": f"{(rel['recall_sensitivity'] - bm['recall_sensitivity'])*100:+.2f}%"},
                    {"Metric": "Precision", "Base Model": f"{bm['precision']*100:.2f}%", "Reliability System": f"{rel['precision']*100:.2f}%", "Delta / Change": f"{(rel['precision'] - bm['precision'])*100:+.2f}%"},
                    {"Metric": "Negative Predictive Value", "Base Model": f"{bm['npv']*100:.2f}%", "Reliability System": f"{rel['npv']*100:.2f}%", "Delta / Change": f"+{(rel['npv'] - bm['npv'])*100:+.2f}%"},
                    {"Metric": "F1 Score", "Base Model": f"{bm['f1_score']:.4f}", "Reliability System": f"{rel['f1_score']:.4f}", "Delta / Change": f"{(rel['f1_score'] - bm['f1_score']):+.4f}"},
                    {"Metric": "Prediction Coverage", "Base Model": "100.0%", "Reliability System": f"{ba_data['coverage_pct']:.1f}%", "Delta / Change": f"-{100 - ba_data['coverage_pct']:.1f}%"},
                    {"Metric": "Human Review Rate", "Base Model": "0.0%", "Reliability System": f"{ba_data['withhold_pct']:.1f}%", "Delta / Change": f"+{ba_data['withhold_pct']:.1f}%"},
                ])
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)

            with col_safe:
                st.markdown("**Error Reduction & Clinical Safety**")
                st.markdown(
                    f"""
                    - **False Positives Withheld:** **{sf['errors_withheld_from_automated_release']['false_positives_withheld']:,}** ({sf['errors_withheld_from_automated_release']['false_positives_withheld_pct']:.1f}% of Base Model FPs)
                    - **False Negatives Withheld:** **{sf['errors_withheld_from_automated_release']['false_negatives_withheld']:,}** ({sf['errors_withheld_from_automated_release']['false_negatives_withheld_pct']:.1f}% of Base Model FNs)
                    - **Total Errors Prevented from Release:** **{sf['errors_withheld_from_automated_release']['total_errors_withheld']:,}** ({sf['errors_withheld_from_automated_release']['total_errors_withheld_pct']:.1f}%)
                    """
                )
                chart_path = PROJECT_ROOT / "outputs" / "before_after" / "base_vs_reliability_metrics.png"
                if chart_path.exists():
                    st.image(str(chart_path), caption="Base Model vs Selective Reliability System (N=16,724 vs N=7,349)")

    # TAB 2: Agent-Wise Breakdown
    with tab2:
        st.subheader("Agent-Wise Diagnostic & Routing Breakdown")
        st.info("Mathematical Integrity Rule: Classification metrics (Accuracy, F1, AUROC) are strictly reported as 'N/A' for agents where no independent ground truth exists.")

        col_a1, col_a2 = st.columns(2)

        with col_a1:
            st.markdown("### 1. Quality Agent")
            if ba_data and "quality_transitions" in ba_data:
                qt = ba_data["quality_transitions"]
                st.markdown(f"**Initial Cohort Quality ($N = {qt['n_total_images']:,}$):**")
                st.write(f"- **GOOD:** {qt['initial_distribution']['good']:,} ({qt['initial_distribution']['pct_good']:.1f}%)")
                st.write(f"- **DEGRADED:** {qt['initial_distribution']['degraded']:,} ({qt['initial_distribution']['pct_degraded']:.1f}%)")
                st.write(f"- **POOR:** {qt['initial_distribution']['poor']:,} ({qt['initial_distribution']['pct_poor']:.1f}%)")

                st.markdown(f"**Quality Transitions After Repair ($N = {qt['n_repairs_evaluated']:,}$ Repairs):**")
                st.write(f"- **Poor -> Good Recovery:** {qt['poor_to_good_recovery_rate_pct']:.1f}% ({qt['transitions_breakdown']['poor_to_good']:,} cases)")
                st.write(f"- **Poor -> Degraded:** {qt['transitions_breakdown']['poor_to_degraded']:,} cases")
                st.write(f"- **Poor -> Poor (Unchanged):** {qt['transitions_breakdown']['poor_to_poor']:,} cases ({qt['quality_unchanged_rate_pct']:.1f}%)")
                st.write(f"- **Quality Worsened Rate:** {qt['quality_worsening_rate_pct']:.2f}% (0 cases)")
                st.caption(f"F1 / Accuracy: `{qt['f1_score']}`")

            st.markdown("---")
            st.markdown("### 2. OOD Agent")
            if ba_data and "ood_agent" in ba_data:
                ood = ba_data["ood_agent"]
                st.markdown(f"**Representation-Space Distribution ($N = {ood['n_total']:,}$):**")
                st.write(f"- **IN_DISTRIBUTION:** {ood['counts']['in_distribution']:,} ({ood['percentages']['in_distribution']:.2f}%)")
                st.write(f"- **BORDERLINE:** {ood['counts']['borderline']:,} ({ood['percentages']['borderline']:.2f}%) -> 100% Escalate")
                st.write(f"- **SEVERE:** {ood['counts']['severe']:,} ({ood['percentages']['severe']:.2f}%) -> 100% Reject")
                m_stats = ood["mahalanobis_distance_stats"]
                st.write(f"- **Mahalanobis Distance:** Mean = {m_stats.get('mean', 0):.2f}, Median = {m_stats.get('median', 0):.2f}, 95th % = {m_stats.get('p95', 0):.2f}")
                st.caption(f"OOD AUROC / Accuracy: `{ood['f1_score']}`")

        with col_a2:
            st.markdown("### 3. Uncertainty Agent")
            if ba_data and "uncertainty_agent" in ba_data:
                unc = ba_data["uncertainty_agent"]
                st.markdown(f"**Uncertainty Distribution ($N = {unc['n_total']:,}$):**")
                st.write(f"- **LOW Uncertainty:** {unc['low_count']:,} ({unc['pct_low']:.1f}%) | Released: {unc['groups']['LOW']['released_count']:,}")
                st.write(f"- **HIGH Uncertainty:** {unc['high_count']:,} ({unc['pct_high']:.1f}%) | Released: {unc['groups']['HIGH']['released_count']:,} (post-repair verified)")

            st.markdown("---")
            st.markdown("### 4. Decision Agent")
            if ba_data and "decision_agent" in ba_data:
                dec = ba_data["decision_agent"]
                st.markdown(f"**Initial Action Routing ($N = {dec['n_total']:,}$):**")
                st.write(f"- **ACCEPT:** {dec['action_counts']['ACCEPT']:,} ({dec['action_percentages']['ACCEPT']:.1f}%)")
                st.write(f"- **REPAIR:** {dec['action_counts']['REPAIR']:,} ({dec['action_percentages']['REPAIR']:.1f}%)")
                st.write(f"- **ESCALATE:** {dec['action_counts']['ESCALATE']:,} ({dec['action_percentages']['ESCALATE']:.1f}%)")
                st.write(f"- **REJECT:** {dec['action_counts']['REJECT']:,} ({dec['action_percentages']['REJECT']:.1f}%)")
                st.caption(f"Decision Accuracy: `{dec['accuracy']}`")

            st.markdown("---")
            st.markdown("### 5. Verification Agent")
            if ba_data and "verification_agent" in ba_data:
                ver = ba_data["verification_agent"]
                st.markdown(f"**Gatekeeping Outcomes ($N = {ver['n_entering_verification']:,}$ Repaired Cases):**")
                st.write(f"- **Verified & Released:** **{ver['verified_count']:,} ({ver['verified_rate_pct']:.1f}%)**")
                st.write(f"- **Escalated & Withheld:** **{ver['escalated_count']:,} ({ver['escalated_rate_pct']:.1f}%)**")
                st.markdown("**Failure Reasons Breakdown:**")
                reasons = ver["failure_reasons"]
                st.write(f"- Unresolved Poor Quality: {reasons.get('quality_unresolved_poor_remained_poor', 0):,} cases")
                st.write(f"- Partial Poor to Degraded: {reasons.get('quality_partial_poor_to_degraded', 0):,} cases")
                st.write(f"- Confidence Guard Exceeded (Delta < -0.01): {reasons.get('confidence_non_degradation_guard_exceeded', 0):,} cases")
                st.caption(f"Verification Accuracy: `{ver['accuracy']}`")

    # TAB 3: Before vs After Repair
    with tab3:
        st.subheader("Paired & Synthetic Before vs After Repair Evaluation")

        if ba_data and "paired_repair_benchmark" in ba_data:
            pb = ba_data["paired_repair_benchmark"]
            st.markdown(f"#### Paired Diagnostic Performance (Exact Same Image Cohort, $N_{{before}} = N_{{after}} = {pb['n_paired']}$)")
            st.caption("Demonstrating whether image sharpening directly improved downstream classification metrics on the exact same patient images.")

            mb = pb["metrics_before"]
            ma = pb["metrics_after"]
            md = pb["metrics_delta"]

            df_paired = pd.DataFrame([
                {"Metric": "Accuracy", "Before Repair": f"{mb['accuracy']*100:.2f}%", "After Repair": f"{ma['accuracy']*100:.2f}%", "Delta": f"{md['accuracy_delta']*100:+.2f}%"},
                {"Metric": "Precision", "Before Repair": f"{mb['precision']*100:.2f}%", "After Repair": f"{ma['precision']*100:.2f}%", "Delta": f"{md['precision_delta']*100:+.2f}%"},
                {"Metric": "Recall / Sensitivity", "Before Repair": f"{mb['recall_sensitivity']*100:.2f}%", "After Repair": f"{ma['recall_sensitivity']*100:.2f}%", "Delta": f"{md['recall_delta']*100:+.2f}%"},
                {"Metric": "Specificity", "Before Repair": f"{mb['specificity']*100:.2f}%", "After Repair": f"{ma['specificity']*100:.2f}%", "Delta": f"{md['specificity_delta']*100:+.2f}%"},
                {"Metric": "F1 Score", "Before Repair": f"{mb['f1_score']:.4f}", "After Repair": f"{ma['f1_score']:.4f}", "Delta": f"{md['f1_delta']:+.4f}"},
                {"Metric": "Negative Predictive Value", "Before Repair": f"{mb['npv']*100:.2f}%", "After Repair": f"{ma['npv']*100:.2f}%", "Delta": f"{md['npv_delta']*100:+.2f}%"},
            ])
            st.dataframe(df_paired, use_container_width=True, hide_index=True)

            sig = pb["signal_deltas"]
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("**Continuous Laplacian Variance (Blur Metric):**")
                lap = sig["laplacian_variance"]
                st.write(f"- Mean: {lap['mean_before']:.1f} -> **{lap['mean_after']:.1f}** (Delta: +{lap['mean_delta']:.1f})")
                st.write(f"- Median: {lap['median_before']:.1f} -> **{lap['median_after']:.1f}**")
                st.write(f"- Improvement Rate: **{lap['pct_improved']:.1f}%**")
            with col_s2:
                st.markdown("**Model Confidence & Stability:**")
                conf = sig["model_confidence"]
                stab = sig["prediction_stability"]
                st.write(f"- Mean Confidence: {conf['mean_before']:.4f} -> {conf['mean_after']:.4f} (Delta: {conf['mean_delta']:+.4f})")
                st.write(f"- Prediction Flips: **{stab['total_flips']}** flips (Stability Rate: **{stab['stability_pct']:.1f}%**)")

        st.markdown("---")
        st.subheader("Controlled Synthetic Defect Benchmarks")
        st.caption("Since natural NIH images have no ground-truth noise or exposure defect labels, controlled benchmarks evaluate NLMeans and CLAHE repairs.")

        col_sn, col_se = st.columns(2)
        with col_sn:
            st.markdown("#### Synthetic Noise Benchmark (NLMeans)")
            if ba_data and "synthetic_noise_benchmark" in ba_data:
                s_noise = ba_data["synthetic_noise_benchmark"]
                st.write(f"- Evaluated Images: {s_noise.get('n_samples', 0)}")
                st.write(f"- Corruption: Additive Gaussian Noise (sigma = {s_noise.get('corruption_severity', 0)})")
                snr = s_noise.get("snr_stats", {})
                st.write(f"- Original SNR: {snr.get('mean_snr_original', 0):.2f} dB")
                st.write(f"- Corrupted SNR: {snr.get('mean_snr_corrupted', 0):.2f} dB")
                st.write(f"- Repaired SNR: **{snr.get('mean_snr_repaired', 0):.2f} dB**")
                st.caption(s_noise.get("disclaimer", ""))

        with col_se:
            st.markdown("#### Synthetic Exposure Benchmark (CLAHE)")
            if ba_data and "synthetic_exposure_benchmark" in ba_data:
                s_exp = ba_data["synthetic_exposure_benchmark"]
                st.write(f"- Evaluated Images: {s_exp.get('n_samples', 0)}")
                st.write(f"- Corruption: Exposure Shift (severity = {s_exp.get('corruption_severity', 0)})")
                intens = s_exp.get("intensity_stats", {})
                st.write(f"- Original Intensity: {intens.get('mean_intensity_original', 0):.1f}")
                st.write(f"- Corrupted Intensity: {intens.get('mean_intensity_corrupted', 0):.1f}")
                st.write(f"- Repaired Intensity: **{intens.get('mean_intensity_repaired', 0):.1f}**")
                st.caption(s_exp.get("disclaimer", ""))

        st.markdown("---")
        st.subheader("Experimental Repair Diagnostic Impact")
        st.caption("Experimental Paired Validation: Testing whether fresh DenseNet-121 inference after Repair Agent processing improves pneumonia classification. (Clearly labeled: Experimental Paired Validation; not the full test set).")

        paired_diag_path = PROJECT_ROOT / "outputs" / "final_paired_repair_diagnostic_summary.json"
        if paired_diag_path.exists():
            try:
                with open(paired_diag_path, encoding="utf-8") as f:
                    diag_summary = json.load(f)

                sm = diag_summary.get("sample_size", {})
                bm = diag_summary.get("before_metrics", {})
                am = diag_summary.get("after_metrics", {})
                deltas = diag_summary.get("metric_deltas", {})
                transitions = diag_summary.get("four_transitions", {})
                stab = diag_summary.get("prediction_stability_pct", 100.0)
                rt_breakdown = diag_summary.get("repair_type_breakdown", {})

                st.markdown(f"**Cohort Overview:** Total Candidates: **{sm.get('candidate_count', 0)}** | Repaired: **{sm.get('repair_applied_count', 0)}** | Skipped: **{sm.get('repair_skipped_count', 0)}**")

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Before Accuracy", f"{bm.get('accuracy', 0)*100:.2f}%")
                with col_m2:
                    st.metric("After Accuracy", f"{am.get('accuracy', 0)*100:.2f}%")
                with col_m3:
                    st.metric("Accuracy Delta", f"{deltas.get('accuracy_delta', 0)*100:+.2f}%")
                with col_m4:
                    st.metric("Prediction Stability", f"{stab:.1f}%")

                st.markdown("#### Four-Way Diagnostic Transitions")
                col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                with col_t1:
                    st.metric("Correct → Correct", f"{transitions.get('Correct->Correct', 0)}")
                with col_t2:
                    st.metric("Correct → Incorrect", f"{transitions.get('Correct->Incorrect', 0)}", delta="Harmful", delta_color="inverse")
                with col_t3:
                    st.metric("Incorrect → Correct", f"{transitions.get('Incorrect->Correct', 0)}", delta="Helpful", delta_color="normal")
                with col_t4:
                    st.metric("Incorrect → Incorrect", f"{transitions.get('Incorrect->Incorrect', 0)}")

                st.markdown("#### Repair Type Breakdown")
                rows_rt = []
                for rtype, rdata in rt_breakdown.items():
                    rows_rt.append({
                        "Repair Type": rtype.capitalize(),
                        "N": rdata.get("n", 0),
                        "Accuracy Before": f"{rdata.get('accuracy_before', 0)*100:.1f}%",
                        "Accuracy After": f"{rdata.get('accuracy_after', 0)*100:.1f}%",
                        "Accuracy Delta": f"{rdata.get('accuracy_delta', 0)*100:+.1f}%",
                        "Incorrect → Correct": rdata.get("incorrect_to_correct", 0),
                        "Correct → Incorrect": rdata.get("correct_to_incorrect", 0),
                        "Stability Rate": f"{rdata.get('prediction_stability', 100.0):.1f}%",
                        "Note": rdata.get("note", ""),
                    })
                if rows_rt:
                    st.dataframe(pd.DataFrame(rows_rt), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error loading paired diagnostic summary: {e}")

    # TAB 4: Safety & Confusion Matrices
    with tab4:
        st.subheader("Confusion Matrices Across Populations")
        st.caption("All confusion matrices explicitly display their respective cohort size (N) and ground truth counts.")

        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            p_bm = PROJECT_ROOT / "outputs" / "before_after" / "base_model_confusion_matrix.png"
            if p_bm.exists():
                st.image(str(p_bm), caption="Standalone Base Model (Full Test Set, N=16,724)")

        with col_cm2:
            p_rel = PROJECT_ROOT / "outputs" / "before_after" / "reliability_system_confusion_matrix.png"
            if p_rel.exists():
                st.image(str(p_rel), caption="Reliability-Aware System (Released Subset, N=7,349)")

        st.markdown("---")
        col_cm3, col_cm4 = st.columns(2)
        with col_cm3:
            p_rep = PROJECT_ROOT / "outputs" / "before_after" / "repair_paired_confusion_matrices.png"
            if p_rep.exists():
                st.image(str(p_rep), caption="Paired Repair Cohort: Before vs After Repair")

        with col_cm4:
            p_qt = PROJECT_ROOT / "outputs" / "before_after" / "quality_transition_matrix.png"
            if p_qt.exists():
                st.image(str(p_qt), caption="Quality Transition Matrix (N=12,082 Repairs)")

    # TAB 5: Data Explorer
    with tab5:
        st.subheader("Evaluation Dataset Record Explorer")
        st.caption("Browse individual patient records and filter by pipeline action.")
        if csv_path.exists():
            try:
                df_csv = pd.read_csv(csv_path, nrows=5000)
                actions_list = ["All"] + sorted(df_csv["final_action"].dropna().unique().tolist())
                selected_action = st.selectbox("Filter by Final Action:", actions_list, key="action_filter")

                if selected_action != "All":
                    df_filtered = df_csv[df_csv["final_action"] == selected_action]
                else:
                    df_filtered = df_csv

                cols_to_show = [
                    c for c in [
                        "image_id", "patient_id", "ground_truth_name", "final_action",
                        "needs_human_review", "prediction_released", "raw_model_score",
                        "quality_label", "after_repair_quality_label", "ood_level",
                        "uncertainty_level", "repair_applied", "verification_status",
                        "delta_confidence",
                    ] if c in df_filtered.columns
                ]
                st.dataframe(df_filtered[cols_to_show].head(200), use_container_width=True, hide_index=True)
                st.caption(f"Showing matching rows (total in partition: {len(df_filtered):,})")
            except Exception as e:
                st.error(f"Error loading evaluation CSV: {e}")


if __name__ == "__main__":
    main()
