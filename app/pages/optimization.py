"""
CarePredict - Page 4: Capacity & Financial Optimization
Interactive decision threshold tuning, editable asymmetric healthcare cost parameters,
capacity-constrained patient prioritization, and financial sensitivity analysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go

from app.config import (
    DEFAULT_THRESHOLD,
    DEFAULT_COST_INTERVENTION,
    DEFAULT_COST_UNFILLED,
    DEFAULT_OUTREACH_EFFECTIVENESS,
    DEFAULT_CAPACITY,
    COLORS
)
from app.data_loader import (
    load_validation_features,
    load_validation_predictions,
    load_threshold_benchmarks
)
from app.financial_engine import (
    evaluate_historical_cost,
    evaluate_prospective_cost,
    generate_threshold_cost_curve,
    apply_capacity_constraint,
    compute_sensitivity_matrix
)
from app.components import render_kpi_card, render_callout

def render_optimization_page():
    st.markdown("### Operational Capacity & Financial Optimization")
    st.markdown(
        "Evaluate decision thresholds, test asymmetric healthcare cost scenarios, "
        "and prioritize patient outreach under finite staff capacity constraints."
    )

    # -------------------------------------------------------------
    # User-Configurable Financial Assumptions Control Bar
    # -------------------------------------------------------------
    with st.expander("Configure Healthcare Financial & Operational Assumptions", expanded=True):
        st.caption("Adjust cost parameters to model your health system's specific economics. Default values reflect the project benchmark scenario.")

        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        with fc1:
            c_intervene = st.number_input(
                "Intervention Cost ($)",
                min_value=1.0, max_value=100.0,
                value=st.session_state.get('cost_intervene', DEFAULT_COST_INTERVENTION),
                step=1.0,
                help="Cost of phone call, navigation assistance, or reminder outreach."
            )
        with fc2:
            c_unfilled = st.number_input(
                "Unfilled Slot Loss ($)",
                min_value=25.0, max_value=1000.0,
                value=st.session_state.get('cost_unfilled', DEFAULT_COST_UNFILLED),
                step=25.0,
                help="Direct economic loss of an unused clinical provider slot."
            )
        with fc3:
            eff_pct = st.slider(
                "Outreach Efficacy (r)",
                min_value=10, max_value=60,
                value=int(st.session_state.get('outreach_eff', DEFAULT_OUTREACH_EFFECTIVENESS * 100)),
                step=5,
                help="Assumed percentage of contacted no-shows persuaded to attend."
            )
            effectiveness = eff_pct / 100.0
        with fc4:
            threshold = st.slider(
                "Decision Threshold",
                min_value=0.04, max_value=0.50,
                value=float(st.session_state.get('sel_threshold', DEFAULT_THRESHOLD)),
                step=0.01,
                help="Threshold above which appointments are flagged for outreach."
            )
        with fc5:
            capacity = st.number_input(
                "Outreach Capacity (Visits)",
                min_value=50, max_value=15000,
                value=int(st.session_state.get('daily_cap', DEFAULT_CAPACITY)),
                step=50,
                help="Maximum number of patient calls or interventions staff can perform."
            )

        if st.button("Restore Default Benchmark Assumptions"):
            st.session_state['cost_intervene'] = DEFAULT_COST_INTERVENTION
            st.session_state['cost_unfilled'] = DEFAULT_COST_UNFILLED
            st.session_state['outreach_eff'] = int(DEFAULT_OUTREACH_EFFECTIVENESS * 100)
            st.session_state['sel_threshold'] = DEFAULT_THRESHOLD
            st.session_state['daily_cap'] = DEFAULT_CAPACITY
            st.rerun()

    opt_tab1, opt_tab2, opt_tab3 = st.tabs([
        "Threshold Cost Modeling (Historical Validation)",
        "Capacity-Constrained Prioritization",
        "Economic Sensitivity Analysis"
    ])

    # Load validation data for ground-truth benchmark evaluation
    val_preds = load_validation_predictions()
    y_true = val_preds['y_true'].values
    y_prob = val_preds['y_prob_unweighted'].values

    # =============================================================
    # TAB 1: Threshold Cost Modeling (Historical & Prospective)
    # =============================================================
    with opt_tab1:
        st.markdown("#### Modeled Financial Performance on Chronological Cohort")
        
        eval_mode = st.radio(
            "Evaluation Paradigm",
            options=["Historical Validation (Ground Truth Outcomes Known)", "Prospective Planning Mode (Expected Outcomes from Probabilities)"],
            horizontal=True
        )

        if eval_mode == "Historical Validation (Ground Truth Outcomes Known)":
            st.caption(f"Evaluated on May 20–31, 2016 validation period (N = 20,539 visits | Base no-show rate = 20.08%).")

            res = evaluate_historical_cost(
                y_true=y_true,
                y_prob=y_prob,
                threshold=threshold,
                c_intervene=c_intervene,
                c_unfilled=c_unfilled,
                effectiveness=effectiveness
            )

            render_callout(
                f"<b>Threshold Policy Benchmark:</b> Selected threshold: <b>0.15</b> — the lowest-cost candidate among evaluated thresholds "
                f"under the specified financial assumptions (Intervention: ${c_intervene:.0f}, Slot Loss: ${c_unfilled:.0f}, Outreach Efficacy: {eff_pct}%). "
                f"Estimated net savings of <b>${res['net_savings']:,.0f}</b> represent a modeled validation scenario on 20,539 historical visits, not measured hospital savings.",
                box_type="notice"
            )

            # KPI Metrics
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_kpi_card("Modeled Net Savings", f"${res['net_savings']:,.0f}", f"{res['pct_cost_reduction']:.1f}% reduction vs baseline")
            with k2:
                render_kpi_card("Total Policy Cost", f"${res['total_cost']:,.0f}", f"Baseline: ${res['baseline_cost']:,.0f}")
            with k3:
                render_kpi_card("No-Show Recall", f"{res['recall']*100:.1f}%", f"{res['tp']:,} of {res['actual_noshows']:,} missed visits")
            with k4:
                render_kpi_card("Flagged Patient Workload", f"{res['flagged_count']:,}", f"{res['pct_flagged']:.1f}% of cohort (Prec: {res['precision']*100:.1f}%)")

            st.markdown("<br>", unsafe_allow_html=True)

            # Threshold Cost Curve
            col_curve, col_cm = st.columns([3, 2])
            with col_curve:
                df_curve, fig_curve, lowest_thresh = generate_threshold_cost_curve(
                    y_true=y_true,
                    y_prob=y_prob,
                    selected_threshold=threshold,
                    c_intervene=c_intervene,
                    c_unfilled=c_unfilled,
                    effectiveness=effectiveness
                )
                st.plotly_chart(fig_curve, use_container_width=True)

            with col_cm:
                st.markdown("##### Confusion Matrix at Threshold " + f"{threshold:.2f}")
                cm_z = [[res['tn'], res['fp']], [res['fn'], res['tp']]]
                cm_text = [[f"TN: {res['tn']:,}", f"FP: {res['fp']:,}"], [f"FN: {res['fn']:,}", f"TP: {res['tp']:,}"]]
                
                fig_cm = ff.create_annotated_heatmap(
                    z=cm_z,
                    x=['Predicted Attend', 'Predicted No-Show'],
                    y=['Actual Attend', 'Actual No-Show'],
                    annotation_text=cm_text,
                    colorscale=[[0, '#e0f2fe'], [1, '#0284c7']]
                )
                fig_cm.update_layout(
                    margin=dict(l=80, r=20, t=40, b=40),
                    height=320,
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown("##### Precomputed Benchmark Comparison (Validation Set)")
            bench_df = load_threshold_benchmarks()
            st.dataframe(bench_df, use_container_width=True)

        else:
            # Prospective Mode: Future outcomes are unknown; expected value integration
            st.caption("Prospective Planning: Realized attendance outcomes are unknown. Metrics represent expected modeled values derived from predicted probabilities.")

            res_p = evaluate_prospective_cost(
                y_prob=y_prob,
                threshold=threshold,
                c_intervene=c_intervene,
                c_unfilled=c_unfilled,
                effectiveness=effectiveness
            )

            render_callout(
                f"<b>Prospective Decision Support:</b> Under operating threshold <b>{threshold:.2f}</b>, "
                f"the system forecasts <b>{res_p['flagged_count']:,} appointments</b> ({res_p['pct_flagged']:.1f}%) "
                f"flagged for intervention. Expected net savings are modeled at <b>${res_p['net_savings']:,.0f}</b> "
                f"relative to an unmitigated baseline cost of ${res_p['baseline_cost']:,.0f}.",
                box_type="notice"
            )

            pk1, pk2, pk3, pk4 = st.columns(4)
            with pk1:
                render_kpi_card("Expected Net Savings", f"${res_p['net_savings']:,.0f}", f"{res_p['pct_cost_reduction']:.1f}% expected reduction")
            with pk2:
                render_kpi_card("Expected Policy Cost", f"${res_p['total_cost']:,.0f}", f"Unmitigated: ${res_p['baseline_cost']:,.0f}")
            with pk3:
                render_kpi_card("Expected Prevented Visits", f"{res_p['expected_prevented']:.1f}", f"At {eff_pct}% outreach effectiveness")
            with pk4:
                render_kpi_card("Scheduled Outreach Vol", f"{res_p['flagged_count']:,}", f"{res_p['pct_flagged']:.1f}% of cohort flagged")

    # =============================================================
    # TAB 2: Capacity-Constrained Prioritization
    # =============================================================
    with opt_tab2:
        st.markdown("#### Patient Outreach Prioritization under Staff Capacity Caps")
        st.markdown(
            "In practice, clinic staff cannot manually contact every flagged patient. "
            "When flagged appointments exceed available staff capacity, CarePredict prioritizes patients strictly by predicted risk."
        )

        render_callout(
            "<b>Ethical & Governance Mandate:</b> Capacity prioritization strictly governs the sequencing of proactive "
            "supportive outreach (e.g. personal reminder calls, transportation navigation). No patient is denied, delayed, "
            "or restricted from clinical care based on predictions or staffing thresholds.",
            box_type="notice"
        )

        val_features_df = load_validation_features()
        val_merged = val_features_df.copy()
        val_merged['predicted_noshow_prob'] = y_prob
        val_merged['flagged_for_intervention'] = val_merged['predicted_noshow_prob'] >= threshold

        prioritized_df, cap_stats = apply_capacity_constraint(val_merged, daily_capacity=capacity)

        cp1, cp2, cp3, cp4 = st.columns(4)
        with cp1:
            render_kpi_card("Available Outreach Capacity", f"{capacity:,}", "Interventions / Period")
        with cp2:
            render_kpi_card("Appointments Flagged", f"{cap_stats['total_flagged']:,}", f"At threshold {threshold:.2f}")
        with cp3:
            render_kpi_card("Reached within Capacity", f"{cap_stats['reached_within_capacity']:,}", f"{cap_stats['capacity_coverage_pct']:.1f}% capacity coverage")
        with cp4:
            render_kpi_card("Unreached Shortfall", f"{cap_stats['unreached_shortfall']:,}", "Exceeds staff bandwidth")

        if cap_stats['unreached_shortfall'] > 0:
            render_callout(
                f"<b>Capacity Shortfall Alert:</b> Flagged volume ({cap_stats['total_flagged']:,}) exceeds capacity ({capacity:,}). "
                f"CarePredict has prioritized the top {capacity:,} highest-risk patients for intervention, "
                f"leaving {cap_stats['unreached_shortfall']:,} flagged patients for automated low-touch SMS.",
                box_type="warning"
            )

        st.markdown("##### Prioritized Patient Outreach Queue (Top 100)")
        disp_cols = [c for c in ['priority_rank', 'within_capacity', 'predicted_noshow_prob', 'lead_days', 'age', 'gender', 'prior_noshow_rate', 'prior_appointments'] if c in prioritized_df.columns]
        
        display_prioritized = prioritized_df[disp_cols].head(100).copy()
        display_prioritized['predicted_noshow_prob'] = display_prioritized['predicted_noshow_prob'].map("{:.1%}".format)
        st.dataframe(display_prioritized, use_container_width=True, hide_index=True)

        csv_priority = prioritized_df.head(capacity).to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download Priority Outreach Queue (Top {min(capacity, len(prioritized_df)):,} CSV)",
            data=csv_priority,
            file_name="carepredict_prioritized_outreach_list.csv",
            mime="text/csv"
        )

    # =============================================================
    # TAB 3: Economic Sensitivity Analysis
    # =============================================================
    with opt_tab3:
        st.markdown("#### Scenario Sensitivity Analysis")
        st.markdown(
            "Explore how net financial savings vary across different outreach effectiveness rates "
            "and unfilled appointment slot costs."
        )

        st.caption(f"Evaluated at decision threshold = {threshold:.2f} and intervention cost = ${c_intervene:.2f}.")

        sens_df = compute_sensitivity_matrix(
            y_true=y_true,
            y_prob=y_prob,
            threshold=threshold,
            c_intervene=c_intervene
        )
        st.dataframe(sens_df, use_container_width=True, hide_index=True)

