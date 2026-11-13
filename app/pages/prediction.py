"""
CarePredict - Page 3: No-Show Prediction & SHAP Explainability
Supports Individual Custom Prediction (Mode A) and Bulk CSV Prediction (Mode B)
with local probability-space SHAP explanations and global risk driver analysis.
"""

from datetime import date, timedelta
import pandas as pd
import numpy as np
import streamlit as st

from app.config import (
    DEFAULT_THRESHOLD,
    APPROVED_FEATURES,
    TEMPLATE_PATH,
    SHAP_BASELINE_PROBABILITY
)
from app.model_service import predict_single, predict_batch, get_risk_tier
from app.feature_builder import derive_individual_features, validate_and_prepare_batch_csv
from app.shap_service import (
    explain_single_prediction_probability,
    create_local_waterfall_chart,
    create_global_importance_chart,
    initialize_shap_explainers
)
from app.data_loader import load_global_shap_importance
from app.components import render_kpi_card, render_callout, render_risk_badge

def render_prediction_page():
    st.markdown("### No-Show Prediction & Explainability Engine")
    st.markdown(
        "Generate point-in-time risk predictions using the champion unweighted XGBoost model. "
        "Every prediction is paired with additive probability-space SHAP explanations."
    )

    pred_mode = st.radio(
        "Select Prediction Mode",
        options=["Individual Appointment (Custom Inputs)", "Bulk Batch Prediction (CSV Upload)", "Global Model Risk Drivers"],
        horizontal=True
    )

    # =========================================================================
    # MODE A: Individual Appointment Prediction
    # =========================================================================
    if pred_mode == "Individual Appointment (Custom Inputs)":
        st.markdown("#### Mode A: Individual Patient Appointment Assessment")

        with st.form("individual_prediction_form"):
            col_p1, col_p2, col_p3 = st.columns(3)

            # Column 1: Patient Profile
            with col_p1:
                st.markdown("##### 1. Patient Clinical Profile")
                age = st.number_input("Patient Age (Years)", min_value=0, max_value=120, value=42, step=1)
                gender = st.selectbox("Gender", options=["Female", "Male"])
                scholarship = st.checkbox("Bolsa Família Welfare Enrolled", value=False)
                hypertension = st.checkbox("Hypertension Diagnosed", value=False)
                diabetes = st.checkbox("Diabetes Diagnosed", value=False)
                alcoholism = st.checkbox("Documented Alcoholism", value=False)
                handicap_level = st.selectbox("Physical Disability Count", options=[0, 1, 2, 3, 4], index=0)

            # Column 2: Appointment Scheduling Details
            with col_p2:
                st.markdown("##### 2. Appointment Scheduling")
                today = date.today()
                scheduled_date = st.date_input("Scheduling Date (Day 0)", value=today)
                default_appt_date = today + timedelta(days=14)
                appointment_date = st.date_input("Scheduled Visit Date", value=default_appt_date)
                
                # Dynamic preview of derived temporal values
                derived_lead = max(0, (appointment_date - scheduled_date).days)
                st.caption(f"Derived Lead Days: **{derived_lead} calendar days**")
                st.caption(f"Same-Day Walk-In: **{'Yes' if derived_lead == 0 else 'No'}**")
                st.caption(f"Appointment Day: **{appointment_date.strftime('%A')}**")

            # Column 3: Patient Historical Fidelity
            with col_p3:
                st.markdown("##### 3. Prior Attendance History")
                is_first_patient = st.checkbox("First-Time Patient (Zero Past History)", value=False)
                
                if not is_first_patient:
                    prior_appts = st.number_input("Total Prior Appointments", min_value=1, max_value=100, value=3, step=1)
                    prior_noshows = st.number_input("Prior Missed Visits", min_value=0, max_value=prior_appts, value=1, step=1)
                    days_since_last = st.number_input("Days Since Preceding Visit", min_value=1, max_value=730, value=30, step=1)
                    derived_rate = (prior_noshows / prior_appts * 100.0) if prior_appts > 0 else 0.0
                    st.caption(f"Derived Prior No-Show Rate: **{derived_rate:.1f}%**")
                else:
                    prior_appts = 0
                    prior_noshows = 0
                    days_since_last = -1
                    st.caption("First-time visit conventions: prior_appts=0, days_gap=-1")

            st.markdown("##### 4. Decision Policy Configuration")
            threshold = st.slider(
                "Operational Decision Threshold (Prob >= Threshold → Flag for Outreach)",
                min_value=0.05, max_value=0.50, value=DEFAULT_THRESHOLD, step=0.01,
                help="Default 0.15 is the lowest-cost threshold under illustrative hospital cost assumptions."
            )

            submit_btn = st.form_submit_button("Calculate Prediction & Explain Risk Drivers", type="primary")

        if submit_btn:
            try:
                # 1. Derive and validate feature dictionary
                features_dict = derive_individual_features(
                    age=age,
                    gender_str=gender,
                    scholarship=scholarship,
                    hypertension=hypertension,
                    diabetes=diabetes,
                    alcoholism=alcoholism,
                    handicap_level=handicap_level,
                    scheduled_dt=scheduled_date,
                    appointment_dt=appointment_date,
                    is_first_time=is_first_patient,
                    prior_appts=prior_appts,
                    prior_missed=prior_noshows,
                    days_since_last=days_since_last
                )

                # 2. Run model inference
                pred_result = predict_single(features_dict, threshold=threshold)
                prob_ns = pred_result["prob_noshow"]
                prob_att = pred_result["prob_attend"]

                st.markdown("---")
                st.markdown("#### Prediction Results & Actionable Recommendation")

                # KPI row
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    render_kpi_card("Predicted No-Show Risk", f"{prob_ns * 100.0:.1f}%", f"Attendance: {prob_att * 100.0:.1f}%")
                with r2:
                    st.markdown(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-label">Operational Risk Tier</div>
                            <div style="margin-top: 8px;">{render_risk_badge(pred_result['risk_tier'])}</div>
                            <div class="kpi-subtext" style="margin-top: 8px;">Threshold: {threshold:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with r3:
                    decision_str = "FLAG FOR OUTREACH" if pred_result["flagged"] else "STANDARD CARE"
                    decision_sub = f"Prob ({prob_ns:.2f}) >= Threshold ({threshold:.2f})" if pred_result["flagged"] else f"Prob ({prob_ns:.2f}) < Threshold ({threshold:.2f})"
                    render_kpi_card("Intervention Decision", decision_str, decision_sub)
                with r4:
                    render_kpi_card("Baseline Population Risk", f"{SHAP_BASELINE_PROBABILITY * 100.0:.1f}%", "Starting reference point")

                # Recommendation box
                action_type = "warning" if pred_result["flagged"] else "notice"
                st.markdown(
                    f"""
                    <div class="{action_type}-box">
                        <b>Clinical Operations Recommendation:</b> {pred_result['risk_action']}<br>
                        <small><i>Note: Model outputs represent risk stratification for allocating supportive resources, not deterministic certainty. Never use predictions to restrict or deprioritize clinical access.</i></small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # 3. Compute Probability-Space SHAP Explanation
                st.markdown("#### Probability-Space SHAP Risk Attribution")
                with st.spinner("Computing additive SHAP explanation..."):
                    shap_exp = explain_single_prediction_probability(features_dict, predicted_prob=prob_ns)

                st.markdown(f"**Clinical Narrative:** {shap_exp['narrative']}")

                # Waterfall Chart
                fig_waterfall = create_local_waterfall_chart(shap_exp)
                st.plotly_chart(fig_waterfall, use_container_width=True)

                # Factor Breakdown Tables
                c_pos, c_neg = st.columns(2)
                with c_pos:
                    st.markdown("##### Top Factors Elevating No-Show Risk (+)")
                    if shap_exp['top_risk_increasing']:
                        df_pos_show = pd.DataFrame(shap_exp['top_risk_increasing'])[['feature', 'feature_value', 'shap_pct']]
                        df_pos_show.columns = ['Feature', 'Value', 'Impact (% pts)']
                        df_pos_show['Impact (% pts)'] = df_pos_show['Impact (% pts)'].map("+{:.2f}% pts".format)
                        st.dataframe(df_pos_show, use_container_width=True, hide_index=True)
                    else:
                        st.write("No features pushed this prediction upward.")

                with c_neg:
                    st.markdown("##### Top Protective Factors Reducing Risk (-)")
                    if shap_exp['top_risk_reducing']:
                        df_neg_show = pd.DataFrame(shap_exp['top_risk_reducing'])[['feature', 'feature_value', 'shap_pct']]
                        df_neg_show.columns = ['Feature', 'Value', 'Impact (% pts)']
                        df_neg_show['Impact (% pts)'] = df_neg_show['Impact (% pts)'].map("{:.2f}% pts".format)
                        st.dataframe(df_neg_show, use_container_width=True, hide_index=True)
                    else:
                        st.write("No features pushed this prediction downward.")

                # Advanced Expandable Raw-Margin View
                with st.expander("Technical Engineering View (Raw-Margin Log-Odds SHAP)", expanded=False):
                    st.markdown(
                        r"""
                        **Raw-Margin Space:** In raw-margin space ($z \in (-\infty, +\infty)$), decision tree leaf values sum linearly. 
                        Passing raw margin through sigmoid $\sigma(z) = 1 / (1 + e^{-z})$ reconstructs the pipeline probability with exact numerical parity.
                        """
                    )
                    pipe = load_champion_model()
                    raw_margin = float(pipe.named_steps['clf'].predict(pipe.named_steps['prep'].transform(pd.DataFrame([features_dict])[APPROVED_FEATURES]), output_margin=True))
                    st.write(f"- Raw Output Margin $z$: **{raw_margin:.4f}**")
                    st.write(rf"- Sigmoid Probability $\sigma(z)$: **{1.0 / (1.0 + np.exp(-raw_margin)):.4f}** (Matches predicted: {prob_ns:.4f})")

            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

    # =========================================================================
    # MODE B: Bulk Prediction Through CSV Upload
    # =========================================================================
    elif pred_mode == "Bulk Batch Prediction (CSV Upload)":
        st.markdown("#### Mode B: Bulk Batch Appointment Risk Scoring")
        st.markdown(
            "Upload a batch CSV file to score multiple scheduled appointments simultaneously. "
            "The pipeline validates column formats, derives Day-0 features, and tags appointments exceeding the decision threshold."
        )

        col_b1, col_b2 = st.columns([2, 1])
        with col_b2:
            st.markdown("##### Download CSV Template")
            if TEMPLATE_PATH.exists():
                with open(TEMPLATE_PATH, 'rb') as f:
                    st.download_button(
                        label="Download Appointment Template (CSV)",
                        data=f,
                        file_name="carepredict_appointment_template.csv",
                        mime="text/csv"
                    )

        with col_b1:
            uploaded_file = st.file_uploader("Upload Scheduled Appointments (CSV)", type=["csv"])
            bulk_threshold = st.slider("Batch Decision Threshold", min_value=0.05, max_value=0.50, value=DEFAULT_THRESHOLD, step=0.01)

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.info(f"Loaded CSV with **{len(df_upload):,} records** and {len(df_upload.columns)} columns.")

                valid_df, invalid_df, warnings = validate_and_prepare_batch_csv(df_upload)

                if not invalid_df.empty:
                    st.warning(f"**{len(invalid_df)} records were rejected** due to data validation errors.")
                    with st.expander("Inspect Rejected Records"):
                        st.dataframe(invalid_df[['_rejection_reason'] + [c for c in invalid_df.columns if c != '_rejection_reason']])

                if valid_df.empty:
                    st.error("No valid appointment records found to score.")
                    return

                with st.spinner(f"Scoring {len(valid_df):,} valid appointments..."):
                    scored_df = predict_batch(valid_df, threshold=bulk_threshold)

                st.success(f"Successfully scored **{len(scored_df):,} appointments**!")

                # Batch Summary KPIs
                flagged_count = int(scored_df['flagged_for_intervention'].sum())
                pct_flagged = (flagged_count / len(scored_df)) * 100.0
                mean_risk = scored_df['predicted_noshow_prob'].mean() * 100.0

                bk1, bk2, bk3, bk4 = st.columns(4)
                with bk1:
                    render_kpi_card("Scored Appointments", f"{len(scored_df):,}")
                with bk2:
                    render_kpi_card("Mean Predicted Risk", f"{mean_risk:.1f}%")
                with bk3:
                    render_kpi_card("Flagged for Outreach", f"{flagged_count:,}", f"{pct_flagged:.1f}% of cohort")
                with bk4:
                    render_kpi_card("Threshold Applied", f"{bulk_threshold:.2f}")

                # Risk Filter
                st.markdown("##### Filter Scored Appointments Table")
                selected_tiers = st.multiselect(
                    "Filter by Operational Risk Tier",
                    options=["Critical Risk", "High Risk", "Moderate Risk", "Low Risk"],
                    default=["Critical Risk", "High Risk", "Moderate Risk", "Low Risk"]
                )
                filtered_scored = scored_df[scored_df['risk_tier'].isin(selected_tiers)]

                display_cols = [c for c in ['appointment_id', 'patient_id', 'lead_days', 'age', 'predicted_noshow_prob', 'risk_tier', 'flagged_for_intervention', 'recommended_action'] if c in filtered_scored.columns]
                st.dataframe(filtered_scored[display_cols].sort_values('predicted_noshow_prob', ascending=False), use_container_width=True)

                # Download Results Button
                csv_results = scored_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Complete Scored Batch (CSV)",
                    data=csv_results,
                    file_name="carepredict_scored_appointments.csv",
                    mime="text/csv"
                )

                # On-Demand Local SHAP for single row in batch
                st.markdown("---")
                st.markdown("##### Inspect Individual Row Explanation from Batch")
                row_idx = st.number_input("Select Row Index to Explain", min_value=0, max_value=len(scored_df)-1, value=0, step=1)
                if st.button("Generate SHAP Explanation for Selected Row"):
                    selected_row_features = scored_df.iloc[row_idx][APPROVED_FEATURES].to_dict()
                    sel_prob = float(scored_df.iloc[row_idx]['predicted_noshow_prob'])
                    with st.spinner("Computing explanation for selected record..."):
                        row_shap = explain_single_prediction_probability(selected_row_features, predicted_prob=sel_prob)
                    st.markdown(f"**Row {row_idx} Clinical Narrative:** {row_shap['narrative']}")
                    st.plotly_chart(create_local_waterfall_chart(row_shap), use_container_width=True)

            except Exception as e:
                st.error(f"Batch Processing Error: {str(e)}")

    # =========================================================================
    # TAB 3: Global Model Risk Drivers
    # =========================================================================
    else:
        st.markdown("#### Global Model Risk Drivers (Probability Space)")
        st.markdown(
            "Overall feature importance ranking across the patient population, "
            "quantifying each feature's mean absolute impact in percentage points on predicted no-show probability."
        )

        render_callout(
            "<b>Attribution vs. Causality Caveat:</b> SHAP values measure statistical credit assignment within the XGBoost tree ensemble. "
            "They do not imply direct counterfactual causality. For example, high lead time correlates with elevated risk, but shortening "
            "lead times administratively without consulting patients will not guarantee attendance if transportation or work barriers persist.",
            box_type="warning"
        )

        df_global = load_global_shap_importance()
        st.plotly_chart(create_global_importance_chart(df_global), use_container_width=True)

        st.markdown("##### Global Feature Importance Table")
        st.dataframe(
            df_global[['rank', 'original_feature', 'mean_abs_shap_pct', 'mean_signed_shap_pct']].rename(
                columns={
                    'rank': 'Rank',
                    'original_feature': 'Predictor Feature',
                    'mean_abs_shap_pct': 'Mean |SHAP| (% pts)',
                    'mean_signed_shap_pct': 'Net Signed Shift (% pts)'
                }
            ),
            use_container_width=True,
            hide_index=True
        )
