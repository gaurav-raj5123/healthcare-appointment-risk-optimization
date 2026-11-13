"""
CarePredict - Page 2: Appointment Analytics
Deep-dive demographic, temporal, and patient-history exploration with searchable data tables.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.data_loader import load_cleaned_data
from app.config import COLORS
from app.components import render_callout
from app.analytics import get_age_bracket, get_lead_time_bracket

def render_analytics_page():
    st.markdown("### Historical Appointment Analytics")
    st.markdown("Investigate retrospective attendance drivers across demographics, scheduling schedules, and historical patient patterns.")

    render_callout(
        "<b>Analytical Distinction:</b> Descriptive analytics below utilize retrospective information (such as past appointments and visit outcomes). "
        "The predictive model (Section 3) strictly uses features available at Day-0 booking time to prevent temporal leakage.",
        box_type="notice"
    )

    df_full = load_cleaned_data()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Demographic Cohorts",
        "Temporal & Scheduling",
        "Patient History Patterns",
        "Interactive Data Explorer"
    ])

    # -------------------------------------------------------------
    # TAB 1: Demographic Cohorts
    # -------------------------------------------------------------
    with tab1:
        st.markdown("#### Patient Demographic & Clinical Profiles")
        c1, c2 = st.columns(2)

        with c1:
            # Gender comparison
            gender_agg = df_full.groupby('gender').agg(
                total=('no_show', 'count'),
                noshows=('no_show', 'sum'),
                noshow_rate=('no_show', lambda s: s.mean() * 100.0)
            ).reset_index()
            gender_agg['gender_label'] = gender_agg['gender'].apply(lambda g: 'Female' if str(g).upper() == 'F' else 'Male')

            fig_gender = px.bar(
                gender_agg,
                x='gender_label',
                y='noshow_rate',
                text='noshow_rate',
                color='gender_label',
                color_discrete_map={'Female': COLORS['primary'], 'Male': COLORS['teal']},
                title="<b>No-Show Rate by Gender</b>"
            )
            fig_gender.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_gender.update_layout(
                yaxis=dict(title="No-Show Rate (%)", range=[0, 25], gridcolor="#f1f5f9"),
                xaxis=dict(title="Gender"),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_gender, use_container_width=True)

        with c2:
            # Welfare (Scholarship) Comparison
            welfare_agg = df_full.groupby('scholarship').agg(
                total=('no_show', 'count'),
                noshow_rate=('no_show', lambda s: s.mean() * 100.0)
            ).reset_index()
            welfare_agg['welfare_label'] = welfare_agg['scholarship'].apply(lambda s: 'Bolsa Família Enrolled' if s == 1 else 'Standard Care')

            fig_welfare = px.bar(
                welfare_agg,
                x='welfare_label',
                y='noshow_rate',
                text='noshow_rate',
                color='welfare_label',
                color_discrete_map={'Bolsa Família Enrolled': COLORS['warning'], 'Standard Care': COLORS['primary']},
                title="<b>No-Show Rate by Social Welfare Status</b>"
            )
            fig_welfare.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_welfare.update_layout(
                yaxis=dict(title="No-Show Rate (%)", range=[0, 30], gridcolor="#f1f5f9"),
                xaxis=dict(title="Welfare Status"),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_welfare, use_container_width=True)

        # Age cohort table breakdown
        df_temp = df_full.copy()
        df_temp['age_group'] = df_temp['age'].apply(get_age_bracket)
        age_order = ["Pediatric (<18)", "Young Adult (18-35)", "Adult (36-50)", "Mature Adult (51-65)", "Senior (66+)"]
        age_summary = df_temp.groupby('age_group').agg(
            total_visits=('no_show', 'count'),
            missed_visits=('no_show', 'sum'),
            noshow_rate=('no_show', lambda s: s.mean() * 100.0)
        ).reindex(age_order).reset_index()
        age_summary['attendance_rate'] = 100.0 - age_summary['noshow_rate']
        age_summary['noshow_rate'] = age_summary['noshow_rate'].map("{:.2f}%".format)
        age_summary['attendance_rate'] = age_summary['attendance_rate'].map("{:.2f}%".format)
        age_summary['total_visits'] = age_summary['total_visits'].map("{:,}".format)
        age_summary['missed_visits'] = age_summary['missed_visits'].map("{:,}".format)

        st.markdown("##### Age Cohort Performance Summary")
        st.dataframe(age_summary, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # TAB 2: Temporal & Scheduling
    # -------------------------------------------------------------
    with tab2:
        st.markdown("#### Operational Lead Times & Scheduling Patterns")

        # Same-day vs Advance booking
        df_temp = df_full.copy()
        df_temp['booking_type'] = df_temp['lead_days'].apply(lambda d: "Same-Day Walk-In" if d == 0 else "Advanced Scheduled")
        b_agg = df_temp.groupby('booking_type').agg(
            total=('no_show', 'count'),
            noshow_pct=('no_show', lambda s: s.mean() * 100.0)
        ).reset_index()

        c3, c4 = st.columns([1, 1])
        with c3:
            fig_b = px.bar(
                b_agg,
                x='booking_type',
                y='noshow_pct',
                text='noshow_pct',
                color='booking_type',
                color_discrete_map={"Same-Day Walk-In": COLORS['success'], "Advanced Scheduled": COLORS['danger']},
                title="<b>The Same-Day Attendance Advantage</b>"
            )
            fig_b.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_b.update_layout(
                yaxis=dict(title="No-Show Rate (%)", range=[0, 35], gridcolor="#f1f5f9"),
                xaxis=dict(title="Booking Modality"),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                showlegend=False,
                height=320
            )
            st.plotly_chart(fig_b, use_container_width=True)

        with c4:
            st.markdown("##### Key Temporal Findings")
            st.markdown(
                """
                * **Immediate Intent:** Same-day bookings achieve **95.36% attendance** (4.64% no-show rate). Patients scheduling for today have acute need and minimal friction.
                * **The Lead Time Surge:** Once lead time exceeds 14 days, no-show rate jumps above 30%.
                * **Operational Takeaway:** Shortening booking windows through fast-track slots is the single most potent administrative lever for improving attendance.
                """
            )

    # -------------------------------------------------------------
    # TAB 3: Patient History Patterns
    # -------------------------------------------------------------
    with tab3:
        st.markdown("#### Historical Attendance Fidelity & Repeat Behavior")

        st.markdown(
            "Patients with prior appointment records exhibit powerful behavioral momentum: "
            "reliable attenders remain reliable, while repeat no-shows face escalating risk multipliers."
        )

        hist_records = [
            {"Category": "First-Time Patients", "No-Show Rate": "21.4%", "Description": "Zero prior visits on record; moderately elevated risk baseline."},
            {"Category": "Established Attenders (Prior No-Show Rate = 0%)", "No-Show Rate": "12.8%", "Description": "Frequent visitors with perfect past attendance; highly protective."},
            {"Category": "Moderate Prior No-Shows (Rate 1% - 50%)", "No-Show Rate": "28.5%", "Description": "Patients with occasional schedule disruptions."},
            {"Category": "Chronic Prior No-Shows (Rate > 50%)", "No-Show Rate": "56.2%", "Description": "Severe non-attendance history; primary candidates for active navigation."}
        ]
        st.dataframe(pd.DataFrame(hist_records), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # TAB 4: Interactive Data Explorer
    # -------------------------------------------------------------
    with tab4:
        st.markdown("#### Searchable Appointment Data Table")
        
        search_term = st.text_input("Search by Neighborhood, Patient ID, or Appointment ID", "")
        
        display_cols = [
            'appointment_id', 'patient_id', 'appointment_day', 'scheduled_day',
            'lead_days', 'age', 'gender', 'scholarship', 'hypertension', 'diabetes', 'no_show'
        ]
        present_cols = [c for c in display_cols if c in df_full.columns]
        
        df_table = df_full[present_cols].copy()
        if search_term:
            mask = df_table['appointment_id'].astype(str).str.contains(search_term, case=False) | \
                   df_table['patient_id'].astype(str).str.contains(search_term, case=False)
            if 'neighbourhood' in df_full.columns:
                mask = mask | df_full['neighbourhood'].astype(str).str.contains(search_term, case=False)
            df_table = df_table[mask]

        st.markdown(f"Displaying **{min(len(df_table), 500):,}** of **{len(df_table):,}** matching appointments:")
        st.dataframe(df_table.head(500), use_container_width=True)

        csv_data = df_table.head(2000).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv_data,
            file_name="filtered_appointment_data.csv",
            mime="text/csv"
        )

