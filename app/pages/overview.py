"""
CarePredict - Page 1: Executive Overview
Landing dashboard providing hospital administrators with high-level appointment situation KPIs,
volume trends, and demographic/temporal risk distributions.
"""

import streamlit as st
import pandas as pd
from app.data_loader import load_cleaned_data
from app.components import render_kpi_card, render_callout
from app.analytics import (
    plot_volume_trend,
    plot_attendance_donut,
    plot_lead_time_risk,
    plot_age_distribution_risk,
    plot_weekday_risk,
    plot_comorbidity_comparison
)

def render_overview_page():
    st.markdown("### Executive Overview")
    st.markdown("High-level operational health of outpatient scheduling and attendance patterns.")

    render_callout(
        "<b>Dataset & Cohort Baseline:</b> Evaluated on <b>110,521 cleaned outpatient appointments</b> "
        "(62,298 unique patients) from public SUS clinics in Vitória, Brazil (April 29 – June 8, 2016). "
        "Original raw cohort contained 110,527 records; 6 anomaly records (1 negative age, 5 inverted booking dates) "
        "were removed during Step 1 auditing. Macro no-show rate is calculated directly as <b>20.19%</b> (22,314 missed appointments).",
        box_type="notice"
    )

    df_clean = load_cleaned_data()

    # -------------------------------------------------------------
    # Interactive Filters Section
    # -------------------------------------------------------------
    with st.expander("Filter Executive Overview Data", expanded=False):
        f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
        
        with f_col1:
            min_date = df_clean['appointment_day'].min().date()
            max_date = df_clean['appointment_day'].max().date()
            date_range = st.date_input(
                "Appointment Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Both start and end dates are inclusive."
            )
        with f_col2:
            gender_filter = st.multiselect("Gender", options=["Female", "Male"], default=["Female", "Male"])
        with f_col3:
            min_age, max_age = int(df_clean['age'].min()), int(df_clean['age'].max())
            age_range = st.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
        with f_col4:
            scholarship_filter = st.selectbox("Bolsa Família Welfare", options=["All Patients", "Enrolled Only", "Not Enrolled"])
        with f_col5:
            comorbid_filter = st.selectbox("Chronic Conditions", options=["All Patients", "Hypertension Diagnosed", "Diabetes Diagnosed"])

    # Apply filters safely
    df = df_clean.copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        df = df[(df['appointment_day'].dt.date >= date_range[0]) & (df['appointment_day'].dt.date <= date_range[1])]
    elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
        df = df[df['appointment_day'].dt.date == date_range[0]]

    if len(gender_filter) == 1:
        req_gender = 'F' if gender_filter[0] == 'Female' else 'M'
        df = df[df['gender'].astype(str).str.upper() == req_gender]
    
    df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]

    if scholarship_filter == "Enrolled Only":
        df = df[df['scholarship'] == 1]
    elif scholarship_filter == "Not Enrolled":
        df = df[df['scholarship'] == 0]

    if comorbid_filter == "Hypertension Diagnosed":
        df = df[df['hypertension'] == 1]
    elif comorbid_filter == "Diabetes Diagnosed":
        df = df[df['diabetes'] == 1]

    if df.empty:
        st.warning("No appointments match the selected filter criteria. Please broaden your selection.")
        return

    # -------------------------------------------------------------
    # Executive KPI Cards (Historical vs Model Estimates)
    # -------------------------------------------------------------
    total_appts = len(df)
    unique_patients = df['patient_id'].nunique() if 'patient_id' in df.columns else total_appts
    actual_noshows = int(df['no_show'].sum())
    noshow_rate = (actual_noshows / total_appts) * 100.0 if total_appts > 0 else 0.0
    attend_rate = 100.0 - noshow_rate

    # Modeled high-risk estimate (using empirical 0.15 threshold expectation)
    # At 0.15 threshold, model flags ~64.6% of appointments for proactive outreach
    est_flagged = int(total_appts * 0.6463)
    est_caught = int(actual_noshows * 0.9362)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_kpi_card("Total Appointments", f"{total_appts:,}", f"{unique_patients:,} unique patients")
    with kpi2:
        render_kpi_card("Overall No-Show Rate", f"{noshow_rate:.1f}%", f"{actual_noshows:,} missed appointments")
    with kpi3:
        render_kpi_card("Attendance Rate", f"{attend_rate:.1f}%", f"{total_appts - actual_noshows:,} attended visits")
    with kpi4:
        render_kpi_card("Estimated Flagged Workload", f"{est_flagged:,}", "At candidate 0.15 decision threshold")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Charts Row 1: Volume Trend & Outcome Distribution
    # -------------------------------------------------------------
    col_c1, col_c2 = st.columns([3, 2])
    with col_c1:
        st.plotly_chart(plot_volume_trend(df), use_container_width=True)
    with col_c2:
        st.plotly_chart(plot_attendance_donut(df), use_container_width=True)

    # -------------------------------------------------------------
    # Charts Row 2: Lead Time Horizon & Age Demographics
    # -------------------------------------------------------------
    col_c3, col_c4 = st.columns(2)
    with col_c3:
        st.plotly_chart(plot_lead_time_risk(df), use_container_width=True)
    with col_c4:
        st.plotly_chart(plot_age_distribution_risk(df), use_container_width=True)

    # -------------------------------------------------------------
    # Charts Row 3: Weekday Patterns & Clinical Comorbidities
    # -------------------------------------------------------------
    col_c5, col_c6 = st.columns(2)
    with col_c5:
        st.plotly_chart(plot_weekday_risk(df), use_container_width=True)
    with col_c6:
        st.plotly_chart(plot_comorbidity_comparison(df), use_container_width=True)

