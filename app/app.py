"""
CarePredict - Appointment Intelligence & Capacity Optimization
Main Streamlit Application Entry Point
"""

import sys
from pathlib import Path

# Ensure project root is the primary search path for unambiguous package resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
if sys.path[0] != str(ROOT_DIR):
    if str(ROOT_DIR) in sys.path:
        sys.path.remove(str(ROOT_DIR))
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from app.config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_ICON
)
from app.components import inject_custom_css, render_header
from app.pages.overview import render_overview_page
from app.pages.analytics import render_analytics_page
from app.pages.prediction import render_prediction_page
from app.pages.optimization import render_optimization_page

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="CarePredict | Appointment Intelligence",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom healthcare aesthetic CSS
inject_custom_css()

def main():
    # -------------------------------------------------------------
    # Sidebar Navigation & Model Specifications
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown(f"## {APP_ICON} CarePredict")
        st.caption("Healthcare Operations Decision Support System")

        nav_choice = st.radio(
            "Navigation",
            options=[
                "Executive Overview",
                "Appointment Analytics",
                "No-Show Prediction & Explainability",
                "Capacity & Financial Optimization"
            ],
            index=0
        )

        st.markdown("---")
        st.markdown("##### Champion Model Specification")
        st.markdown(
            """
            * **Architecture:** Unweighted XGBoost
            * **Primary Metric:** PR-AUC = 0.3584
            * **ROC-AUC:** 0.7382
            * **Brier Calibration:** 0.1431 (ECE: 0.0240)
            * **Decision Threshold:** 0.15 (Default)
            * **Pipeline Status:** Step 1–6 Frozen
            """
        )

        st.markdown("---")
        with st.expander("About this Prototype & Limitations"):
            st.markdown(
                """
                **Dataset Context:**  
                110,521 cleaned outpatient appointments from public SUS clinics in Vitória, Brazil (April–June 2016).

                **Chronological Partitions:**
                * Train: Apr 29 – May 19 (63,532 rows)
                * Validation: May 20 – May 31 (20,539 rows)
                * Holdout: Jun 1 – Jun 8 (26,450 rows)

                **Ethical & Clinical Governance:**  
                * This application is a decision-support prototype, not a clinical diagnostic system.
                * Predictions must support human-led supportive outreach and must **never** be used to deny, delay, or restrict patient care.
                * Financial calculations are modeled scenarios under illustrative assumptions, not measured operational savings.
                """
            )

        st.caption("CarePredict v1.0 | Machine Learning Portfolio")

    # -------------------------------------------------------------
    # Top Header Banner
    # -------------------------------------------------------------
    render_header(APP_TITLE, APP_SUBTITLE)

    # -------------------------------------------------------------
    # Page Routing
    # -------------------------------------------------------------
    if nav_choice == "Executive Overview":
        render_overview_page()
    elif nav_choice == "Appointment Analytics":
        render_analytics_page()
    elif nav_choice == "No-Show Prediction & Explainability":
        render_prediction_page()
    elif nav_choice == "Capacity & Financial Optimization":
        render_optimization_page()

    # Footer
    st.markdown("---")
    st.caption("CarePredict Decision Support Platform | Developed with Scikit-Learn, XGBoost, SHAP, and Streamlit.")

if __name__ == "__main__":
    main()
