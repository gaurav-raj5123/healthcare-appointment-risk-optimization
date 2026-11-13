"""
CarePredict - Application Configuration
Paths, constants, risk thresholds, and visual design parameters.
"""

import os
from pathlib import Path

# Resolve Project Root dynamically
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# Application Metadata
APP_TITLE = "CarePredict | Appointment Intelligence & Capacity Optimization"
APP_SUBTITLE = "Predict appointment non-attendance. Understand patient risk. Optimize intervention capacity."
APP_ICON = "🏥"

# Data & Model Artifact Paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

CLEANED_DATA_PATH = DATA_DIR / "cleaned_patient_data.csv"
TRAIN_FEATURES_PATH = DATA_DIR / "train_features.csv"
VAL_FEATURES_PATH = DATA_DIR / "val_features.csv"
TEST_FEATURES_PATH = DATA_DIR / "test_features.csv"
VAL_PREDICTIONS_PATH = DATA_DIR / "val_predictions_xgb_unweighted.csv"
FEATURE_METADATA_PATH = DATA_DIR / "feature_metadata.json"
LEAKAGE_AUDIT_PATH = DATA_DIR / "leakage_audit_report.json"
SHAP_GROUPED_PATH = DATA_DIR / "shap_probability_grouped_feature_importance.csv"
SHAP_LOCAL_PATH = DATA_DIR / "shap_probability_local_explanations.csv"
THRESHOLD_COMPARISON_PATH = DATA_DIR / "step5_threshold_comparison.csv"
THRESHOLD_JSON_PATH = DATA_DIR / "step5_threshold_cost_analysis.json"

MODEL_PATH = MODELS_DIR / "xgboost_unweighted.joblib"
CALIBRATED_MODEL_PATH = MODELS_DIR / "xgboost_calibrated.joblib"
TEMPLATE_PATH = APP_DIR / "templates" / "appointment_template.csv"
CSS_PATH = APP_DIR / "assets" / "style.css"

# Approved Feature Schema (Must match trained XGBoost pipeline exactly)
APPROVED_FEATURES = [
    "age",
    "gender",
    "scholarship",
    "hypertension",
    "diabetes",
    "alcoholism",
    "has_handicap",
    "lead_days",
    "same_day_booking",
    "appointment_dow",
    "scheduled_dow",
    "appointment_month",
    "prior_appointments",
    "prior_noshows",
    "prior_noshow_rate",
    "days_since_last_appointment",
    "is_first_appointment",
]

# Baseline Expected Value from Step 6 Dual-Space SHAP
SHAP_BASELINE_PROBABILITY = 0.18646  # 18.65%

# Default Financial & Policy Parameters
DEFAULT_THRESHOLD = 0.15
DEFAULT_COST_INTERVENTION = 10.0      # $10 per flagged patient
DEFAULT_COST_UNFILLED = 200.0         # $200 per unused appointment slot
DEFAULT_OUTREACH_EFFECTIVENESS = 0.30 # 30% reduction in no-show probability
DEFAULT_CAPACITY = 500                # Default daily staff outreach cap

# Risk Tiers (Operational categories, not clinical diagnoses)
# Mutually exclusive, non-overlapping probability intervals:
#   Low Risk:      0.00 <= p < 0.15
#   Moderate Risk: 0.15 <= p < 0.40
#   High Risk:     0.40 <= p < 0.70
#   Critical Risk: 0.70 <= p <= 1.00
RISK_TIERS = {
    "Low Risk": {
        "min": 0.0,
        "max": 0.15,
        "color": "#16a34a",
        "action": "Standard automated reminder; no manual staff action required.",
        "badge_class": "badge-low"
    },
    "Moderate Risk": {
        "min": 0.15,
        "max": 0.40,
        "color": "#d97706",
        "action": "Two-way automated SMS reminder with 1-click confirmation/reschedule.",
        "badge_class": "badge-moderate"
    },
    "High Risk": {
        "min": 0.40,
        "max": 0.70,
        "color": "#ea580c",
        "action": "Personal outreach call by clinic scheduler; identify schedule barriers.",
        "badge_class": "badge-high"
    },
    "Critical Risk": {
        "min": 0.70,
        "max": 1.0,
        "color": "#dc2626",
        "action": "Comprehensive care navigation: clinical navigator call, transit voucher, or telehealth conversion.",
        "badge_class": "badge-critical"
    }
}

# Theme Color Palette
COLORS = {
    "primary": "#0284c7",       # Sky 600
    "primary_dark": "#0369a1",  # Sky 700
    "teal": "#0d9488",          # Teal 600
    "teal_dark": "#0f766e",     # Teal 700
    "background": "#f8fafc",    # Slate 50
    "surface": "#ffffff",
    "border": "#e2e8f0",        # Slate 200
    "text": "#0f172a",          # Slate 900
    "text_muted": "#64748b",    # Slate 500
    "success": "#16a34a",       # Emerald 600
    "warning": "#d97706",       # Amber 600
    "danger": "#dc2626",        # Red 600
}

