"""
CarePredict - Data and Artifact Loader
Fast, cached loading of datasets, metadata, and benchmark results.
"""

import json
import pandas as pd
import streamlit as st
from app.config import (
    CLEANED_DATA_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    VAL_PREDICTIONS_PATH,
    FEATURE_METADATA_PATH,
    SHAP_GROUPED_PATH,
    THRESHOLD_COMPARISON_PATH,
    THRESHOLD_JSON_PATH,
)

@st.cache_data(show_spinner=False)
def load_cleaned_data() -> pd.DataFrame:
    """Loads full audited cleaned patient dataset (110,521 rows)."""
    df = pd.read_csv(CLEANED_DATA_PATH)
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
    df['appointment_day'] = pd.to_datetime(df['appointment_day'])
    # Derive lead_days if not already present
    if 'lead_days' not in df.columns:
        df['lead_days'] = (df['appointment_day'].dt.normalize() - df['scheduled_day'].dt.normalize()).dt.days
    return df

@st.cache_data(show_spinner=False)
def load_validation_features() -> pd.DataFrame:
    """Loads chronological validation feature dataset (20,539 rows)."""
    df = pd.read_csv(VAL_FEATURES_PATH)
    if 'scheduled_day' in df.columns:
        df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
    if 'appointment_day' in df.columns:
        df['appointment_day'] = pd.to_datetime(df['appointment_day'])
    return df

@st.cache_data(show_spinner=False)
def load_train_features() -> pd.DataFrame:
    """Loads chronological training feature dataset (63,532 rows)."""
    df = pd.read_csv(TRAIN_FEATURES_PATH)
    return df

@st.cache_data(show_spinner=False)
def load_validation_predictions() -> pd.DataFrame:
    """Loads saved validation predictions from the champion unweighted XGBoost model."""
    return pd.read_csv(VAL_PREDICTIONS_PATH)

@st.cache_data(show_spinner=False)
def load_feature_metadata() -> dict:
    """Loads the feature metadata catalog and split architecture."""
    with open(FEATURE_METADATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_global_shap_importance() -> pd.DataFrame:
    """Loads precomputed probability-space global feature importance ranking."""
    return pd.read_csv(SHAP_GROUPED_PATH)

@st.cache_data(show_spinner=False)
def load_threshold_benchmarks() -> pd.DataFrame:
    """Loads Step 5 threshold cost comparison table."""
    return pd.read_csv(THRESHOLD_COMPARISON_PATH)

@st.cache_data(show_spinner=False)
def load_threshold_cost_analysis() -> dict:
    """Loads Step 5 serialized threshold cost JSON analysis."""
    with open(THRESHOLD_JSON_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

