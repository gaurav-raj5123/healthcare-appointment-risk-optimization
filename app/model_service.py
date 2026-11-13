"""
CarePredict - Model Service
Loads and serves the champion unweighted XGBoost model for individual and batch inference.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from app.config import MODEL_PATH, APPROVED_FEATURES, RISK_TIERS, DEFAULT_THRESHOLD

@st.cache_resource(show_spinner=False)
def load_champion_model():
    """
    Loads and caches the serialized champion unweighted XGBoost pipeline.
    Pipeline includes:
      - 'prep': ColumnTransformer (StandardScaler + OneHotEncoder + passthrough)
      - 'clf': XGBClassifier (scale_pos_weight=1.0)
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Champion model not found at: {MODEL_PATH}")
    pipe = joblib.load(MODEL_PATH)
    # Ensure public attribute enable_categorical is False for consistent inference & SHAP
    if hasattr(pipe, 'named_steps') and 'clf' in pipe.named_steps:
        pipe.named_steps['clf'].enable_categorical = False
    return pipe

def get_risk_tier(prob: float) -> dict:
    """
    Classifies predicted probability into mutually exclusive operational risk tiers:
      - Low Risk:      0.00 <= prob < 0.15
      - Moderate Risk: 0.15 <= prob < 0.40
      - High Risk:     0.40 <= prob < 0.70
      - Critical Risk: 0.70 <= prob <= 1.00
    Raises ValueError on invalid values (negative, > 1.0, NaN, infinity).
    """
    if prob is None or not isinstance(prob, (int, float, np.number)) or np.isnan(prob) or np.isinf(prob):
        raise ValueError(f"Invalid probability value for risk tier: {prob}. Must be a finite numeric value.")
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f"Probability out of bounds: {prob}. Must be between 0.0 and 1.0.")

    if 0.0 <= prob < 0.15:
        tier_name = "Low Risk"
    elif 0.15 <= prob < 0.40:
        tier_name = "Moderate Risk"
    elif 0.40 <= prob < 0.70:
        tier_name = "High Risk"
    elif 0.70 <= prob <= 1.0:
        tier_name = "Critical Risk"
    else:
        raise ValueError(f"Uncategorized probability: {prob}")

    tier_info = RISK_TIERS[tier_name]
    return {
        "name": tier_name,
        "color": tier_info["color"],
        "action": tier_info["action"],
        "badge_class": tier_info["badge_class"]
    }

def predict_single(features_dict: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Runs inference for a single appointment record.
    Returns probability, attendance probability, risk tier, and intervention decision.
    """
    pipe = load_champion_model()
    # Construct 1-row DataFrame with exact approved feature columns
    df_row = pd.DataFrame([features_dict])[APPROVED_FEATURES]
    prob_noshow = float(pipe.predict_proba(df_row)[0, 1])
    prob_attend = float(1.0 - prob_noshow)
    flagged = bool(prob_noshow >= threshold)
    risk_info = get_risk_tier(prob_noshow)
    
    return {
        "prob_noshow": prob_noshow,
        "prob_attend": prob_attend,
        "flagged": flagged,
        "threshold_used": threshold,
        "risk_tier": risk_info["name"],
        "risk_color": risk_info["color"],
        "risk_action": risk_info["action"],
        "badge_class": risk_info["badge_class"]
    }

def predict_batch(df_features: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """
    Runs batch inference on a DataFrame containing approved features.
    Appends probability columns, risk tiers, and intervention decision.
    """
    pipe = load_champion_model()
    X = df_features[APPROVED_FEATURES].copy()
    probs = pipe.predict_proba(X)[:, 1]
    
    df_out = df_features.copy()
    df_out['predicted_noshow_prob'] = probs.astype(float)
    df_out['predicted_attend_prob'] = (1.0 - probs).astype(float)
    df_out['flagged_for_intervention'] = df_out['predicted_noshow_prob'] >= threshold
    
    tiers = [get_risk_tier(p) for p in probs]
    df_out['risk_tier'] = [t["name"] for t in tiers]
    df_out['recommended_action'] = [t["action"] for t in tiers]
    df_out['threshold_evaluated'] = threshold
    
    return df_out

