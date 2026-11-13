"""
CarePredict - SHAP Explainability Service
Implements probability-space and raw-margin TreeExplainer explanations,
mapping transformed features to original clinical concepts with exact additivity.
"""

import scipy.sparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import shap

from app.config import (
    APPROVED_FEATURES,
    SHAP_BASELINE_PROBABILITY,
    SHAP_GROUPED_PATH,
    COLORS
)
from app.model_service import load_champion_model
from app.data_loader import load_train_features

@st.cache_resource(show_spinner=False)
def initialize_shap_explainers():
    """
    Initializes and caches both Probability-Space and Raw-Margin TreeExplainers
    using the exact methodology from Step 6:
      - 200 rows sampled from train_features.csv (Seed=42)
      - SHAP Independent masker evaluates effective background size of 100 rows
      - Interventional feature perturbation
      - Exact transformer mapping for aggregating one-hot columns
    """
    pipe = load_champion_model()
    prep = pipe.named_steps['prep']
    clf = pipe.named_steps['clf']
    clf.enable_categorical = False

    train_df = load_train_features()
    rng = np.random.default_rng(42)
    bg_indices = rng.choice(len(train_df), size=200, replace=False)
    bg_indices.sort()
    df_bg = train_df.iloc[bg_indices][APPROVED_FEATURES]

    X_bg_trans = prep.transform(df_bg)
    if scipy.sparse.issparse(X_bg_trans):
        X_bg_trans = X_bg_trans.toarray()

    # Probability-Space Explainer
    explainer_prob = shap.TreeExplainer(
        clf,
        data=X_bg_trans,
        model_output='probability',
        feature_perturbation='interventional'
    )

    # Raw-Margin Explainer
    explainer_raw = shap.TreeExplainer(
        clf,
        data=X_bg_trans,
        model_output='raw',
        feature_perturbation='interventional'
    )

    # Establish exact transformed-to-original column mapping
    feature_names_out = list(prep.get_feature_names_out())
    col_to_orig = {}
    for trans_name, trans_obj, input_cols in prep.transformers_:
        if trans_name == 'remainder' and trans_obj == 'drop':
            continue
        if trans_name == 'cat':
            for col_name, cat_list in zip(input_cols, trans_obj.categories_):
                kept_cats = cat_list[1:] if trans_obj.drop == 'first' else cat_list
                for cat_val in kept_cats:
                    col_to_orig[f'cat__{col_name}_{cat_val}'] = col_name
        elif trans_name in ('num', 'bin'):
            for col_name in input_cols:
                col_to_orig[f'{trans_name}__{col_name}'] = col_name

    return {
        "explainer_prob": explainer_prob,
        "explainer_raw": explainer_raw,
        "prep": prep,
        "clf": clf,
        "feature_names_out": feature_names_out,
        "col_to_orig": col_to_orig
    }

def explain_single_prediction_probability(features_dict: dict, predicted_prob: float) -> dict:
    """
    Computes exact probability-space SHAP attribution for a single appointment.
    Returns:
      - base_value: Baseline probability (18.65%)
      - feature_contributions: List of dicts with original feature, value, impact (% pts)
      - top_risk_increasing: Top 5 factors pushing toward no-show
      - top_risk_reducing: Top 5 factors pushing toward attendance
      - narrative: Clinician-friendly plain text interpretation
    """
    services = initialize_shap_explainers()
    explainer_prob = services["explainer_prob"]
    prep = services["prep"]
    feature_names_out = services["feature_names_out"]
    col_to_orig = services["col_to_orig"]

    df_row = pd.DataFrame([features_dict])[APPROVED_FEATURES]
    X_trans = prep.transform(df_row)
    if scipy.sparse.issparse(X_trans):
        X_trans = X_trans.toarray()

    shap_obj = explainer_prob(X_trans)
    if len(shap_obj.values.shape) == 3:
        raw_vals = shap_obj.values[0, :, 1]
    else:
        raw_vals = shap_obj.values[0]

    # Base value extraction
    if isinstance(explainer_prob.expected_value, (list, np.ndarray)):
        base_val = float(explainer_prob.expected_value[1] if len(explainer_prob.expected_value) > 1 else explainer_prob.expected_value[0])
    else:
        base_val = float(explainer_prob.expected_value)

    # Group transformed columns back to 17 approved original features
    grouped_contribs = []
    for orig in APPROVED_FEATURES:
        cols_idx = [i for i, col in enumerate(feature_names_out) if col_to_orig.get(col) == orig]
        if cols_idx:
            contrib_prob = float(np.sum(raw_vals[cols_idx]))
            contrib_pct = float(contrib_prob * 100.0)
            orig_val = features_dict.get(orig, 0)
            grouped_contribs.append({
                "feature": orig,
                "feature_value": orig_val,
                "shap_prob": contrib_prob,
                "shap_pct": contrib_pct,
                "abs_shap_pct": abs(contrib_pct)
            })

    df_contribs = pd.DataFrame(grouped_contribs)

    # Separate positive (risk increasing) and negative (risk reducing)
    pos_drivers = df_contribs[df_contribs['shap_pct'] > 0].sort_values('shap_pct', ascending=False)
    neg_drivers = df_contribs[df_contribs['shap_pct'] < 0].sort_values('shap_pct', ascending=True)

    top_risk_increasing = pos_drivers.head(5).to_dict(orient='records')
    top_risk_reducing = neg_drivers.head(5).to_dict(orient='records')

    # Plain-language narrative generator
    narrative_points = []
    if not pos_drivers.empty:
        top_pos = pos_drivers.iloc[0]
        narrative_points.append(
            f"The primary driver elevating no-show risk is **{format_feature_label(top_pos['feature'])}** "
            f"(value: {top_pos['feature_value']}), adding **+{top_pos['shap_pct']:.2f}% pts** to predicted probability."
        )
    if not neg_drivers.empty:
        top_neg = neg_drivers.iloc[0]
        narrative_points.append(
            f"The strongest protective factor is **{format_feature_label(top_neg['feature'])}** "
            f"(value: {top_neg['feature_value']}), subtracting **{abs(top_neg['shap_pct']):.2f}% pts** from risk."
        )

    narrative = " ".join(narrative_points) if narrative_points else "Feature contributions are balanced around the population baseline."

    return {
        "base_value_prob": base_val,
        "base_value_pct": base_val * 100.0,
        "predicted_prob": predicted_prob,
        "predicted_pct": predicted_prob * 100.0,
        "df_contributions": df_contribs.sort_values('abs_shap_pct', ascending=False).reset_index(drop=True),
        "top_risk_increasing": top_risk_increasing,
        "top_risk_reducing": top_risk_reducing,
        "narrative": narrative
    }

def format_feature_label(col_name: str) -> str:
    """Formats technical feature keys into human-readable clinical labels."""
    labels = {
        "lead_days": "Scheduling Lead Time (Days)",
        "age": "Patient Age",
        "same_day_booking": "Same-Day Walk-In Booking",
        "prior_noshow_rate": "Historical No-Show Rate",
        "prior_appointments": "Total Prior Appointments",
        "days_since_last_appointment": "Days Since Last Visit",
        "appointment_dow": "Appointment Day of Week",
        "scholarship": "Bolsa Família Welfare Enrollment",
        "scheduled_dow": "Scheduled Day of Week",
        "gender": "Patient Gender",
        "diabetes": "Diabetes Diagnosis",
        "prior_noshows": "Prior Missed Visits Count",
        "hypertension": "Hypertension Diagnosis",
        "alcoholism": "Alcoholism History",
        "appointment_month": "Appointment Month",
        "has_handicap": "Physical Disability Status",
        "is_first_appointment": "First-Time Patient Flag"
    }
    return labels.get(col_name, col_name.replace("_", " ").title())

def create_local_waterfall_chart(explanation: dict) -> go.Figure:
    """
    Renders an interactive Plotly horizontal bar / waterfall explanation
    showing feature shifts starting from baseline probability to final prediction.
    """
    base_pct = explanation["base_value_pct"]
    pred_pct = explanation["predicted_pct"]
    df_contribs = explanation["df_contributions"].copy()

    # Filter for meaningful contributors (top 8 by absolute magnitude)
    top_df = df_contribs.head(8).copy()
    top_df['label'] = [format_feature_label(f) for f in top_df['feature']]

    # Color mapping: Red for risk increasing (+), Green for risk reducing (-)
    colors = [COLORS["danger"] if val > 0 else COLORS["success"] for val in top_df['shap_pct']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top_df['label'],
        x=top_df['shap_pct'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.1)', width=1)),
        text=[f"{v:+.2f}% pts" for v in top_df['shap_pct']],
        textposition="outside",
        hoverinfo="y+x"
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Individual Risk Attribution</b> (Baseline: {base_pct:.1f}% → Predicted: {pred_pct:.1f}%)",
            font=dict(size=14, color=COLORS["text"])
        ),
        xaxis=dict(
            title="Percentage Point Shift from Population Baseline (% pts)",
            zeroline=True,
            zerolinecolor="#94a3b8",
            zerolinewidth=1.5,
            gridcolor="#f1f5f9"
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12)
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=220, r=60, t=50, b=40),
        height=380
    )

    return fig

def create_global_importance_chart(df_global: pd.DataFrame) -> go.Figure:
    """Renders interactive Plotly horizontal bar chart of global feature importance."""
    df = df_global.sort_values('mean_abs_shap_pct', ascending=True).tail(10).copy()
    df['label'] = [format_feature_label(f) for f in df['original_feature']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['label'],
        x=df['mean_abs_shap_pct'],
        orientation='h',
        marker=dict(
            color=COLORS["primary"],
            line=dict(color=COLORS["primary_dark"], width=1)
        ),
        text=[f"{v:.2f}% pts" for v in df['mean_abs_shap_pct']],
        textposition="outside",
        hoverinfo="y+x"
    ))

    fig.update_layout(
        title=dict(
            text="<b>Top 10 Global No-Show Risk Drivers</b> (Probability Space)",
            font=dict(size=15, color=COLORS["text"])
        ),
        xaxis=dict(
            title="Mean Absolute Impact on Predicted Probability (% points)",
            gridcolor="#f1f5f9"
        ),
        yaxis=dict(tickfont=dict(size=12)),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=230, r=60, t=50, b=40),
        height=420
    )

    return fig

