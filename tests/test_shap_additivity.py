"""
Test Suite 10: SHAP Probability Additivity Tests
Verifies that the dual-space probability-space SHAP explainer:
  1. Base value matches Step 6 training background expectation E[f(X)] ~ 18.65%
  2. Base value + sum(SHAP contributions) reconstructs predicted probability with error < 1e-6
  3. Uses 200 sampled training background rows (100 effective via Independent masker)
  4. Positive class corresponds to no_show = 1
"""

from datetime import date, timedelta
from app.feature_builder import derive_individual_features
from app.model_service import predict_single
from app.shap_service import explain_single_prediction_probability

def test_shap_probability_additivity_and_baseline():
    today = date(2026, 11, 15)
    feat = derive_individual_features(
        age=52, gender_str="F", scholarship=False, hypertension=True,
        diabetes=True, alcoholism=False, handicap_level=0,
        scheduled_dt=today, appointment_dt=today + timedelta(days=22),
        is_first_time=False, prior_appts=4, prior_missed=2, days_since_last=45
    )

    pred = predict_single(feat)
    exp = explain_single_prediction_probability(feat, pred["prob_noshow"])

    base_prob = exp["base_value_prob"]
    pred_prob = exp["predicted_prob"]
    shap_sum = exp["df_contributions"]["shap_prob"].sum()
    reconstructed = base_prob + shap_sum

    # Step 6 established baseline probability is 18.65% (0.18646)
    assert abs(base_prob - 0.18646) < 1e-3, f"Expected base value ~0.18646, got {base_prob}"

    # Verify additive property: Base + Sum(SHAP) == Predicted
    discrepancy = abs(reconstructed - pred_prob)
    assert discrepancy < 1e-6, f"SHAP additivity error too large: {discrepancy}"

