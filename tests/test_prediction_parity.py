"""
Test Suite 6: Individual vs. Batch Prediction Parity Tests
Verifies that individual prediction mode and batch prediction mode produce identical
predicted probabilities and risk tier classifications for identical input records.
Tested archetypes:
  1. First-time patient
  2. Repeat patient with perfect prior attendance (0 missed)
  3. Repeat patient with multiple missed appointments
  4. Same-day walk-in appointment
  5. Long-lead scheduling appointment (45 days)
"""

from datetime import date, timedelta
import pandas as pd
import numpy as np
from app.feature_builder import derive_individual_features
from app.model_service import predict_single, predict_batch

def test_prediction_parity_across_archetypes():
    ref_date = date(2026, 11, 15)

    test_cases = [
        # Case 1: First-time patient
        {
            "name": "First-Time Patient",
            "age": 28, "gender_str": "F", "scholarship": False, "hypertension": False,
            "diabetes": False, "alcoholism": False, "handicap_level": 0,
            "scheduled_dt": ref_date, "appointment_dt": ref_date + timedelta(days=10),
            "is_first_time": True, "prior_appts": 0, "prior_missed": 0, "days_since_last": -1
        },
        # Case 2: Repeat patient with 0 missed visits
        {
            "name": "Reliable Repeat Attender",
            "age": 62, "gender_str": "M", "scholarship": False, "hypertension": True,
            "diabetes": False, "alcoholism": False, "handicap_level": 1,
            "scheduled_dt": ref_date, "appointment_dt": ref_date + timedelta(days=14),
            "is_first_time": False, "prior_appts": 5, "prior_missed": 0, "days_since_last": 25
        },
        # Case 3: Repeat patient with multiple missed visits
        {
            "name": "Chronic No-Show Repeat Patient",
            "age": 41, "gender_str": "F", "scholarship": True, "hypertension": False,
            "diabetes": False, "alcoholism": False, "handicap_level": 0,
            "scheduled_dt": ref_date, "appointment_dt": ref_date + timedelta(days=21),
            "is_first_time": False, "prior_appts": 4, "prior_missed": 2, "days_since_last": 14
        },
        # Case 4: Same-day walk-in
        {
            "name": "Same-Day Walk-In Appointment",
            "age": 50, "gender_str": "M", "scholarship": False, "hypertension": True,
            "diabetes": True, "alcoholism": False, "handicap_level": 0,
            "scheduled_dt": ref_date, "appointment_dt": ref_date,  # Lead = 0
            "is_first_time": True, "prior_appts": 0, "prior_missed": 0, "days_since_last": -1
        },
        # Case 5: Long-lead appointment
        {
            "name": "Long-Lead Horizon Patient",
            "age": 33, "gender_str": "F", "scholarship": False, "hypertension": False,
            "diabetes": False, "alcoholism": False, "handicap_level": 0,
            "scheduled_dt": ref_date, "appointment_dt": ref_date + timedelta(days=45),  # Lead = 45
            "is_first_time": False, "prior_appts": 2, "prior_missed": 1, "days_since_last": 60
        }
    ]

    for case in test_cases:
        feat_dict = derive_individual_features(
            age=case["age"],
            gender_str=case["gender_str"],
            scholarship=case["scholarship"],
            hypertension=case["hypertension"],
            diabetes=case["diabetes"],
            alcoholism=case["alcoholism"],
            handicap_level=case["handicap_level"],
            scheduled_dt=case["scheduled_dt"],
            appointment_dt=case["appointment_dt"],
            is_first_time=case["is_first_time"],
            prior_appts=case["prior_appts"],
            prior_missed=case["prior_missed"],
            days_since_last=case["days_since_last"]
        )

        single_res = predict_single(feat_dict, threshold=0.15)

        df_batch_in = pd.DataFrame([feat_dict])
        batch_res = predict_batch(df_batch_in, threshold=0.15).iloc[0]

        single_prob = single_res["prob_noshow"]
        batch_prob = batch_res["predicted_noshow_prob"]

        diff = abs(single_prob - batch_prob)
        assert diff < 1e-6, (
            f"Parity failure for {case['name']}: single={single_prob}, batch={batch_prob}, diff={diff}"
        )
        assert single_res["risk_tier"] == batch_res["risk_tier"], (
            f"Risk tier mismatch for {case['name']}: single={single_res['risk_tier']}, batch={batch_res['risk_tier']}"
        )
        assert single_res["flagged"] == batch_res["flagged_for_intervention"], (
            f"Flagged decision mismatch for {case['name']}"
        )

