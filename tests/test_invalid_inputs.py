"""
Test Suite 9: Invalid Input Handling & Robustness Tests
Verifies that invalid demographic, temporal, and patient history inputs are cleanly rejected
with informative exceptions, and batch CSV processor isolates invalid records into invalid_df.
"""

from datetime import date, timedelta
import pandas as pd
import pytest
from app.feature_builder import derive_individual_features, validate_and_prepare_batch_csv

def test_negative_age_rejected():
    today = date(2026, 11, 15)
    with pytest.raises(ValueError, match="Age must be between 0 and 115"):
        derive_individual_features(
            age=-5, gender_str="F", scholarship=False, hypertension=False,
            diabetes=False, alcoholism=False, handicap_level=0,
            scheduled_dt=today, appointment_dt=today + timedelta(days=5),
            is_first_time=True
        )

def test_overage_rejected():
    today = date(2026, 11, 15)
    with pytest.raises(ValueError, match="Age must be between 0 and 115"):
        derive_individual_features(
            age=130, gender_str="F", scholarship=False, hypertension=False,
            diabetes=False, alcoholism=False, handicap_level=0,
            scheduled_dt=today, appointment_dt=today + timedelta(days=5),
            is_first_time=True
        )

def test_inverted_dates_rejected():
    today = date(2026, 11, 15)
    with pytest.raises(ValueError, match="cannot precede the scheduling date"):
        derive_individual_features(
            age=40, gender_str="M", scholarship=False, hypertension=False,
            diabetes=False, alcoholism=False, handicap_level=0,
            scheduled_dt=today, appointment_dt=today - timedelta(days=1),  # Inverted!
            is_first_time=True
        )

def test_prior_noshows_exceeding_appointments_rejected():
    today = date(2026, 11, 15)
    with pytest.raises(ValueError, match="cannot exceed total prior appointments"):
        derive_individual_features(
            age=40, gender_str="M", scholarship=False, hypertension=False,
            diabetes=False, alcoholism=False, handicap_level=0,
            scheduled_dt=today, appointment_dt=today + timedelta(days=5),
            is_first_time=False,
            prior_appts=2,
            prior_missed=3,  # 3 > 2!
            days_since_last=10
        )

def test_batch_csv_isolates_invalid_rows():
    # Construct a DataFrame with 1 valid row and 2 invalid rows
    df_raw = pd.DataFrame([
        {
            "appointment_id": 1,
            "patient_id": 101,
            "scheduled_day": "2026-11-01",
            "appointment_day": "2026-11-10",
            "age": 35,
            "gender": "F",
            "prior_appointments": 1,
            "prior_noshows": 0,
            "days_since_last_appointment": 15
        },
        {
            "appointment_id": 2,
            "patient_id": 102,
            "scheduled_day": "2026-11-10",
            "appointment_day": "2026-11-05",  # Inverted date!
            "age": 28,
            "gender": "M",
            "prior_appointments": 0,
            "prior_noshows": 0
        },
        {
            "appointment_id": 3,
            "patient_id": 103,
            "scheduled_day": "2026-11-01",
            "appointment_day": "2026-11-10",
            "age": -3,  # Negative age!
            "gender": "F",
            "prior_appointments": 0,
            "prior_noshows": 0
        }
    ])

    valid_df, invalid_df, warnings = validate_and_prepare_batch_csv(df_raw)
    
    assert len(valid_df) == 1, f"Expected 1 valid record, got {len(valid_df)}"
    assert len(invalid_df) == 2, f"Expected 2 invalid records, got {len(invalid_df)}"
    assert "_rejection_reason" in invalid_df.columns

