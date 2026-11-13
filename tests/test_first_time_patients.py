"""
Test Suite 8: First-Time Patient Convention Tests
Verifies that first-time patients strictly follow the Step 3 missing-value conventions:
  - prior_appointments = 0
  - prior_noshows = 0
  - prior_noshow_rate = 0.0
  - days_since_last_appointment = -1
  - is_first_appointment = 1
And returning patients strictly enforce:
  - is_first_appointment = 0
  - days_since_last_appointment >= 0
"""

from datetime import date, timedelta
import pytest
from app.feature_builder import derive_individual_features

def test_first_time_patient_conventions():
    today = date(2026, 11, 15)
    feat = derive_individual_features(
        age=30, gender_str="F", scholarship=False, hypertension=False,
        diabetes=False, alcoholism=False, handicap_level=0,
        scheduled_dt=today, appointment_dt=today + timedelta(days=7),
        is_first_time=True
    )
    
    assert feat["prior_appointments"] == 0
    assert feat["prior_noshows"] == 0
    assert feat["prior_noshow_rate"] == 0.0
    assert feat["days_since_last_appointment"] == -1
    assert feat["is_first_appointment"] == 1

def test_first_time_contradiction_rejected():
    today = date(2026, 11, 15)
    with pytest.raises(ValueError, match="First-time patient indicator is True"):
        derive_individual_features(
            age=30, gender_str="F", scholarship=False, hypertension=False,
            diabetes=False, alcoholism=False, handicap_level=0,
            scheduled_dt=today, appointment_dt=today + timedelta(days=7),
            is_first_time=True,
            prior_appts=2  # Contradiction
        )

def test_returning_patient_negative_days_rejected():
    today = date(2026, 11, 15)
    with pytest.raises(ValueError, match="must be non-negative for returning patients"):
        derive_individual_features(
            age=45, gender_str="M", scholarship=False, hypertension=True,
            diabetes=False, alcoholism=False, handicap_level=0,
            scheduled_dt=today, appointment_dt=today + timedelta(days=7),
            is_first_time=False,
            prior_appts=3,
            prior_missed=1,
            days_since_last=-1  # Invalid for returning patients!
        )

def test_returning_patient_valid_conventions():
    today = date(2026, 11, 15)
    feat = derive_individual_features(
        age=45, gender_str="M", scholarship=False, hypertension=True,
        diabetes=False, alcoholism=False, handicap_level=0,
        scheduled_dt=today, appointment_dt=today + timedelta(days=7),
        is_first_time=False,
        prior_appts=4,
        prior_missed=1,
        days_since_last=30
    )
    
    assert feat["prior_appointments"] == 4
    assert feat["prior_noshows"] == 1
    assert feat["prior_noshow_rate"] == 0.25
    assert feat["days_since_last_appointment"] == 30
    assert feat["is_first_appointment"] == 0

