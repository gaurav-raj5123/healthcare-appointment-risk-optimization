"""
Test Suite 4: Dataset Count and Macro KPI Tests
Verifies that the cleaned analytical dataset has exact audited dimensions:
  - Total Cleaned Appointments: 110,521 (6 anomalies removed from 110,527 raw)
  - Unique Patients: 62,298
  - No-Shows: 22,314 (20.1898% -> 20.19%)
  - Attended: 88,207 (79.8102% -> 79.81%)
  - Sunday appointment count: 0 (empirically verified, not assumed)
"""

import pandas as pd
from app.data_loader import load_cleaned_data

def test_cleaned_dataset_row_count():
    df = load_cleaned_data()
    assert len(df) == 110521, f"Expected 110,521 cleaned rows, got {len(df)}"

def test_cleaned_dataset_unique_patients():
    df = load_cleaned_data()
    assert df['patient_id'].nunique() == 62298, f"Expected 62,298 unique patients, got {df['patient_id'].nunique()}"

def test_cleaned_dataset_outcomes():
    df = load_cleaned_data()
    noshow_count = int(df['no_show'].sum())
    attended_count = len(df) - noshow_count
    
    assert noshow_count == 22314, f"Expected 22,314 no-shows, got {noshow_count}"
    assert attended_count == 88207, f"Expected 88,207 attended, got {attended_count}"
    
    noshow_rate = noshow_count / len(df)
    assert abs(noshow_rate - 0.201898) < 1e-4

def test_sunday_appointments_genuinely_zero():
    df = load_cleaned_data()
    sunday_visits = (df['appointment_day'].dt.dayofweek == 6).sum()
    assert sunday_visits == 0, f"Expected 0 Sunday appointments, found {sunday_visits}"

def test_six_anomalies_excluded():
    df = load_cleaned_data()
    # Age < 0 was 1 row in raw
    assert (df['age'] < 0).sum() == 0
    # Inverted dates (appointment_day < scheduled_day) were 5 rows in raw
    assert (df['lead_days'] < 0).sum() == 0

