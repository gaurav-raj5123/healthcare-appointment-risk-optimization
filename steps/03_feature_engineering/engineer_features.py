import os
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_patient_data.csv')
TRAIN_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'train_features.csv')
TEST_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'test_features.csv')
METADATA_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'feature_metadata.json')

# Temporal Split Cutoff: Train on April/May 2016, Test on unseen future June 2016 appointments
TEMPORAL_CUTOFF_DATE = '2016-06-01'

def engineer_features(input_path=INPUT_DATA_PATH):
    print("=" * 70)
    print("STEP 3: FEATURE ENGINEERING PIPELINE (PREDICTION-TIME COMPLIANT)")
    print("=" * 70)
    print(f"Loading cleaned dataset from: {input_path}")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"Loaded {initial_count:,} appointments.")
    
    # Standardize target
    df['no_show'] = df['disengaged'].astype(int)  # 1 = No-show, 0 = Show
    
    # Ensure proper datetime parsing
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
    df['appointment_day'] = pd.to_datetime(df['appointment_day'])
    
    # ---------------------------------------------------------
    # 1. CORE STATIC CLINICAL & DEMOGRAPHIC FEATURES
    # ---------------------------------------------------------
    print("\n--- Engineering Core Static Features ---")
    # gender: 1 = Female, 0 = Male
    df['gender'] = (df['gender'].astype(str).str.upper() == 'F').astype(int)
    
    # has_handicap: binary indicator for physical disability (raw data had 0-4)
    df['has_handicap'] = (df['handicap'] > 0).astype(int)
    
    # Ensure binary integer types
    for col in ['scholarship', 'hypertension', 'diabetes', 'alcoholism', 'sms_received']:
        df[col] = df[col].astype(int)
        
    print(f"Static features verified: age, gender, scholarship, hypertension, diabetes, alcoholism, has_handicap, sms_received")

    # ---------------------------------------------------------
    # 2. CORE TEMPORAL & OPERATIONAL FEATURES
    # ---------------------------------------------------------
    print("\n--- Engineering Core Temporal Features ---")
    # lead_days: days between scheduling and appointment
    df['lead_days'] = (df['appointment_day'].dt.normalize() - df['scheduled_day'].dt.normalize()).dt.days
    
    # same_day_booking: binary flag for 0 lead days (distinct low-risk operational segment)
    df['same_day_booking'] = (df['lead_days'] == 0).astype(int)
    
    # Day of week (0 = Monday, ..., 5 = Saturday)
    df['appointment_dow'] = df['appointment_day'].dt.dayofweek
    df['scheduled_dow'] = df['scheduled_day'].dt.dayofweek
    
    # Appointment month (4 = April, 5 = May, 6 = June)
    df['appointment_month'] = df['appointment_day'].dt.month
    
    print(f"Temporal features created: lead_days, same_day_booking, appointment_dow, scheduled_dow, appointment_month")

    # ---------------------------------------------------------
    # 3. CORE PATIENT HISTORY FEATURES (POINT-IN-TIME, ZERO LEAKAGE)
    # ---------------------------------------------------------
    print("\n--- Engineering Core Patient Behavioral History (Zero-Leakage) ---")
    # Sort chronologically by patient and appointment date so rolling statistics are strictly past-only
    df = df.sort_values(by=['patient_id', 'appointment_day', 'scheduled_day', 'appointment_id']).reset_index(drop=True)
    
    # prior_appointments: number of appointments prior to this visit
    df['prior_appointments'] = df.groupby('patient_id').cumcount()
    
    # prior_noshows: cumulative missed appointments strictly prior to this visit
    df['prior_noshows'] = df.groupby('patient_id')['no_show'].cumsum() - df['no_show']
    
    # prior_noshow_rate: ratio of prior misses (0.0 for first-time visitors with 0 prior visits)
    df['prior_noshow_rate'] = np.where(
        df['prior_appointments'] > 0,
        df['prior_noshows'] / df['prior_appointments'],
        0.0
    ).round(4)
    
    # is_first_appointment: binary flag distinguishing first visit from returning patients
    df['is_first_appointment'] = (df['prior_appointments'] == 0).astype(int)
    
    # days_since_last_appointment: interval since previous appointment (-1 for first visits)
    prev_appt_date = df.groupby('patient_id')['appointment_day'].shift(1)
    df['days_since_last_appointment'] = (df['appointment_day'] - prev_appt_date).dt.days.fillna(-1).astype(int)
    
    print(f"Patient history features created:")
    print(f"  First-time visits: {df['is_first_appointment'].sum():,} ({df['is_first_appointment'].mean()*100:.2f}%)")
    print(f"  Repeat visits:     {(df['is_first_appointment'] == 0).sum():,} ({(df['is_first_appointment'] == 0).mean()*100:.2f}%)")
    print(f"  Mean prior no-show rate (repeat): {df[df['is_first_appointment'] == 0]['prior_noshow_rate'].mean():.4f}")

    # ---------------------------------------------------------
    # 4. STRICT TEMPORAL TRAIN / TEST SPLIT
    # ---------------------------------------------------------
    print("\n--- Performing Strict Temporal Train/Test Split ---")
    train_mask = df['appointment_day'] < pd.Timestamp(TEMPORAL_CUTOFF_DATE, tz=df['appointment_day'].dt.tz)
    test_mask = df['appointment_day'] >= pd.Timestamp(TEMPORAL_CUTOFF_DATE, tz=df['appointment_day'].dt.tz)
    
    df_train = df[train_mask].copy().sort_values(by=['appointment_day', 'scheduled_day']).reset_index(drop=True)
    df_test = df[test_mask].copy().sort_values(by=['appointment_day', 'scheduled_day']).reset_index(drop=True)
    
    print(f"Temporal Cutoff Date: {TEMPORAL_CUTOFF_DATE}")
    print(f"  Train set: {len(df_train):,} appointments ({len(df_train)/len(df)*100:.1f}%) | "
          f"Dates: {df_train['appointment_day'].min().strftime('%Y-%m-%d')} to {df_train['appointment_day'].max().strftime('%Y-%m-%d')} | "
          f"No-show rate: {df_train['no_show'].mean()*100:.2f}%")
    print(f"  Test set:  {len(df_test):,} appointments ({len(df_test)/len(df)*100:.1f}%) | "
          f"Dates: {df_test['appointment_day'].min().strftime('%Y-%m-%d')} to {df_test['appointment_day'].max().strftime('%Y-%m-%d')} | "
          f"No-show rate: {df_test['no_show'].mean()*100:.2f}%")

    # ---------------------------------------------------------
    # 5. INTEGRITY CHECKS & LEAKAGE AUDIT
    # ---------------------------------------------------------
    print("\n--- Running Automated Integrity & Anti-Leakage Audits ---")
    assert len(df_train) + len(df_test) == len(df), "Row count mismatch in temporal split!"
    assert df_train['appointment_day'].max() < df_test['appointment_day'].min(), "Temporal leakage detected! Train overlap with Test."
    assert (df['prior_noshows'] <= df['prior_appointments']).all(), "Logical error: prior_noshows > prior_appointments!"
    assert ((df['prior_noshow_rate'] >= 0.0) & (df['prior_noshow_rate'] <= 1.0)).all(), "Invalid prior_noshow_rate range!"
    assert (df['lead_days'] >= 0).all(), "Negative lead time detected!"
    print("All 5 integrity assertions PASSED successfully!")

    # ---------------------------------------------------------
    # 6. SAVE OUTPUT DATASETS & FEATURE CATALOG
    # ---------------------------------------------------------
    # Order columns cleanly: Identifiers/Dates -> Target -> Features
    feature_cols = [
        'age', 'gender', 'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'has_handicap', 'sms_received',
        'lead_days', 'same_day_booking', 'appointment_dow', 'scheduled_dow', 'appointment_month',
        'prior_appointments', 'prior_noshows', 'prior_noshow_rate', 'days_since_last_appointment', 'is_first_appointment'
    ]
    meta_cols = ['appointment_id', 'patient_id', 'scheduled_day', 'appointment_day', 'no_show']
    ordered_cols = meta_cols + feature_cols
    
    df_train_out = df_train[ordered_cols]
    df_test_out = df_test[ordered_cols]
    
    os.makedirs(os.path.dirname(TRAIN_OUTPUT_PATH), exist_ok=True)
    df_train_out.to_csv(TRAIN_OUTPUT_PATH, index=False)
    df_test_out.to_csv(TEST_OUTPUT_PATH, index=False)
    print(f"\nSaved Train features ({len(df_train_out):,} rows) to: {TRAIN_OUTPUT_PATH}")
    print(f"Saved Test features  ({len(df_test_out):,} rows) to: {TEST_OUTPUT_PATH}")
    
    # Feature catalog metadata
    feature_catalog = {
        "pipeline_version": "1.0",
        "temporal_cutoff": TEMPORAL_CUTOFF_DATE,
        "train_rows": len(df_train_out),
        "test_rows": len(df_test_out),
        "target": {
            "name": "no_show",
            "type": "binary",
            "values": {"0": "Show (Attended)", "1": "No-Show (Missed)"},
            "train_rate": round(float(df_train['no_show'].mean()), 4),
            "test_rate": round(float(df_test['no_show'].mean()), 4)
        },
        "features": {
            "core_static": [
                {"name": "age", "type": "continuous", "description": "Patient age in years"},
                {"name": "gender", "type": "binary", "description": "1 = Female, 0 = Male"},
                {"name": "scholarship", "type": "binary", "description": "Welfare assistance enrollment (Bolsa Familia)"},
                {"name": "hypertension", "type": "binary", "description": "Diagnosed hypertension"},
                {"name": "diabetes", "type": "binary", "description": "Diagnosed diabetes"},
                {"name": "alcoholism", "type": "binary", "description": "Documented alcoholism"},
                {"name": "has_handicap", "type": "binary", "description": "Physical disability indicator"},
                {"name": "sms_received", "type": "binary", "description": "Whether an SMS reminder was sent"}
            ],
            "core_temporal": [
                {"name": "lead_days", "type": "continuous", "description": "Waiting time in calendar days from booking to visit"},
                {"name": "same_day_booking", "type": "binary", "description": "1 if appointment booked on the same day (lead_days == 0)"},
                {"name": "appointment_dow", "type": "categorical_nominal", "description": "Day of week of appointment (0=Monday..5=Saturday)"},
                {"name": "scheduled_dow", "type": "categorical_nominal", "description": "Day of week when booking was scheduled (0=Monday..5=Saturday)"},
                {"name": "appointment_month", "type": "categorical_nominal", "description": "Calendar month of appointment (4=Apr, 5=May, 6=Jun)"}
            ],
            "core_patient_history": [
                {"name": "prior_appointments", "type": "discrete_count", "description": "Number of previous appointments for this patient"},
                {"name": "prior_noshows", "type": "discrete_count", "description": "Number of previous missed appointments for this patient"},
                {"name": "prior_noshow_rate", "type": "continuous_ratio", "description": "Ratio of prior missed visits (0.0 for first-time visitors)"},
                {"name": "days_since_last_appointment", "type": "discrete_count", "description": "Days since patient's previous visit (-1 for first-time visits)"},
                {"name": "is_first_appointment", "type": "binary", "description": "1 if patient has no prior recorded visits, 0 if returning"}
            ]
        }
    }
    
    with open(METADATA_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(feature_catalog, f, indent=2)
    print(f"Saved Feature Catalog metadata to: {METADATA_OUTPUT_PATH}")
    print("\nSTEP 3 FEATURE ENGINEERING COMPLETE.")
    return df_train_out, df_test_out

if __name__ == '__main__':
    engineer_features()
