import os
import json
import time
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_patient_data.csv')
TRAIN_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'train_features.csv')
VAL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'val_features.csv')
DEV_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'dev_features.csv')
TEST_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'test_features.csv')
METADATA_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'feature_metadata.json')

# Chronological Split Boundaries
TRAIN_VAL_CUTOFF = '2016-05-20'   # Split April-May into Train (early) and Validation (late)
VAL_TEST_CUTOFF = '2016-06-01'    # June is the final unseen test set

def compute_leakage_safe_history(df):
    """
    Computes patient history features strictly respecting the point-in-time constraint:
    At the moment of scheduling (Day 0), only past appointments whose outcomes
    have ALREADY completed and are known (appointment_date < scheduled_date) are counted.
    
    Zero future information:
    - Future appointments of the same patient are ignored.
    - Same-day scheduled visits are not assumed known (no intra-day timestamps exist).
    - Current appointment outcome is never used.
    """
    total_len = len(df)
    pids = df['patient_id'].values
    sched_dates = df['scheduled_day'].dt.normalize().values
    appt_dates = df['appointment_day'].dt.normalize().values
    noshows = df['no_show'].values
    appt_ids = df['appointment_id'].values

    from collections import defaultdict
    patient_appts = defaultdict(list)
    for idx in range(total_len):
        patient_appts[pids[idx]].append((
            appt_dates[idx],      # date of visit
            noshows[idx],         # outcome (0 or 1)
            sched_dates[idx],     # when this visit was booked
            appt_ids[idx],        # appointment id
            idx                   # original row index
        ))

    prior_appts_arr = np.zeros(total_len, dtype=int)
    prior_noshows_arr = np.zeros(total_len, dtype=int)
    days_since_last_arr = np.full(total_len, -1, dtype=int)

    for pid, visits in patient_appts.items():
        if len(visits) <= 1:
            continue
        
        for v in visits:
            v_appt_date, v_no_show, v_sched_date, v_appt_id, v_idx = v
            
            # Completed past visits strictly prior to this booking calendar date
            past_completed = [w for w in visits if w[0] < v_sched_date]
            
            if past_completed:
                n_completed = len(past_completed)
                n_noshows = sum(w[1] for w in past_completed)
                last_completed_date = max(w[0] for w in past_completed)
                days_gap = int((v_appt_date - last_completed_date) / np.timedelta64(1, 'D'))
                
                prior_appts_arr[v_idx] = n_completed
                prior_noshows_arr[v_idx] = n_noshows
                days_since_last_arr[v_idx] = days_gap

    prior_noshow_rate_arr = np.zeros(total_len, dtype=float)
    mask_has_prior = prior_appts_arr > 0
    prior_noshow_rate_arr[mask_has_prior] = np.round(
        prior_noshows_arr[mask_has_prior] / prior_appts_arr[mask_has_prior], 4
    )
    is_first_appt_arr = (prior_appts_arr == 0).astype(int)

    return prior_appts_arr, prior_noshows_arr, prior_noshow_rate_arr, days_since_last_arr, is_first_appt_arr

def engineer_features(input_path=INPUT_DATA_PATH):
    print("=" * 75)
    print("STEP 3: FEATURE ENGINEERING PIPELINE (POINT-IN-TIME LEAKAGE-SAFE)")
    print("=" * 75)
    print(f"Loading cleaned dataset from: {input_path}")
    t0 = time.time()
    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"Loaded {initial_count:,} appointments.")
    
    assert set(df['no_show'].unique()) <= {0, 1}, "Target variable no_show must be binary 0 or 1!"
    
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
    df['appointment_day'] = pd.to_datetime(df['appointment_day'])
    
    # ---------------------------------------------------------
    # 1. CORE STATIC CLINICAL & DEMOGRAPHIC FEATURES
    # (sms_received excluded from predictors: future info at Day 0)
    # ---------------------------------------------------------
    print("\n--- Engineering Core Static Features ---")
    df['gender'] = (df['gender'].astype(str).str.upper() == 'F').astype(int)
    df['has_handicap'] = (df['handicap'] > 0).astype(int)
    for col in ['scholarship', 'hypertension', 'diabetes', 'alcoholism']:
        df[col] = df[col].astype(int)
        
    print("Static features verified: age, gender, scholarship, hypertension, diabetes, alcoholism, has_handicap")

    # ---------------------------------------------------------
    # 2. CORE TEMPORAL & OPERATIONAL FEATURES
    # ---------------------------------------------------------
    print("\n--- Engineering Core Temporal Features ---")
    df['lead_days'] = (df['appointment_day'].dt.normalize() - df['scheduled_day'].dt.normalize()).dt.days
    df['same_day_booking'] = (df['lead_days'] == 0).astype(int)
    df['appointment_dow'] = df['appointment_day'].dt.dayofweek
    df['scheduled_dow'] = df['scheduled_day'].dt.dayofweek
    df['appointment_month'] = df['appointment_day'].dt.month
    print("Temporal features created: lead_days, same_day_booking, appointment_dow, scheduled_dow, appointment_month")

    # ---------------------------------------------------------
    # 3. CORE PATIENT HISTORY FEATURES (STRICT POINT-IN-TIME)
    # ---------------------------------------------------------
    print("\n--- Engineering Point-in-Time Patient History (Zero-Leakage) ---")
    p_appts, p_noshows, p_rate, d_last, is_first = compute_leakage_safe_history(df)
    df['prior_appointments'] = p_appts
    df['prior_noshows'] = p_noshows
    df['prior_noshow_rate'] = p_rate
    df['days_since_last_appointment'] = d_last
    df['is_first_appointment'] = is_first
    
    print(f"Patient history features created:")
    print(f"  First-time visits (0 completed past): {df['is_first_appointment'].sum():,} ({df['is_first_appointment'].mean()*100:.2f}%)")
    print(f"  Returning visits (>=1 completed past): {(df['is_first_appointment'] == 0).sum():,} ({(df['is_first_appointment'] == 0).mean()*100:.2f}%)")
    print(f"  Mean prior no-show rate (returning):  {df[df['is_first_appointment'] == 0]['prior_noshow_rate'].mean():.4f}")

    # ---------------------------------------------------------
    # 4. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
    # ---------------------------------------------------------
    print("\n--- Performing Chronological 3-Way Split (Train / Val / Test) ---")
    train_mask = df['appointment_day'] < pd.Timestamp(TRAIN_VAL_CUTOFF, tz=df['appointment_day'].dt.tz)
    val_mask = (df['appointment_day'] >= pd.Timestamp(TRAIN_VAL_CUTOFF, tz=df['appointment_day'].dt.tz)) & \
               (df['appointment_day'] < pd.Timestamp(VAL_TEST_CUTOFF, tz=df['appointment_day'].dt.tz))
    dev_mask = df['appointment_day'] < pd.Timestamp(VAL_TEST_CUTOFF, tz=df['appointment_day'].dt.tz)
    test_mask = df['appointment_day'] >= pd.Timestamp(VAL_TEST_CUTOFF, tz=df['appointment_day'].dt.tz)
    
    df_train = df[train_mask].copy().sort_values(by=['appointment_day', 'scheduled_day']).reset_index(drop=True)
    df_val = df[val_mask].copy().sort_values(by=['appointment_day', 'scheduled_day']).reset_index(drop=True)
    df_dev = df[dev_mask].copy().sort_values(by=['appointment_day', 'scheduled_day']).reset_index(drop=True)
    df_test = df[test_mask].copy().sort_values(by=['appointment_day', 'scheduled_day']).reset_index(drop=True)
    
    print(f"Split Partitions:")
    print(f"  1. Train (Early Dev): {len(df_train):,} rows ({len(df_train)/len(df)*100:.1f}%) | "
          f"Dates: {df_train['appointment_day'].min().strftime('%Y-%m-%d')} to {df_train['appointment_day'].max().strftime('%Y-%m-%d')} | "
          f"No-shows: {df_train['no_show'].sum():,} ({df_train['no_show'].mean()*100:.2f}%)")
    print(f"  2. Val   (Late Dev):  {len(df_val):,} rows ({len(df_val)/len(df)*100:.1f}%) | "
          f"Dates: {df_val['appointment_day'].min().strftime('%Y-%m-%d')} to {df_val['appointment_day'].max().strftime('%Y-%m-%d')} | "
          f"No-shows: {df_val['no_show'].sum():,} ({df_val['no_show'].mean()*100:.2f}%)")
    print(f"  -- Dev   (Full Dev):  {len(df_dev):,} rows ({len(df_dev)/len(df)*100:.1f}%) | "
          f"Dates: {df_dev['appointment_day'].min().strftime('%Y-%m-%d')} to {df_dev['appointment_day'].max().strftime('%Y-%m-%d')} | "
          f"No-shows: {df_dev['no_show'].sum():,} ({df_dev['no_show'].mean()*100:.2f}%)")
    print(f"  3. Test  (Final/Expl):{len(df_test):,} rows ({len(df_test)/len(df)*100:.1f}%) | "
          f"Dates: {df_test['appointment_day'].min().strftime('%Y-%m-%d')} to {df_test['appointment_day'].max().strftime('%Y-%m-%d')} | "
          f"No-shows: {df_test['no_show'].sum():,} ({df_test['no_show'].mean()*100:.2f}%)")

    # ---------------------------------------------------------
    # 5. INTEGRITY CHECKS & LEAKAGE AUDIT ASSERTIONS
    # ---------------------------------------------------------
    print("\n--- Running Automated Integrity & Anti-Leakage Audits ---")
    assert len(df_train) + len(df_val) + len(df_test) == len(df), "Row count mismatch in 3-way split!"
    assert len(df_dev) == len(df_train) + len(df_val), "Dev set row count mismatch!"
    assert df_train['appointment_day'].max() < df_val['appointment_day'].min(), "Temporal overlap between Train and Val!"
    assert df_val['appointment_day'].max() < df_test['appointment_day'].min(), "Temporal overlap between Val and Test!"
    assert (df['prior_noshows'] <= df['prior_appointments']).all(), "Logical error: prior_noshows > prior_appointments!"
    assert ((df['prior_noshow_rate'] >= 0.0) & (df['prior_noshow_rate'] <= 1.0)).all(), "Invalid prior_noshow_rate range!"
    assert (df['days_since_last_appointment'] >= -1).all(), "Invalid days_since_last_appointment!"
    assert (df['lead_days'] >= 0).all(), "Negative lead time detected!"
    print("All 8 integrity assertions PASSED successfully!")

    # ---------------------------------------------------------
    # 6. SAVE OUTPUT DATASETS & FEATURE CATALOG
    # ---------------------------------------------------------
    feature_cols = [
        'age', 'gender', 'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'has_handicap',
        'lead_days', 'same_day_booking', 'appointment_dow', 'scheduled_dow', 'appointment_month',
        'prior_appointments', 'prior_noshows', 'prior_noshow_rate', 'days_since_last_appointment', 'is_first_appointment'
    ]
    meta_cols = ['appointment_id', 'patient_id', 'scheduled_day', 'appointment_day', 'sms_received', 'no_show']
    ordered_cols = meta_cols + feature_cols
    
    df_train_out = df_train[ordered_cols]
    df_val_out = df_val[ordered_cols]
    df_dev_out = df_dev[ordered_cols]
    df_test_out = df_test[ordered_cols]
    
    os.makedirs(os.path.dirname(TRAIN_OUTPUT_PATH), exist_ok=True)
    df_train_out.to_csv(TRAIN_OUTPUT_PATH, index=False)
    df_val_out.to_csv(VAL_OUTPUT_PATH, index=False)
    df_dev_out.to_csv(DEV_OUTPUT_PATH, index=False)
    df_test_out.to_csv(TEST_OUTPUT_PATH, index=False)
    
    print(f"\nSaved Train features ({len(df_train_out):,} rows) to: {TRAIN_OUTPUT_PATH}")
    print(f"Saved Val features   ({len(df_val_out):,} rows) to: {VAL_OUTPUT_PATH}")
    print(f"Saved Dev features   ({len(df_dev_out):,} rows) to: {DEV_OUTPUT_PATH}")
    print(f"Saved Test features  ({len(df_test_out):,} rows) to: {TEST_OUTPUT_PATH}")
    
    feature_catalog = {
        "pipeline_version": "1.2",
        "prediction_point": "At scheduling time (Day 0) - strictly past-completed appointments only",
        "split_architecture": {
            "train": {
                "date_range": [
                    df_train['appointment_day'].min().strftime('%Y-%m-%d'),
                    df_train['appointment_day'].max().strftime('%Y-%m-%d')
                ],
                "rows": len(df_train),
                "noshow_rate": round(float(df_train['no_show'].mean()), 4),
                "positive_count": int(df_train['no_show'].sum()),
                "negative_count": int((df_train['no_show'] == 0).sum()),
                "purpose": "Model fitting during exploration and tuning"
            },
            "validation": {
                "date_range": [
                    df_val['appointment_day'].min().strftime('%Y-%m-%d'),
                    df_val['appointment_day'].max().strftime('%Y-%m-%d')
                ],
                "rows": len(df_val),
                "noshow_rate": round(float(df_val['no_show'].mean()), 4),
                "positive_count": int(df_val['no_show'].sum()),
                "negative_count": int((df_val['no_show'] == 0).sum()),
                "purpose": "Model comparison, hyperparameter selection, threshold optimization, probability calibration"
            },
            "full_dev": {
                "date_range": [
                    df_dev['appointment_day'].min().strftime('%Y-%m-%d'),
                    df_dev['appointment_day'].max().strftime('%Y-%m-%d')
                ],
                "rows": len(df_dev),
                "noshow_rate": round(float(df_dev['no_show'].mean()), 4),
                "positive_count": int(df_dev['no_show'].sum()),
                "negative_count": int((df_dev['no_show'] == 0).sum()),
                "purpose": "Full April-May development set to refit chosen model before final test evaluation"
            },
            "test": {
                "date_range": [
                    df_test['appointment_day'].min().strftime('%Y-%m-%d'),
                    df_test['appointment_day'].max().strftime('%Y-%m-%d')
                ],
                "rows": len(df_test),
                "noshow_rate": round(float(df_test['no_show'].mean()), 4),
                "positive_count": int(df_test['no_show'].sum()),
                "negative_count": int((df_test['no_show'] == 0).sum()),
                "purpose": "Final evaluation set (June 2016; treated as exploratory test set because initially used in early benchmarks)"
            }
        },
        "target": {
            "name": "no_show",
            "type": "binary",
            "values": {"0": "Show (Attended)", "1": "No-Show (Missed)"}
        },
        "features": {
            "explicit_list": feature_cols,
            "core_static": [
                {"name": "age", "type": "continuous", "description": "Patient age in years"},
                {"name": "gender", "type": "binary", "description": "1 = Female, 0 = Male"},
                {"name": "scholarship", "type": "binary", "description": "Welfare assistance enrollment (Bolsa Familia)"},
                {"name": "hypertension", "type": "binary", "description": "Diagnosed hypertension"},
                {"name": "diabetes", "type": "binary", "description": "Diagnosed diabetes"},
                {"name": "alcoholism", "type": "binary", "description": "Documented alcoholism"},
                {"name": "has_handicap", "type": "binary", "description": "Physical disability indicator (handicap > 0)"}
            ],
            "core_temporal": [
                {"name": "lead_days", "type": "continuous", "description": "Waiting time in calendar days from booking to visit"},
                {"name": "same_day_booking", "type": "candidate_binary", "description": "Derived from lead_days == 0 (tested via ablation)"},
                {"name": "appointment_dow", "type": "categorical_nominal", "description": "Day of week of appointment (0=Mon..5=Sat)"},
                {"name": "scheduled_dow", "type": "categorical_nominal", "description": "Day of week booking was scheduled (0=Mon..5=Sat)"},
                {"name": "appointment_month", "type": "categorical_nominal", "description": "Calendar month of appointment (4=Apr, 5=May, 6=Jun)"}
            ],
            "core_patient_history": [
                {"name": "prior_appointments", "type": "discrete_count", "description": "Number of strictly prior completed appointments (appointment_date < scheduled_date)"},
                {"name": "prior_noshows", "type": "discrete_count", "description": "Number of strictly prior completed missed visits"},
                {"name": "prior_noshow_rate", "type": "continuous_ratio", "description": "Ratio of prior missed visits (0.0 for first-time visitors)"},
                {"name": "days_since_last_appointment", "type": "discrete_count", "description": "Days since previous completed visit (-1 for first-time visitors)"},
                {"name": "is_first_appointment", "type": "binary", "description": "1 if no prior completed visits, 0 if returning"}
            ],
            "excluded_from_predictors": [
                {"name": "sms_received", "reason": "Future operational info at Day 0 booking; preserved in metadata for intervention analysis only."},
                {"name": "patient_id", "reason": "Identifier only."},
                {"name": "appointment_id", "reason": "Arbitrary row identifier."}
            ]
        }
    }
    
    with open(METADATA_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(feature_catalog, f, indent=2)
    print(f"Saved Feature Catalog metadata to: {METADATA_OUTPUT_PATH}")
    print(f"\nSTEP 3 FEATURE ENGINEERING COMPLETE in {time.time() - t0:.2f}s.")
    return df_train_out, df_val_out, df_dev_out, df_test_out

if __name__ == '__main__':
    engineer_features()
