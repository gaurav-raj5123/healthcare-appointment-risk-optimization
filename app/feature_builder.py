"""
CarePredict - Feature Builder & Validator
Validates inputs and derives point-in-time features adhering to the training schema.
"""

from datetime import date, datetime
import pandas as pd
import numpy as np
from app.config import APPROVED_FEATURES, TEMPLATE_PATH

def derive_individual_features(
    age: int,
    gender_str: str,
    scholarship: bool,
    hypertension: bool,
    diabetes: bool,
    alcoholism: bool,
    handicap_level: int,
    scheduled_dt: date,
    appointment_dt: date,
    is_first_time: bool,
    prior_appts: int = 0,
    prior_missed: int = 0,
    days_since_last: int = -1
) -> dict:
    """
    Constructs and derives the exact 17 features required by the trained model.
    Enforces strict consistency rules matching the training pipeline:
      - Historical visits strictly prior to scheduling: historical_date < scheduled_date
      - Zero current-outcome leakage
      - First-time visitor conventions: prior_appts=0, prior_noshows=0, prior_rate=0.0, days_gap=-1, is_first=1
      - Returning visitor conventions: prior_appts>0, 0 <= prior_noshows <= prior_appts, days_gap >= 0, is_first=0
    """
    # 1. Validation checks
    if appointment_dt < scheduled_dt:
        raise ValueError(
            f"Appointment date ({appointment_dt}) cannot precede the scheduling date ({scheduled_dt})."
        )
    
    if age < 0 or age > 115:
        raise ValueError(f"Age must be between 0 and 115 years (provided: {age}).")

    if prior_appts < 0:
        raise ValueError(f"Prior appointments count cannot be negative (provided: {prior_appts}).")

    if prior_missed < 0:
        raise ValueError(f"Prior no-shows count cannot be negative (provided: {prior_missed}).")

    if prior_missed > prior_appts:
        raise ValueError(
            f"Prior no-shows ({prior_missed}) cannot exceed total prior appointments ({prior_appts})."
        )

    # 2. Temporal derivations
    lead_days = (appointment_dt - scheduled_dt).days
    same_day_booking = 1 if lead_days == 0 else 0
    appointment_dow = appointment_dt.weekday()  # 0=Monday..6=Sunday
    scheduled_dow = scheduled_dt.weekday()
    appointment_month = appointment_dt.month

    # 3. Static demographics & comorbidities
    gender_bin = 1 if str(gender_str).strip().upper() in ('F', 'FEMALE', '1') else 0
    has_handicap = 1 if int(handicap_level) > 0 else 0

    # 4. Patient history logic (Strict Point-in-Time Parity with Step 3)
    if is_first_time and prior_appts > 0:
        raise ValueError(
            f"First-time patient indicator is True, but prior appointments count is {prior_appts}. "
            "First-time patients must have 0 prior appointments."
        )

    if is_first_time or prior_appts == 0:
        p_appts = 0
        p_noshows = 0
        p_rate = 0.0
        days_last = -1  # Step 3 frozen convention for first-time visitors
        first_flag = 1
    else:
        p_appts = int(prior_appts)
        p_noshows = int(prior_missed)
        p_rate = round(float(p_noshows / p_appts), 4)
        if days_since_last < 0:
            raise ValueError(
                f"Days since last appointment ({days_since_last}) must be non-negative for returning patients. "
                "The -1 convention is strictly reserved for first-time visitors with no prior visits."
            )
        days_last = int(days_since_last)
        first_flag = 0

    feature_dict = {
        "age": int(age),
        "gender": int(gender_bin),
        "scholarship": int(bool(scholarship)),
        "hypertension": int(bool(hypertension)),
        "diabetes": int(bool(diabetes)),
        "alcoholism": int(bool(alcoholism)),
        "has_handicap": int(has_handicap),
        "lead_days": int(lead_days),
        "same_day_booking": int(same_day_booking),
        "appointment_dow": int(appointment_dow),
        "scheduled_dow": int(scheduled_dow),
        "appointment_month": int(appointment_month),
        "prior_appointments": int(p_appts),
        "prior_noshows": int(p_noshows),
        "prior_noshow_rate": float(p_rate),
        "days_since_last_appointment": int(days_last),
        "is_first_appointment": int(first_flag)
    }

    # Verify that all 17 approved features are present
    assert all(k in feature_dict for k in APPROVED_FEATURES), "Missing approved feature in derivation!"
    return feature_dict

def validate_and_prepare_batch_csv(df_upload: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Validates a bulk uploaded CSV against expected schema.
    Returns:
      - valid_df: Ready for model prediction (with approved features).
      - invalid_df: Rejected records with error reasons.
      - warnings: General processing warnings.
    """
    df = df_upload.copy()
    errors = []
    warnings = []

    # Check if raw appointment format or pre-engineered feature format
    raw_required = ['scheduled_day', 'appointment_day', 'age', 'gender']
    feat_required = APPROVED_FEATURES

    has_raw = all(c in df.columns for c in raw_required)
    has_features = all(c in df.columns for c in feat_required)

    if not has_raw and not has_features:
        missing = [c for c in raw_required if c not in df.columns]
        raise ValueError(f"Uploaded CSV is missing mandatory columns: {missing}. Please use the template.")

    rejected_records = []
    valid_records = []

    if has_features:
        # Pre-engineered format: validate bounds and schema integrity
        for idx, row in df.iterrows():
            row_errors = []
            if pd.isnull(row['age']) or row['age'] < 0 or row['age'] > 115:
                row_errors.append(f"Invalid age ({row.get('age')}); must be 0-115")
            if pd.isnull(row['lead_days']) or row['lead_days'] < 0:
                row_errors.append(f"Invalid lead_days ({row.get('lead_days')}); cannot be negative")
            p_appts = row.get('prior_appointments', 0)
            p_noshows = row.get('prior_noshows', 0)
            if p_appts < 0:
                row_errors.append(f"prior_appointments cannot be negative ({p_appts})")
            if p_noshows < 0:
                row_errors.append(f"prior_noshows cannot be negative ({p_noshows})")
            if p_noshows > p_appts:
                row_errors.append(f"prior_noshows ({p_noshows}) > prior_appointments ({p_appts})")
            
            row_dict = row.to_dict()
            if row_errors:
                row_dict['_rejection_reason'] = "; ".join(row_errors)
                rejected_records.append(row_dict)
            else:
                valid_records.append(row_dict)

    else:
        # Raw format: derive features
        df['scheduled_day'] = pd.to_datetime(df['scheduled_day'], errors='coerce')
        df['appointment_day'] = pd.to_datetime(df['appointment_day'], errors='coerce')

        for idx, row in df.iterrows():
            row_errors = []
            row_dict = row.to_dict()

            if pd.isnull(row['scheduled_day']) or pd.isnull(row['appointment_day']):
                row_errors.append("Invalid or unparseable date format")
            elif row['appointment_day'].date() < row['scheduled_day'].date():
                row_errors.append(f"appointment_day ({row['appointment_day'].date()}) is prior to scheduled_day ({row['scheduled_day'].date()})")

            if pd.isnull(row['age']) or row['age'] < 0 or row['age'] > 115:
                row_errors.append(f"Invalid age ({row.get('age')}); must be 0-115")

            if row_errors:
                row_dict['_rejection_reason'] = "; ".join(row_errors)
                rejected_records.append(row_dict)
                continue

            try:
                # Fill missing optionals with defaults
                scholarship = bool(row.get('scholarship', 0))
                hypertension = bool(row.get('hypertension', 0))
                diabetes = bool(row.get('diabetes', 0))
                alcoholism = bool(row.get('alcoholism', 0))
                handicap_level = int(row.get('handicap', row.get('has_handicap', 0)))

                prior_appts = int(row.get('prior_appointments', 0))
                prior_noshows = int(row.get('prior_noshows', 0))
                is_first = True if prior_appts == 0 else False
                days_since_last = int(row.get('days_since_last_appointment', -1 if is_first else 30))

                feat_dict = derive_individual_features(
                    age=int(row['age']),
                    gender_str=str(row['gender']),
                    scholarship=scholarship,
                    hypertension=hypertension,
                    diabetes=diabetes,
                    alcoholism=alcoholism,
                    handicap_level=handicap_level,
                    scheduled_dt=row['scheduled_day'].date(),
                    appointment_dt=row['appointment_day'].date(),
                    is_first_time=is_first,
                    prior_appts=prior_appts,
                    prior_missed=prior_noshows,
                    days_since_last=days_since_last
                )
                # Preserve identifier columns if present
                for id_col in ['appointment_id', 'patient_id', 'patient_name', 'appointment_time']:
                    if id_col in row and not pd.isnull(row[id_col]):
                        feat_dict[id_col] = row[id_col]

                valid_records.append(feat_dict)

            except Exception as e:
                row_dict['_rejection_reason'] = str(e)
                rejected_records.append(row_dict)

    valid_df = pd.DataFrame(valid_records)
    invalid_df = pd.DataFrame(rejected_records)

    return valid_df, invalid_df, warnings

def generate_csv_template() -> pd.DataFrame:
    """Generates a sample DataFrame for downloadable template."""
    return pd.DataFrame([
        {
            "appointment_id": 1001,
            "patient_id": 5001,
            "scheduled_day": "2026-11-01",
            "appointment_day": "2026-11-15",
            "age": 34,
            "gender": "F",
            "scholarship": 0,
            "hypertension": 0,
            "diabetes": 0,
            "alcoholism": 0,
            "handicap": 0,
            "prior_appointments": 2,
            "prior_noshows": 0,
            "days_since_last_appointment": 45
        },
        {
            "appointment_id": 1002,
            "patient_id": 5002,
            "scheduled_day": "2026-11-10",
            "appointment_day": "2026-11-10",
            "age": 68,
            "gender": "M",
            "scholarship": 0,
            "hypertension": 1,
            "diabetes": 1,
            "alcoholism": 0,
            "handicap": 1,
            "prior_appointments": 8,
            "prior_noshows": 1,
            "days_since_last_appointment": 21
        },
        {
            "appointment_id": 1003,
            "patient_id": 5003,
            "scheduled_day": "2026-10-15",
            "appointment_day": "2026-11-12",
            "age": 19,
            "gender": "M",
            "scholarship": 1,
            "hypertension": 0,
            "diabetes": 0,
            "alcoholism": 0,
            "handicap": 0,
            "prior_appointments": 0,
            "prior_noshows": 0,
            "days_since_last_appointment": -1
        }
    ])

