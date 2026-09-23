import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, 'data', 'raw', 'KaggleV2-May-2016.csv')
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_patient_data.csv')

def clean_data(input_csv=DEFAULT_INPUT, output_csv=DEFAULT_OUTPUT):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    initial_len = len(df)
    
    # 1. Rename columns to fix typos and standardize
    rename_dict = {
        'PatientId': 'patient_id',
        'AppointmentID': 'appointment_id',
        'Gender': 'gender',
        'ScheduledDay': 'scheduled_day',
        'AppointmentDay': 'appointment_day',
        'Age': 'age',
        'Neighbourhood': 'neighbourhood',
        'Scholarship': 'scholarship',
        'Hipertension': 'hypertension',
        'Diabetes': 'diabetes',
        'Alcoholism': 'alcoholism',
        'Handcap': 'handicap',
        'SMS_received': 'sms_received',
        'No-show': 'no_show'
    }
    df = df.rename(columns=rename_dict)
    
    # 2. Parse Datetimes
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
    df['appointment_day'] = pd.to_datetime(df['appointment_day'])
    
    # Format patient_id as int64 (avoid scientific notation float)
    df['patient_id'] = df['patient_id'].astype(np.int64)
    
    # 3. Calculate waiting time / lead time in days
    df['lead_days'] = (df['appointment_day'].dt.normalize() - df['scheduled_day'].dt.normalize()).dt.days
    
    # 4. Filter anomalies
    age_valid = df['age'] >= 0
    lead_valid = df['lead_days'] >= 0
    
    clean_mask = age_valid & lead_valid
    dropped_count = initial_len - clean_mask.sum()
    print(f"Dropped {dropped_count} invalid records (Age < 0: {(~age_valid).sum()}, Lead Days < 0: {(~lead_valid).sum()})")
    
    df_clean = df[clean_mask].copy()
    
    # 5. Standardize Target Variable: 1 = No-show (did not attend), 0 = Show (attended)
    df_clean['no_show'] = (df_clean['no_show'].str.strip().str.lower() == 'yes').astype(int)
    assert set(df_clean['no_show'].unique()) <= {0, 1}, "Target variable contains values other than 0 and 1!"
    
    # Quick sanity verification
    print("\n--- Cleaned Dataset Summary ---")
    print(f"Final records: {len(df_clean):,} (retained {len(df_clean)/initial_len*100:.4f}%)")
    print(f"Unique patients: {df_clean['patient_id'].nunique():,}")
    print(f"No-Show rate: {df_clean['no_show'].mean()*100:.2f}% ({df_clean['no_show'].sum():,} missed appointments)")
    print(f"Lead time range: {df_clean['lead_days'].min()} to {df_clean['lead_days'].max()} days")
    print(f"Age range: {df_clean['age'].min()} to {df_clean['age'].max()} years")
    
    # Ensure output dir exists and save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_clean.to_csv(output_csv, index=False)
    print(f"\nSaved cleaned dataset to: {output_csv}")
    return df_clean

if __name__ == '__main__':
    clean_data()
