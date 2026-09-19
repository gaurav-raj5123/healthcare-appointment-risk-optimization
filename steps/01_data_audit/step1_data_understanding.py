import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'KaggleV2-May-2016.csv')

def main():
    print(f"Loading {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH)
    
    print("\n--- 1. BASIC SHAPE & INFO ---")
    print(f"Total Records (Appointments): {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print("Columns:", list(df.columns))
    print("\nData Types & Non-Null Counts:")
    print(df.info())
    
    print("\n--- 2. FIRST 5 ROWS ---")
    print(df.head())
    
    print("\n--- 3. UNIQUE PATIENTS & APPOINTMENTS ---")
    n_unique_patients = df['PatientId'].nunique()
    n_unique_appts = df['AppointmentID'].nunique()
    print(f"Unique Patients: {n_unique_patients:,}")
    print(f"Unique Appointments: {n_unique_appts:,}")
    print(f"Average appointments per patient: {len(df)/n_unique_patients:.2f}")
    
    print("\n--- 4. MISSING VALUES & DUPLICATES ---")
    missing = df.isnull().sum()
    print("Missing values per column:\n", missing[missing > 0] if missing.sum() > 0 else "Zero missing values across all columns!")
    print(f"Exact Duplicate Rows: {df.duplicated().sum()}")
    print(f"Duplicate AppointmentID: {df.duplicated(subset=['AppointmentID']).sum()}")
    
    print("\n--- 5. TARGET VARIABLE (No-show) DISTRIBUTION ---")
    target_counts = df['No-show'].value_counts()
    target_pct = df['No-show'].value_counts(normalize=True) * 100
    for val in target_counts.index:
        print(f"  {val}: {target_counts[val]:,} ({target_pct[val]:.2f}%)")
    
    print("\n--- 6. NUMERICAL FEATURE DISTRIBUTIONS & ANOMALIES ---")
    print("Age Summary:")
    print(df['Age'].describe())
    
    neg_age = df[df['Age'] < 0]
    print(f"\nRecords with Age < 0: {len(neg_age)}")
    if len(neg_age) > 0:
        print(neg_age[['PatientId', 'Age', 'ScheduledDay', 'AppointmentDay', 'No-show']])
        
    old_age = df[df['Age'] > 100]
    print(f"Records with Age > 100: {len(old_age)}")
    
    print("\n--- 7. BINARY / CATEGORICAL DISTRIBUTIONS ---")
    for col in ['Gender', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']:
        print(f"\nValue counts for '{col}':")
        print(df[col].value_counts().to_dict())
        
    print("\n--- 8. DATE & TIME ANALYSIS (LEAD TIME SANITY CHECK) ---")
    df['ScheduledDay_dt'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay_dt'] = pd.to_datetime(df['AppointmentDay'])
    
    print(f"ScheduledDay Range: {df['ScheduledDay_dt'].min()} to {df['ScheduledDay_dt'].max()}")
    print(f"AppointmentDay Range: {df['AppointmentDay_dt'].min()} to {df['AppointmentDay_dt'].max()}")
    
    lead_days = (df['AppointmentDay_dt'].dt.normalize() - df['ScheduledDay_dt'].dt.normalize()).dt.days
    print(f"\nLead time (Appointment Day - Scheduled Day in days):")
    print(lead_days.describe())
    
    neg_lead = (lead_days < 0).sum()
    print(f"Records where AppointmentDay < ScheduledDay (Data error): {neg_lead}")
    same_day = (lead_days == 0).sum()
    print(f"Same-day appointments (Scheduled & attended same day): {same_day:,} ({same_day/len(df)*100:.2f}%)")

if __name__ == '__main__':
    main()
