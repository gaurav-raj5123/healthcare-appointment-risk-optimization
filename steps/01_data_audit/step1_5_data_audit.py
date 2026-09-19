import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'KaggleV2-May-2016.csv')

def run_step_1_5_audit():
    print(f"Loading {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH)
    
    df['ScheduledDay_dt'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay_dt'] = pd.to_datetime(df['AppointmentDay'])
    df['lead_days'] = (df['AppointmentDay_dt'].dt.normalize() - df['ScheduledDay_dt'].dt.normalize()).dt.days

    print("=" * 60)
    print("STEP 1.5 — RIGOROUS DATA AUDIT REPORT")
    print("=" * 60)

    print("\n# 1. Exact dataset shape:")
    print(df.shape)

    print("\n# 2. Unique patients:")
    print(df['PatientId'].nunique())

    print("\n# 3. Appointments per patient summary stats:")
    print(df.groupby('PatientId').size().describe())

    print("\n# 4. Negative ages:")
    neg_ages = df[df['Age'] < 0]
    print(neg_ages[['PatientId', 'Age', 'ScheduledDay', 'AppointmentDay', 'No-show']])

    print("\n# 5. Negative lead times (appointments booked after the appointment happened):")
    neg_leads = df[df['lead_days'] < 0][['PatientId', 'ScheduledDay', 'AppointmentDay', 'lead_days', 'No-show']]
    print(neg_leads)

    print("\n# 6. Lead-time distribution:")
    print(df['lead_days'].describe())

    print("\n# 7. Target distribution:")
    print(df['No-show'].value_counts(normalize=True))
    print(df['No-show'].value_counts())

    print("\n# 8. Duplicate appointments (AppointmentID):")
    print(f"Duplicates: {df['AppointmentID'].duplicated().sum()}")

    print("\n# 9. Duplicate patient + appointment combinations:")
    print(f"Duplicates: {df.duplicated(subset=['PatientId', 'AppointmentID']).sum()}")

    print("\n# 10. Distribution of appointments per patient (counts):")
    appt_counts = df.groupby('PatientId').size().value_counts().sort_index()
    print("First 15 counts (1 to 15 appointments):")
    print(appt_counts.head(15))
    print("\nTop 5 extreme repeat patients:")
    print(appt_counts.tail(5))
    
    single_appt_patients = (appt_counts.get(1, 0))
    total_patients = df['PatientId'].nunique()
    repeat_patients = total_patients - single_appt_patients
    repeat_appts = df.shape[0] - single_appt_patients
    
    print("\n--- REPEAT PATIENT ANALYSIS & DATA LEAKAGE IMPLICATIONS ---")
    print(f"Patients with exactly 1 appointment: {single_appt_patients:,} ({single_appt_patients / total_patients * 100:.2f}%)")
    print(f"Patients with 2+ appointments: {repeat_patients:,} ({repeat_patients / total_patients * 100:.2f}%)")
    print(f"Appointments belonging to repeat patients: {repeat_appts:,} ({repeat_appts / df.shape[0] * 100:.2f}%)")
    
    print("\n--- TIMELINE & TEMPORAL SPLIT AUDIT ---")
    print(f"ScheduledDay range:   {df['ScheduledDay_dt'].min()} to {df['ScheduledDay_dt'].max()}")
    print(f"AppointmentDay range: {df['AppointmentDay_dt'].min()} to {df['AppointmentDay_dt'].max()}")
    print(f"Total calendar span of AppointmentDay: {(df['AppointmentDay_dt'].max() - df['AppointmentDay_dt'].min()).days} days")

if __name__ == '__main__':
    run_step_1_5_audit()
