import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CLEAN_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_patient_data.csv')

def main():
    print(f"Loading {CLEAN_PATH}...")
    df = pd.read_csv(CLEAN_PATH)

    print("--- SMS_received Basic Stats ---")
    print(df['sms_received'].value_counts(dropna=False))
    print(df['sms_received'].value_counts(normalize=True))

    print("\n--- SMS_received by Lead Days (Waiting Time) ---")
    same_day = df[df['lead_days'] == 0]
    print(f"Total Same-Day appointments (lead_days = 0): {len(same_day):,}")
    print("SMS_received on same-day appointments:")
    print(same_day['sms_received'].value_counts())

    one_day = df[df['lead_days'] == 1]
    print(f"\nTotal 1-day lead appointments: {len(one_day):,}")
    print("SMS_received on 1-day lead appointments:")
    print(one_day['sms_received'].value_counts())

    two_days = df[df['lead_days'] == 2]
    print(f"\nTotal 2-day lead appointments: {len(two_days):,}")
    print("SMS_received on 2-day lead appointments:")
    print(two_days['sms_received'].value_counts())

    bins = [-1, 0, 1, 2, 3, 7, 14, 30, 60, 180]
    labels = ['0 days (Same day)', '1 day', '2 days', '3 days', '4-7 days', '8-14 days', '15-30 days', '31-60 days', '61+ days']
    df['lead_bin'] = pd.cut(df['lead_days'], bins=bins, labels=labels)

    sms_by_bin = df.groupby('lead_bin', observed=False).agg(
        total_appts=('appointment_id', 'count'),
        sms_sent=('sms_received', 'sum'),
        sms_rate=('sms_received', 'mean'),
        noshow_rate=('no_show', 'mean')
    )
    sms_by_bin['sms_rate_pct'] = (sms_by_bin['sms_rate'] * 100).round(2)
    sms_by_bin['noshow_rate_pct'] = (sms_by_bin['noshow_rate'] * 100).round(2)
    print("\n--- SMS Rate and No-Show Rate across Lead Time Bins ---")
    print(sms_by_bin[['total_appts', 'sms_sent', 'sms_rate_pct', 'noshow_rate_pct']])

    c = df.groupby(['lead_bin', 'sms_received'], observed=False)['no_show'].agg(['count', 'mean']).unstack()
    c.columns = ['No_SMS_N', 'SMS_N', 'No_SMS_NoShow_%', 'SMS_NoShow_%']
    c['No_SMS_NoShow_%'] = (c['No_SMS_NoShow_%']*100).round(2)
    c['SMS_NoShow_%'] = (c['SMS_NoShow_%']*100).round(2)
    c['Reduction (% points)'] = (c['No_SMS_NoShow_%'] - c['SMS_NoShow_%']).round(2)
    print("\n--- Controlled SMS Impact Table ---")
    print(c.to_string())

if __name__ == '__main__':
    main()
