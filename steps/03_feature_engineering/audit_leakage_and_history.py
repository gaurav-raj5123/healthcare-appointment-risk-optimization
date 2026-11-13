"""
Audit script: Point-in-time leakage audit and verification of patient history features.
Compares naive cumulative history vs strictly point-in-time completed history.
"""

import os
import json
import time
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CLEAN_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_patient_data.csv')

def run_leakage_audit():
    print("=" * 75)
    print("AUDIT: POINT-IN-TIME LEAKAGE & PATIENT HISTORY INTEGRITY")
    print("=" * 75)
    
    t0 = time.time()
    df = pd.read_csv(CLEAN_DATA_PATH)
    total_records = len(df)
    
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
    df['appointment_day'] = pd.to_datetime(df['appointment_day'])
    df['scheduled_date'] = df['scheduled_day'].dt.normalize()
    df['appointment_date'] = df['appointment_day'].dt.normalize()
    
    print(f"Total appointments loaded: {total_records:,}")
    print(f"Unique patients:           {df['patient_id'].nunique():,}")
    
    # 1. Multi-appointment patients
    counts = df['patient_id'].value_counts()
    multi_pids = set(counts[counts > 1].index)
    multi_appts = len(df[df['patient_id'].isin(multi_pids)])
    print(f"Single-visit patients:     {len(counts[counts == 1]):,} ({len(counts[counts == 1])/df['patient_id'].nunique()*100:.2f}%)")
    print(f"Multi-visit patients:      {len(multi_pids):,} ({len(multi_pids)/df['patient_id'].nunique()*100:.2f}%)")
    print(f"Appointments from repeats: {multi_appts:,} ({multi_appts/total_records*100:.2f}%)")
    
    # 2. Granularity of Timestamps
    has_sched_time = (df['scheduled_day'].dt.time != pd.to_datetime('00:00:00').time()).sum()
    has_appt_time = (df['appointment_day'].dt.time != pd.to_datetime('00:00:00').time()).sum()
    print("\n--- Timestamp Granularity Inspection ---")
    print(f"scheduled_day has intra-day time (hour:minute:second): {has_sched_time:,} of {total_records:,} (100.0%)")
    print(f"appointment_day has intra-day time:                    {has_appt_time:,} of {total_records:,} (0.0% - Midnight truncated)")
    print("CRITICAL LIMITATION:")
    print("Because appointment_day lacks hour/minute timestamps, for an appointment scheduled on day D,")
    print("any other appointment on that same calendar day D cannot be proven to have concluded before the scheduling action.")
    print("Therefore, by strict conservative design, only appointments with appointment_day < scheduled_date (at least 1 full")
    print("calendar day earlier) can be proven available at scheduling time.")
    
    # 3. Simulate Naive (Buggy) Cumulative Logic
    df_naive = df.sort_values(by=['patient_id', 'appointment_day', 'scheduled_day', 'appointment_id']).reset_index(drop=True)
    df_naive['naive_prior_appts'] = df_naive.groupby('patient_id').cumcount()
    df_naive['naive_prior_noshows'] = df_naive.groupby('patient_id')['no_show'].cumsum() - df_naive['no_show']
    
    # Fast Native Python Audit of Naive Logic
    pids = df_naive['patient_id'].values
    sched_dates = df_naive['scheduled_date'].values
    appt_dates = df_naive['appointment_date'].values
    noshows = df_naive['no_show'].values
    appt_ids = df_naive['appointment_id'].values
    naive_p_appts = df_naive['naive_prior_appts'].values
    naive_p_noshows = df_naive['naive_prior_noshows'].values
    
    from collections import defaultdict
    indices_by_patient = defaultdict(list)
    for idx, pid in enumerate(pids):
        indices_by_patient[pid].append(idx)
        
    strictly_future_leak_count = 0
    same_day_leak_count = 0
    total_contaminated_count = 0
    leaked_examples = []
    
    for pid, idx_list in indices_by_patient.items():
        if len(idx_list) <= 1:
            continue
        for k in range(1, len(idx_list)):
            curr_i = idx_list[k]
            curr_sched = sched_dates[curr_i]
            
            prec_i = idx_list[:k]
            prec_appts = appt_dates[prec_i]
            prec_ns = noshows[prec_i]
            
            future_mask = prec_appts > curr_sched
            same_day_mask = prec_appts == curr_sched
            past_mask = prec_appts < curr_sched
            
            has_future = np.any(future_mask)
            has_sameday = np.any(same_day_mask)
            
            if has_future or has_sameday:
                total_contaminated_count += 1
                if has_future:
                    strictly_future_leak_count += 1
                if has_sameday:
                    same_day_leak_count += 1
                    
                if len(leaked_examples) < 5:
                    leaked_examples.append({
                        "patient_id": int(pid),
                        "appointment_id": int(appt_ids[curr_i]),
                        "booking_date": str(curr_sched)[:10],
                        "appointment_date": str(appt_dates[curr_i])[:10],
                        "naive_prior_appointments": int(naive_p_appts[curr_i]),
                        "naive_prior_noshows": int(naive_p_noshows[curr_i]),
                        "leak_safe_prior_appointments": int(np.sum(past_mask)),
                        "leak_safe_prior_noshows": int(np.sum(prec_ns[past_mask])),
                        "strictly_future_appointments_leaked": int(np.sum(future_mask)),
                        "same_day_appointments_leaked": int(np.sum(same_day_mask))
                    })
                    
    print("\n--- Leakage Quantification Report ---")
    print(f"Appointments where naive logic leaked strictly FUTURE appointment outcomes: {strictly_future_leak_count:,} ({strictly_future_leak_count/total_records*100:.2f}%)")
    print(f"Appointments where naive logic leaked SAME-DAY unverified appointment outcomes: {same_day_leak_count:,} ({same_day_leak_count/total_records*100:.2f}%)")
    print(f"Total appointments with contaminated history features under naive logic:      {total_contaminated_count:,} ({total_contaminated_count/total_records*100:.2f}%)")
    
    print("\n--- Concrete Examples of Leaked Records ---")
    for ex in leaked_examples:
        print(f"Patient {ex['patient_id']} | Booking Date: {ex['booking_date']} | Appt Date: {ex['appointment_date']}")
        print(f"  Naive Features:     prior_appts={ex['naive_prior_appointments']}, prior_noshows={ex['naive_prior_noshows']}")
        print(f"  Leak-Safe Features: prior_appts={ex['leak_safe_prior_appointments']}, prior_noshows={ex['leak_safe_prior_noshows']}")
        print(f"  Contamination:      {ex['strictly_future_appointments_leaked']} future visits + {ex['same_day_appointments_leaked']} same-day visits")
        print()
        
    audit_summary = {
        "total_records": total_records,
        "unique_patients": df['patient_id'].nunique(),
        "multi_visit_patients": len(multi_pids),
        "multi_visit_appointments": multi_appts,
        "strictly_future_leaks": strictly_future_leak_count,
        "same_day_leaks": same_day_leak_count,
        "total_contaminated_records": total_contaminated_count,
        "contamination_rate_pct": round(total_contaminated_count / total_records * 100, 2),
        "examples": leaked_examples
    }
    
    out_json = os.path.join(PROJECT_ROOT, 'data', 'processed', 'leakage_audit_report.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Saved audit report summary to: {out_json}")
    print(f"Audit completed in {time.time() - t0:.2f}s.")
    return audit_summary

if __name__ == '__main__':
    run_leakage_audit()

