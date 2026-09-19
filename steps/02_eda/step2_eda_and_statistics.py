import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

def p(text=""):
    print(text, flush=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CLEAN_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_patient_data.csv')

def run_step_2_analysis():
    p("=" * 70)
    p("STEP 2: COMPREHENSIVE EDA & STATISTICAL ASSOCIATION ANALYSIS")
    p("=" * 70)
    
    p(f"Loading cleaned dataset from {CLEAN_PATH}...")
    df = pd.read_csv(CLEAN_PATH)
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
    df['appointment_day'] = pd.to_datetime(df['appointment_day'])
    
    total_n = len(df)
    overall_noshow_rate = df['disengaged'].mean()
    p(f"Total Appointments: {total_n:,}")
    p(f"Overall Disengagement (No-Show) Rate: {overall_noshow_rate*100:.2f}%\n")
    
    # 1. DEMOGRAPHIC ANALYSIS: AGE & GENDER
    p("-" * 50)
    p("1. DEMOGRAPHIC FACTORS: AGE & GENDER")
    p("-" * 50)
    
    show_ages = df[df['disengaged'] == 0]['age']
    noshow_ages = df[df['disengaged'] == 1]['age']
    
    t_stat, t_pval = stats.ttest_ind(noshow_ages, show_ages, equal_var=False)
    
    p(f"Age (Show Mean +/- Std):      {show_ages.mean():.1f} +/- {show_ages.std():.1f} years (Median: {show_ages.median()})")
    p(f"Age (No-Show Mean +/- Std):   {noshow_ages.mean():.1f} +/- {noshow_ages.std():.1f} years (Median: {noshow_ages.median()})")
    p(f"Welch's t-test p-value:       {t_pval:.2e} (t-stat: {t_stat:.2f})")
    
    age_bins = [-1, 12, 19, 39, 59, 120]
    age_labels = ['Child (0-12)', 'Adolescent (13-19)', 'Young Adult (20-39)', 'Middle-Aged (40-59)', 'Senior (60+)']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    
    age_group_summary = df.groupby('age_group', observed=False)['disengaged'].agg(
        total_appts='count',
        noshow_count='sum',
        noshow_rate='mean'
    )
    age_group_summary['noshow_rate_pct'] = (age_group_summary['noshow_rate'] * 100).round(2)
    p("\nNo-Show Rate by Age Group:")
    p(age_group_summary[['total_appts', 'noshow_count', 'noshow_rate_pct']].to_string())
    
    gender_summary = df.groupby('gender')['disengaged'].agg(
        total_appts='count',
        noshow_count='sum',
        noshow_rate='mean'
    )
    gender_summary['noshow_rate_pct'] = (gender_summary['noshow_rate'] * 100).round(2)
    p("\nNo-Show Rate by Gender:")
    p(gender_summary[['total_appts', 'noshow_count', 'noshow_rate_pct']].to_string())
    
    gender_table = pd.crosstab(df['gender'], df['disengaged'])
    chi2_g, p_g, _, _ = stats.chi2_contingency(gender_table)
    p(f"Gender Chi-square: {chi2_g:.2f}, p-value: {p_g:.4f}")
    
    # 2. CLINICAL COMORBIDITIES & ODDS RATIOS
    p("\n" + "-" * 50)
    p("2. CLINICAL COMORBIDITIES & ODDS RATIOS")
    p("-" * 50)
    
    df['has_handicap'] = (df['handicap'] > 0).astype(int)
    comorbidities = ['hypertension', 'diabetes', 'alcoholism', 'has_handicap']
    df['comorbidity_count'] = df['hypertension'] + df['diabetes'] + df['alcoholism'] + df['has_handicap']
    
    comorb_results = []
    for col in comorbidities + ['scholarship']:
        table = pd.crosstab(df[col], df['disengaged'])
        chi2, p_val, dof, _ = stats.chi2_contingency(table)
        
        a = table.loc[1, 1]
        b = table.loc[1, 0]
        c = table.loc[0, 1]
        d = table.loc[0, 0]
        odds_ratio = (a * d) / (b * c)
        
        log_or = np.log(odds_ratio)
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_lower = np.exp(log_or - 1.96 * se_log_or)
        ci_upper = np.exp(log_or + 1.96 * se_log_or)
        
        rate_exposed = df[df[col] == 1]['disengaged'].mean() * 100
        rate_unexposed = df[df[col] == 0]['disengaged'].mean() * 100
        
        comorb_results.append({
            'Factor': col,
            'Rate_Exposed_%': round(rate_exposed, 2),
            'Rate_Unexposed_%': round(rate_unexposed, 2),
            'Odds_Ratio': round(odds_ratio, 3),
            '95% CI': f"[{ci_lower:.2f}, {ci_upper:.2f}]",
            'Chi2': round(chi2, 2),
            'p-value': f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
        })
        
    comorb_df = pd.DataFrame(comorb_results)
    p(comorb_df.to_string(index=False))
    
    p("\nComorbidity Burden (Count of chronic conditions 0 to 4):")
    burden_summary = df.groupby('comorbidity_count')['disengaged'].agg(
        total_appts='count',
        noshow_rate='mean'
    )
    burden_summary['noshow_rate_pct'] = (burden_summary['noshow_rate'] * 100).round(2)
    p(burden_summary[['total_appts', 'noshow_rate_pct']].to_string())
    
    # 3. OPERATIONAL FACTORS
    p("\n" + "-" * 50)
    p("3. OPERATIONAL FACTORS: LEAD TIME & DAY OF WEEK")
    p("-" * 50)
    
    df['appt_day_of_week'] = df['appointment_day'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    day_summary = df.groupby('appt_day_of_week', observed=False)['disengaged'].agg(
        total_appts='count',
        noshow_rate='mean'
    ).reindex(day_order)
    day_summary['noshow_rate_pct'] = (day_summary['noshow_rate'] * 100).round(2)
    p("No-Show Rate by Appointment Day of Week:")
    p(day_summary[['total_appts', 'noshow_rate_pct']].to_string())
    
    lead_bins = [-1, 0, 3, 7, 14, 30, 180]
    lead_labels = ['Same Day (0d)', 'Short (1-3d)', '1 Week (4-7d)', '2 Weeks (8-14d)', 'Month (15-30d)', 'Long (>30d)']
    df['lead_category'] = pd.cut(df['lead_days'], bins=lead_bins, labels=lead_labels)
    
    lead_summary = df.groupby('lead_category', observed=False)['disengaged'].agg(
        total_appts='count',
        noshow_rate='mean'
    )
    lead_summary['noshow_rate_pct'] = (lead_summary['noshow_rate'] * 100).round(2)
    p("\nNo-Show Rate by Lead Time Category:")
    p(lead_summary[['total_appts', 'noshow_rate_pct']].to_string())
    
    # 4. GEOGRAPHIC VARIATION
    p("\n" + "-" * 50)
    p("4. GEOGRAPHIC VARIATION: NEIGHBOURHOOD")
    p("-" * 50)
    
    hood_stats = df.groupby('neighbourhood')['disengaged'].agg(
        total_appts='count',
        noshow_rate='mean'
    )
    hood_stats_100 = hood_stats[hood_stats['total_appts'] >= 100]
    p(f"Total Unique Neighbourhoods: {df['neighbourhood'].nunique()}")
    p("\nTop 5 Highest Disengagement Neighbourhoods (min 100 appts):")
    top_5_high = hood_stats_100.sort_values(by='noshow_rate', ascending=False).head(5)
    top_5_high['noshow_rate_pct'] = (top_5_high['noshow_rate'] * 100).round(2)
    p(top_5_high[['total_appts', 'noshow_rate_pct']].to_string())
    
    p("\nTop 5 Lowest Disengagement Neighbourhoods (min 100 appts):")
    top_5_low = hood_stats_100.sort_values(by='noshow_rate', ascending=True).head(5)
    top_5_low['noshow_rate_pct'] = (top_5_low['noshow_rate'] * 100).round(2)
    p(top_5_low[['total_appts', 'noshow_rate_pct']].to_string())
    
    p("\nSTEP 2 ANALYSIS COMPLETE.")

if __name__ == '__main__':
    run_step_2_analysis()
