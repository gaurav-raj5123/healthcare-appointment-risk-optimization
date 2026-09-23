import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_patient_data.csv')
FIG_DIR = os.path.join(PROJECT_ROOT, 'steps', '02_eda', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Set global plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

def main():
    print(f"Loading cleaned data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
    df['appointment_day'] = pd.to_datetime(df['appointment_day'])
    df['no_show'] = df['disengaged']  # 1 = No-show, 0 = Show
    
    print("\n--- Generating Figure 1: Numerical Distributions (Age & Lead Days) ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Age histplot
    sns.histplot(df['age'], bins=35, kde=True, ax=axes[0, 0], color='#2b5c8f')
    axes[0, 0].set_title("Age Distribution (Mean: 37.1, Median: 37)")
    axes[0, 0].set_xlabel("Age (Years)")
    axes[0, 0].set_ylabel("Appointment Count")
    
    # Age boxplot
    sns.boxplot(x='no_show', y='age', data=df, ax=axes[0, 1], palette=['#4daf4a', '#e41a1c'])
    axes[0, 1].set_xticklabels(['Show (0)', 'No-Show (1)'])
    axes[0, 1].set_title("Age by Appointment Outcome (No-Show is younger)")
    axes[0, 1].set_xlabel("Outcome")
    axes[0, 1].set_ylabel("Age (Years)")
    
    # Lead time histplot (clipped to 60 days for visibility)
    sns.histplot(df[df['lead_days'] <= 60]['lead_days'], bins=30, kde=False, ax=axes[1, 0], color='#d95f02')
    axes[1, 0].set_title("Lead Days Distribution (<= 60d) - 34.9% at Day 0")
    axes[1, 0].set_xlabel("Lead Days (Waiting Time)")
    axes[1, 0].set_ylabel("Appointment Count")
    
    # Lead time boxplot (log scale for outliers)
    sns.boxplot(x='no_show', y=np.log1p(df['lead_days']), data=df, ax=axes[1, 1], palette=['#4daf4a', '#e41a1c'])
    axes[1, 1].set_xticklabels(['Show (0)', 'No-Show (1)'])
    axes[1, 1].set_title("Log(1 + Lead Days) by Outcome")
    axes[1, 1].set_xlabel("Outcome")
    axes[1, 1].set_ylabel("Log(1 + Lead Days)")
    
    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, '01_numerical_distributions.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    print("\n--- Generating Figure 2: Target Imbalance (Show vs No-Show) ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Barplot of counts
    counts = df['no_show'].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=axes[0], palette=['#2b83ba', '#d7191c'])
    axes[0].set_xticklabels(['Show (0)', 'No-Show (1)'])
    axes[0].set_title("Appointment Outcome Frequency")
    axes[0].set_ylabel("Number of Appointments")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 1500, f"{v:,} ({v/len(df)*100:.1f}%)", ha='center', fontweight='bold')
    axes[0].set_ylim(0, 100000)
    
    # Donut pie chart
    axes[1].pie(counts.values, labels=['Show (79.8%)', 'No-Show (20.2%)'], colors=['#2b83ba', '#d7191c'],
               autopct='%1.1f%%', startangle=140, pctdistance=0.8, wedgeprops=dict(width=0.4, edgecolor='w'))
    axes[1].set_title("Class Ratio: ~4 : 1 Imbalance")
    
    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, '02_target_imbalance.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    print("\n--- Generating Figure 3: Categorical Univariate Analysis ---")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    cat_cols = ['gender', 'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'sms_received']
    titles = ['Gender', 'Welfare Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'SMS Reminder Received']
    
    for idx, (col, title) in enumerate(zip(cat_cols, titles)):
        r, c = divmod(idx, 3)
        val_counts = df[col].value_counts()
        sns.barplot(x=val_counts.index.astype(str), y=val_counts.values, ax=axes[r, c], color='#4575b4')
        axes[r, c].set_title(f"{title} (Feature: {col})")
        axes[r, c].set_ylabel("Count")
        for i, val in enumerate(val_counts.values):
            axes[r, c].text(i, val * 0.5, f"{val:,}\n({val/len(df)*100:.1f}%)", ha='center', color='white', fontweight='bold')
            
    plt.tight_layout()
    fig3_path = os.path.join(FIG_DIR, '03_categorical_univariate.png')
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    print("\n--- Generating Figure 4: Bivariate Risk Curves ---")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    # 1. Age groups vs no-show rate
    age_bins = [-1, 12, 19, 39, 59, 120]
    age_labels = ['Child\n(0-12)', 'Adolescent\n(13-19)', 'Young Adult\n(20-39)', 'Middle-Aged\n(40-59)', 'Senior\n(60+)']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    age_rate = df.groupby('age_group', observed=False)['no_show'].mean() * 100
    sns.barplot(x=age_rate.index, y=age_rate.values, ax=axes[0, 0], palette='Blues_r')
    axes[0, 0].axhline(df['no_show'].mean()*100, color='red', linestyle='--', label=f"Average ({df['no_show'].mean()*100:.1f}%)")
    axes[0, 0].set_title("No-Show Rate Across Patient Life Stages")
    axes[0, 0].set_ylabel("No-Show Rate (%)")
    axes[0, 0].legend()
    for i, v in enumerate(age_rate.values):
        axes[0, 0].text(i, v + 0.8, f"{v:.1f}%", ha='center', fontweight='bold')
    axes[0, 0].set_ylim(0, 32)
    
    # 2. Lead time bins vs no-show rate
    lead_bins = [-1, 0, 3, 7, 14, 30, 60, 365]
    lead_labels = ['Same day', '1-3d', '4-7d', '8-14d', '15-30d', '31-60d', '60+d']
    df['lead_bin'] = pd.cut(df['lead_days'], bins=lead_bins, labels=lead_labels)
    lead_rate = df.groupby('lead_bin', observed=False)['no_show'].mean() * 100
    sns.barplot(x=lead_rate.index, y=lead_rate.values, ax=axes[0, 1], palette='Oranges')
    axes[0, 1].axhline(df['no_show'].mean()*100, color='red', linestyle='--', label=f"Average ({df['no_show'].mean()*100:.1f}%)")
    axes[0, 1].set_title("No-Show Rate by Lead Time (Waiting Days)")
    axes[0, 1].set_ylabel("No-Show Rate (%)")
    axes[0, 1].legend()
    for i, v in enumerate(lead_rate.values):
        axes[0, 1].text(i, v + 0.8, f"{v:.1f}%", ha='center', fontweight='bold')
    axes[0, 1].set_ylim(0, 40)
    
    # 3. Day of week vs no-show rate
    df['day_of_week'] = df['appointment_day'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    day_rate = df.groupby('day_of_week', observed=False)['no_show'].mean().reindex(day_order) * 100
    sns.barplot(x=day_rate.index, y=day_rate.values, ax=axes[1, 0], palette='Purples')
    axes[1, 0].axhline(df['no_show'].mean()*100, color='red', linestyle='--', label=f"Average ({df['no_show'].mean()*100:.1f}%)")
    axes[1, 0].set_title("No-Show Rate by Appointment Day of Week")
    axes[1, 0].set_ylabel("No-Show Rate (%)")
    axes[1, 0].legend()
    for i, v in enumerate(day_rate.values):
        axes[1, 0].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')
    axes[1, 0].set_ylim(0, 28)
    
    # 4. Medical conditions vs no-show rate
    cond_names = ['Hypertension', 'Diabetes', 'Handicap', 'Alcoholism', 'Scholarship']
    cond_cols = ['hypertension', 'diabetes', 'has_handicap', 'alcoholism', 'scholarship']
    df['has_handicap'] = (df['handicap'] > 0).astype(int)
    
    yes_rates = [df[df[c] == 1]['no_show'].mean() * 100 for c in cond_cols]
    no_rates = [df[df[c] == 0]['no_show'].mean() * 100 for c in cond_cols]
    
    cond_df = pd.DataFrame({
        'Condition': cond_names * 2,
        'Status': ['Has Condition (1)'] * len(cond_names) + ['No Condition (0)'] * len(cond_names),
        'NoShowRate': yes_rates + no_rates
    })
    sns.barplot(x='Condition', y='NoShowRate', hue='Status', data=cond_df, ax=axes[1, 1], palette=['#d73027', '#4575b4'])
    axes[1, 1].set_title("No-Show Rate by Clinical Condition & Welfare")
    axes[1, 1].set_ylabel("No-Show Rate (%)")
    axes[1, 1].set_ylim(0, 30)
    axes[1, 1].legend(title="")
    
    plt.tight_layout()
    fig4_path = os.path.join(FIG_DIR, '04_bivariate_risk_curves.png')
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")

    print("\n--- Generating Figure 5: SMS and No-Show Rate by Lead-Time Group ---")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    lead_bins_sms = [-1, 0, 3, 7, 14, 30, 60, 365]
    lead_labels_sms = ['Same day (0d)', '1-3 days', '4-7 days', '8-14 days', '15-30 days', '31-60 days', '60+ days']
    df['lead_bin_sms'] = pd.cut(df['lead_days'], bins=lead_bins_sms, labels=lead_labels_sms)
    
    sms_strat = df.groupby(['lead_bin_sms', 'sms_received'], observed=False)['no_show'].mean().unstack() * 100
    sms_strat.columns = ['No SMS Received (0)', 'SMS Reminder Received (1)']
    
    sms_strat.plot(kind='bar', ax=ax, color=['#d7191c', '#2c7bb6'], width=0.7)
    ax.set_title("SMS and No-Show Rate by Lead-Time Group", fontsize=14, fontweight='bold')
    ax.set_ylabel("No-Show Rate (%)")
    ax.set_xlabel("Appointment Lead Time Window")
    ax.set_xticklabels(lead_labels_sms, rotation=15)
    ax.legend(title="")
    ax.set_ylim(0, 45)
    
    # Annotate the observed difference on each bar pair (starting from 4-7 days where SMS is active)
    for i in range(2, len(lead_labels_sms)):
        no_sms_val = sms_strat.iloc[i, 0]
        sms_val = sms_strat.iloc[i, 1]
        diff = no_sms_val - sms_val
        ax.text(i + 0.17, sms_val + 1.0, f"-{diff:.1f}%\nDiff", ha='center', color='#1a9641', fontweight='bold', fontsize=9)

    plt.tight_layout()
    fig5_path = os.path.join(FIG_DIR, '05_sms_leadtime_paradox.png')
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved: {fig5_path}")

    print("\n--- Generating Figure 6: Prior Patient Behavior (Causal History & Rate) ---")
    # Chronologically sort appointments to simulate true historical causal calculation
    df_sorted = df.sort_values(by=['patient_id', 'appointment_day', 'scheduled_day']).copy()
    
    # Calculate expanding prior appointments, prior no-shows, and prior no-show rate strictly before current row
    df_sorted['prior_appointments'] = df_sorted.groupby('patient_id').cumcount()
    df_sorted['prior_noshows'] = df_sorted.groupby('patient_id')['no_show'].cumsum() - df_sorted['no_show']
    df_sorted['prior_noshow_rate'] = np.where(
        df_sorted['prior_appointments'] > 0,
        df_sorted['prior_noshows'] / df_sorted['prior_appointments'],
        np.nan
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Prior appointments vs no-show rate
    df_sorted['prior_appt_bin'] = pd.cut(df_sorted['prior_appointments'], bins=[-1, 0, 1, 2, 4, 100], 
                                         labels=['0 (First Visit)', '1 Prior', '2 Prior', '3-4 Prior', '5+ Prior'])
    prior_appt_rate = df_sorted.groupby('prior_appt_bin', observed=False)['no_show'].mean() * 100
    sns.barplot(x=prior_appt_rate.index, y=prior_appt_rate.values, ax=axes[0], hue=prior_appt_rate.index, palette='crest', legend=False)
    axes[0].set_title("No-Show Rate by Prior Total Visits")
    axes[0].set_ylabel("Current No-Show Rate (%)")
    for i, v in enumerate(prior_appt_rate.values):
        axes[0].text(i, v + 0.6, f"{v:.1f}%", ha='center', fontweight='bold')
    axes[0].set_ylim(0, 26)
    
    # 2. Prior no-shows count vs no-show rate (for repeat patients)
    repeat_df = df_sorted[df_sorted['prior_appointments'] > 0].copy()
    repeat_df['prior_noshow_cap'] = repeat_df['prior_noshows'].clip(upper=4).astype(int).astype(str)
    repeat_df.loc[repeat_df['prior_noshow_cap'] == '4', 'prior_noshow_cap'] = '4+'
    
    prior_ns_rate = repeat_df.groupby('prior_noshow_cap', observed=False)['no_show'].mean() * 100
    order_ns = ['0', '1', '2', '3', '4+']
    prior_ns_rate = prior_ns_rate.reindex(order_ns)
    
    sns.barplot(x=prior_ns_rate.index, y=prior_ns_rate.values, ax=axes[1], hue=prior_ns_rate.index, palette='Reds', legend=False)
    axes[1].set_title("No-Show Rate by Past Missed Count")
    axes[1].set_xlabel("Number of Prior No-Shows")
    axes[1].set_ylabel("Current No-Show Rate (%)")
    for i, v in enumerate(prior_ns_rate.values):
        axes[1].text(i, v + 1.0, f"{v:.1f}%", ha='center', fontweight='bold')
    axes[1].set_ylim(0, 50)

    # 3. Prior no-show rate (%) vs current no-show rate
    rate_bins = [-0.01, 0.0, 0.25, 0.50, 0.75, 1.0]
    rate_labels = ['0% (Never missed)', '1-25%', '26-50%', '51-75%', '76-100%']
    repeat_df['prior_rate_bin'] = pd.cut(repeat_df['prior_noshow_rate'], bins=rate_bins, labels=rate_labels)
    prior_ratio_rate = repeat_df.groupby('prior_rate_bin', observed=False)['no_show'].mean() * 100

    sns.barplot(x=prior_ratio_rate.index, y=prior_ratio_rate.values, ax=axes[2], hue=prior_ratio_rate.index, palette='YlOrRd', legend=False)
    axes[2].set_title("No-Show Rate by Historical Miss Ratio")
    axes[2].set_xlabel("Prior No-Show Rate (%)")
    axes[2].set_ylabel("Current No-Show Rate (%)")
    axes[2].tick_params(axis='x', rotation=15)
    for i, v in enumerate(prior_ratio_rate.values):
        axes[2].text(i, v + 1.0, f"{v:.1f}%", ha='center', fontweight='bold')
    axes[2].set_ylim(0, 50)
    
    plt.tight_layout()
    fig6_path = os.path.join(FIG_DIR, '06_patient_history_effect.png')
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"Saved: {fig6_path}")

    print("\n--- Generating Figure 7: Correlation Heatmap ---")
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_features = ['age', 'lead_days', 'scholarship', 'hypertension', 'diabetes', 
                        'alcoholism', 'has_handicap', 'sms_received', 'no_show']
    corr = df[numeric_features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='vlag', vmin=-0.15, vmax=0.4, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Correlation Matrix of Observable Features & Target (no_show)", fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    fig7_path = os.path.join(FIG_DIR, '07_multivariate_correlation.png')
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    print(f"Saved: {fig7_path}")
    print("\nAll 7 EDA figures successfully generated!")

if __name__ == '__main__':
    main()
