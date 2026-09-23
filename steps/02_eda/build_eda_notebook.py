import nbformat as nbf
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, 'steps', '02_eda', '01_eda.ipynb')

def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Business framing
    cells.append(nbf.v4.new_markdown_cell("""# 🏥 Healthcare Appointment No-Show Prediction EDA

### **Business Question:**
> *"Which scheduled appointments are at high risk of a patient not attending (no-show), what observable clinical and operational factors are associated with that risk, and how can the clinic use these predictions to prioritize reminders and schedule capacity?"*

---

## 🎯 The 5-Stage EDA Framework
* **Stage 1 — Understand the distributions:** Summary statistics and density curves for Age and Lead Time.
* **Stage 2 — Understand the target:** Imbalance ratio, prevalence, and evaluation metric decisions (PR-AUC vs Accuracy).
* **Stage 3 — Univariate analysis:** Frequency distributions across demographics, chronic conditions, and social assistance.
* **Stage 4 — Bivariate analysis:** Statistical associations between observable features and the `no_show` target.
* **Stage 5 — Multivariate analysis:** Feature interactions, the SMS x Lead Time confounding paradox, and correlation structure.

---
"""))

    # Section 1: Load cleaned data
    cells.append(nbf.v4.new_markdown_cell("## 1. Load Cleaned Data\nWe load the pre-audited dataset where date anomalies and negative ages have already been sanitized."))
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress minor library deprecation warnings for clean presentation
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)

data_path = os.path.join('..', '..', 'data', 'processed', 'cleaned_patient_data.csv')
df = pd.read_csv(data_path)

# Ensure datetime types
df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
df['appointment_day'] = pd.to_datetime(df['appointment_day'])
df['no_show'] = df['disengaged']  # 1 = No-show, 0 = Show

print(f"Dataset successfully loaded: {df.shape[0]:,} rows, {df.shape[1]} columns.")
"""))

    # Section 2: Dataset overview
    cells.append(nbf.v4.new_markdown_cell("## 2. Dataset Overview\nInspecting data types, non-null counts, and unique patient volume."))
    cells.append(nbf.v4.new_code_cell("""print("Column Data Types & Non-Nulls:")
print(df.info())
print("\\nFirst 5 Records:")
df.head()
"""))
    cells.append(nbf.v4.new_code_cell("""unique_patients = df['patient_id'].nunique()
total_appts = len(df)
print(f"Total Appointments: {total_appts:,}")
print(f"Unique Patients: {unique_patients:,}")
print(f"Average appointments per patient: {total_appts / unique_patients:.2f}")
"""))

    # Section 3: Target distribution
    cells.append(nbf.v4.new_markdown_cell("""## 3. Target Distribution (`no_show`)
* `0` = Show (Patient attended scheduled appointment)
* `1` = No-Show (Patient did not attend scheduled appointment)

### 💡 Key Modeling Decision:
The target exhibits an **80/20 class imbalance**. A trivial model predicting "Show" for every appointment achieves 80% accuracy, but is completely useless to a clinic.  
**Decision:** We will evaluate models using **PR-AUC (Precision-Recall AUC), Recall, F1-Score, and Cost-Optimized Thresholds**, not baseline accuracy.
"""))
    cells.append(nbf.v4.new_code_cell("""target_counts = df['no_show'].value_counts()
target_pct = df['no_show'].value_counts(normalize=True) * 100

print("Target Distribution:")
for k, v in target_counts.items():
    label = "No-Show (Did Not Attend)" if k == 1 else "Show (Attended)"
    print(f"  {k} ({label}): {v:,} ({target_pct[k]:.2f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
bar_labels = ['Show (0)', 'No-Show (1)']
sns.barplot(x=bar_labels, y=target_counts.values, ax=axes[0], hue=bar_labels, palette=['#2b83ba', '#d7191c'], legend=False)
axes[0].set_title('Appointment Outcome Distribution (Count)')
axes[0].set_ylabel('Count')

axes[1].pie(target_counts.values, labels=['Show (79.8%)', 'No-Show (20.2%)'], colors=['#2b83ba', '#d7191c'],
            autopct='%1.1f%%', startangle=140, pctdistance=0.8, wedgeprops=dict(width=0.4, edgecolor='w'))
axes[1].set_title('Target Proportion: ~4:1 Imbalance')
plt.tight_layout()
plt.show()
"""))

    # Section 4: Numerical feature analysis
    cells.append(nbf.v4.new_markdown_cell("## 4. Numerical Feature Analysis: Age & Lead Time"))
    cells.append(nbf.v4.new_code_cell("""print("Numerical Summary Statistics:")
df[['age', 'lead_days']].describe()
"""))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Age distribution
sns.histplot(df['age'], bins=35, kde=True, ax=axes[0], color='#2b5c8f')
axes[0].set_title('Patient Age Distribution')
axes[0].set_xlabel('Age (Years)')
axes[0].set_ylabel('Number of Appointments')

# Lead time distribution (clipped to 60 days)
sns.histplot(df[df['lead_days'] <= 60]['lead_days'], bins=30, ax=axes[1], color='#d95f02')
axes[1].set_title('Lead Time Distribution (Waiting Days <= 60)')
axes[1].set_xlabel('Lead Days (Booking to Appointment)')
axes[1].set_ylabel('Number of Appointments')

plt.tight_layout()
plt.show()

print(f"Same-day bookings (lead_days == 0): {(df['lead_days'] == 0).sum():,} ({(df['lead_days'] == 0).mean()*100:.2f}%)")
"""))

    # Section 5: Categorical feature analysis
    cells.append(nbf.v4.new_markdown_cell("## 5. Categorical Feature Analysis (Univariate)"))
    cells.append(nbf.v4.new_code_cell("""cat_cols = ['gender', 'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'sms_received']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for idx, col in enumerate(cat_cols):
    r, c = divmod(idx, 3)
    counts = df[col].value_counts()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=axes[r, c], color='#4575b4')
    axes[r, c].set_title(f"Feature: {col}")
    axes[r, c].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
"""))

    # Section 6: No-show analysis (Bivariate)
    cells.append(nbf.v4.new_markdown_cell("## 6. No-Show Analysis (Bivariate Risk Curves)\nConnecting every observable factor back to the business target: **What increases or decreases no-show risk?**"))
    cells.append(nbf.v4.new_code_cell("""# 6.1 Age Life Stages vs No-Show
age_bins = [-1, 12, 19, 39, 59, 120]
age_labels = ['Child (0-12)', 'Adolescent (13-19)', 'Young Adult (20-39)', 'Middle-Aged (40-59)', 'Senior (60+)']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

age_rate = df.groupby('age_group', observed=False)['no_show'].mean() * 100
print("No-Show Rate Across Life Stages:")
print(age_rate.round(2))

plt.figure(figsize=(9, 4))
sns.barplot(x=age_rate.index, y=age_rate.values, hue=age_rate.index, palette='Blues_r', legend=False)
plt.axhline(df['no_show'].mean() * 100, color='red', linestyle='--', label=f"Average ({df['no_show'].mean()*100:.1f}%)")
plt.title("No-Show Rate by Patient Life Stage")
plt.ylabel("No-Show Rate (%)")
plt.legend()
plt.show()
"""))
    cells.append(nbf.v4.new_code_cell("""# 6.2 Lead Time Groups vs No-Show
lead_bins = [-1, 0, 3, 7, 14, 30, 60, 365]
lead_labels = ['Same day (0d)', '1-3d', '4-7d', '8-14d', '15-30d', '31-60d', '60+d']
df['lead_bin'] = pd.cut(df['lead_days'], bins=lead_bins, labels=lead_labels)

lead_rate = df.groupby('lead_bin', observed=False)['no_show'].mean() * 100
print("No-Show Rate Across Lead Time Groups:")
print(lead_rate.round(2))

plt.figure(figsize=(10, 4.5))
sns.barplot(x=lead_rate.index, y=lead_rate.values, hue=lead_rate.index, palette='Oranges', legend=False)
plt.axhline(df['no_show'].mean() * 100, color='red', linestyle='--', label=f"Average ({df['no_show'].mean()*100:.1f}%)")
plt.title("No-Show Rate by Lead Time (Waiting Days)")
plt.ylabel("No-Show Rate (%)")
plt.legend()
plt.show()
"""))
    cells.append(nbf.v4.new_code_cell("""# 6.3 Clinical Conditions & Welfare Odds Ratios
df['has_handicap'] = (df['handicap'] > 0).astype(int)
from scipy import stats

factors = ['hypertension', 'diabetes', 'alcoholism', 'has_handicap', 'scholarship']
for f in factors:
    tab = pd.crosstab(df[f], df['no_show'])
    or_val = (tab.loc[1, 1] * tab.loc[0, 0]) / (tab.loc[1, 0] * tab.loc[0, 1])
    chi2, p_val, _, _ = stats.chi2_contingency(tab)
    print(f"{f:15s} | Exposed: {df[df[f]==1]['no_show'].mean()*100:.2f}% | Unexposed: {df[df[f]==0]['no_show'].mean()*100:.2f}% | Odds Ratio: {or_val:.3f} | p-value: {p_val:.2e}")
"""))

    # Section 7: Patient history analysis
    cells.append(nbf.v4.new_markdown_cell("""## 7. Patient History Analysis (Causal Behavioral Features)
Investigating whether historical attendance associates with future attendance **without data leakage**.  
We sort chronologically and compute rolling cumulative statistics strictly prior to each appointment, including both prior no-show count and prior no-show rate.
"""))
    cells.append(nbf.v4.new_code_cell("""# Chronological sort
df_sorted = df.sort_values(by=['patient_id', 'appointment_day', 'scheduled_day']).copy()

# Causal rolling features (Zero-Leakage)
df_sorted['prior_appointments'] = df_sorted.groupby('patient_id').cumcount()
df_sorted['prior_noshows'] = df_sorted.groupby('patient_id')['no_show'].cumsum() - df_sorted['no_show']
df_sorted['prior_noshow_rate'] = np.where(
    df_sorted['prior_appointments'] > 0,
    df_sorted['prior_noshows'] / df_sorted['prior_appointments'],
    np.nan
)

repeat_patients = df_sorted[df_sorted['prior_appointments'] > 0].copy()
repeat_patients['prior_noshow_capped'] = repeat_patients['prior_noshows'].clip(upper=4).astype(int).astype(str)
repeat_patients.loc[repeat_patients['prior_noshow_capped'] == '4', 'prior_noshow_capped'] = '4+'

ns_rate_by_history = repeat_patients.groupby('prior_noshow_capped', observed=False)['no_show'].mean() * 100

# Rate bins for prior_noshow_rate
rate_bins = [-0.01, 0.0, 0.25, 0.50, 0.75, 1.0]
rate_labels = ['0% (Never missed)', '1-25%', '26-50%', '51-75%', '76-100% (Always missed)']
repeat_patients['prior_rate_bin'] = pd.cut(repeat_patients['prior_noshow_rate'], bins=rate_bins, labels=rate_labels)
ns_rate_by_ratio = repeat_patients.groupby('prior_rate_bin', observed=False)['no_show'].mean() * 100

print("Current No-Show Rate by Prior Missed Appointments Count:")
print(ns_rate_by_history.round(2))
print("\\nCurrent No-Show Rate by Prior No-Show Rate (Ratio):")
print(ns_rate_by_ratio.round(2))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Left: Count of prior no-shows
sns.barplot(x=ns_rate_by_history.index, y=ns_rate_by_history.values, ax=axes[0], hue=ns_rate_by_history.index, palette='Reds', legend=False)
axes[0].set_title("No-Show Rate by Prior No-Show Count")
axes[0].set_xlabel("Number of Prior Missed Appointments")
axes[0].set_ylabel("Current No-Show Rate (%)")

# Right: Ratio of prior no-shows (prior_noshow_rate)
sns.barplot(x=ns_rate_by_ratio.index, y=ns_rate_by_ratio.values, ax=axes[1], hue=ns_rate_by_ratio.index, palette='YlOrRd', legend=False)
axes[1].set_title("No-Show Rate by Prior No-Show Rate (%)")
axes[1].set_xlabel("Historical No-Show Ratio")
axes[1].set_ylabel("Current No-Show Rate (%)")
axes[1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.show()
"""))

    # Section 8: Multivariate relationships
    cells.append(nbf.v4.new_markdown_cell(r"""## 8. Multivariate Relationships: The SMS x Lead Time Paradox
In aggregate, appointments that received an SMS reminder observed a higher no-show rate (27.5% vs 16.7%).  
**Why? Confounding by lead time.** Reminders were only dispatched to appointments scheduled $\ge 3$ days in advance. No-show rates differ between SMS and non-SMS appointments after stratifying by lead time.
"""))
    cells.append(nbf.v4.new_code_cell("""sms_strat = df.groupby(['lead_bin', 'sms_received'], observed=False)['no_show'].mean().unstack() * 100
sms_strat.columns = ['No SMS (0)', 'SMS Sent (1)']
sms_strat['Difference (% points)'] = sms_strat['No SMS (0)'] - sms_strat['SMS Sent (1)']
print(sms_strat.round(2))

plt.figure(figsize=(11, 5))
sms_strat[['No SMS (0)', 'SMS Sent (1)']].plot(kind='bar', figsize=(11, 5), color=['#d7191c', '#2c7bb6'])
plt.title("SMS and No-Show Rate by Lead-Time Group")
plt.ylabel("No-Show Rate (%)")
plt.xlabel("Lead Time Window")
plt.legend()
plt.tight_layout()
plt.show()
"""))
    cells.append(nbf.v4.new_code_cell("""# Correlation Matrix of Numerical & Binary Variables
numeric_features = ['age', 'lead_days', 'scholarship', 'hypertension', 'diabetes', 
                    'alcoholism', 'has_handicap', 'sms_received', 'no_show']
corr = df[numeric_features].corr()

plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='vlag', vmin=-0.15, vmax=0.4, square=True)
plt.title("Correlation Matrix of Observable Features")
plt.show()
"""))

    # Section 9: EDA conclusions
    cells.append(nbf.v4.new_markdown_cell(r"""## 9. EDA Conclusions & Modeling Decisions

| Business Question / Observation | Empirical Finding | Modeling Decision |
| :--- | :--- | :--- |
| **Q1: Who has higher no-show rates?** | Adolescents and Young Adults (13–39) and Welfare (`Scholarship`) recipients have higher observed no-show rates. Patients with chronic conditions (Hypertension, Diabetes) observe lower no-show rates. | Include age life-stage groupings, comorbidity burden scores, and socioeconomic flags as features. |
| **Q2: Does appointment timing matter?** | Same-day visits observe a **4.65%** no-show rate; appointments with lead times $\ge 1$ week observe no-show rates exceeding **25%**. | Include `is_same_day` binary flag, continuous `lead_days`, and log-transformed `log_lead_days`. |
| **Q3: Does prior behavior matter?** | Patients with previous no-shows observe significantly higher subsequent no-show rates (up to **39%** for patients who missed >50% of prior visits). | Engineer causal rolling features (`prior_appointments`, `prior_noshows`, and `prior_noshow_rate`) without lookahead bias. |
| **Q4: Is communication associated with attendance?** | In appointments with lead times $\ge 3$ days, appointments with SMS reminders observed no-show rates **3% to 8.4% lower** than those without reminders. | Include `sms_received` and interaction features (`sms_x_lead_days`). |
| **Q5: Is target imbalanced?** | 80% Show vs 20% No-Show. | **Do not evaluate on accuracy.** Use **PR-AUC, ROC-AUC, F1-Score, and Expected Value Threshold Optimization**. |

---
**Next Step:** Proceed to `steps/03_feature_engineering/` to formalize these causal and operational features into the final modeling dataset!
"""))

    nb.cells = cells
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Jupyter Notebook successfully built at: {NOTEBOOK_PATH}")

if __name__ == '__main__':
    build_notebook()
