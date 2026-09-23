import nbformat as nbf
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, 'steps', '03_feature_engineering', '02_feature_engineering.ipynb')

def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Framing
    cells.append(nbf.v4.new_markdown_cell(r"""# ⚙️ Step 3: Feature Engineering & Temporal Train/Test Split

### **The Core Principle: Prediction-Time Availability**
> *"Would the clinic staff know this information at the moment the patient schedules the appointment?"*

At prediction time:
* **Allowed:** Patient age, gender, historical chronic conditions, booking timestamp, appointment date, lead time, and all past appointment attendance records strictly prior to this visit.
* **Strictly Prohibited:** Current visit outcome, future appointment dates/outcomes, or any information generated during or after the visit.

---

## 🏗️ The 3-Layer Feature Architecture
1. **Layer 1 — Core Static Features:** Demographics (`age`, `gender`), socioeconomic status (`scholarship`), clinical comorbidities (`hypertension`, `diabetes`, `alcoholism`, `has_handicap`), and communication (`sms_received`).
2. **Layer 2 — Core Temporal Features:** Operational timing (`lead_days`, `same_day_booking`, `appointment_dow`, `scheduled_dow`, `appointment_month`).
3. **Layer 3 — Core Patient History Features (Point-in-Time, Zero-Leakage):** `prior_appointments`, `prior_noshows`, `prior_noshow_rate`, `days_since_last_appointment`, `is_first_appointment`.
"""))

    # Section 1: Load engineered datasets
    cells.append(nbf.v4.new_markdown_cell("## 1. Load Engineered Train & Test Datasets\nGenerated via the prediction-time compliant pipeline script `engineer_features.py`."))
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 5)

train_path = os.path.join('..', '..', 'data', 'processed', 'train_features.csv')
test_path = os.path.join('..', '..', 'data', 'processed', 'test_features.csv')
meta_path = os.path.join('..', '..', 'data', 'processed', 'feature_metadata.json')

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

with open(meta_path, 'r') as f:
    catalog = json.load(f)

print(f"Train set: {len(df_train):,} rows, {df_train.shape[1]} columns")
print(f"Test set:  {len(df_test):,} rows, {df_test.shape[1]} columns")
print(f"Total features cataloged: {len(catalog['features']['core_static']) + len(catalog['features']['core_temporal']) + len(catalog['features']['core_patient_history'])}")
"""))

    # Section 2: Temporal Split Inspection
    cells.append(nbf.v4.new_markdown_cell("## 2. Temporal Train / Test Split Verification\nStrict calendar partition ensuring the model trains exclusively on past appointments and evaluates on unseen future appointments."))
    cells.append(nbf.v4.new_code_cell("""df_train['appointment_day'] = pd.to_datetime(df_train['appointment_day'])
df_test['appointment_day'] = pd.to_datetime(df_test['appointment_day'])

print(f"Training Date Range:   {df_train['appointment_day'].min().date()} to {df_train['appointment_day'].max().date()}")
print(f"Testing Date Range:    {df_test['appointment_day'].min().date()} to {df_test['appointment_day'].max().date()}")
print(f"Temporal Overlap Check: Max Train < Min Test? -> {df_train['appointment_day'].max() < df_test['appointment_day'].min()}")

split_summary = pd.DataFrame({
    'Dataset': ['Train (April/May 2016)', 'Test (June 2016)'],
    'Appointments': [len(df_train), len(df_test)],
    'Share (%)': [len(df_train)/ (len(df_train) + len(df_test)) * 100, len(df_test) / (len(df_train) + len(df_test)) * 100],
    'No-Show Rate (%)': [df_train['no_show'].mean() * 100, df_test['no_show'].mean() * 100]
})
print("\\nSplit Summary Table:")
print(split_summary.to_string(index=False))
"""))

    # Section 3: Point-in-Time History Demonstration
    cells.append(nbf.v4.new_markdown_cell("""## 3. Patient History Feature Demonstration (Zero-Leakage Audit)
To verify that no future information leaks into the past, let's examine a repeat patient with multiple appointments.
Notice how `prior_appointments`, `prior_noshows`, and `prior_noshow_rate` update strictly on past visits.
"""))
    cells.append(nbf.v4.new_code_cell("""# Find a repeat patient with at least 3 visits
repeat_pids = df_train[df_train['prior_appointments'] >= 2]['patient_id'].unique()
example_pid = repeat_pids[0]

example_history = df_train[df_train['patient_id'] == example_pid][
    ['patient_id', 'appointment_day', 'no_show', 'prior_appointments', 'prior_noshows', 'prior_noshow_rate', 'days_since_last_appointment', 'is_first_appointment']
].sort_values(by='appointment_day')

print(f"Chronological point-in-time progression for Patient ID: {example_pid}")
example_history
"""))

    # Section 4: Signal Analysis of Historical Features
    cells.append(nbf.v4.new_markdown_cell("## 4. Signal Analysis: First-Time vs Returning Patients & Prior No-Show Rates"))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. First-time vs Returning
first_vs_return = df_train.groupby('is_first_appointment')['no_show'].agg(['count', 'mean'])
first_vs_return['mean'] = first_vs_return['mean'] * 100
first_labels = ['Returning Patient (0)', 'First-Time Patient (1)']

sns.barplot(x=first_labels, y=first_vs_return['mean'].values, ax=axes[0], hue=first_labels, palette=['#2b83ba', '#fdae61'], legend=False)
axes[0].set_title("No-Show Rate: First-Time vs Returning Patients")
axes[0].set_ylabel("Observed No-Show Rate (%)")
for i, v in enumerate(first_vs_return['mean'].values):
    axes[0].text(i, v + 0.5, f"{v:.2f}%\\n(N={first_vs_return['count'].iloc[i]:,})", ha='center', fontweight='bold')
axes[0].set_ylim(0, 26)

# 2. Returning patients by prior no-show rate
repeat_train = df_train[df_train['is_first_appointment'] == 0].copy()
rate_bins = [-0.01, 0.0, 0.25, 0.50, 0.75, 1.0]
rate_labels = ['0% (Never missed)', '1-25%', '26-50%', '51-75%', '76-100% (Always missed)']
repeat_train['rate_group'] = pd.cut(repeat_train['prior_noshow_rate'], bins=rate_bins, labels=rate_labels)

rate_group_stats = repeat_train.groupby('rate_group', observed=False)['no_show'].agg(['count', 'mean'])
rate_group_stats['mean'] = rate_group_stats['mean'] * 100

sns.barplot(x=rate_group_stats.index, y=rate_group_stats['mean'].values, ax=axes[1], hue=rate_group_stats.index, palette='YlOrRd', legend=False)
axes[1].set_title("No-Show Rate by Historical Miss Ratio (prior_noshow_rate)")
axes[1].set_xlabel("Historical Miss Ratio")
axes[1].set_ylabel("Observed No-Show Rate (%)")
axes[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(rate_group_stats['mean'].values):
    axes[1].text(i, v + 0.8, f"{v:.1f}%", ha='center', fontweight='bold')
axes[1].set_ylim(0, 48)

plt.tight_layout()
plt.show()
"""))

    # Section 5: Feature Correlations with Target
    cells.append(nbf.v4.new_markdown_cell("## 5. Linear Feature Correlations with Target (`no_show`)"))
    cells.append(nbf.v4.new_code_cell("""feature_cols = [
    'age', 'gender', 'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'has_handicap', 'sms_received',
    'lead_days', 'same_day_booking', 'appointment_dow', 'scheduled_dow', 'appointment_month',
    'prior_appointments', 'prior_noshows', 'prior_noshow_rate', 'days_since_last_appointment', 'is_first_appointment'
]

corrs = df_train[feature_cols].apply(lambda col: col.corr(df_train['no_show'])).sort_values()

plt.figure(figsize=(10, 6))
colors = ['#d7191c' if v > 0 else '#2c7bb6' for v in corrs.values]
corrs.plot(kind='barh', color=colors)
plt.title("Feature Correlations with Target (no_show) on Training Set", fontsize=13, fontweight='bold')
plt.xlabel("Pearson Correlation (r)")
plt.axvline(0, color='black', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Feature correlations with no_show:")
print(corrs.round(3))
"""))

    # Section 6: Summary & Handoff to Modeling
    cells.append(nbf.v4.new_markdown_cell("""## 6. Milestone Summary & Step 4 Readiness

| Feature Category | Features Included | Modeling Readiness |
| :--- | :--- | :--- |
| **Static Demographics & Clinical** | `age`, `gender`, `scholarship`, `hypertension`, `diabetes`, `alcoholism`, `has_handicap`, `sms_received` | Clean binary & numeric scales ready for both linear & tree-based models. |
| **Temporal & Operational** | `lead_days`, `same_day_booking`, `appointment_dow`, `scheduled_dow`, `appointment_month` | Captures booking horizons and day-of-week volume without leakage. |
| **Patient Behavioral History** | `prior_appointments`, `prior_noshows`, `prior_noshow_rate`, `days_since_last_appointment`, `is_first_appointment` | Strictly point-in-time; provides strongest predictive separation. |

### 🚀 Next Step:
Proceed to **`steps/04_modeling/`** to benchmark baseline estimators, Logistic Regression, Random Forest, and LightGBM/XGBoost on the temporal split!
"""))

    nb.cells = cells
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Jupyter Notebook successfully built at: {NOTEBOOK_PATH}")

if __name__ == '__main__':
    build_notebook()
