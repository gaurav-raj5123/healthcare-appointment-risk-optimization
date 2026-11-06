import nbformat as nbf
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, 'steps', '04_modeling', '03_model_benchmarking.ipynb')

def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Framing
    cells.append(nbf.v4.new_markdown_cell(r"""# 🤖 Step 4: Leakage-Safe Model Benchmarking & Pipeline Evaluation

### **Objective:**
Evaluate candidate model architectures on the **chronological validation set** (May 20 - May 31, 2016) with strict point-in-time features to establish:
1. **The Deception of Accuracy:** Why a naive 80% accuracy baseline is completely useless to a healthcare clinic (Recall = 0.0%).
2. **Model Progression:**
   * **Baseline Models:** Majority Class Dummy & Stratified Prior Random Guessing.
   * **Model 1 — Logistic Regression:** Interpretable linear baseline with Standard Scaling and One-Hot Encoding for nominal days of the week.
   * **Model 2 — Random Forest:** Non-linear ensemble capturing complex threshold effects.
   * **Model 3 — XGBoost:** Gradient-boosted decision trees optimizing the Precision-Recall tradeoff under severe class imbalance (~4:1).
3. **Probability Calibration & Brier Score:** Why Brier Score is a proper scoring rule rather than pure calibration, and why uncalibrated class weighting shifts probabilities away from base rates.
4. **Ablation Studies:**
   * Quantifying incremental value of `same_day_booking` over continuous `lead_days`.
   * Quantifying incremental lift across feature groups (Static vs Static+Temporal vs Full).
"""))

    # Section 1: Load Data & Pre-Trained Pipelines
    cells.append(nbf.v4.new_markdown_cell("## 1. Load Chronological Validation Set & Pre-Trained Pipelines"))
    cells.append(nbf.v4.new_code_cell("""import os
import joblib
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 5)

val_path     = os.path.join('..', '..', 'data', 'processed', 'val_features.csv')
test_path    = os.path.join('..', '..', 'data', 'processed', 'test_features.csv')
models_dir   = os.path.join('..', '..', 'models')
summary_path = os.path.join('..', '..', 'data', 'processed', 'model_benchmark_summary.csv')
json_path    = os.path.join('..', '..', 'data', 'processed', 'model_benchmark_results.json')

df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path)

with open(json_path, 'r', encoding='utf-8') as f:
    bench_data = json.load(f)

meta_cols = ['appointment_id', 'patient_id', 'scheduled_day', 'appointment_day', 'sms_received', 'no_show']
feature_cols = [c for c in df_val.columns if c not in meta_cols]

X_val = df_val[feature_cols]
y_val = df_val['no_show'].values

X_test = df_test[feature_cols]
y_test = df_test['no_show'].values

print(f"Validation Set: {len(df_val):,} appointments | No-Show Rate: {y_val.mean()*100:.2f}%")
print(f"Test Set (Exploratory): {len(df_test):,} appointments | No-Show Rate: {y_test.mean()*100:.2f}%")
print(f"Features Verified ({len(feature_cols)}): {feature_cols}")

# Load pipelines
pipe_lr = joblib.load(os.path.join(models_dir, 'logistic_regression.joblib'))
pipe_rf = joblib.load(os.path.join(models_dir, 'random_forest.joblib'))
pipe_xgb = joblib.load(os.path.join(models_dir, 'xgboost.joblib'))
pipe_xgb_cal = joblib.load(os.path.join(models_dir, 'xgboost_calibrated.joblib'))
print("Successfully loaded pre-trained pipelines: LR, RF, XGBoost (Weighted), and Calibrated XGBoost!")
"""))

    # Section 2: Executive Benchmark Results Table
    cells.append(nbf.v4.new_markdown_cell("## 2. Validation Benchmark Performance Table\nEvaluated on the chronological validation set (May 20 - May 31, 2016)."))
    cells.append(nbf.v4.new_code_cell("""df_summary = pd.read_csv(summary_path)
print("Validation Benchmark Performance Table:")
df_summary
"""))

    # Section 3: Precision-Recall Curves
    cells.append(nbf.v4.new_markdown_cell("""## 3. Precision-Recall Curves (Primary Evaluation Metric)
With ~4:1 class imbalance (20% no-show prevalence), **PR-AUC (Average Precision)** is the definitive metric separating clinical utility from majority-class guessing.
"""))
    cells.append(nbf.v4.new_code_cell("""models_dict = {
    'Logistic Regression': pipe_lr,
    'Random Forest': pipe_rf,
    'XGBoost (Weighted)': pipe_xgb
}

plt.figure(figsize=(9, 6))
baseline_prevalence = y_val.mean()
plt.axhline(baseline_prevalence, color='black', linestyle='--', label=f'Majority Baseline ({baseline_prevalence*100:.1f}%)')

colors = {'Logistic Regression': '#2b83ba', 'Random Forest': '#fdae61', 'XGBoost (Weighted)': '#d7191c'}

for name, model in models_dict.items():
    y_prob = model.predict_proba(X_val)[:, 1]
    prec, rec, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    plt.plot(rec, prec, label=f'{name} (PR-AUC = {pr_auc:.4f})', color=colors[name], lw=2.2)

plt.title('Precision-Recall Curves on Chronological Validation Set', fontsize=13, fontweight='bold')
plt.xlabel('Recall (Fraction of No-Shows Caught)')
plt.ylabel('Precision (Fraction of Flagged Appointments that Miss)')
plt.legend(loc='upper right', frameon=True)
plt.xlim(0, 1)
plt.ylim(0, 0.75)
plt.tight_layout()
plt.show()
"""))

    # Section 4: ROC Curves
    cells.append(nbf.v4.new_markdown_cell("## 4. Receiver Operating Characteristic (ROC) Curves"))
    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9, 6))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Chance (AUC = 0.50)')

for name, model in models_dict.items():
    y_prob = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc_val = roc_auc_score(y_val, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (ROC-AUC = {auc_val:.4f})', color=colors[name], lw=2.2)

plt.title('ROC Curves on Chronological Validation Set', fontsize=13, fontweight='bold')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall / Sensitivity)')
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()
"""))

    # Section 5: Probability Calibration & Reliability Curves
    cells.append(nbf.v4.new_markdown_cell("""## 5. Probability Calibration & Reliability Curves (Brier Score Analysis)
Brier Score is a proper scoring rule decomposing into Calibration (Reliability) and Discrimination (Resolution).
An unweighted model yields well-calibrated probabilities close to empirical rates, whereas `scale_pos_weight` shifts the distribution upward to elevate default recall.
"""))
    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

y_prob_xgb = pipe_xgb.predict_proba(X_val)[:, 1]
y_prob_cal = pipe_xgb_cal.predict_proba(X_val)[:, 1]

prob_true_xgb, prob_pred_xgb = calibration_curve(y_val, y_prob_xgb, n_bins=10)
prob_true_cal, prob_pred_cal = calibration_curve(y_val, y_prob_cal, n_bins=10)

brier_xgb = brier_score_loss(y_val, y_prob_xgb)
brier_cal = brier_score_loss(y_val, y_prob_cal)

plt.plot(prob_pred_xgb, prob_true_xgb, 's-', color='#d7191c', label=f'XGBoost Weighted (Brier = {brier_xgb:.4f})')
plt.plot(prob_pred_cal, prob_true_cal, 'o-', color='#2b83ba', label=f'XGBoost Calibrated (Brier = {brier_cal:.4f})')

plt.title('Reliability Diagram (Calibration Curves)', fontsize=13, fontweight='bold')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Observed Fraction of No-Shows')
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()
"""))

    # Section 6: Feature Group & same_day_booking Ablation
    cells.append(nbf.v4.new_markdown_cell("""## 6. Ablation Studies (Validated on Holdout Set)
1. **Incremental Value of `same_day_booking`:** Does adding a binary flag derived from `lead_days == 0` add predictive power to tree models?
2. **Feature Group Incremental Lift:** Measuring performance progression from Static -> Static+Temporal -> Full (with Patient History).
"""))
    cells.append(nbf.v4.new_code_cell("""ab_sameday = bench_data['ablation_same_day_booking']
ab_groups = bench_data['ablation_feature_groups']

print("--- Ablation 1: same_day_booking ---")
print(f"XGBoost WITH same_day_booking:    PR-AUC = {ab_sameday['pr_auc_with']:.4f}")
print(f"XGBoost WITHOUT same_day_booking: PR-AUC = {ab_sameday['pr_auc_without']:.4f}")
print(f"Delta: {ab_sameday['delta_pr_auc']:+.4f} -> {ab_sameday['conclusion']}")

print("\\n--- Ablation 2: Feature Group Incremental Lift ---")
ab_df = pd.DataFrame([
    {'Group': 'A: Static Features Only (7 feat)', 'PR_AUC': ab_groups['group_a_static']['pr_auc'], 'ROC_AUC': ab_groups['group_a_static']['roc_auc'], 'Incremental_Lift': '-'},
    {'Group': 'B: Static + Temporal (12 feat)', 'PR_AUC': ab_groups['group_b_static_temporal']['pr_auc'], 'ROC_AUC': ab_groups['group_b_static_temporal']['roc_auc'], 'Incremental_Lift': f"+{ab_groups['group_b_static_temporal']['lift_over_static']:.4f}"},
    {'Group': 'C: Full Model (+ History, 17 feat)', 'PR_AUC': ab_groups['group_c_full']['pr_auc'], 'ROC_AUC': ab_groups['group_c_full']['roc_auc'], 'Incremental_Lift': f"+{ab_groups['group_c_full']['lift_over_temporal']:.4f}"}
])
ab_df
"""))

    # Section 7: Interpretability - Logistic Regression Odds Ratios
    cells.append(nbf.v4.new_markdown_cell("""## 7. Interpretability: Logistic Regression Odds Ratios
Because continuous features are standardized and days of week are one-hot encoded, exponentiating the learned coefficients ($e^{\\beta}$) provides directional **Odds Ratios**.
"""))
    cells.append(nbf.v4.new_code_cell("""cat_names = list(pipe_lr.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(['appointment_dow', 'scheduled_dow', 'appointment_month']))
num_names = ['age', 'lead_days', 'prior_appointments', 'prior_noshows', 'prior_noshow_rate', 'days_since_last_appointment']
bin_names = ['gender', 'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'has_handicap', 'same_day_booking', 'is_first_appointment']
all_feature_names = cat_names + num_names + bin_names

coefs = pipe_lr.named_steps['clf'].coef_[0]
odds_ratios = np.exp(coefs)

df_or = pd.DataFrame({
    'Feature': all_feature_names,
    'Coefficient': coefs,
    'Odds_Ratio': odds_ratios
}).sort_values(by='Odds_Ratio', ascending=False).reset_index(drop=True)

plt.figure(figsize=(10, 7))
colors_or = ['#d7191c' if v > 1.0 else '#2c7bb6' for v in df_or['Odds_Ratio']]
plt.barh(df_or['Feature'], df_or['Odds_Ratio'], color=colors_or)
plt.axvline(1.0, color='black', linestyle='--', alpha=0.8, label='Neutral Odds (OR = 1.0)')
plt.title('Logistic Regression Feature Odds Ratios (Directional Risk Drivers)', fontsize=13, fontweight='bold')
plt.xlabel('Odds Ratio (exp(beta))')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

print("Top 5 Risk Factors (OR > 1.0):")
print(df_or.head(5).to_string(index=False))
print("\\nTop 5 Protective / Adherence Factors (OR < 1.0):")
print(df_or.tail(5).to_string(index=False))
"""))

    # Section 8: XGBoost Feature Importance
    cells.append(nbf.v4.new_markdown_cell("## 8. XGBoost Feature Importance (Gain)"))
    cells.append(nbf.v4.new_code_cell("""xgb_model = pipe_xgb.named_steps['clf']
xgb_importances = xgb_model.feature_importances_

df_imp = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance_Gain': xgb_importances
}).sort_values(by='Importance_Gain', ascending=False).reset_index(drop=True)

plt.figure(figsize=(10, 6.5))
plt.barh(df_imp['Feature'].head(12)[::-1], df_imp['Importance_Gain'].head(12)[::-1], color='#2b5c8f')
plt.title('Top 12 Most Important Features in XGBoost (Gain)', fontsize=13, fontweight='bold')
plt.xlabel('Relative Feature Importance (Gain)')
plt.tight_layout()
plt.show()

print("Top 10 Feature Importances in XGBoost:")
print(df_imp.head(10).to_string(index=False))
"""))

    # Section 9: Handoff to Step 5
    cells.append(nbf.v4.new_markdown_cell(r"""## 9. Handoff to Step 5: Cost-Sensitive Threshold Optimization

### **Key Conclusions from Benchmarking:**
1. **XGBoost achieves the highest discriminative performance** on unseen validation appointments (PR-AUC = 0.3597, ROC-AUC = 0.7385), outperforming Logistic Regression (0.3524) and Random Forest (0.3546).
2. **Class weighting vs Calibration trade-off:** `scale_pos_weight` inflates raw output probabilities to roughly balance false positives and false negatives at 0.50 threshold, creating higher recall but worsening raw Brier Score. Unweighted/calibrated models produce well-calibrated probabilities (Brier = 0.1431 vs baseline 0.2008).
3. **`same_day_booking` is redundant for tree models** (delta PR-AUC = +0.0024).
4. **Temporal and Patient History features** provide almost all predictive lift over demographic baselines.

### 🎯 Next Milestone: Step 5 — Cost-Sensitive Threshold Optimization
In clinical operations:
* **Cost of False Negative (Missed No-Show):** \$200 (idle provider slot, lost clinic capacity).
* **Cost of False Positive (Unnecessary Reminder/Call):** \$10 (staff outreach cost).

In **Step 5**, we will sweep decision thresholds $p \in [0.05, 0.95]$ on calibrated probabilities to determine the exact optimal operational cutoff minimizing total expected clinical loss!
"""))

    nb.cells = cells
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Jupyter Notebook successfully built at: {NOTEBOOK_PATH}")

if __name__ == '__main__':
    build_notebook()
