import nbformat as nbf
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, 'steps', '04_modeling', '03_model_benchmarking.ipynb')

def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Framing
    cells.append(nbf.v4.new_markdown_cell(r"""# 🤖 Step 4: Model Benchmarking & Pipeline Evaluation

### **Objective:**
Evaluate candidate model architectures on the **unseen future June test set** (temporal validation) to establish:
1. **The Deception of Accuracy:** Why a naive 81.5% accuracy baseline is completely useless to a healthcare clinic.
2. **Model Progression:**
   * **Baseline Models:** Majority Class Dummy & Stratified Random Guessing.
   * **Model 1 — Logistic Regression:** Interpretable linear baseline with Standard Scaling and One-Hot Encoding for nominal days of the week.
   * **Model 2 — Random Forest:** Non-linear ensemble capturing complex threshold effects.
   * **Model 3 — XGBoost:** Gradient-boosted decision trees optimizing the Precision-Recall tradeoff under severe class imbalance (~4:1).
3. **Ablation Study:** Quantifying whether `same_day_booking` provides incremental lift over raw `lead_days`.
"""))

    # Section 1: Load Data & Pre-Trained Pipelines
    cells.append(nbf.v4.new_markdown_cell("## 1. Load Unseen Temporal Test Set & Pre-Trained Pipelines"))
    cells.append(nbf.v4.new_code_cell("""import os
import joblib
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress minor library transform warnings
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 5)

test_path = os.path.join('..', '..', 'data', 'processed', 'test_features.csv')
models_dir = os.path.join('..', '..', 'models')
summary_path = os.path.join('..', '..', 'data', 'processed', 'model_benchmark_summary.csv')

df_test = pd.read_csv(test_path)
meta_cols = ['appointment_id', 'patient_id', 'scheduled_day', 'appointment_day', 'sms_received', 'no_show']
feature_cols = [c for c in df_test.columns if c not in meta_cols]

X_test = df_test[feature_cols]
y_test = df_test['no_show'].values

print(f"Loaded Unseen Test Set: {len(df_test):,} records | Baseline No-Show Rate: {y_test.mean()*100:.2f}%")
print(f"Features: {feature_cols}")

# Load pipelines
pipe_lr = joblib.load(os.path.join(models_dir, 'logistic_regression.joblib'))
pipe_rf = joblib.load(os.path.join(models_dir, 'random_forest.joblib'))
pipe_xgb = joblib.load(os.path.join(models_dir, 'xgboost.joblib'))
print("Successfully loaded: Logistic Regression, Random Forest, and XGBoost pipelines!")
"""))

    # Section 2: Executive Benchmark Results Table
    cells.append(nbf.v4.new_markdown_cell("## 2. Model Benchmark Summary Table\nEvaluated strictly on the unseen chronological future test set (June 1 - June 8, 2016)."))
    cells.append(nbf.v4.new_code_cell("""df_summary = pd.read_csv(summary_path)
print("Model Performance Matrix (Ranked by PR-AUC):")
df_summary
"""))

    # Section 3: Precision-Recall Curves (Primary Metric for Class Imbalance)
    cells.append(nbf.v4.new_markdown_cell("""## 3. Precision-Recall Curves (Primary Metric)
In healthcare no-show prediction where 80% attend and only ~20% miss, **PR-AUC** measures the model's true capability to surface at-risk appointments without being fooled by the majority class.
"""))
    cells.append(nbf.v4.new_code_cell("""models_dict = {
    'Logistic Regression': pipe_lr,
    'Random Forest': pipe_rf,
    'XGBoost': pipe_xgb
}

plt.figure(figsize=(9, 6))

# Plot baseline prevalence
baseline_prevalence = y_test.mean()
plt.axhline(baseline_prevalence, color='black', linestyle='--', label=f'Baseline Prevalence ({baseline_prevalence*100:.1f}%)')

colors = {'Logistic Regression': '#2b83ba', 'Random Forest': '#fdae61', 'XGBoost': '#d7191c'}

for name, model in models_dict.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    plt.plot(rec, prec, label=f'{name} (PR-AUC = {pr_auc:.3f})', color=colors[name], lw=2.2)

plt.title('Precision-Recall Curves on Unseen Future Appointments', fontsize=13, fontweight='bold')
plt.xlabel('Recall (Fraction of Missed Appointments Caught)')
plt.ylabel('Precision (Fraction of Flagged Appointments that Truly No-Show)')
plt.legend(loc='upper right', frameon=True)
plt.xlim(0, 1)
plt.ylim(0, 0.7)
plt.tight_layout()
plt.show()
"""))

    # Section 4: ROC Curves
    cells.append(nbf.v4.new_markdown_cell("## 4. Receiver Operating Characteristic (ROC) Curves"))
    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9, 6))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Chance (AUC = 0.50)')

for name, model in models_dict.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (ROC-AUC = {auc_val:.3f})', color=colors[name], lw=2.2)

plt.title('ROC Curves on Unseen Future Appointments', fontsize=13, fontweight='bold')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall / Sensitivity)')
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()
"""))

    # Section 5: Confusion Matrices Side-by-Side
    cells.append(nbf.v4.new_markdown_cell("## 5. Confusion Matrices (Default 0.5 Decision Threshold)"))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for idx, (name, model) in enumerate(models_dict.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Show (0)', 'No-Show (1)'])
    disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
    axes[idx].set_title(f"{name}\\nRecall: {(cm[1,1]/(cm[1,0]+cm[1,1]))*100:.1f}% | Acc: {(cm[0,0]+cm[1,1])/cm.sum()*100:.1f}%")

plt.tight_layout()
plt.show()
"""))

    # Section 6: Interpretability - Logistic Regression Odds Ratios
    cells.append(nbf.v4.new_markdown_cell("""## 6. Interpretability: Logistic Regression Odds Ratios
Because Logistic Regression is trained with standardized continuous features and one-hot encoded nominal days of week, exponentiating the learned coefficients ($e^{\\beta}$) yields the exact directional **Odds Ratios**.
"""))
    cells.append(nbf.v4.new_code_cell("""# Extract feature names after ColumnTransformer preprocessing
cat_names = list(pipe_lr.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(['appointment_dow', 'scheduled_dow', 'appointment_month']))
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

    # Section 7: XGBoost Feature Importance
    cells.append(nbf.v4.new_markdown_cell("## 7. XGBoost Feature Importance (Gain)"))
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

    # Section 8: Ablation Finding & Step 5 Readiness
    cells.append(nbf.v4.new_markdown_cell(r"""## 8. Ablation Study & Step 5 Handoff

### **Ablation Key Finding:**
* **XGBoost with `same_day_booking`**: PR-AUC = `0.3646`
* **XGBoost without `same_day_booking`**: PR-AUC = `0.3637`
* **Delta**: `+0.0009` (negligible). As hypothesized, gradient boosted trees split naturally on `lead_days == 0`.

### 🎯 Next Milestone: Step 5 — Cost-Sensitive Threshold Optimization
Currently, models use an arbitrary `0.50` decision threshold.  
In an outpatient clinic:
* **Cost of False Negative (Missed No-Show):** \$200 (idle physician hour, lost capacity).
* **Cost of False Positive (Unnecessary Reminder/Call):** \$10 (staff outreach time).

In **Step 5**, we will construct an economic cost matrix and find the exact optimal probability threshold $p^*$ that maximizes clinical cost savings!
"""))

    nb.cells = cells
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Jupyter Notebook successfully built at: {NOTEBOOK_PATH}")

if __name__ == '__main__':
    build_notebook()
