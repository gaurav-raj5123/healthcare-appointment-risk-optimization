# Healthcare Appointment No-Show Risk & Capacity Optimization

## 🎯 Business Problem
**Which scheduled appointments are at high risk of a patient not attending (no-show), and what clinical, operational, and socioeconomic factors contribute to that risk?**

In outpatient provider networks, missed appointments (no-shows) disrupt clinical workflows, leave high-cost provider capacity idle, and compromise preventative care continuity.

---

## 📁 Repository Structure

```
PATIENT CHURN PREDICTION/
├── data/
│   ├── raw/                  # Original raw dataset (KaggleV2-May-2016.csv)
│   └── processed/            # Audited, cleaned dataset (cleaned_patient_data.csv)
│
├── steps/
│   ├── 01_data_audit/        # Step 1 & 1.5: Schema inspection, anomaly cleaning, leakage audit
│   │   ├── step1_data_understanding.py
│   │   ├── step1_clean_and_save.py
│   │   └── step1_5_data_audit.py
│   │
│   ├── 02_eda/               # Step 2: Statistical association, odds ratios, SMS confounder analysis
│   │   ├── 01_eda.ipynb      # Complete 9-stage EDA notebook with all figures & statistical tests
│   │   ├── generate_eda_figures.py # Standalone script generating high-resolution publication figures
│   │   └── figures/          # Exported high-resolution charts
│   │
│   ├── 03_feature_engineering/ # Step 3: Prediction-time feature engineering & temporal split
│   │   ├── engineer_features.py      # Production pipeline generating train/test feature sets
│   │   ├── build_features_notebook.py # Generator for feature engineering notebook
│   │   └── 02_feature_engineering.ipynb # Interactive verification & correlation notebook
│   │
│   ├── 04_modeling/            # Step 4 & 5: Baseline, Logistic Regression, Random Forest, XGBoost
│   ├── 05_evaluation/          # Step 6, 7 & 8: PR-AUC, Threshold optimization, Error analysis
│   └── 06_shap_explainability/ # Step 9 & 10: SHAP risk drivers & Clinical intervention dashboard
│
└── utils/                    # Helper scripts and environment diagnostics
    ├── check_dates.py
    └── test_imports.py
```

---

## 🚀 Execution Guide
All step scripts can be run directly from the project root:

```bash
# Step 1: Run Data Cleaning
python steps/01_data_audit/step1_clean_and_save.py

# Step 1.5: Run Leakage & Patient Audit
python steps/01_data_audit/step1_5_data_audit.py

# Step 2: Generate All Publication EDA Figures
python steps/02_eda/generate_eda_figures.py

# Step 3: Run Prediction-Time Feature Engineering & Temporal Split
python steps/03_feature_engineering/engineer_features.py

# Or explore interactively:
# Open steps/02_eda/01_eda.ipynb or steps/03_feature_engineering/02_feature_engineering.ipynb
```
