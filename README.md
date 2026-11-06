# Healthcare Appointment No-Show Risk & Capacity Optimization

## 🎯 Business Problem
**Which scheduled appointments are at high risk of a patient not attending (no-show), and what clinical, operational, and socioeconomic factors contribute to that risk?**

In outpatient healthcare provider networks, missed appointments (no-shows) disrupt clinical workflows, leave high-cost provider capacity idle, and compromise preventative care continuity.

* **Prediction Point:** At booking time (Day 0) when an appointment is scheduled.
* **Target:** `no_show = 1` (patient missed scheduled visit), `0` (patient attended).

---

## 📁 Repository Structure

```
PATIENT CHURN PREDICTION/
├── data/
│   ├── raw/                  # Original raw dataset (KaggleV2-May-2016.csv)
│   └── processed/            # Audited, cleaned dataset and feature partitions
│       ├── cleaned_patient_data.csv    # Anomaly-free dataset (110,521 rows)
│       ├── train_features.csv          # Train partition: Apr 29 - May 19 (63,532 rows)
│       ├── val_features.csv            # Validation partition: May 20 - May 31 (20,539 rows)
│       ├── dev_features.csv            # Full Dev partition: Apr 29 - May 31 (84,071 rows)
│       ├── test_features.csv           # Unseen Test partition: Jun 1 - Jun 8 (26,450 rows)
│       ├── feature_metadata.json       # Feature catalog and split architecture
│       ├── leakage_audit_report.json   # Quantified point-in-time leakage audit report
│       ├── model_benchmark_summary.csv # Model performance benchmark table
│       └── model_benchmark_results.json# Detailed benchmark metrics and ablation study
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
│   ├── 03_feature_engineering/ # Step 3: Leakage-safe feature engineering & chronological split
│   │   ├── audit_leakage_and_history.py # Standalone point-in-time leakage audit & test script
│   │   ├── engineer_features.py         # Production pipeline generating train/val/test feature sets
│   │   ├── build_features_notebook.py   # Generator for feature engineering notebook
│   │   └── 02_feature_engineering.ipynb # Interactive verification & correlation notebook
│   │
│   ├── 04_modeling/            # Step 4: Model benchmarking (Baselines, Logistic Regression, RF, XGBoost)
│   │   ├── train_and_evaluate.py        # End-to-end training, PR-AUC benchmarking, ablation & calibration
│   │   ├── build_modeling_notebook.py   # Generator for modeling notebook
│   │   └── 03_model_benchmarking.ipynb  # Interactive ROC, PR, calibration curves & importance
│   │
│   ├── 05_evaluation/          # Step 5 & 6: Threshold optimization, Cost-utility curve, Error analysis
│   └── 06_shap_explainability/ # Step 7 & 8: SHAP risk drivers & Clinical intervention dashboard
│
├── models/                     # Serialized production pipelines
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── xgboost_calibrated.joblib
│
└── utils/                      # Helper scripts and environment diagnostics
    ├── check_dates.py
    └── test_imports.py
```

---

## 🛡️ Anti-Leakage & Experimental Design Architecture

1. **Prediction-Time Constraint (Day 0 Availability):**
   * Operational predictors are strictly restricted to information available when an appointment is booked.
   * `sms_received` is excluded from Day-0 predictors because reminders are dispatched days after scheduling.
   * Identifiers (`patient_id`, `appointment_id`) are excluded from model features.
2. **Strict Point-in-Time Patient History:**
   * An appointment outcome can only enter a patient's historical features if its appointment date has already passed prior to the current scheduling date (`appointment_day < scheduled_date`).
   * Eliminates 27,525 contaminated records (24.9% of all appointments) caused by naive cumulative operations.
3. **Chronological 3-Way Partitioning:**
   * **Training (Early Dev):** April 29 – May 19, 2016 (63,532 appointments, 20.95% no-show rate).
   * **Validation (Late Dev):** May 20 – May 31, 2016 (20,539 appointments, 20.08% no-show rate). Used for model selection, feature ablation, threshold optimization, and probability calibration.
   * **Final Test (Unseen Future):** June 1 – June 8, 2016 (26,450 appointments, 18.46% no-show rate). Preserved for final evaluation.

---

## 📊 Benchmark Results (Validation Set)

| Model | PR-AUC | ROC-AUC | Recall | Precision | F1-Score | Accuracy | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Weighted)** | **0.3597** | **0.7385** | 0.8521 | 0.3071 | **0.4515** | 0.5844 | 0.2166 | 0.2503 |
| **XGBoost (Unweighted)** | 0.3584 | 0.7382 | 0.0347 | 0.4719 | 0.0646 | 0.7984 | **0.1431** | **0.0240** |
| **XGBoost (Calibrated)** | 0.3584 | 0.7382 | 0.0524 | 0.4244 | 0.0932 | 0.7955 | 0.1439 | 0.0359 |
| **Random Forest** | 0.3546 | 0.7365 | 0.8586 | 0.3055 | 0.4507 | 0.5797 | 0.2123 | 0.2467 |
| **Logistic Regression** | 0.3524 | 0.7302 | **0.8887** | 0.3005 | 0.4491 | 0.5623 | 0.2209 | 0.2602 |
| **Majority Baseline** | 0.2008 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | **0.7992** | 0.2008 | 0.2008 |
| **Stratified Baseline** | 0.1996 | 0.4963 | 0.2027 | 0.1951 | 0.1988 | 0.6720 | 0.3280 | 0.3280 |

---

## 🚀 Execution Guide

```bash
# Step 1: Run Data Cleaning
python steps/01_data_audit/step1_clean_and_save.py

# Step 1.5: Run Point-in-Time Leakage Audit
python steps/03_feature_engineering/audit_leakage_and_history.py

# Step 2: Generate All Publication EDA Figures
python steps/02_eda/generate_eda_figures.py

# Step 3: Run Prediction-Time Feature Engineering & Chronological Split
python steps/03_feature_engineering/engineer_features.py

# Step 4: Run Model Benchmarking, Calibration & Ablation
python steps/04_modeling/train_and_evaluate.py
```
