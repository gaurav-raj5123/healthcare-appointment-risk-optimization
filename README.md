# Healthcare Patient Churn & Care Disengagement Prediction

## 🎯 Business Problem
**Which patients are at high risk of disengaging from a healthcare provider, and what clinical, operational, and socioeconomic factors contribute to that risk?**

In outpatient provider networks, patient churn manifests as missed appointments and broken continuity of care, which accelerates chronic disease complications and costs health systems millions in idle clinical capacity.

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
│   │   ├── step2_eda_and_statistics.py
│   │   └── analyze_sms.py
│   │
│   ├── 03_feature_engineering/ # Step 3: Causal rolling features, operational & clinical encodings
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

# Step 2: Run Statistical EDA & Odds Ratio Analysis
python steps/02_eda/step2_eda_and_statistics.py

# Step 2: Run SMS Confounder Analysis
python steps/02_eda/analyze_sms.py
```
