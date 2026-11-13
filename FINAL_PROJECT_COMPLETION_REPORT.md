# Healthcare Appointment No-Show Prediction & Capacity Optimization
## Final Milestone Completion, Technical Audit & Project Freeze Report

**Date:** November 11, 2026  
**Project Status:** COMPLETE & FROZEN  
**Selected Champion Model:** Unweighted XGBoost (`models/xgboost_unweighted.joblib`)  
**Lowest-Cost Evaluated Decision Threshold:** $0.15$ (Recall: $93.62\%$, Modeled Financial Savings: $\$98,920$ under illustrative assumptions)  
**Interpretability Framework:** Dual-Space SHAP (Raw-Margin & Interventional Probability Space)  

---

## 1. Project Completion Summary

The Healthcare Appointment No-Show Prediction and Capacity Optimization project has successfully reached its final milestone. All engineering stages—from raw data auditing to game-theoretic SHAP explainability—have been implemented, mathematically verified, end-to-end executed, and documented to reproducible, interview-defensible standards.

The project demonstrates an end-to-end, reproducible machine learning prototype with zero violations detected by the implemented point-in-time leakage audit, designed to evaluate outpatient appointment non-attendance risk at booking time. The project is delivered as a defensible portfolio piece with clean code, reproducible notebooks, automated testing scripts, visual documentation, a 5-slide stakeholder presentation, and an interview preparation guide.

---

## 2. Final Repository Structure

```
PATIENT NO-SHOW PREDICTION/
├── .gitignore                                  # Standardized git ignore rules
├── requirements.txt                            # Pinned Python package dependencies
├── README.md                                   # Root documentation (17 comprehensive sections)
├── INTERVIEW_PREPARATION.md                    # Technical defense guide (pitches + 13 Q&As)
├── FINAL_PROJECT_COMPLETION_REPORT.md          # Formal audit & milestone sign-off (this document)
├── presentation.pptx                           # Executive presentation (5 slides, 16:9 widescreen)
├── generate_presentation.py                    # Standalone script for generating presentation.pptx
│
├── data/
│   ├── raw/
│   │   └── KaggleV2-May-2016.csv              # Original raw Kaggle dataset (110,527 rows)
│   └── processed/
│       ├── cleaned_patient_data.csv            # Cleaned, anomaly-free dataset (110,521 rows)
│       ├── train_features.csv                  # Chronological train set: Apr 29 - May 19 (63,532 rows)
│       ├── val_features.csv                    # Chronological validation set: May 20 - May 31 (20,539 rows)
│       ├── test_features.csv                   # Chronological holdout set: Jun 1 - Jun 8 (26,450 rows)
│       ├── dev_features.csv                    # Combined train + val development set (84,071 rows)
│       ├── feature_metadata.json               # Schema metadata, feature types, and chronological dates
│       ├── leakage_audit_report.json           # Quantified audit report (27,525 contaminated rows remediated)
│       ├── model_benchmark_summary.csv         # Full model benchmarking evaluation metrics
│       ├── model_benchmark_results.json        # Serialized benchmark scores and feature ablation data
│       ├── step5_threshold_comparison.csv      # Complete threshold-by-threshold financial cost analysis
│       ├── val_predictions_xgb_unweighted.csv  # Out-of-fold validation predictions & actual outcomes
│       └── shap_summary_probability_space.csv  # Mean probability-space SHAP feature contributions
│
├── docs/
│   ├── architecture_diagram.png                # High-resolution (300 DPI) 9-stage pipeline architecture
│   └── presentation.pptx                       # Stakeholder presentation copy for docs folder
│
├── models/
│   ├── logistic_regression.joblib              # Serialized Logistic Regression pipeline
│   ├── random_forest.joblib                    # Serialized Random Forest pipeline
│   ├── xgboost.joblib                          # Serialized Class-Weighted XGBoost model
│   ├── xgboost_unweighted.joblib               # Selected Champion: Unweighted XGBoost model
│   └── xgboost_calibrated.joblib               # Calibrated XGBoost model
│
├── steps/
│   ├── 01_data_audit/
│   │   ├── step1_data_understanding.py         # Schema parsing and statistical distributions
│   │   ├── step1_clean_and_save.py             # Data cleansing and anomaly removal
│   │   └── step1_5_data_audit.py               # Patient appointment patterns & leakage inspection
│   │
│   ├── 02_eda/
│   │   ├── 01_eda.ipynb                        # Interactive 9-stage EDA notebook (executed, 0 errors)
│   │   ├── generate_eda_figures.py             # Publication figure generation script
│   │   └── figures/                            # 6 high-resolution exported EDA charts
│   │
│   ├── 03_feature_engineering/
│   │   ├── audit_leakage_and_history.py        # Automated point-in-time leakage audit script
│   │   ├── engineer_features.py                # Production Day-0 feature engineering pipeline
│   │   ├── build_features_notebook.py          # Generator for feature engineering notebook
│   │   └── 02_feature_engineering.ipynb        # Interactive feature verification & correlation notebook
│   │
│   ├── 04_modeling/
│   │   ├── train_and_evaluate.py               # Model benchmarking, calibration & ablation engine
│   │   ├── build_modeling_notebook.py          # Generator for modeling notebook
│   │   └── 03_model_benchmarking.ipynb         # Interactive benchmark comparison notebook
│   │
│   ├── 05_evaluation/
│   │   ├── evaluate_thresholds.py              # Threshold optimization & financial cost modeling script
│   │   ├── build_threshold_notebook.py         # Generator for threshold evaluation notebook
│   │   ├── 04_threshold_and_cost_optimization.ipynb # Interactive threshold optimization notebook
│   │   └── figures/                            # Exported confusion matrix charts
│   │
│   └── 06_shap_explainability/
│       ├── run_shap_analysis.py                # Dual-space SHAP analysis engine (raw & probability space)
│       ├── build_shap_notebook.py              # Generator for SHAP notebook
│       ├── 05_shap_interpretability.ipynb      # Interactive dual-space SHAP notebook
│       ├── README.md                           # Comprehensive technical SHAP documentation
│       └── figures/                            # Dual-space summary, beeswarm, waterfall, and dependence plots
│
└── utils/
    ├── check_dates.py                          # Date validation utility
    └── test_imports.py                         # Environment diagnostic utility
```

---

## 3. Deliverables Created & Updated in Final Milestone

| Deliverable | Path | Description & Quality Standard |
| :--- | :--- | :--- |
| **Root Documentation** | `README.md` | Comprehensive 17-section technical guide with formulas, metrics, architecture diagrams, and clinical interpretation. |
| **Architecture Diagram** | `docs/architecture_diagram.png` | 16x10 inch, 300 DPI high-resolution visual layout illustrating the complete 9-stage data and modeling lifecycle. |
| **Executive Presentation** | `presentation.pptx` & `docs/presentation.pptx` | Professional 5-slide 16:9 widescreen stakeholder slide deck with embedded high-resolution figures. |
| **Presentation Generator** | `generate_presentation.py` | Standalone, reproducible Python script using `python-pptx` to build the executive presentation. |
| **Interview Defense Guide** | `INTERVIEW_PREPARATION.md` | In-depth technical preparation document containing 30-sec pitch, 2-min overview, 5-min walkthrough, and 13 defensible interview Q&As. |
| **SHAP Documentation** | `steps/06_shap_explainability/README.md` | Dedicated technical guide detailing dual-space methodology, background dataset effective size, and additivity tolerances. |
| **Environment Specification** | `requirements.txt` | Cleaned and pinned dependencies (`scikit-learn`, `xgboost`, `shap`, `pandas`, `numpy`, `python-pptx`, `nbclient`, `ipykernel`). |
| **Completion Audit Report** | `FINAL_PROJECT_COMPLETION_REPORT.md` | Milestone completion report and technical freeze audit sign-off (this document). |

---

## 4. Verification Checks & Actual Results

### A. Environment & Dynamic Path Audit
* **Hardcoded Path Audit:** Complete regex search across all `.py`, `.ipynb`, `.json`, `.csv`, and `.md` files confirmed **zero hardcoded absolute machine paths** (e.g., `C:\Users\...`). All paths are dynamically resolved via `Path(__file__).resolve().parent` or notebook-relative paths.
* **Environment Compatibility:** Tested under Python 3.14 / 3.10+ with `scikit-learn==1.9.0`, `xgboost==3.4.1`, and `shap==0.52.0`. Unused dependencies (`streamlit`, `lightgbm`) were safely pruned.

### B. Notebook End-to-End Execution Verification
All 5 interactive project notebooks were executed sequentially from clean state using `nbclient.NotebookClient`:
1. `steps/02_eda/01_eda.ipynb`: Executed with **0 errors**. All 9 analytical stages and figures rendered.
2. `steps/03_feature_engineering/02_feature_engineering.ipynb`: Executed with **0 errors**. Correlation matrices, distribution plots, and point-in-time leakage checks verified.
3. `steps/04_modeling/03_model_benchmarking.ipynb`: Executed with **0 errors**. Benchmark tables, ROC curves, PR curves, calibration curves, and feature importance charts rendered.
4. `steps/05_evaluation/04_threshold_and_cost_optimization.ipynb`: Executed with **0 errors**. Threshold cost curves, F1/Recall trade-off plots, and confusion matrices ($0.50$ vs $0.15$) rendered.
5. `steps/06_shap_explainability/05_shap_interpretability.ipynb`: Executed with **0 errors**. Dual-space beeswarm plots, waterfall plots, dependence plots, and local patient explanations rendered.

### C. Step 6 Dual-Space SHAP Technical Audit
1. **Background Dataset & Effective Sample Size:**
   * **Rows Supplied:** 200 rows sampled from `data/processed/train_features.csv` (`random_state=42`).
   * **Internal Masker Behavior:** SHAP's `shap.maskers.Independent` internally caps reference samples at `max_samples=100`.
   * **Effective Background Size:** **100 observations** (subsampled deterministically via `sklearn.utils.shuffle(X, n_samples=100, random_state=0)`).
   * **Explainers Parity:** Verified that `explainer_raw` and `explainer_prob` evaluate on the **exact same 100 observations** ($0.00\times 10^0$ difference between internal background matrices).
2. **Prediction Invariance:**
   * Changing `clf.enable_categorical = False` on the public estimator resolved SHAP's C++ categorical check cleanly without monkey-patching.
   * Max absolute difference in `predict_proba`: **$0.00 \times 10^0$** (exact numerical invariance).
   * Max absolute difference in raw margins: **$0.00 \times 10^0$** (exact numerical invariance).
3. **Additivity & Reconstruction Errors (2,000 Validation Appointments):**
   * **Probability-Space Max Additivity Error:** $2.32 \times 10^{-7}$ ($< 10^{-4}$ tolerance — **PASSED**).
   * **Probability-Space Mean Additivity Error:** $4.29 \times 10^{-8}$.
   * **Raw-Margin Max Additivity Error:** $4.37 \times 10^{-6}$ ($< 10^{-4}$ tolerance — **PASSED**).
   * **Raw-Margin Mean Additivity Error:** $4.71 \times 10^{-7}$.
   * **Raw Sigmoid Probability Parity:** Reconstructed probabilities match pipeline outputs within $1.96 \times 10^{-8}$.
   * **Baseline Expected Probability:** $\phi_0 = 18.65\%$ ($0.18646$).

### D. Model Benchmarking & Metric Reconciliation
All results verified against `data/processed/model_benchmark_summary.csv`, `data/processed/model_benchmark_results.json`, and direct inference from serialized models on the chronological validation cohort (May 20–31, 2016; 20,539 appointments):

| Model Architecture | PR-AUC | ROC-AUC | Recall | Precision | F1-Score | Accuracy | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Weighted)** | **0.3597** | **0.7385** | 0.8521 | 0.3071 | **0.4515** | 0.5844 | 0.2166 | 0.2503 |
| **XGBoost (Sigmoid Calibrated, 5-Fold CV)** | 0.3588 | 0.7383 | 0.0909 | 0.4252 | 0.1498 | 0.7928 | 0.1445 | 0.0359 |
| **XGBoost (Unweighted - Champion)** | 0.3584 | 0.7382 | 0.0347 | **0.4719** | 0.0646 | **0.7984** | **0.1431** | **0.0240** |
| **Random Forest** | 0.3546 | 0.7365 | 0.8586 | 0.3055 | 0.4507 | 0.5797 | 0.2123 | 0.2467 |
| **Logistic Regression** | 0.3524 | 0.7302 | **0.8887** | 0.3005 | 0.4491 | 0.5623 | 0.2209 | 0.2602 |
| **Majority Baseline** | 0.2008 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.7992 | 0.2008 | 0.2008 |
| **Stratified Baseline** | 0.1996 | 0.4963 | 0.2027 | 0.1951 | 0.1988 | 0.6720 | 0.3280 | 0.3280 |

* **Discrepancy Investigation & Resolution:** An earlier documentation draft cited lower benchmark values (e.g. PR-AUC 0.3392 for class-weighted XGBoost, 0.3168 for Random Forest, 0.3016 for Logistic Regression). A comprehensive audit tracing `train_and_evaluate.py`, `03_model_benchmarking.ipynb`, `model_benchmark_summary.csv`, and direct out-of-fold inference on the serialized `.joblib` models confirmed that those lower figures were a documentation transcription error. The actual serialized pipelines and summary tables confirm the higher values listed above.
* **Why Unweighted XGBoost is Champion:** Although class-weighted XGBoost achieved a nominally similar PR-AUC (0.3597 vs 0.3584) and high raw recall at 0.50, it severely degraded probability calibration ($\text{Brier} = 0.2166$, $\text{ECE} = 0.2503$). Unweighted XGBoost preserved pristine probabilistic calibration ($\text{Brier} = 0.1431$, $\text{ECE} = 0.0240$). Preserving true base rates is essential for downstream asymmetric cost optimization and probability-space SHAP.

### E. Threshold Selection & Financial Cost Model Reconciliation
Verified against `data/processed/step5_threshold_comparison.csv` and `data/processed/step5_threshold_cost_analysis.json` on the validation cohort (20,539 appointments) under illustrative parameters ($C_{\text{intervention}} = \$10$, $C_{\text{noshow}} = \$200$, outreach efficacy $r = 30\%$):
* **Baseline (Zero Intervention):** Total Cost = **$\$824,800$** | Net Savings = $\$0$.
* **Default Threshold ($0.50$):** Recall = **$3.47\%$** (captures 143 of 4,124 no-shows) | Precision = $47.19\%$ | Total Cost = **$\$819,250$** | Net Savings = $\$5,550$.
* **Lowest-Cost Evaluated Candidate ($0.15$):** Recall = **$93.62\%$** (captures 3,861 of 4,124 no-shows) | Precision = $29.09\%$ | Flagged = 13,274 ($64.63\%$) | Total Cost = **$\$725,880$** | Net Savings vs Baseline = **$\$98,920$** | Incremental Net Savings vs 0.50 = **$\$93,370$** ($11.4\%$ cost reduction).
* **Qualification of Financial Claims:** 0.15 was the lowest-cost threshold among the evaluated candidates under the specified illustrative cost and intervention-effectiveness assumptions. Estimated savings of $\$98,920$ represent a modeled financial scenario on the validation cohort, not measured operational savings or proven real-world outcomes.

---

## 5. Known Limitations & Scope Boundaries

1. **Chronological Holdout Qualification:** June 1–8 (26,450 appointments) was retained as a chronological holdout, excluded from model training and validation. Exploratory analysis of its predictions had previously occurred during the project, so it is not presented as a completely untouched final test set.
2. **Dataset Generalizability:** Sourced from public clinics in Vitória, Brazil (2016). External validation on modern EHR systems is recommended prior to live deployment.
3. **Short Observational Horizon:** Data spans 6 weeks, which prevents estimating year-round seasonal fluctuations (e.g., flu season, winter holidays).
4. **Illustrative Business Cost Parameters:** The financial optimization assumes $\$10$ intervention cost, $\$200$ missed appointment penalty, and $30\%$ outreach conversion. While realistic, these parameters are customizable to specific hospital accounting models.
5. **Attribution vs. Causality:** SHAP values measure statistical attribution within the model, not clinical causality. Interventions must address the underlying barriers rather than artificially altering feature proxies.

---

## 6. Confirmation of Immutability (Steps 1–6)

* **No Models Retrained:** All serialized models in `models/` remain identical to their historical checkpoint versions.
* **No Feature Engineering Alterations:** Feature definitions, metadata, and column order remain untouched.
* **No Partition Changes:** Chronological train/validation/holdout date ranges and row counts remain strictly preserved.
* **Zero Fabrication:** All reported metrics, cost savings, and SHAP values are derived directly from serialized execution outputs.

---

## 7. Final Readiness Assessment & Declaration of Freeze

### Readiness Assessment
* **Portfolio Showcase:** Ready for immediate publication on GitHub.
* **Interview Defense:** Thoroughly supported by `INTERVIEW_PREPARATION.md` and `presentation.pptx`.
* **Code Quality:** Execution-verified, modular, and reproducible with zero violations detected by point-in-time leakage audit.

### Declaration of Project Freeze
**Steps 1 through 6, the final documentation, visual assets, presentations, and technical audits are hereby declared OFFICIALLY FROZEN.** No further architectural modifications, retraining passes, or restructuring will be performed. The project represents a complete, defensible, reproducible machine learning prototype.

---
*Report certified and frozen on November 11, 2026.*
