# Healthcare Appointment No-Show Prediction & Risk Optimization
## Comprehensive Technical Interview Preparation & Defense Guide

This document is engineered to prepare candidates to defend this machine learning engineering project in technical, clinical, and leadership interviews. Every explanation, metric, and rationale aligns with the verified codebase and serialized execution artifacts.

---

### Part A: 30-Second Elevator Pitch

> *"I engineered an end-to-end, reproducible machine learning prototype to predict and mitigate outpatient appointment no-shows using 110.5k cleaned appointments from Brazil. By designing a point-in-time patient history feature store, the automated leakage audit confirmed zero point-in-time violations. Partitioning the data chronologically across time, we benchmarked multiple ML architectures. Our unweighted XGBoost model achieved a PR-AUC of 0.3584, ROC-AUC of 0.7382, and superior probability calibration (Brier score 0.1431, ECE 0.0240). Evaluating decision thresholds under an illustrative asymmetric healthcare cost matrix ($10 outreach vs $200 unfilled slot, 30% outreach effectiveness), threshold 0.15 emerged as the lowest-cost candidate among evaluated thresholds, increasing recall from 3.47% to 93.62% and reducing modeled costs by an estimated $98,920 across 20,539 validation visits (a modeled financial scenario, not measured operational savings). Finally, I deployed dual-space interventional SHAP explanations to translate log-odds into direct percentage-point risk shifts for clinic navigation teams."*

---

### Part B: 2-Minute Project Overview

> *"Outpatient no-shows create a dual crisis in healthcare: they cost the US healthcare system over $150 billion annually in stranded capacity and worsen patient outcomes through delayed preventive care.
> 
> In this project, I used a public dataset of 110,521 cleaned clinical visits from Vitória, Brazil. My primary technical focus was sound experimental design and data leakage prevention. In healthcare, random K-fold cross-validation inflates performance because future patient behavior leaks into the past. I established a strict chronological split: training on April 29 to May 19 (63,532 appointments), validating on May 20 to May 31 (20,539 appointments), and retaining June 1 to June 8 (26,450 appointments) as a chronological holdout excluded from model training and validation. Exploratory analysis of June predictions had previously occurred during the project, so it is not presented as a completely untouched final test set.
> 
> For feature engineering, I built 17 approved Day-0 predictors across static demographics, booking temporal features, and rolling patient histories. Crucially, patient historical aggregates (prior appointments and no-show rates) were computed strictly prior to the booking timestamp, eliminating 27,525 contaminated records identified in naive cumulative logic, and `sms_received` was intentionally excluded because SMS delivery is dispatched days after booking.
> 
> Evaluating on Precision-Recall AUC due to the 20% class imbalance, an unweighted XGBoost classifier was selected as champion (PR-AUC 0.3584, ROC-AUC 0.7382, Brier score 0.1431, ECE 0.0240). While class-weighted XGBoost achieved high default recall, it severely degraded probability calibration (Brier score 0.2166, ECE 0.2503). Unweighted XGBoost preserved pristine calibration, making it the ideal foundation for downstream threshold optimization.
> 
> In production, default 0.50 probability thresholds fail under class imbalance, capturing only 3.47% of no-shows. Implementing an illustrative healthcare cost function ($10 intervention vs $200 missed visit, 30% outreach effectiveness), I evaluated candidate decision thresholds. Threshold 0.15 was the lowest-cost candidate among evaluated thresholds, increasing recall from 3.47% to 93.62% and reducing modeled costs from $824,800 to $725,880—an estimated modeled net savings of $98,920 across the validation cohort ($93,370 vs threshold 0.50).
> 
> Finally, because clinicians cannot interpret raw log-odds, I implemented dual-space SHAP using an interventional TreeExplainer (supplying 200 training observations, with 100 effective background observations evaluated internally by SHAP's Independent masker). This allowed us to present additive, percentage-point explanations starting from an empirical baseline of 18.65%, identifying scheduling lead time (+10.79% pts) and age (+3.28% pts) as primary risk drivers while establishing strict governance protocols ensuring risk scores are used exclusively for supportive care, never for care restriction."*

---

### Part C: 5-Minute In-Depth Technical Walkthrough

#### 1. Problem Formulation & Data Sanitization (Step 1)
* **Context:** Binary classification predicting `no_show = 1` vs `no_show = 0` at Day-0 booking time.
* **Sanitization:** Out of 110,527 raw records, exactly one record contained an impossible age (`-1`), and 5 records had inverted timestamps where scheduling occurred after the appointment date. Dropping these yielded 110,521 valid appointments with 0 missing values.

#### 2. Exploratory Data Analysis & The SMS Paradox (Step 2)
* **Class Imbalance:** Base no-show rate of 20.19% across the full cleaned dataset (88,208 attendees vs 22,313 no-shows).
* **The SMS Paradox:** Patients who received an SMS reminder showed a 27.57% no-show rate compared to 16.70% for non-recipients. Bivariate analysis revealed this was an operational selection confounder: SMS reminders were only dispatched for appointments with lead times $\ge 3$ days. Same-day appointments (which never receive SMS reminders) had a sub-5% no-show rate, whereas appointments scheduled weeks in advance had higher baseline risk and were the primary recipients of SMS messages.

#### 3. Leakage Prevention & Temporal Partitioning (Step 3)
* **Chronological Temporal Split:**
  * **Train Set:** April 29 – May 19, 2016 ($N = 63,532$, no-show rate: $20.95\%$)
  * **Validation Set:** May 20 – May 31, 2016 ($N = 20,539$, no-show rate: $20.08\%$)
  * **Chronological Holdout:** June 1 – June 8, 2016 ($N = 26,450$, no-show rate: $18.46\%$)
  * *Holdout Qualification:* June 1–8 was retained as a chronological holdout, excluded from model training and validation. Exploratory analysis of its predictions had previously occurred, so it is not presented as a completely untouched final test set.
* **Point-in-Time Patient History:** Computed `prior_appointments`, `prior_noshows`, and `prior_noshow_rate` strictly using appointments where `historical_appointment_day < current_scheduled_date`. Automated audit confirmed 0 violations across all partitions.
* **Exclusion of `sms_received`:** SMS reminders are sent days after scheduling. Including `sms_received` for Day-0 booking-time triage is an operational distribution mismatch and target leakage.

#### 4. Model Benchmarking & Metric Selection (Step 4)
* **Preprocessing Pipeline:** Scikit-learn `ColumnTransformer` with `OneHotEncoder(drop='first')` for nominal features and `StandardScaler` for continuous features, producing a clean numerical feature matrix.
* **Metric Selection:** Primary evaluation on PR-AUC (Precision-Recall Area Under Curve) due to the $20\%$ positive class imbalance.
* **Validation Benchmark Results (Verified from `model_benchmark_summary.csv`):**
  * Baseline (Majority Class): PR-AUC = 0.2008 | ROC-AUC = 0.5000 | Brier = 0.2008 | ECE = 0.2008
  * Baseline (Stratified Prior): PR-AUC = 0.1996 | ROC-AUC = 0.4963 | Brier = 0.3280 | ECE = 0.3280
  * Logistic Regression: PR-AUC = 0.3524 | ROC-AUC = 0.7302 | Brier = 0.2209 | ECE = 0.2602
  * Random Forest: PR-AUC = 0.3546 | ROC-AUC = 0.7365 | Brier = 0.2123 | ECE = 0.2467
  * XGBoost (Class Weighted): PR-AUC = 0.3597 | ROC-AUC = 0.7385 | Brier = 0.2166 | ECE = 0.2503
  * **XGBoost (Unweighted - Champion):** **PR-AUC = 0.3584** | **ROC-AUC = 0.7382** | **Brier = 0.1431** | **ECE = 0.0240**
  * XGBoost (Sigmoid Calibrated, 5-Fold CV): PR-AUC = 0.3588 | ROC-AUC = 0.7383 | Brier = 0.1445 | ECE = 0.0359
* **Why Unweighted XGBoost Won:** Class-weighted XGBoost achieved high default recall but distorted posterior probabilities ($\text{Brier} = 0.2166$, $\text{ECE} = 0.2503$). Unweighted XGBoost preserved pristine probabilistic calibration ($\text{Brier} = 0.1431$, $\text{ECE} = 0.0240$), enabling downstream asymmetric threshold optimization without probability distortion.

#### 5. Operational Threshold Evaluation & Business Cost Modeling (Step 5)
* **Illustrative Cost Matrix (Validation Cohort: 20,539 Appointments):**
  * Cost of Calling / Navigation Outreach ($C_{\text{intervention}}$): $10 per patient.
  * Cost of Unfilled Clinical Slot ($C_{\text{noshow}}$): $200 per missed appointment.
  * Outreach Effectiveness ($r$): Assumed 30% relative risk reduction in no-shows for contacted patients.
* **Cost Equation:**
  $$\text{Total Cost} = \text{Flagged} \times \$10 + [\text{FN} + (1 - r) \times \text{TP}] \times \$200$$
* **Threshold Comparison:**
  * **No Intervention Baseline:** Total Cost = **$824,800** | Net Savings = $0.
  * **Default (0.50):** Recall = 3.47%, Precision = 47.19%, Flagged = 303 (1.48%), Total Cost = **$819,250**, Net Savings = $5,550.
  * **Lowest-Cost Evaluated Candidate (0.15):** Recall = 93.62%, Precision = 29.09%, Flagged = 13,274 (64.63%), Total Cost = **$725,880**, Net Savings vs Baseline = **$98,920**, Incremental Net Savings vs 0.50 = **$93,370** ($11.4\%$ cost reduction).
  * *Qualification:* 0.15 was the lowest-cost threshold among the evaluated candidates under the specified illustrative cost and intervention-effectiveness assumptions. Estimated savings of $98,920 represent a modeled financial scenario on the validation cohort, not measured operational savings.

#### 6. Dual-Space SHAP Interpretability & Governance (Step 6)
* **Architecture:** Implemented two parallel explainers using interventional TreeExplainer:
  1. **Raw-Margin Space:** Explains raw log-odds $z \in (-\infty, +\infty)$ with exact margin additivity (max error $4.37 \times 10^{-6}$) and sigmoid probability parity (max error $1.96 \times 10^{-8}$).
  2. **Probability Space:** Explains direct probability shifts $P(\text{no-show} \mid x) = \phi_0 + \sum \phi_i$ starting from baseline $\phi_0 = 18.65\%$ with exact probability additivity (max error $2.32 \times 10^{-7}$).
* **Background Sample Size:** 200 rows supplied from training data (`random_state=42`); 100 effective background observations evaluated internally by TreeExplainer via SHAP's default `max_samples=100`. Both explainers evaluate on the exact same 100 observations ($0.00\times 10^0$ difference).
* **Key Findings:** Scheduling lead time (`lead_days`) is the primary operational lever, contributing an average magnitude of 10.79 percentage points to predicted risk, followed by age (+3.28% pts) and same-day booking (+1.77% pts).
* **Clinical Governance:** Established that SHAP values measure statistical attribution within the model, not real-world clinical causation. High-risk scores must be used strictly for supportive navigation (reminders, transit assistance), never to deprioritize patients.

---

### Part D: Technical Questions & Defensible Answers

#### 1. Why did you use chronological temporal splitting instead of standard K-fold cross-validation?
> *"In clinical scheduling, patient appointments are sequential time-series events. Standard K-fold cross-validation randomly shuffles rows across time, allowing a model to train on appointments from June while evaluating on appointments from May. This introduces look-ahead temporal leakage, artificially inflating validation metrics. By using a strict forward temporal split—training on April 29 to May 19, validating on May 20 to May 31, and retaining June 1 to June 8 as a chronological holdout—we accurately simulate production deployment where the model can only use historical data to forecast future appointments. June 1–8 was retained as a chronological holdout, excluded from model training and validation; exploratory analysis of its predictions had previously occurred, so it is not presented as an untouched test set."*

#### 2. How exactly was data leakage prevented during feature engineering?
> *"Data leakage was prevented at two levels:
> 1. **Temporal Horizon:** All patient historical aggregates (`prior_appointments`, `prior_noshows`, `prior_noshow_rate`) were computed using a point-in-time calculation that filtered strictly for records where `historical_appointment_day < current_scheduled_date`. For same-day bookings where scheduled date equals appointment date, we strictly utilized prior sequential booking records, ensuring no same-day or future outcome was incorporated into the patient's history. The automated leakage audit confirmed zero violations detected.
> 2. **Pipeline Preprocessing:** All transformers (`OneHotEncoder`, `StandardScaler`) were fit strictly on the training partition and then applied (`transform`) to validation and holdout partitions, preventing global distributional statistics from leaking into evaluation sets."*

#### 3. Why was `sms_received` excluded from Day-0 prediction features?
> *"During exploratory data analysis, we identified that SMS reminders are dispatched several days after an appointment is booked, typically 2 to 4 days prior to the visit. In clinical practice, Day-0 triage happens at the moment of scheduling. At booking time, the clinic does not know whether an SMS will be successfully sent or delivered. Including `sms_received` creates an operational distribution mismatch and target leakage. Furthermore, EDA revealed that same-day visits never receive SMS, confounding SMS delivery with lead time."*

#### 4. How were historical patient behavior features constructed, and what edge cases were handled?
> *"We grouped the sanitized appointments by `patient_id` and sorted chronologically by `appointment_date` and `scheduled_date`. For each appointment, we calculated:
> - `prior_appointments`: Total historical visits prior to the booking timestamp.
> - `prior_noshows`: Total missed appointments prior to the booking timestamp.
> - `prior_noshow_rate`: `prior_noshows / prior_appointments` (set to 0.0 for first-time patients).
> - `days_since_last_appointment`: Calendar days since the preceding appointment (set to -1 for first-time patients).
> - `is_first_appointment`: Binary indicator (1 if first visit, 0 otherwise).
> To prevent prefix collision during modeling and SHAP aggregation, all feature names were strictly isolated in metadata without relying on substring heuristics."*

#### 5. Why did you choose PR-AUC over ROC-AUC as the primary model selection metric?
> *"In our dataset, only 20.08% of validation appointments result in a no-show. When evaluated on imbalanced binary classification, ROC-AUC can be deceptively optimistic because the False Positive Rate denominator is dominated by the large majority class (attendees). A large increase in false positives causes only a small change in FPR. In contrast, Precision-Recall AUC (PR-AUC) focuses directly on the minority class (`no_show = 1`), penalizing false positives heavily through precision. The unweighted XGBoost model achieved a PR-AUC of 0.3584 (compared to the 0.2008 baseline), proving superior precision at clinically useful recall levels."*

#### 6. Why did unweighted XGBoost emerge as the champion over class-weighted XGBoost?
> *"Applying `scale_pos_weight = 3.77` in XGBoost forced the trees to heavily penalize false negatives during splitting. While this artificially raised raw recall at the default 0.50 threshold (85.21%), it distorted the predicted probabilities toward the extremes, degrading probability calibration (Brier score degraded from 0.1431 to 0.2166; ECE degraded from 0.0240 to 0.2503). Unweighted XGBoost achieved essentially identical discrimination (PR-AUC 0.3584 vs 0.3597) while preserving smooth, well-calibrated probabilities across the spectrum. This allowed us to perform threshold tuning cleanly in Step 5 without probability distortion."*

#### 7. How does decision threshold tuning affect recall, precision, and clinical utility?
> *"Standard classification packages default to a 0.50 probability threshold, which implicitly assumes that a false positive and a false negative carry equal cost. In outpatient scheduling, a False Negative (unfilled slot, $200 loss) is 20 times more expensive than a False Positive ($10 outreach call). At 0.50, the model achieved only 3.47% recall (capturing 143 of 4,124 missed visits). By tuning the threshold down to 0.15, we accept lower precision (29.09%) in exchange for 93.62% recall, capturing 3,861 of 4,124 no-shows. In clinical operations, calling ~3.4 patients to capture 1 true no-show is highly cost-effective under asymmetric costs."*

#### 8. How was the illustrative business cost calculated?
> *"The business cost model calculates the modeled clinical expense over the validation cohort:
> $$\text{Total Cost} = \text{Flagged Patients} \times C_{\text{intervention}} + [\text{FN} + (1 - r) \times \text{TP}] \times C_{\text{noshow}}$$
> - $C_{\text{intervention}} = \$10$: Staff call or automated outreach per flagged patient.
> - $C_{\text{noshow}} = \$200$: Clinical opportunity loss for an unattended slot.
> - $r = 0.30$: Outreach effectiveness (assumes 30% of contacted no-shows are persuaded to attend).
> At threshold 0.15, 13,274 patients are flagged (cost: $132,740), 263 false negatives occur ($52,600), and 3,861 true positives yield 2,702.7 unprevented no-shows ($540,540), totaling $725,880. Compared to the no-intervention baseline ($824,800), the estimated net savings is $98,920 ($93,370 incremental savings vs threshold 0.50). 0.15 was the lowest-cost threshold among the evaluated candidates under the specified illustrative cost and intervention-effectiveness assumptions. Estimated savings of $98,920 represent a modeled financial scenario on the validation cohort, not measured operational savings."*

#### 9. What is SHAP, and how does TreeExplainer work?
> *"SHAP (SHapley Additive exPlanations) is an axiomatic approach from cooperative game theory that assigns each feature an additive importance value for a specific prediction. For tree ensembles, Lundberg et al.'s `TreeExplainer` calculates exact Shapley values in polynomial time $\mathcal{O}(TLD^2)$ rather than exponential time. It recursively traverses decision trees, tracking the proportion of training background paths that satisfy split conditions when specific features are conditioned upon, guaranteeing local accuracy, missingness, and consistency."*

#### 10. What is the fundamental difference between raw-margin SHAP and probability-space SHAP?
> *"In tree models, raw-margin SHAP explains predictions in log-odds space ($z \in (-\infty, +\infty)$), where the tree leaf values naturally sum: $z(x) = \phi_0^{\text{raw}} + \sum \phi_i^{\text{raw}}$. Passing $z$ through a sigmoid function yields the probability.
> 
> However, because the sigmoid function is non-linear, raw log-odds contributions do not sum additively to the final probability ($\sigma(\sum \phi_i) \neq \sum \sigma(\phi_i)$). To make explanations actionable for healthcare staff, probability-space SHAP integrates over an empirical background dataset using interventional perturbation. This yields direct, intuitive percentage-point shifts:
> $$P(\text{no-show} \mid x) = \phi_0 + \sum_{i=1}^n \phi_i$$
> Starting from the empirical baseline $\phi_0 = 18.65\%$, a feature contribution of $+0.10$ means an exact $+10.0$ percentage-point addition to the patient's predicted risk. In our implementation, 200 rows were supplied from training data, with 100 effective background observations evaluated internally by TreeExplainer via SHAP's default `max_samples=100` cap. Both explainers evaluate on the exact same 100 background observations."*

#### 11. What are the technical limitations of interventional SHAP?
> *"Interventional TreeExplainer simulates missing features by marginalizing over independent distributions from the background dataset (`feature_perturbation='interventional'`). This breaks empirical correlations between features. When features are correlated in clinical reality—such as `prior_appointments`, `prior_noshows`, and `prior_noshow_rate`—the algorithm can theoretically query synthetic feature combinations that rarely occur together. While point-in-time engineering ensured logical consistency, practitioners must evaluate grouped conceptual features collectively rather than assuming complete independence."*

#### 12. Why are SHAP explanations statistical attributions rather than causal clinical drivers?
> *"SHAP quantifies statistical credit assignment within the mathematical model; it does not measure counterfactual causal relationships in the physical world. For example, observing that `lead_days` contributes $+10.79\%$ percentage points to predicted risk demonstrates that the tree ensemble relies heavily on lead time to partition risk. It does not prove that administratively shortening appointment windows by mandate will cause patients to attend, particularly if systemic transportation or work schedule barriers remain unaddressed. Confusing model attribution with clinical causality risks counterproductive policy interventions."*

#### 13. What are the project's primary limitations and recommendations for future iterations?
> *"1. **Geographic Specificity:** Data originates from a single public healthcare system in Vitória, Brazil; transportability to US private clinics requires retraining and re-calibration.
> 2. **Static Comorbidities:** Hypertension and diabetes were captured as binary flags without severity markers (e.g., HbA1c levels or blood pressure readings).
> 3. **Unobserved Confounders:** Patient socioeconomic status, distance to clinic, and public transit access were unobserved.
> 4. **Future Enhancements:** Future iterations should test multi-modal features (clinical notes, weather forecasts) and conduct prospective randomized controlled trials (A/B testing) to measure real-world outreach conversion rates."*
