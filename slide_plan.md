# Slide Plan — Healthcare No-Show Prediction & Capacity Optimization

## Slide 1: TITLE
- **Action Title:** Predicting Healthcare No-Shows to Optimize Clinical Capacity
- **Single Idea:** Project identity, author, scope in one glance
- **Visual:** Large project title, subtitle tagline, 3 mini KPI badges (110K visits, 93.6% recall, $98.9K savings)
- **Bullets:** None (title slide)
- **Speaker Notes:** Full elevator pitch, dataset source, technologies used

## Slide 2: PROBLEM
- **Action Title:** One in Five Patients Misses Their Appointment
- **Single Idea:** The business cost and clinical harm of no-shows
- **Visual:** Giant "20.19%" KPI in accent cyan, 3 supporting KPI cards (110,521 cleaned visits, $200 slot cost, 62,298 patients)
- **Bullets:** Idle capacity & lost revenue / Clinical deterioration / Outreach teams cannot call everyone
- **Speaker Notes:** US healthcare system stats, $150B annual cost, Vitória Brazil dataset context

## Slide 3: DATA & LEAKAGE PREVENTION
- **Action Title:** Chronological Splits and Zero Leakage Ensure Honest Evaluation
- **Single Idea:** Temporal integrity is the foundation — no peeking at the future
- **Visual:** 3-block flow diagram (Train → Val → Holdout with date ranges), 2 KPI cards (27,525 contaminated rows remediated, 0 leakage violations)
- **Bullets:** Strict point-in-time patient history / sms_received excluded (post-booking) / 6 anomalies removed
- **Speaker Notes:** Random K-fold would leak; chronological split simulates production deployment; holdout was previously explored

## Slide 4: APPROACH / ARCHITECTURE
- **Action Title:** A Six-Stage Pipeline from Raw Data to Explainable Decisions
- **Single Idea:** End-to-end flow showing how each stage feeds the next
- **Visual:** Horizontal flow diagram with 6 rounded rectangles + arrows: Data Audit → EDA → Feature Eng. (17 features) → XGBoost → Threshold Opt. → SHAP
- **Bullets:** 17 Day-0 features across 3 domains / Chronological validation / Probability-space explanations
- **Speaker Notes:** Feature domains (static demographics, booking temporal, patient history), why XGBoost, why dual-space SHAP

## Slide 5: MODEL BENCHMARK
- **Action Title:** Unweighted XGBoost Wins on Calibration, Not Just Accuracy
- **Single Idea:** Comparing 4 models — weighted XGB has higher PR-AUC but terrible calibration
- **Visual:** Native grouped bar chart (PR-AUC and Brier Score for 4 models), winner highlighted in cyan, KPI card with ECE 0.024
- **Bullets:** PR-AUC primary metric (20% class imbalance) / Calibration essential for threshold tuning / Class weighting degraded Brier 0.14→0.22
- **Speaker Notes:** Full model comparison table, why PR-AUC over ROC-AUC, calibration vs discrimination tradeoff

## Slide 6: THRESHOLD & COST OPTIMIZATION
- **Action Title:** Threshold 0.15 Captures 94% of No-Shows and Saves $98,920
- **Single Idea:** Moving from default 0.50 to cost-optimized 0.15 — dramatic recall improvement
- **Visual:** 4 KPI cards ($824.8K baseline, $725.9K policy cost, $98.9K savings, 93.62% recall), confusion matrix image
- **Bullets:** Default 0.50 catches only 3.5% of no-shows / Cost formula: Flagged×$10 + unmitigated×$200 / Modeled scenario, not measured savings
- **Speaker Notes:** Full cost formula, why 0.15 is lowest-cost among evaluated candidates, asymmetric cost rationale

## Slide 7: SHAP EXPLAINABILITY
- **Action Title:** Lead Time Dominates Risk; SHAP Makes Every Prediction Transparent
- **Single Idea:** Probability-space SHAP lets clinicians understand exactly why a patient is flagged
- **Visual:** SHAP probability bar importance PNG (fitted with aspect ratio), 3 top-feature KPI badges
- **Bullets:** Base probability 18.65% + feature shifts = prediction / Lead days: ±10.8% pts / Attribution ≠ causality
- **Speaker Notes:** Dual-space architecture (raw margin + probability), additivity axiom (<2.4e-7 error), top 6 features, clinical actionability

## Slide 8: GOVERNANCE & LIMITATIONS
- **Action Title:** Predictions Must Support Patients, Never Restrict Access
- **Single Idea:** Non-punitive clinical deployment principles and honest limitations
- **Visual:** 4 governance principle cards with icons
- **Bullets:** Non-punitive mandate / Attribution ≠ causality / Single-city 2016 dataset / Financial estimates are modeled
- **Speaker Notes:** Detailed governance framework, transportability concerns, holdout was previously explored

## Slide 9: CONCLUSION & NEXT STEPS
- **Action Title:** A Complete, Interview-Ready ML Pipeline with 57 Verified Tests
- **Single Idea:** What was delivered and where it could go next
- **Visual:** 4 summary KPI cards (6 pipeline stages, 57 tests passed, 4-tab dashboard, 17 features), CarePredict dashboard screenshot
- **Bullets:** Full reproducibility: Steps 1-6 + CarePredict dashboard / Next: multi-site validation, temporal recalibration, EHR integration
- **Speaker Notes:** Technical stack summary, dashboard capabilities, interview defense points
