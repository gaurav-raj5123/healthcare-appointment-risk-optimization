# Step 6: Dual-Space SHAP Model Interpretability & Clinical Governance

This directory contains the implementation, verification scripts, notebook, and visual artifacts for **Step 6 (Dual-Space SHAP Interpretability)** of the Healthcare Appointment No-Show Prediction project.

---

## 1. Overview & Dual-Space Architecture

To bridge the gap between technical machine learning engineering and clinical decision-making, Step 6 implements a **Dual-Space SHAP (SHapley Additive exPlanations)** framework for the selected unweighted XGBoost model (`models/xgboost_unweighted.joblib`).

```
                            [ Trained Unweighted XGBoost ]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [ Raw-Margin Explainer ]                        [ Probability-Space Explainer ]
     • Output: Log-odds margin z                     • Output: Probability P(no-show | x)
     • Space: (-∞, +∞)                               • Space: [0.0, 1.0] (or 0% to 100%)
     • Formula: z(x) = φ0_raw + ∑ φi_raw             • Formula: P(y=1|x) = φ0 + ∑ φi
     • Role: Engineering verification baseline       • Role: Primary clinical stakeholder view
```

1. **Technical Reference Space (Raw-Margin SHAP):**
   * Operates in the tree ensemble's natural unconstrained leaf margin space ($z \in (-\infty, +\infty)$).
   * Verifies exact additive properties and mathematical consistency before applying non-linear link functions.
   * Sigmoid transformation $\sigma(z) = \frac{1}{1 + e^{-z}}$ reconstructs pipeline probabilities within numerical precision ($1.96 \times 10^{-8}$).

2. **Primary Stakeholder Space (Probability-Space SHAP):**
   * Uses an interventional TreeExplainer to marginalize expectations over a representative reference population.
   * Explains feature contributions directly in intuitive **probability units** and **percentage points (% pts)**.
   * Every patient's prediction begins at the baseline expected probability **$\phi_0 = 18.65\%$ ($0.18646$)**, with individual clinical and scheduling features adding or subtracting percentage points to form the final predicted no-show risk.

---

## 2. Background Dataset & Effective Sample Size Verification

* **Rows Supplied from Training Partition:** 200 rows sampled from `data/processed/train_features.csv` (`random_state=42`).
* **Internal Masker Behavior:** When an empirical background matrix is passed to `shap.TreeExplainer(clf, data=X_bg_trans, feature_perturbation='interventional')`, SHAP instantiates an internal `shap.maskers.Independent` masker. By default, `Independent` applies `max_samples=100`, subsampling the supplied dataset deterministically via `sklearn.utils.shuffle(X, n_samples=100, random_state=0)`.
* **Effective Background Size:** **100 observations** evaluated internally.
* **Explainer Background Parity:** Both `explainer_raw` and `explainer_prob` evaluate on the **exact same 100 background observations** ($0.00\times 10^0$ difference between internal background matrices).
* **Strict Training Isolation:** Background data is sampled exclusively from the chronological training partition (`train_features.csv`), ensuring training isolation and preventing feature contamination into validation or chronological holdout sets.

---

## 3. Mathematical Verification & Additivity Assertions

All assertions are verified during automated execution (`run_shap_analysis.py`) on $N = 2,000$ validation appointments (`Seed=42`):

| Verification Check | Metric / Formula | Result | Tolerance Status |
| :--- | :--- | :---: | :---: |
| **Probability Additivity** | $\max \|\phi_0 + \sum \phi_i - P(\text{no-show} \mid x)\|$ | **$2.32 \times 10^{-7}$** | PASSED ($< 10^{-4}$) |
| **Mean Probability Error** | $\text{mean} \|\phi_0 + \sum \phi_i - P(\text{no-show} \mid x)\|$ | **$4.29 \times 10^{-8}$** | PASSED |
| **Raw Margin Additivity** | $\max \|\phi_0^{\text{raw}} + \sum \phi_i^{\text{raw}} - z(x)\|$ | **$4.37 \times 10^{-6}$** | PASSED ($< 10^{-4}$) |
| **Mean Raw Margin Error** | $\text{mean} \|\phi_0^{\text{raw}} + \sum \phi_i^{\text{raw}} - z(x)\|$ | **$4.71 \times 10^{-7}$** | PASSED |
| **Sigmoid Parity** | $\max \|\sigma(z(x)) - P(\text{no-show} \mid x)\|$ | **$1.96 \times 10^{-8}$** | PASSED ($< 10^{-4}$) |
| **Categorical Flag Parity** | Max diff after `clf.enable_categorical = False` | **$0.00 \times 10^0$** | Exact Parity |
| **Baseline Expected Prob** | $E[f(X)] = \phi_0$ | **$18.65\%$** ($0.18646$) | Verified |

---

## 4. Global Feature Importance in Probability Space

Evaluated across the 2,000 validation cohort appointments and aggregated to original Day-0 features:

$$\text{Base Expected Probability: } \phi_0 = 18.65\% \quad (0.18646)$$

| Rank | Feature | Mean \|SHAP\| (% Points) | Mean \|SHAP\| (Prob Units) | Mean Signed Shift (% pts) | Operational & Clinical Interpretation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `lead_days` | **10.79% pts** | 0.10786 | +2.11% pts | Primary operational lever; scheduling lag exponentially increases forgetting and scheduling friction. |
| **2** | `age` | **3.28% pts** | 0.03278 | +0.51% pts | Bimodal profile: protective in adults 50+, higher risk in adolescents and young adults. |
| **3** | `same_day_booking` | **1.77% pts** | 0.01767 | +0.08% pts | Same-day walk-in scheduling strongly suppresses predicted no-show risk. |
| **4** | `prior_noshow_rate` | **0.76% pts** | 0.00756 | +0.12% pts | Historical non-attendance is a potent risk multiplier for repeat no-shows. |
| **5** | `prior_appointments` | **0.70% pts** | 0.00700 | -0.07% pts | Established patients with regular clinical engagement attend more reliably. |
| **6** | `days_since_last_appointment` | **0.62% pts** | 0.00619 | +0.11% pts | Longer gaps between visits reflect weaker care continuity. |
| **7** | `appointment_dow` | **0.48% pts** | 0.00477 | +0.20% pts | Friday and Monday visits show slightly higher risk. |
| **8** | `scholarship` | **0.42% pts** | 0.00423 | +0.08% pts | Welfare enrollment reflects socioeconomic barriers (transit costs, hourly wage inflexibility). |
| **9** | `scheduled_dow` | **0.32% pts** | 0.00323 | +0.07% pts | Minor scheduling day variations. |
| **10** | `gender` | **0.30% pts** | 0.00302 | +0.08% pts | Minor predictive weight. |

---

## 5. Local Patient Case Studies (Probability Space)

* **Case 1: High-Risk True Positive (Sample Row 361 | Predicted = 76.84% | Actual = No-Show):**
  * Baseline probability: $18.65\%$.
  * Driven upward by `prior_noshow_rate` (**+22.53% pts**), `age` (**+19.96% pts**), `prior_noshows` (**+10.71% pts**), and `days_since_last_appointment` (**+6.27% pts**), overcoming lead-time effects (**-6.92% pts**).
  * Recommended action: Proactive personal outreach call by bilingual patient navigator + transportation voucher.

* **Case 2: Low-Risk True Negative (Sample Row 714 | Predicted = 0.66% | Actual = Show-Up):**
  * Baseline probability: $18.65\%$.
  * Driven downward by `lead_days` (**-8.94% pts**), `prior_appointments` (**-4.20% pts**), `age` (**-2.87% pts**), and `same_day_booking` (**-1.49% pts**).
  * Recommended action: Standard automated SMS notification; zero clinical staff intervention required.

* **Case 3: False-Positive Error Case (Sample Row 1110 | Predicted = 66.01% $\ge 0.15$ | Actual = Show-Up):**
  * Flagged for intervention under the $0.15$ candidate decision threshold due to `prior_noshow_rate` (**+19.44% pts**), `age` (**+19.17% pts**), and `prior_noshows` (**+12.81% pts**), yet the patient attended.
  * Clinical Governance Lesson: **Model predictions are probabilistic risk stratifiers, not deterministic certainties.** Outreach interventions must always remain supportive (navigation, reminders), never punitive or restrictive.

---

## 6. Clinical Governance & Methodological Boundaries

1. **Statistical Attribution vs. Clinical Causality:** A $+10.79\%$ point attribution for `lead_days` measures statistical credit assignment inside the tree ensemble. It does not imply that administratively shortening appointment windows will causally ensure attendance if root schedule barriers remain unaddressed.
2. **Interventional Masking with Correlated Predictors:** Interventional masking samples feature subsets independently, which can query off-manifold synthetic points for correlated features (e.g., `prior_appointments` and `prior_noshow_rate`). Grouped conceptual evaluation is recommended.
3. **Ethical Non-Discrimination:** Risk scores must **never** be used to deny appointments, delay scheduling, or deprioritize care.

---

## 7. Execution Guide

```bash
# Run standalone dual-space SHAP analysis pipeline
python steps/06_shap_explainability/run_shap_analysis.py

# Rebuild interactive SHAP notebook
python steps/06_shap_explainability/build_shap_notebook.py
```
