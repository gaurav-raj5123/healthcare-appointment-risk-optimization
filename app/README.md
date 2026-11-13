# CarePredict | Streamlit Decision Support Application

**CarePredict** is an interactive, production-grade Streamlit web application designed for healthcare clinic operations managers and clinical staff. It operationalizes the frozen machine learning model and dual-space SHAP explainability pipeline from the Healthcare Appointment No-Show Prediction project into an intuitive, decision-support interface.

---

## 🚀 Quickstart & Launch Instructions

### Prerequisites
* Python 3.10+
* Virtual environment activated with project dependencies installed

### Launch Command
From the project root directory, run:

```bash
streamlit run app/app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

### Automated Verification Test Suite
Before running or deploying the dashboard, execute the automated 57-test verification suite:

```bash
pytest tests -v
```
All 57 tests pass with 100% test coverage across schema parity, risk tiers, point-in-time integrity, SHAP additivity, and financial models.

---

## 🏛️ Application Architecture & Modular Structure

The dashboard is structured into decoupled, maintainable service modules under `app/`:

```
app/
├── app.py                      # Main application entry point & sidebar routing
├── config.py                   # Paths, constants, risk thresholds, color palette
├── data_loader.py              # Cached loading of dataframes and benchmark artifacts
├── model_service.py            # Model loading, single & batch probability scoring
├── feature_builder.py          # Input validation, temporal & point-in-time feature derivation
├── shap_service.py             # Probability-space & raw-margin TreeExplainer explanations
├── financial_engine.py         # Cost modeling, threshold curves, capacity prioritization
├── analytics.py                # Demographic, temporal, and patient-history aggregations
├── components.py               # Reusable KPI cards, styled badges, callout boxes
│
├── pages/
│   ├── overview.py             # Section 1: Executive Overview & KPI Dashboard
│   ├── analytics.py            # Section 2: Demographic & Temporal Analytics
│   ├── prediction.py           # Section 3: Individual & Bulk Prediction + Local/Global SHAP
│   └── optimization.py         # Section 4: Capacity & Financial Optimization
│
├── assets/
│   └── style.css               # Clean healthcare aesthetics (light slate theme, teal accents)
│
└── templates/
    └── appointment_template.csv# Sample CSV input template for bulk batch scoring
```

---

## 🖥️ Application Features & Navigation Sections

### 1. Executive Overview
* **Cohort Distinction & Real-Time KPI Cards:** Clearly distinguishes the raw dataset ($110,527$ records) from the audited analytical cohort ($110,521$ visits, $62,298$ unique patients). Displays macro no-show rate ($20.19\%$, $22,314$ missed), attendance rate ($79.81\%$, $88,207$ attended), selected decision threshold ($0.15$), and validation cost savings ($\$98,920$).
* **Interactive Filtering:** Date range filter and demographic cohort slicing (gender, welfare status *Bolsa Família*, chronic conditions).
* **Operational Visualizations:** Attendance donut breakdown, lead time risk curve across 6 bins (`0 Days (Same-Day)`, `1-3 Days`, `4-7 Days`, `8-14 Days`, `15-30 Days`, `31+ Days`), non-overlapping age cohorts (Pediatric `<18`, Young Adult `18-35`, Adult `36-50`, Mature Adult `51-65`, Senior `66+`), day-of-week volume/risk (with Sunday explicitly noted as 0 occurrences in cleaned cohort), and clinical comorbidity comparisons.

### 2. Appointment Analytics
* **Demographic Cohort Analysis:** Non-overlapping age brackets and clinical diagnoses.
* **Temporal Patterns:** Contrasting same-day bookings ($95.36\%$ attendance) vs. extended lead horizons ($>32\%$ no-show rate for $>14$ days).
* **Patient History Patterns:** Evaluates attendance momentum across first-time vs. established returning patients without point-in-time leakage.
* **Data Explorer:** Searchable, sortable appointment-level table with CSV export.

### 3. No-Show Prediction & SHAP Explainability
* **Mutually Exclusive Operational Risk Tiers:**
  * **Low Risk:** $[0.0, 0.15)$ — Routine care, standard automated confirmation.
  * **Moderate Risk:** $[0.15, 0.40)$ — Automated 48-hour SMS reminder + scheduling confirmation.
  * **High Risk:** $[0.40, 0.70)$ — Direct nurse outreach call + transport support navigation.
  * **Critical Risk:** $[0.70, 1.00]$ — Multi-channel intensive outreach, provider alert, flexible rebooking.
* **Mode A: Individual Prediction (Custom User Inputs):**
  * Clinical, scheduling, and historical attendance form with strict boundary validation (disallowing inverted scheduling dates, impossible age $>115$ or $<0$, negative visit counts, and non-sensical returning/first-time status combinations).
  * Strict point-in-time conventions: first-time appointments enforce `prior_appointments=0`, `prior_noshows=0`, `prior_noshow_rate=0.0`, `days_since_last_appointment=-1`, `is_first_appointment=1`.
  * **Additive Probability-Space SHAP:** Starting from baseline $E[f(X)] = 18.65\%$, renders an interactive waterfall chart displaying exact feature contributions in percentage points ($+2.11\%$ pts for lead time, $-1.49\%$ pts for same-day booking, etc.) with additivity parity ($|\sum \phi_i - (p - \phi_0)| < 10^{-6}$).
  * Generates plain-language clinical narratives and security-hardened UI components with strict `html.escape()` sanitization.
* **Mode B: Bulk Batch CSV Upload:**
  * Validates column schema and data constraints, routing invalid rows into a quarantined dataframe with specific rejection reasons.
  * Vectorized inference retaining float precision for unrounded parity with individual scoring ($< 10^{-6}$).
  * Searchable results table with risk tier filters, complete CSV export, and on-demand local SHAP explanations.
* **Global Model Risk Drivers:**
  * Top predictive drivers in probability space with explicit non-causal attribution disclaimers.

### 4. Capacity & Financial Optimization
* **User-Configurable Financial Parameters:**
  * Intervention Cost ($C_{\text{intervention}}$): default $\$10$.
  * Unfilled Appointment Cost ($C_{\text{noshow}}$): default $\$200$.
  * Outreach Effectiveness ($r$): default $30\%$ ($0.30$).
  * Decision Threshold: default $0.15$ (*Selected threshold: 0.15 — the lowest-cost candidate among evaluated thresholds under the specified financial assumptions*).
  * Daily Outreach Capacity: default $500$ appointments.
* **Dual Financial Evaluation Modes:**
  * **Historical Validation Mode (Ground Truth Known):** Evaluates exact confusion matrix and cost savings on the chronological validation cohort ($20,539$ visits), yielding $\$824,800$ baseline cost, $\$725,880$ policy cost, $\$98,920$ net savings ($12.0\%$ reduction), and $93.62\%$ recall.
  * **Prospective Planning Mode (Expected Outcomes):** Computes expected cost and net savings from predicted probabilities when ground truth is unknown ($E[\text{cost}] = p \cdot (1 - r) \cdot C_{\text{noshow}} + C_{\text{int}}$ vs. $p \cdot C_{\text{noshow}}$).
* **Capacity-Constrained Prioritization:**
  * Strict descending-probability sort for outreach queue.
  * Identifies appointments within capacity vs. unreached shortfall.
  * Displays prioritized queue with CSV export.
* **Scenario Sensitivity Analysis:**
  * 2D sensitivity matrix across outreach effectiveness ($10\%$ to $50\%$) and outreach costs ($\$5$ to $\$25$).
* **Non-Punitive Clinical Governance Callout:** Prominent clinical directive ensuring predictions are never used to deprioritize or deny care.

---

## 📋 Bulk CSV Upload Schema

To score batches of appointments, upload a CSV adhering to either:
1. **Raw Appointment Schema:**
   * `appointment_id` (optional identifier)
   * `patient_id` (optional identifier)
   * `scheduled_day` (YYYY-MM-DD)
   * `appointment_day` (YYYY-MM-DD)
   * `age` (integer, 0–120)
   * `gender` ('F' or 'M')
   * `scholarship` (0 or 1)
   * `hypertension` (0 or 1)
   * `diabetes` (0 or 1)
   * `alcoholism` (0 or 1)
   * `handicap` (0 to 4)
   * `prior_appointments` (integer $\ge 0$)
   * `prior_noshows` (integer $\le$ prior_appointments)
   * `days_since_last_appointment` (integer, -1 for first-time visits)

2. **Pre-Engineered Feature Schema:** Containing the 17 approved feature columns directly.

A downloadable sample template is provided within the application and at `app/templates/appointment_template.csv`.

---

## ⚖️ Clinical Governance & Limitations

* **Decision Support, Not Diagnosis:** CarePredict is an operational risk-stratification prototype designed to optimize outreach resources (reminders, navigation, transit support).
* **Non-Punitive Mandate:** Model predictions must **never** be used to deny, delay, or deprioritize patient appointments.
* **Illustrative Financial Modeling:** Financial calculations represent modeled scenarios under stated assumptions, not measured hospital accounting savings.
* **Attribution vs. Causality:** SHAP values represent model credit assignment, not clinical causation.

