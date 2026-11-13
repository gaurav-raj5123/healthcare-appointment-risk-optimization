"""
Script to generate the professional 5-slide project presentation:
presentation.pptx (and docs/presentation.pptx)
"""

import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

os.makedirs('docs', exist_ok=True)
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

DARK_BG = RGBColor(15, 23, 42)        # #0f172a
CARD_BG = RGBColor(30, 41, 59)        # #1e293b
WHITE = RGBColor(248, 250, 252)       # #f8fafc
MUTED = RGBColor(148, 163, 184)       # #94a3b8
CYAN = RGBColor(56, 189, 248)         # #38bdf8
GREEN = RGBColor(52, 211, 153)        # #34d399

def apply_background(slide):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = DARK_BG
    bg.line.fill.background()
    return bg

def add_header(slide, title_text, category_text):
    tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.9))
    tf = tx_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p0 = tf.paragraphs[0]
    p0.text = category_text.upper()
    p0.font.size = Pt(10)
    p0.font.bold = True
    p0.font.color.rgb = CYAN
    p0.space_after = Pt(2)
    
    p1 = tf.add_paragraph()
    p1.text = title_text
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

def add_card(slide, left, top, width, height, title, body_bullets, border_color=None):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = Pt(1.5)
    else:
        card.line.color.rgb = RGBColor(51, 65, 85)
        card.line.width = Pt(1)
        
    tx_box = slide.shapes.add_textbox(Inches(left + 0.25), Inches(top + 0.2), Inches(width - 0.5), Inches(height - 0.4))
    tf = tx_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    if title:
        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.size = Pt(13)
        p0.font.bold = True
        p0.font.color.rgb = CYAN
        p0.space_after = Pt(6)
    
    for i, bullet in enumerate(body_bullets):
        if not title and i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = bullet
        p.font.size = Pt(10.5)
        p.font.color.rgb = WHITE
        p.space_after = Pt(4)
        p.level = 0
    return card

# -------------------------------------------------------------
# SLIDE 1: Problem Statement & Objectives
# -------------------------------------------------------------
s1 = prs.slides.add_slide(blank_layout)
apply_background(s1)
add_header(s1, "Predicting & Mitigating Healthcare Appointment No-Shows", "Executive Project Overview | End-to-End ML Prototype")

add_card(s1, 0.8, 1.5, 5.6, 4.4, "The Clinical & Operational Challenge", [
    "• Chronic Healthcare Inefficiency: Missed appointments cost the US health system over $150B annually and create severe care gaps.",
    "• Clinic Baseline Disruption: Out of 110,521 cleaned visits, 20.19% resulted in no-shows, stranding clinical staff and diagnostic equipment.",
    "• Asymmetric Clinical Impact: Missed chronic disease follow-ups (hypertension, diabetes) sharply elevate emergency admission risks.",
    "• Operational Constraints: Clinic outreach teams have finite calling hours and cannot manually contact 100% of booked patients.",
    "• Objective: Build an end-to-end, reproducible ML prototype to reliably stratify no-show risk at booking time and optimize outreach."
], CYAN)

add_card(s1, 6.8, 1.5, 5.7, 4.4, "Engineering & Delivery Objectives", [
    "• Strict Temporal Evaluation: Avoid random K-fold leakage; validate exclusively on chronological cohorts.",
    "• Point-in-Time Data Integrity: Engineer rolling patient historical metrics strictly prior to appointment creation.",
    "• Asymmetric Cost Modeling: Evaluate decision thresholds using an illustrative cost matrix ($10 call vs $200 unfilled slot).",
    "• Dual-Space Explainability: Provide exact log-odds attributions for engineers and intuitive percentage-point shifts for clinicians.",
    "• Methodological Rigor: Freeze champion model (unweighted XGBoost) and verify mathematical additivity (< 1e-4 tolerance)."
], GREEN)

add_card(s1, 0.8, 6.1, 11.7, 0.9, "", [
    "DATASET: 110,521 Cleaned Appointments  |  CHAMPION: Unweighted XGBoost (PR-AUC: 0.3584, Brier: 0.1431)  |  EST. SAVINGS: $98,920 (Modeled Scenario)"
])

# -------------------------------------------------------------
# SLIDE 2: Dataset, Quality & Leakage Prevention
# -------------------------------------------------------------
s2 = prs.slides.add_slide(blank_layout)
apply_background(s2)
add_header(s2, "Data Sanitization, Temporal Partitioning & Leakage Prevention", "Data Architecture & Integrity Audit | Steps 1–3")

add_card(s2, 0.8, 1.5, 5.6, 5.4, "Point-in-Time Architecture & Temporal Splits", [
    "• Data Sanitization: Filtered 1 negative age anomaly (age = -1) and 5 inverted-date records; verified 0 null values across 110,521 records.",
    "• Strict Chronological Partitioning:",
    "   - Train (April 29 – May 19, 2016): 63,532 appointments (20.95% no-show)",
    "   - Validation (May 20 – May 31, 2016): 20,539 appointments (20.08% no-show)",
    "   - Chronological Holdout (June 1 – June 8, 2016): 26,450 appointments (18.46% no-show)",
    "• Chronological Holdout Qualification: June 1–8 was excluded from model training/validation. Exploratory analysis of its predictions previously occurred, so it is not presented as an untouched test set.",
    "• Point-in-Time Patient History: Filtered strictly for events where appointment date preceded scheduling date, eliminating 27,525 contaminated records.",
    "• Zero Leakage Audit Violations: Automated audit confirmed 0 point-in-time violations.",
    "• Exclusion of sms_received: SMS status is dispatched after booking; excluding it avoids operational target mismatch."
], CYAN)

img_path = 'steps/02_eda/figures/05_sms_leadtime_paradox.png'
if os.path.exists(img_path):
    s2.shapes.add_picture(img_path, Inches(6.8), Inches(1.5), width=Inches(5.7))

# -------------------------------------------------------------
# SLIDE 3: Feature Engineering & Model Benchmarking
# -------------------------------------------------------------
s3 = prs.slides.add_slide(blank_layout)
apply_background(s3)
add_header(s3, "Feature Engineering, Benchmark Results & Model Selection", "Model Benchmarking & Evaluation | Step 4")

add_card(s3, 0.8, 1.5, 5.8, 5.4, "Model Comparison & Performance Benchmarks", [
    "• 17 Approved Predictor Features across 3 distinct domains:",
    "   - Static Demographics: age, gender, scholarship, chronic comorbidities",
    "   - Booking / Temporal: lead_days, same_day_booking, day of week",
    "   - Historical Behaviour: prior_appointments, prior_noshow_rate, recency",
    "• Evaluation Metric: PR-AUC (Precision-Recall AUC) chosen due to 20% positive class imbalance; baseline PR-AUC = 0.2008.",
    "• Validation Benchmark Results (Verified from model_benchmark_summary.csv):",
    "   - Logistic Regression:       PR-AUC = 0.3524 | ROC-AUC = 0.7302 | Brier = 0.2209",
    "   - Random Forest:             PR-AUC = 0.3546 | ROC-AUC = 0.7365 | Brier = 0.2123",
    "   - XGBoost (Class Weighted):  PR-AUC = 0.3597 | ROC-AUC = 0.7385 | Brier = 0.2166",
    "   - XGBoost (Unweighted):      PR-AUC = 0.3584 | ROC-AUC = 0.7382 | Brier = 0.1431 (Winner!)",
    "• Why Unweighted XGBoost Won: Class weighting degraded probability calibration (Brier: 0.2166, ECE: 0.2503). Unweighted XGBoost preserved pristine calibration (Brier: 0.1431, ECE: 0.0240), essential for threshold tuning."
], GREEN)

arch_img = 'docs/architecture_diagram.png'
if os.path.exists(arch_img):
    s3.shapes.add_picture(arch_img, Inches(6.9), Inches(1.5), width=Inches(5.6))

# -------------------------------------------------------------
# SLIDE 4: Threshold Optimization & Business Cost Analysis
# -------------------------------------------------------------
s4 = prs.slides.add_slide(blank_layout)
apply_background(s4)
add_header(s4, "Decision Threshold Evaluation & Healthcare Cost Matrix", "Operational Decision Analytics | Step 5")

add_card(s4, 0.8, 1.5, 5.6, 5.4, "Asymmetric Healthcare Cost Analysis", [
    "• The Default 0.50 Threshold Failure:",
    "   - At threshold 0.50, recall is only 3.47% (captures 143 of 4,124 no-shows!).",
    "   - Healthcare cost of False Negatives ($200) far exceeds cost of outreach ($10).",
    "• Illustrative Cost Assumptions (Validation Cohort: 20,539 Appointments):",
    "   - Intervention Cost: $10 per flagged patient",
    "   - Cost of Unfilled Clinical Slot: $200 per missed appointment",
    "   - Outreach Effectiveness (r): 30% reduction in no-show probability",
    "• Total Cost Formula:",
    "   Total Cost = Flagged * $10 + [FN + (1 - r) * TP] * $200",
    "• Lowest-Cost Evaluated Operating Point (Threshold = 0.15):",
    "   - Recall increases from 3.47% to 93.62% (identifies 3,861 no-shows)",
    "   - Precision stands at 29.09% (sufficient for low-cost interventions)",
    "   - Modeled cost drops from $824,800 (baseline) to $725,880 ($819,250 at 0.50)",
    "   - Estimated Net Savings: $98,920 vs baseline ($93,370 vs 0.50). Modeled scenario under assumed parameters, not measured operational savings."
], CYAN)

cm_img = 'steps/05_evaluation/figures/confusion_matrix_threshold_comparison.png'
if os.path.exists(cm_img):
    s4.shapes.add_picture(cm_img, Inches(6.8), Inches(1.5), width=Inches(5.7))

# -------------------------------------------------------------
# SLIDE 5: SHAP Explainability, Governance & Limitations
# -------------------------------------------------------------
s5 = prs.slides.add_slide(blank_layout)
apply_background(s5)
add_header(s5, "Dual-Space SHAP Interpretability, Clinical Governance & Limitations", "Model Explainability & Healthcare Governance | Step 6")

add_card(s5, 0.8, 1.5, 5.8, 5.4, "Dual-Space SHAP Insights & Clinical Governance", [
    "• Dual-Space Architecture:",
    "   - Raw Margin (Log-Odds): Engineering baseline; exact sigmoid parity.",
    "   - Probability Space: Clinician-friendly additive explanations:",
    "     P(no-show | x) = Base Value (18.65%) + Sum of Feature Shifts (% pts).",
    "   - Background Dataset: 200 rows supplied from train; 100 effective background observations evaluated internally via SHAP Independent masker.",
    "• Top Empirical Probability Drivers:",
    "   - Lead Days: +10.79% pts (Primary operational lever: delay increases risk).",
    "   - Patient Age: +3.28% pts (Young adults higher risk; seniors protective).",
    "   - Same-Day Booking: +1.77% pts (Immediate appointments attend reliably).",
    "   - Historical No-Show Rate: +0.76% pts (Past non-attendance ramps risk).",
    "• Governance & Non-Discrimination Principles:",
    "   - Statistical Attribution != Causality (Do not reschedule arbitrarily!).",
    "   - Non-Punitive Deployment: Risk scores must ONLY allocate supportive care (navigation, reminders, transit vouchers), NEVER restrict access."
], GREEN)

shap_img = 'steps/06_shap_explainability/figures/shap_prob_bar_importance.png'
if os.path.exists(shap_img):
    s5.shapes.add_picture(shap_img, Inches(6.9), Inches(1.5), width=Inches(5.6))

prs.save('docs/presentation.pptx')
print("Presentation successfully saved to docs/presentation.pptx!")

try:
    prs.save('presentation.pptx')
    print("Presentation successfully saved to presentation.pptx!")
except PermissionError:
    print("Note: presentation.pptx in root directory is currently open in Microsoft PowerPoint and locked for writing.")
    print("docs/presentation.pptx has been successfully updated with all reconciled metrics and qualifications.")
