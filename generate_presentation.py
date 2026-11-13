"""
generate_presentation.py — Professional 9-slide PPTX for Healthcare No-Show Prediction
Redesigned from scratch: dark-theme, KPI cards, native charts, fitted images, speaker notes.
"""

import csv
import json
import os
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu

# ============================================================================
# DESIGN SYSTEM — constants reused everywhere
# ============================================================================
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.6)
COL_W = (SLIDE_W - 2 * MARGIN) / 12  # 12-column grid unit

# Palette
BG        = RGBColor(0x0F, 0x17, 0x2A)  # #0F172A
SURFACE   = RGBColor(0x1E, 0x29, 0x3B)  # #1E293B
BORDER    = RGBColor(0x33, 0x41, 0x55)  # #334155
TEXT      = RGBColor(0xF8, 0xFA, 0xFC)  # #F8FAFC
MUTED     = RGBColor(0x94, 0xA3, 0xB8)  # #94A3B8
CYAN      = RGBColor(0x38, 0xBD, 0xF8)  # #38BDF8
GREEN     = RGBColor(0x34, 0xD3, 0x99)  # #34D399
WARNING   = RGBColor(0xFB, 0xBF, 0x24)  # #FBBF24
RED_SOFT  = RGBColor(0xF8, 0x71, 0x71)  # #F87171

# Font
FONT = 'Calibri'

# ============================================================================
# DATA — loaded from repo artifacts where possible, fallback to verified values
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

def _load_data():
    """Load verified numbers from repo CSVs/JSON."""
    data = {}

    # Model benchmarks
    bench_path = PROJECT_ROOT / 'data' / 'processed' / 'model_benchmark_summary.csv'
    if bench_path.exists():
        with open(bench_path, newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        data['benchmarks'] = {}
        for r in rows:
            name = r['model'].strip()
            data['benchmarks'][name] = {
                'pr_auc': float(r['pr_auc']),
                'roc_auc': float(r['roc_auc']),
                'brier': float(r['brier_score']),
                'ece': float(r['ece']),
            }

    # Threshold comparison
    thresh_path = PROJECT_ROOT / 'data' / 'processed' / 'step5_threshold_comparison.csv'
    if thresh_path.exists():
        with open(thresh_path, newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        data['thresholds'] = {}
        for r in rows:
            t = float(r['threshold'])
            data['thresholds'][t] = {
                'recall': float(r['recall']),
                'precision': float(r['precision']),
                'tp': int(r['true_positives']),
                'fn': int(r['false_negatives']),
                'fp': int(r['false_positives']),
                'tn': int(r['true_negatives']),
                'flagged': int(r['flagged_count']),
                'baseline_cost': float(r['baseline_cost']),
                'expected_cost': float(r['expected_cost']),
                'net_savings': float(r['net_savings']),
            }

    # SHAP feature importance
    shap_path = PROJECT_ROOT / 'data' / 'processed' / 'shap_probability_feature_importance.csv'
    if shap_path.exists():
        with open(shap_path, newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        data['shap_top'] = []
        for r in sorted(rows, key=lambda x: float(x['mean_abs_shap_pct']), reverse=True)[:6]:
            data['shap_top'].append({
                'name': r['clean_name'],
                'pct': float(r['mean_abs_shap_pct']),
            })

    return data

DATA = _load_data()

# Verified project-level constants
TOTAL_RAW = 110527
TOTAL_CLEAN = 110521
UNIQUE_PATIENTS = 62298
NOSHOW_COUNT = 22314
NOSHOW_RATE = 20.19
TRAIN_N = 63532
VAL_N = 20539
HOLDOUT_N = 26450
CONTAMINATED_ROWS = 27525
SHAP_BASELINE = 18.65

# Extract from loaded data with verified fallbacks
bm = DATA.get('benchmarks', {})
xgb_uw = bm.get('XGBoost (Unweighted)', {'pr_auc': 0.3584, 'brier': 0.1431, 'ece': 0.024})
xgb_w  = bm.get('XGBoost (Weighted)', {'pr_auc': 0.3597, 'brier': 0.2166, 'ece': 0.2503})
rf     = bm.get('Random Forest', {'pr_auc': 0.3546, 'brier': 0.2123})
lr     = bm.get('Logistic Regression', {'pr_auc': 0.3524, 'brier': 0.2209})

t15 = DATA.get('thresholds', {}).get(0.15, {
    'recall': 0.9362, 'tp': 3861, 'fn': 263, 'fp': 9413, 'tn': 7002,
    'flagged': 13274, 'baseline_cost': 824800, 'expected_cost': 725880, 'net_savings': 98920,
})

shap_top = DATA.get('shap_top', [
    {'name': 'lead_days', 'pct': 10.79},
    {'name': 'age', 'pct': 3.28},
    {'name': 'same_day_booking', 'pct': 1.77},
    {'name': 'prior_noshow_rate', 'pct': 0.76},
    {'name': 'prior_appointments', 'pct': 0.70},
    {'name': 'days_since_last_appointment', 'pct': 0.62},
])

# Figure paths
FIG = {
    'architecture': PROJECT_ROOT / 'docs' / 'architecture_diagram.png',
    'confusion':    PROJECT_ROOT / 'steps' / '05_evaluation' / 'figures' / 'confusion_matrix_threshold_comparison.png',
    'shap_bar':     PROJECT_ROOT / 'steps' / '06_shap_explainability' / 'figures' / 'shap_prob_bar_importance.png',
    'shap_waterfall': PROJECT_ROOT / 'steps' / '06_shap_explainability' / 'figures' / 'shap_prob_waterfall_high_risk.png',
    'dashboard':    PROJECT_ROOT / 'docs' / 'dashboard_screenshots' / '01_executive_overview.png',
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK_LAYOUT = prs.slide_layouts[6]  # blank


def _set_bg(slide):
    """Set proper slide background via the background.fill API."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = BG


def _add_accent_line(slide, y=None):
    """Thin cyan accent line under the title area."""
    if y is None:
        y = Inches(1.45)
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, MARGIN, y, SLIDE_W - 2 * MARGIN, Pt(2)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = CYAN
    line.line.fill.background()


def _add_category_tag(slide, text, left=None, top=None):
    """Small uppercase category label above the title."""
    left = left or MARGIN
    top = top or Inches(0.35)
    tb = slide.shapes.add_textbox(left, top, Inches(6), Inches(0.3))
    tf = tb.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.text = text.upper()
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = CYAN
    p.font.name = FONT


def _add_title(slide, text, left=None, top=None, width=None, size=None):
    """Add slide title in bold white Calibri."""
    left = left or MARGIN
    top = top or Inches(0.6)
    width = width or (SLIDE_W - 2 * MARGIN)
    size = size or Pt(30)
    tb = slide.shapes.add_textbox(left, top, width, Inches(0.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = size
    p.font.bold = True
    p.font.color.rgb = TEXT
    p.font.name = FONT
    return tb


def _add_subtitle(slide, text, left=None, top=None, width=None, color=None):
    """Add a subtitle line."""
    left = left or MARGIN
    top = top or Inches(1.15)
    width = width or (SLIDE_W - 2 * MARGIN)
    tb = slide.shapes.add_textbox(left, top, width, Inches(0.35))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(16)
    p.font.color.rgb = color or MUTED
    p.font.name = FONT
    return tb


def _add_footer(slide, slide_num, total=9):
    """Footer bar with project name and slide number."""
    # Separator line
    sep = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), SLIDE_H - Inches(0.38), SLIDE_W, Pt(1)
    )
    sep.fill.solid()
    sep.fill.fore_color.rgb = BORDER
    sep.line.fill.background()

    tb = slide.shapes.add_textbox(MARGIN, SLIDE_H - Inches(0.35), Inches(6), Inches(0.3))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "Healthcare No-Show Prediction & Capacity Optimization"
    p.font.size = Pt(9)
    p.font.color.rgb = MUTED
    p.font.name = FONT

    tb2 = slide.shapes.add_textbox(SLIDE_W - MARGIN - Inches(1), SLIDE_H - Inches(0.35), Inches(1), Inches(0.3))
    tf2 = tb2.text_frame
    p2 = tf2.paragraphs[0]
    p2.text = f"{slide_num} / {total}"
    p2.font.size = Pt(9)
    p2.font.color.rgb = MUTED
    p2.font.name = FONT
    p2.alignment = PP_ALIGN.RIGHT


def _add_notes(slide, text):
    """Set speaker notes for a slide."""
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = text


def _add_kpi_card(slide, left, top, width, height, number, label, accent=CYAN, number_size=None):
    """Big-number KPI card with a surface-colored rounded rectangle."""
    number_size = number_size or Pt(44)

    # Card background
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    card.fill.solid()
    card.fill.fore_color.rgb = SURFACE
    card.line.color.rgb = BORDER
    card.line.width = Pt(1)

    # Accent top edge
    edge = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, left, top, width, Pt(3)
    )
    edge.fill.solid()
    edge.fill.fore_color.rgb = accent
    edge.line.fill.background()

    # Number
    tb = slide.shapes.add_textbox(
        left + Inches(0.2), top + Inches(0.2), width - Inches(0.4), Inches(0.8)
    )
    tf = tb.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.text = str(number)
    p.font.size = number_size
    p.font.bold = True
    p.font.color.rgb = accent
    p.font.name = FONT
    p.alignment = PP_ALIGN.CENTER

    # Label
    tb2 = slide.shapes.add_textbox(
        left + Inches(0.15), top + height - Inches(0.5), width - Inches(0.3), Inches(0.4)
    )
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = label
    p2.font.size = Pt(12)
    p2.font.color.rgb = MUTED
    p2.font.name = FONT
    p2.alignment = PP_ALIGN.CENTER

    return card


def _add_bullets(slide, items, left, top, width, height, font_size=None):
    """Add bulleted text with proper XML bullet characters."""
    font_size = font_size or Pt(16)
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True

    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.font.size = font_size
        p.font.color.rgb = TEXT
        p.font.name = FONT
        p.space_after = Pt(8)

        # Set XML bullet character
        pPr = p._pPr
        if pPr is None:
            pPr = p._p.get_or_add_pPr()
        buChar = pPr.makeelement(qn('a:buChar'), {'char': '▸'})
        # Remove any existing bullet
        for existing in pPr.findall(qn('a:buChar')):
            pPr.remove(existing)
        for existing in pPr.findall(qn('a:buNone')):
            pPr.remove(existing)
        pPr.append(buChar)

        # Add space after bullet via run
        run = p.add_run()
        run.text = f" {item}"
        run.font.size = font_size
        run.font.color.rgb = TEXT
        run.font.name = FONT

    return tb


def _add_image_fit(slide, img_path, left, top, max_w, max_h):
    """Add an image preserving aspect ratio within a bounding box."""
    if not os.path.exists(img_path):
        return None
    from PIL import Image as PILImage
    with PILImage.open(img_path) as im:
        img_w, img_h = im.size

    # Calculate scaling
    scale_w = max_w / img_w
    scale_h = max_h / img_h
    scale = min(scale_w, scale_h)
    final_w = int(img_w * scale)
    final_h = int(img_h * scale)

    # Center within bounding box
    offset_x = (max_w - final_w) // 2
    offset_y = (max_h - final_h) // 2

    return slide.shapes.add_picture(
        str(img_path), left + offset_x, top + offset_y, final_w, final_h
    )


def _add_flow_box(slide, left, top, width, height, text, accent=CYAN):
    """Rounded rectangle for architecture flow diagram."""
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    box.fill.solid()
    box.fill.fore_color.rgb = SURFACE
    box.line.color.rgb = accent
    box.line.width = Pt(1.5)

    box.text_frame.word_wrap = True
    box.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    p = box.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT
    p.font.name = FONT
    box.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

    return box


def _add_arrow(slide, left, top, width, height):
    """Right-pointing arrow connector."""
    arrow = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW, left, top, width, height
    )
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = CYAN
    arrow.line.fill.background()
    return arrow


# ============================================================================
# SLIDE 1: TITLE
# ============================================================================
def slide_title():
    s = prs.slides.add_slide(BLANK_LAYOUT)
    _set_bg(s)

    # Category tag
    _add_category_tag(s, "End-to-End Machine Learning Portfolio Project")

    # Title
    _add_title(s, "Healthcare Appointment No-Show\nPrediction & Capacity Optimization",
               top=Inches(1.5), size=Pt(36))

    # Subtitle
    _add_subtitle(s, "A leakage-audited, cost-optimized XGBoost pipeline with dual-space SHAP explainability",
                  top=Inches(2.8), color=MUTED)

    # Accent line
    _add_accent_line(s, y=Inches(3.3))

    # 3 mini KPI badges
    badge_y = Inches(4.0)
    badge_h = Inches(1.6)
    badge_w = Inches(3.2)
    gap = Inches(0.5)
    total_w = 3 * badge_w + 2 * gap
    start_x = (SLIDE_W - total_w) // 2

    _add_kpi_card(s, start_x, badge_y, badge_w, badge_h,
                  "110,521", "Cleaned Appointments Analyzed", CYAN, Pt(42))
    _add_kpi_card(s, start_x + badge_w + gap, badge_y, badge_w, badge_h,
                  "93.62%", "No-Show Recall at τ = 0.15", GREEN, Pt(42))
    _add_kpi_card(s, start_x + 2 * (badge_w + gap), badge_y, badge_w, badge_h,
                  "$98,920", "Modeled Net Savings", WARNING, Pt(42))

    # Author line
    _add_subtitle(s, "Gaurav Raj  ·  Python  ·  XGBoost  ·  SHAP  ·  Streamlit  ·  2024",
                  top=Inches(6.2), color=MUTED)

    _add_footer(s, 1)
    _add_notes(s, (
        "SPEAKER NOTES — Title Slide\n\n"
        "Elevator pitch: This is an end-to-end ML project that predicts whether "
        "a healthcare outpatient will miss their appointment. It was built on "
        "110,521 cleaned appointments from Vitória, Brazil (2016). The pipeline "
        "strictly prevents temporal data leakage via chronological splitting and "
        "point-in-time feature engineering. The champion unweighted XGBoost model "
        "achieves PR-AUC 0.3584 with excellent calibration (Brier 0.1431, ECE 0.024). "
        "At threshold 0.15, recall reaches 93.62% with modeled net savings of $98,920 "
        "on the validation cohort under illustrative cost assumptions "
        "($10 intervention, $200 unfilled slot, 30% outreach success).\n\n"
        "Tech stack: Python, scikit-learn, XGBoost, SHAP, pandas, Streamlit, Plotly."
    ))


# ============================================================================
# SLIDE 2: PROBLEM
# ============================================================================
def slide_problem():
    s = prs.slides.add_slide(BLANK_LAYOUT)
    _set_bg(s)

    _add_category_tag(s, "Business Problem & Motivation")
    _add_title(s, "One in Five Patients Misses Their Appointment")
    _add_accent_line(s)

    # Giant KPI — no-show rate
    _add_kpi_card(s, Inches(0.8), Inches(1.8), Inches(4.5), Inches(2.8),
                  "20.19%", "Baseline No-Show Rate\n22,314 of 110,521 Cleaned Visits",
                  CYAN, Pt(54))

    # Supporting KPIs
    col_left = Inches(5.8)
    card_w = Inches(3.3)
    card_h = Inches(1.2)
    _add_kpi_card(s, col_left, Inches(1.8), card_w, card_h,
                  "$200", "Estimated Cost per Unfilled Slot", WARNING, Pt(36))
    _add_kpi_card(s, col_left + card_w + Inches(0.3), Inches(1.8), card_w, card_h,
                  "62,298", "Unique Patients in Cohort", GREEN, Pt(36))
    _add_kpi_card(s, col_left, Inches(3.2), card_w, card_h,
                  "$10", "Outreach Intervention Cost", CYAN, Pt(36))
    _add_kpi_card(s, col_left + card_w + Inches(0.3), Inches(3.2), card_w, card_h,
                  "20:1", "Cost Asymmetry Ratio", RED_SOFT, Pt(36))

    # Bullets
    _add_bullets(s, [
        "Idle clinical capacity and lost revenue from unused provider time slots",
        "Disrupted chronic disease management leads to preventable ER visits",
        "Outreach teams cannot manually call every patient — need automated prioritization",
    ], MARGIN, Inches(5.0), SLIDE_W - 2 * MARGIN, Inches(1.5), Pt(16))

    _add_footer(s, 2)
    _add_notes(s, (
        "SPEAKER NOTES — Problem\n\n"
        "The dataset comes from the Brazilian public health system (SUS) in Vitória, "
        "Espírito Santo. The 20.19% no-show rate represents 22,314 missed appointments "
        "out of 110,521 cleaned records. The cost assumptions used throughout the project "
        "are illustrative: $200 represents the estimated opportunity cost of an unfilled "
        "clinical slot (provider time, equipment, room), while $10 represents the marginal "
        "cost of an outreach intervention (phone call, SMS, navigator time). "
        "The 20:1 asymmetry means missing a no-show is 20× costlier than a wasted phone call, "
        "making aggressive recall thresholds economically rational. "
        "US healthcare systems lose an estimated $150B+ annually to appointment non-attendance."
    ))


# ============================================================================
# SLIDE 3: DATA & LEAKAGE PREVENTION
# ============================================================================
def slide_data_leakage():
    s = prs.slides.add_slide(BLANK_LAYOUT)
    _set_bg(s)

    _add_category_tag(s, "Data Integrity & Temporal Architecture")
    _add_title(s, "Chronological Splits and Zero Leakage Ensure Honest Evaluation")
    _add_accent_line(s)

    # 3-block chronological split visualization
    block_y = Inches(1.8)
    block_h = Inches(1.4)
    block_w = Inches(3.5)
    gap = Inches(0.3)
    start_x = Inches(0.8)

    # Train block
    _add_flow_box(s, start_x, block_y, block_w, block_h,
                  f"TRAIN\nApr 29 – May 19\n{TRAIN_N:,} appts", CYAN)
    # Arrow
    _add_arrow(s, start_x + block_w + Inches(0.05), block_y + Inches(0.5), Inches(0.2), Inches(0.4))
    # Val block
    _add_flow_box(s, start_x + block_w + gap, block_y, block_w, block_h,
                  f"VALIDATION\nMay 20 – May 31\n{VAL_N:,} appts", GREEN)
    # Arrow
    _add_arrow(s, start_x + 2*block_w + gap + Inches(0.05), block_y + Inches(0.5), Inches(0.2), Inches(0.4))
    # Holdout block
    _add_flow_box(s, start_x + 2*(block_w + gap), block_y, block_w, block_h,
                  f"HOLDOUT\nJun 1 – Jun 8\n{HOLDOUT_N:,} appts", WARNING)

    # KPI cards row
    kpi_y = Inches(3.6)
    kpi_w = Inches(2.8)
    kpi_h = Inches(1.5)
    kpi_gap = Inches(0.35)
    kpi_start = Inches(0.8)

    _add_kpi_card(s, kpi_start, kpi_y, kpi_w, kpi_h,
                  "27,525", "Contaminated Rows\nRemediated", RED_SOFT, Pt(36))
    _add_kpi_card(s, kpi_start + kpi_w + kpi_gap, kpi_y, kpi_w, kpi_h,
                  "0", "Point-in-Time\nLeakage Violations", GREEN, Pt(36))
    _add_kpi_card(s, kpi_start + 2*(kpi_w + kpi_gap), kpi_y, kpi_w, kpi_h,
                  "6", "Anomalous Records\nRemoved", CYAN, Pt(36))
    _add_kpi_card(s, kpi_start + 3*(kpi_w + kpi_gap), kpi_y, kpi_w, kpi_h,
                  "17", "Day-0 Approved\nFeatures", CYAN, Pt(36))

    # Bullets
    _add_bullets(s, [
        "Patient history features computed strictly before each appointment's scheduling date",
        "sms_received excluded — dispatched post-booking, causes operational target mismatch",
        "Holdout was previously explored; not presented as an untouched test set",
    ], MARGIN, Inches(5.4), SLIDE_W - 2 * MARGIN, Inches(1.3), Pt(16))

    _add_footer(s, 3)
    _add_notes(s, (
        "SPEAKER NOTES — Data & Leakage\n\n"
        "Critical engineering decision: we use chronological splits, not random K-fold, "
        "because random splits leak future appointment outcomes into training data. "
        "The point-in-time audit found 27,525 historical records where prior appointments "
        "occurred on or after the current scheduling timestamp — these were properly excluded "
        "during feature computation. The automated audit confirmed zero remaining violations.\n\n"
        "Six anomalies were removed: 1 negative age (-1 years) and 5 records with "
        "appointment dates before their scheduling dates (inverted timestamps).\n\n"
        "sms_received is excluded because it is dispatched by the clinic AFTER the appointment "
        "is booked. Including it would create a target information leak and is not available "
        "at Day 0 when predictions need to be made.\n\n"
        "Honesty caveat: The June 1–8 holdout was explored during earlier development, "
        "so we present it transparently as a previously-explored holdout rather than "
        "claiming it is an untouched test set."
    ))


# ============================================================================
# SLIDE 4: ARCHITECTURE / APPROACH
# ============================================================================
def slide_architecture():
    s = prs.slides.add_slide(BLANK_LAYOUT)
    _set_bg(s)

    _add_category_tag(s, "End-to-End Pipeline Architecture")
    _add_title(s, "A Six-Stage Pipeline from Raw Data to Explainable Decisions")
    _add_accent_line(s)

    # Flow diagram — 6 boxes with arrows
    box_w = Inches(1.65)
    box_h = Inches(1.2)
    arrow_w = Inches(0.25)
    arrow_h = Inches(0.35)
    flow_y = Inches(2.2)
    start_x = Inches(0.5)
    step = box_w + Inches(0.35)

    stages = [
        ("1. DATA\nAUDIT", "110,521\ncleaned"),
        ("2. EDA &\nFEATURES", "17 Day-0\nfeatures"),
        ("3. TEMPORAL\nSPLIT", "Train / Val\n/ Holdout"),
        ("4. XGBOOST\nTRAINING", "Unweighted\nchampion"),
        ("5. THRESHOLD\nOPTIMIZATION", "τ = 0.15\ncost-optimal"),
        ("6. DUAL-SPACE\nSHAP", "Probability\n& log-odds"),
    ]

    colors = [CYAN, CYAN, GREEN, WARNING, GREEN, CYAN]

    for i, (label, sub) in enumerate(stages):
        x = start_x + i * step
        _add_flow_box(s, x, flow_y, box_w, box_h, f"{label}\n─\n{sub}", colors[i])
        if i < len(stages) - 1:
            _add_arrow(s, x + box_w + Inches(0.02), flow_y + Inches(0.42),
                      arrow_w, arrow_h)

    # Feature domain cards below
    domain_y = Inches(4.0)
    domain_w = Inches(3.7)
    domain_h = Inches(1.8)
    domain_gap = Inches(0.35)
    domain_start = Inches(0.8)

    domains = [
        ("STATIC DEMOGRAPHICS", "age, gender, scholarship,\nhypertension, diabetes,\nalcoholism, has_handicap", CYAN),
        ("BOOKING & TEMPORAL", "lead_days, same_day_booking,\nappointment_dow, scheduled_dow,\nappointment_month", GREEN),
        ("PATIENT HISTORY", "prior_appointments, prior_noshows,\nprior_noshow_rate,\ndays_since_last, is_first_appointment", WARNING),
    ]

    for i, (title, features, color) in enumerate(domains):
        x = domain_start + i * (domain_w + domain_gap)
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, x, domain_y, domain_w, domain_h
        ) if False else None  # We'll use the helper

        # Card background
        card = s.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, x, domain_y, domain_w, domain_h
        )
        card.fill.solid()
        card.fill.fore_color.rgb = SURFACE
        card.line.color.rgb = color
        card.line.width = Pt(1.5)

        # Card title
        tb = s.shapes.add_textbox(x + Inches(0.2), domain_y + Inches(0.12),
                                  domain_w - Inches(0.4), Inches(0.3))
        p = tb.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = color
        p.font.name = FONT

        # Card body
        tb2 = s.shapes.add_textbox(x + Inches(0.2), domain_y + Inches(0.5),
                                   domain_w - Inches(0.4), domain_h - Inches(0.6))
        tf2 = tb2.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        p2.text = features
        p2.font.size = Pt(14)
        p2.font.color.rgb = TEXT
        p2.font.name = FONT

    _add_footer(s, 4)
    _add_notes(s, (
        "SPEAKER NOTES — Architecture\n\n"
        "The pipeline has 6 sequential stages. Each stage produces artifacts that feed the next.\n\n"
        "Stage 1 (Data Audit): Raw Kaggle CSV → schema parsing, anomaly detection, cleaning → "
        "110,521 records (6 removed).\n"
        "Stage 2 (EDA & Features): Statistical distributions, risk factor analysis → 17 Day-0 "
        "features engineered across 3 domains (demographics, temporal booking, patient history).\n"
        "Stage 3 (Temporal Split): Chronological partitioning into train (63,532), validation "
        "(20,539), and holdout (26,450). No random shuffling.\n"
        "Stage 4 (XGBoost Training): Benchmark 4 model architectures. Unweighted XGBoost selected "
        "for superior probability calibration (Brier 0.1431 vs 0.2166 weighted).\n"
        "Stage 5 (Threshold Optimization): Cost-utility framework replacing default 0.50 with "
        "0.15, the lowest-cost evaluated candidate.\n"
        "Stage 6 (Dual-Space SHAP): Raw-margin log-odds explainability for engineers + "
        "probability-space percentage-point explanations for clinicians. Baseline E[f(X)] = 18.65%."
    ))


# ============================================================================
# SLIDE 5: MODEL BENCHMARK (with native chart)
# ============================================================================
def slide_benchmark():
    s = prs.slides.add_slide(BLANK_LAYOUT)
    _set_bg(s)

    _add_category_tag(s, "Model Selection & Evaluation — Step 4")
    _add_title(s, "Unweighted XGBoost Wins on Calibration, Not Just Accuracy")
    _add_accent_line(s)

    # Native grouped bar chart: PR-AUC and Brier Score
    chart_data = CategoryChartData()
    chart_data.categories = ['Logistic\nRegression', 'Random\nForest',
                             'XGBoost\n(Weighted)', 'XGBoost\n(Unweighted)']
    chart_data.add_series('PR-AUC', (lr['pr_auc'], rf['pr_auc'],
                                      xgb_w['pr_auc'], xgb_uw['pr_auc']))
    chart_data.add_series('Brier Score', (lr['brier'], rf['brier'],
                                          xgb_w['brier'], xgb_uw['brier']))

    chart_frame = s.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.8), Inches(1.7),
        Inches(8.0), Inches(4.5), chart_data
    )
    chart = chart_frame.chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.color.rgb = TEXT
    chart.legend.font.size = Pt(12)
    chart.legend.font.name = FONT

    # Style chart
    plot = chart.plots[0]
    plot.gap_width = 120

    # Series 0 — PR-AUC (cyan)
    s0 = plot.series[0]
    s0.format.fill.solid()
    s0.format.fill.fore_color.rgb = CYAN
    s0.has_data_labels = True
    s0.data_labels.font.size = Pt(11)
    s0.data_labels.font.color.rgb = TEXT
    s0.data_labels.font.name = FONT
    s0.data_labels.number_format = '0.0000'
    s0.data_labels.label_position = XL_LABEL_POSITION.OUTSIDE_END

    # Series 1 — Brier Score (warning/amber)
    s1 = plot.series[1]
    s1.format.fill.solid()
    s1.format.fill.fore_color.rgb = WARNING
    s1.has_data_labels = True
    s1.data_labels.font.size = Pt(11)
    s1.data_labels.font.color.rgb = TEXT
    s1.data_labels.font.name = FONT
    s1.data_labels.number_format = '0.0000'
    s1.data_labels.label_position = XL_LABEL_POSITION.OUTSIDE_END

    # Axis styling
    cat_axis = chart.category_axis
    cat_axis.tick_labels.font.size = Pt(11)
    cat_axis.tick_labels.font.color.rgb = TEXT
    cat_axis.tick_labels.font.name = FONT
    cat_axis.format.line.color.rgb = BORDER
    cat_axis.has_major_gridlines = False

    val_axis = chart.value_axis
    val_axis.tick_labels.font.size = Pt(10)
    val_axis.tick_labels.font.color.rgb = MUTED
    val_axis.tick_labels.font.name = FONT
    val_axis.format.line.color.rgb = BORDER
    val_axis.major_gridlines.format.line.color.rgb = BORDER
    val_axis.major_gridlines.format.line.dash_style = 2  # dash
    val_axis.minimum_scale = 0
    val_axis.maximum_scale = 0.4

    # Chart area background — set via chart plot's XML
    chart.chart_style = 2

    # KPI cards on right
    kpi_x = Inches(9.3)
    kpi_w = Inches(3.5)
    kpi_h = Inches(1.3)

    _add_kpi_card(s, kpi_x, Inches(1.8), kpi_w, kpi_h,
                  f"{xgb_uw['pr_auc']:.4f}", "PR-AUC (Champion)", CYAN, Pt(38))
    _add_kpi_card(s, kpi_x, Inches(3.3), kpi_w, kpi_h,
                  f"{xgb_uw['brier']:.4f}", "Brier Score (Lower = Better)", GREEN, Pt(38))
    _add_kpi_card(s, kpi_x, Inches(4.8), kpi_w, kpi_h,
                  f"{xgb_uw['ece']:.4f}", "Expected Calibration Error", WARNING, Pt(38))

    _add_footer(s, 5)
    _add_notes(s, (
        "SPEAKER NOTES — Model Benchmark\n\n"
        "Four model architectures were evaluated on the chronological validation set:\n"
        f"- Logistic Regression: PR-AUC {lr['pr_auc']:.4f}, Brier {lr['brier']:.4f}\n"
        f"- Random Forest: PR-AUC {rf['pr_auc']:.4f}, Brier {rf['brier']:.4f}\n"
        f"- XGBoost (Class Weighted): PR-AUC {xgb_w['pr_auc']:.4f}, Brier {xgb_w['brier']:.4f}, ECE {xgb_w['ece']:.4f}\n"
        f"- XGBoost (Unweighted): PR-AUC {xgb_uw['pr_auc']:.4f}, Brier {xgb_uw['brier']:.4f}, ECE {xgb_uw['ece']:.4f}\n\n"
        "Key insight: The class-weighted XGBoost has marginally higher PR-AUC (0.3597 vs 0.3584) "
        "but catastrophically worse calibration (Brier 0.2166, ECE 0.2503). Since we need "
        "well-calibrated probabilities for threshold tuning and cost optimization, the "
        "unweighted variant is the clear winner.\n\n"
        "PR-AUC is the primary metric because with 20% positive class prevalence, ROC-AUC "
        "can be misleadingly optimistic — PR-AUC directly measures precision-recall tradeoffs "
        "in the minority class."
    ))


# ============================================================================
# SLIDE 6: THRESHOLD & COST OPTIMIZATION
# ============================================================================
def slide_threshold():
    s = prs.slides.add_slide(BLANK_LAYOUT)
    _set_bg(s)

    _add_category_tag(s, "Decision Analytics & Cost Optimization — Step 5")
    _add_title(s, "Threshold 0.15 Captures 94% of No-Shows, Saves $98,920")
    _add_accent_line(s)

    # 4 KPI cards across the top
    kpi_y = Inches(1.7)
    kpi_w = Inches(2.8)
    kpi_h = Inches(1.5)
    kpi_gap = Inches(0.3)
    kpi_start = Inches(0.6)

    _add_kpi_card(s, kpi_start, kpi_y, kpi_w, kpi_h,
                  "$824,800", "Baseline Cost\n(Zero-Intervention Policy)", MUTED, Pt(34))
    _add_kpi_card(s, kpi_start + kpi_w + kpi_gap, kpi_y, kpi_w, kpi_h,
                  "$725,880", "Policy Cost at τ = 0.15", CYAN, Pt(34))
    _add_kpi_card(s, kpi_start + 2*(kpi_w + kpi_gap), kpi_y, kpi_w, kpi_h,
                  "$98,920", "Modeled Net Savings\n(12.0% Cost Reduction)", GREEN, Pt(34))
    _add_kpi_card(s, kpi_start + 3*(kpi_w + kpi_gap), kpi_y, kpi_w, kpi_h,
                  "93.62%", "No-Show Recall\n(3,861 of 4,124 Caught)", WARNING, Pt(34))

    # Confusion matrix image on the left
    _add_image_fit(s, str(FIG['confusion']),
                   Inches(0.8), Inches(3.5), Inches(5.5), Inches(3.2))

    # Bullets on the right
    _add_bullets(s, [
        "Default threshold 0.50 catches only 3.47% of no-shows — catastrophic",
        "Cost formula: Flagged × $10 + Unmitigated No-Shows × $200",
        "Modeled financial scenario under stated assumptions, not measured savings",
    ], Inches(6.8), Inches(3.6), Inches(5.8), Inches(2.5), Pt(16))

    _add_footer(s, 6)
    _add_notes(s, (
        "SPEAKER NOTES — Threshold & Cost Optimization\n\n"
        f"Validation cohort: {VAL_N:,} appointments, 4,124 actual no-shows (20.08%).\n\n"
        "At the default decision threshold of 0.50, only 143 of 4,124 no-shows are identified "
        "(3.47% recall). This is because the model outputs well-calibrated probabilities: "
        "with a 20% base rate, very few patients exceed 50% predicted risk.\n\n"
        "The cost-utility framework evaluates each candidate threshold using:\n"
        "  Total Cost = (# Flagged × $10) + [FN + (1 - 0.30) × TP] × $200\n"
        "  where r = 30% is the assumed outreach effectiveness.\n\n"
        f"At τ = 0.15: {t15['tp']} TP, {t15['fn']} FN, {t15['fp']} FP, {t15['tn']} TN.\n"
        f"  Baseline: $824,800 | Policy: $725,880 | Net Savings: $98,920 (12.0%)\n"
        f"  {t15['flagged']:,} patients flagged ({t15['flagged']/VAL_N*100:.1f}% of cohort)\n\n"
        "CAVEAT: These are modeled financial estimates under illustrative assumptions, "
        "not measured hospital accounting savings. The threshold 0.15 is described as "
        "'the lowest-cost candidate among evaluated thresholds under the specified "
        "financial assumptions.'"
    ))


# ============================================================================
# SLIDE 7: SHAP EXPLAINABILITY
# ============================================================================
def slide_shap():
    s = prs.slides.add_slide(BLANK_LAYOUT)
    _set_bg(s)

    _add_category_tag(s, "Model Explainability — Step 6")
    _add_title(s, "Lead Time Dominates Risk; SHAP Makes Every Prediction Transparent")
    _add_accent_line(s)

    # SHAP bar importance image on the left
    _add_image_fit(s, str(FIG['shap_bar']),
                   Inches(0.6), Inches(1.7), Inches(6.5), Inches(4.5))

    # Top-3 feature KPI badges on the right
    kpi_x = Inches(7.6)
    kpi_w = Inches(5.0)
    kpi_h = Inches(1.15)
    kpi_gap = Inches(0.25)

    if len(shap_top) >= 3:
        _add_kpi_card(s, kpi_x, Inches(1.8), kpi_w, kpi_h,
                      f"±{shap_top[0]['pct']:.1f}% pts",
                      f"Lead Days — #1 Predictive Driver", CYAN, Pt(36))
        _add_kpi_card(s, kpi_x, Inches(1.8) + kpi_h + kpi_gap, kpi_w, kpi_h,
                      f"±{shap_top[1]['pct']:.1f}% pts",
                      f"Patient Age — #2 Predictive Driver", GREEN, Pt(36))
        _add_kpi_card(s, kpi_x, Inches(1.8) + 2*(kpi_h + kpi_gap), kpi_w, kpi_h,
                      f"±{shap_top[2]['pct']:.1f}% pts",
                      f"Same-Day Booking — #3 Predictive Driver", WARNING, Pt(36))

    # Bullets
    _add_bullets(s, [
        f"Base probability E[f(X)] = {SHAP_BASELINE}% + per-feature shifts = prediction",
        "Probability-space SHAP: clinicians see percentage-point contributions",
        "Statistical attribution ≠ clinical causation — SHAP shows correlation, not interventions",
    ], Inches(7.6), Inches(5.9), Inches(5.2), Inches(1.2), Pt(14))

    _add_footer(s, 7)
    _add_notes(s, (
        "SPEAKER NOTES — SHAP Explainability\n\n"
        "We implement a Dual-Space SHAP framework:\n"
        "  1. Raw-margin space (log-odds): for ML engineers, exact sigmoid parity.\n"
        "  2. Probability space: for clinicians, percentage-point attributions.\n\n"
        f"Baseline probability E[f(X)] = {SHAP_BASELINE}% (computed from 100 background "
        "observations sampled from training data).\n\n"
        "Top 6 probability-space drivers by mean |SHAP|:\n"
    ) + "\n".join(f"  {i+1}. {f['name']}: ±{f['pct']:.2f}% pts" for i, f in enumerate(shap_top)) + (
        "\n\nAdditivity axiom verified: maximum error |Σφᵢ - (p - φ₀)| ≤ 2.38×10⁻⁷.\n\n"
        "Key clinical insight: Lead days is by far the dominant driver — patients booked "
        "far in advance have dramatically higher no-show risk. Same-day bookings are protective "
        "(only ~4.6% no-show rate). Prior no-show history amplifies risk multiplicatively.\n\n"
        "IMPORTANT: SHAP values represent statistical model credit assignment, not causal "
        "medical mechanisms. We cannot claim that shortening lead time will reduce no-shows — "
        "it may just correlate with other patient behaviors."
    ))


# ============================================================================
# SLIDE 8: GOVERNANCE & LIMITATIONS
# ============================================================================
def slide_governance():
    s = prs.slides.add_slide(BLANK_LAYOUT)
    _set_bg(s)

    _add_category_tag(s, "Clinical Governance & Ethical Deployment")
    _add_title(s, "Predictions Must Support Patients, Never Restrict Access")
    _add_accent_line(s)

    # 4 governance principle cards
    card_w = Inches(5.7)
    card_h = Inches(1.05)
    card_gap = Inches(0.25)
    col1_x = Inches(0.8)
    col2_x = Inches(6.8)
    row1_y = Inches(1.8)

    principles = [
        ("NON-PUNITIVE MANDATE", "Risk scores must only allocate supportive outreach — reminders, transport, navigation. Never deny or delay care.", CYAN, col1_x, row1_y),
        ("ATTRIBUTION ≠ CAUSALITY", "SHAP values measure model credit, not clinical causation. Rescheduling a patient closer does not guarantee attendance.", WARNING, col2_x, row1_y),
        ("ILLUSTRATIVE FINANCIALS", "Cost savings are modeled estimates under assumed parameters ($10/$200/30%), not measured hospital accounting.", GREEN, col1_x, row1_y + card_h + card_gap),
        ("SINGLE-SITE LIMITATIONS", "Trained on 2016 Vitória, Brazil public health data. Requires local recalibration for other healthcare systems.", RED_SOFT, col2_x, row1_y + card_h + card_gap),
    ]

    for title, body, color, x, y in principles:
        card = s.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, x, y, card_w, card_h
        )
        card.fill.solid()
        card.fill.fore_color.rgb = SURFACE
        card.line.color.rgb = color
        card.line.width = Pt(1.5)

        # Accent top edge
        edge = s.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, x, y, card_w, Pt(3)
        )
        edge.fill.solid()
        edge.fill.fore_color.rgb = color
        edge.line.fill.background()

        # Title
        tb = s.shapes.add_textbox(x + Inches(0.25), y + Inches(0.1),
                                  card_w - Inches(0.5), Inches(0.25))
        p = tb.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = color
        p.font.name = FONT

        # Body
        tb2 = s.shapes.add_textbox(x + Inches(0.25), y + Inches(0.4),
                                   card_w - Inches(0.5), card_h - Inches(0.5))
        tf2 = tb2.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        p2.text = body
        p2.font.size = Pt(14)
        p2.font.color.rgb = TEXT
        p2.font.name = FONT

    # Additional honesty bullets
    _add_bullets(s, [
        "Chronological holdout (Jun 1–8) was previously explored during development",
        "Model performance may degrade under distribution shift or population changes",
        "Dashboard predictions are decision-support tools, not clinical diagnoses",
    ], MARGIN, Inches(4.5), SLIDE_W - 2 * MARGIN, Inches(2.0), Pt(16))

    _add_footer(s, 8)
    _add_notes(s, (
        "SPEAKER NOTES — Governance & Limitations\n\n"
        "This slide is critical for interview defense. It demonstrates mature ML engineering "
        "awareness of deployment risks.\n\n"
        "NON-PUNITIVE: In healthcare ML, the greatest ethical risk is using risk predictions "
        "to withhold care from vulnerable populations. Our model explicitly mandates that "
        "predictions only trigger supportive interventions: phone reminders, transportation "
        "vouchers, community health worker outreach, flexible rebooking options.\n\n"
        "ATTRIBUTION ≠ CAUSALITY: Interviewers will probe whether SHAP values can be used "
        "for causal intervention design. The answer is: they show statistical association "
        "within the model, not causal mechanisms. Artificially shortening lead time might "
        "not reduce no-shows if the underlying driver is patient scheduling conflicts.\n\n"
        "FINANCIAL CAVEATS: All cost numbers ($98,920 savings) are illustrative modeled "
        "scenarios, not hospital P&L figures. The cost matrix ($10/$200) represents "
        "plausible but assumed estimates.\n\n"
        "GENERALIZATION: The model was trained on a specific Brazilian public health "
        "cohort from 2016. Deploying to a US hospital, a rural clinic, or a "
        "specialist practice would require complete retraining and recalibration."
    ))


# ============================================================================
# SLIDE 9: CONCLUSION & NEXT STEPS
# ============================================================================
def slide_conclusion():
    s = prs.slides.add_slide(BLANK_LAYOUT)
    _set_bg(s)

    _add_category_tag(s, "Conclusion & Future Directions")
    _add_title(s, "A Complete, Interview-Ready ML Pipeline with 57 Verified Tests")
    _add_accent_line(s)

    # Summary KPIs
    kpi_y = Inches(1.8)
    kpi_w = Inches(2.7)
    kpi_h = Inches(1.4)
    kpi_gap = Inches(0.3)
    kpi_start = Inches(0.8)

    _add_kpi_card(s, kpi_start, kpi_y, kpi_w, kpi_h,
                  "6", "Pipeline Stages\nEnd-to-End", CYAN, Pt(42))
    _add_kpi_card(s, kpi_start + kpi_w + kpi_gap, kpi_y, kpi_w, kpi_h,
                  "57", "Automated Tests\n100% Pass Rate", GREEN, Pt(42))
    _add_kpi_card(s, kpi_start + 2*(kpi_w + kpi_gap), kpi_y, kpi_w, kpi_h,
                  "4-Tab", "CarePredict\nStreamlit Dashboard", WARNING, Pt(42))
    _add_kpi_card(s, kpi_start + 3*(kpi_w + kpi_gap), kpi_y, kpi_w, kpi_h,
                  "17", "Day-0 Features\nLeakage-Audited", CYAN, Pt(42))

    # Dashboard screenshot
    _add_image_fit(s, str(FIG['dashboard']),
                   Inches(0.8), Inches(3.5), Inches(6.5), Inches(3.2))

    # Next steps bullets
    _add_bullets(s, [
        "Full reproducibility: Steps 1–6 frozen, all notebooks executed, CarePredict live",
        "Next: Multi-site validation, temporal recalibration, EHR integration, A/B testing",
        "GitHub: github.com/gaurav-raj5123/healthcare-appointment-risk-optimization",
    ], Inches(7.6), Inches(3.7), Inches(5.2), Inches(2.5), Pt(16))

    _add_footer(s, 9)
    _add_notes(s, (
        "SPEAKER NOTES — Conclusion\n\n"
        "Summary of what was delivered:\n"
        "  1. Leakage-audited, chronologically-split ML pipeline (6 stages)\n"
        "  2. Champion unweighted XGBoost (PR-AUC 0.3584, Brier 0.1431)\n"
        "  3. Cost-optimized threshold 0.15 (93.62% recall, $98,920 modeled savings)\n"
        "  4. Dual-space SHAP explainability (probability + log-odds, additivity < 2.4e-7)\n"
        "  5. CarePredict Streamlit dashboard (4 tabs: overview, analytics, prediction, optimization)\n"
        "  6. 57 automated tests covering data integrity, schema parity, SHAP additivity, "
        "financial formulas, and Streamlit UI integration\n\n"
        "Potential next steps (not implemented, for future work discussion):\n"
        "  - Multi-site validation: retrain on data from a different healthcare system\n"
        "  - Temporal recalibration: adapt to post-COVID scheduling patterns\n"
        "  - EHR integration: connect to Epic/Cerner for real-time Day-0 scoring\n"
        "  - A/B testing: randomized controlled trial of model-guided outreach vs. standard care\n\n"
        "GitHub repository is public and contains all code, data, notebooks, and documentation."
    ))


# ============================================================================
# MAIN — generate all slides and save
# ============================================================================
if __name__ == '__main__':
    slide_title()
    slide_problem()
    slide_data_leakage()
    slide_architecture()
    slide_benchmark()
    slide_threshold()
    slide_shap()
    slide_governance()
    slide_conclusion()

    # Save
    os.makedirs('docs', exist_ok=True)
    prs.save('presentation.pptx')
    prs.save('docs/presentation.pptx')
    print("Presentation saved: presentation.pptx and docs/presentation.pptx")
    print(f"Total slides: {len(prs.slides)}")
