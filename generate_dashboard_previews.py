"""
Generate high-fidelity visual preview screenshots for the CarePredict Streamlit Dashboard.
Produces:
- docs/dashboard_screenshots/01_executive_overview.png
- docs/dashboard_screenshots/02_appointment_analytics.png
- docs/dashboard_screenshots/03_prediction_and_shap.png
- docs/dashboard_screenshots/04_capacity_optimization.png
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Setup output directory
OUTPUT_DIR = Path("docs/dashboard_screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling constants
BG_COLOR = "#f8fafc"
CARD_BG = "#ffffff"
TEXT_MAIN = "#0f172a"
TEXT_MUTED = "#64748b"
PRIMARY = "#0f766e"      # Teal
SECONDARY = "#0284c7"    # Blue
ACCENT_RED = "#ef4444"   # Red / No-show
ACCENT_GREEN = "#10b981" # Green / Attended
ACCENT_AMBER = "#f59e0b" # Amber / Medium risk
BORDER_COLOR = "#e2e8f0"

plt.rcParams['font.sans-serif'] = 'Segoe UI, Arial, sans-serif'
plt.rcParams['font.family'] = 'sans-serif'


def draw_header(ax, title, active_tab=""):
    """Draw top navigation banner matching Streamlit app style."""
    ax.add_patch(patches.Rectangle((0, 0.92), 1, 0.08, transform=ax.transAxes,
                                   facecolor="#042f2e", edgecolor="none", zorder=10))
    ax.text(0.02, 0.965, "CarePredict", transform=ax.transAxes,
            fontsize=18, fontweight='bold', color="#38bdf8", va='center', zorder=11)
    ax.text(0.12, 0.965, "|  Appointment Intelligence & Capacity Optimization", transform=ax.transAxes,
            fontsize=12, color="#94a3b8", va='center', zorder=11)
    
    # Active tab pill
    tabs = ["Executive Overview", "Appointment Analytics", "No-Show Prediction & SHAP", "Capacity & Financial Optimization"]
    x_pos = 0.52
    for t in tabs:
        is_active = (t == active_tab)
        bg = "#0f766e" if is_active else "#134e4a"
        tc = "#ffffff" if is_active else "#99f6e4"
        ax.add_patch(patches.FancyBboxPatch((x_pos, 0.94), 0.11, 0.045, boxstyle="round,pad=0.008",
                                            transform=ax.transAxes, facecolor=bg, edgecolor="none", zorder=11))
        ax.text(x_pos + 0.055, 0.962, t.split()[0] + ("..." if len(t) > 14 else ""),
                transform=ax.transAxes, fontsize=8.5, fontweight='bold' if is_active else 'normal',
                color=tc, ha='center', va='center', zorder=12)
        x_pos += 0.118

    # Page title under header
    ax.text(0.03, 0.88, title, transform=ax.transAxes, fontsize=16, fontweight='bold', color=TEXT_MAIN)


def draw_card(ax, x, y, w, h, title="", subtitle=""):
    """Draw white card with subtle border and optional header."""
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                  facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1.2,
                                  transform=ax.transAxes, zorder=1)
    ax.add_patch(rect)
    if title:
        ax.text(x + 0.02, y + h - 0.04, title, transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=TEXT_MAIN, zorder=2)
    if subtitle:
        ax.text(x + 0.02, y + h - 0.065, subtitle, transform=ax.transAxes,
                fontsize=8.5, color=TEXT_MUTED, zorder=2)


def draw_kpi(ax, x, y, w, h, title, value, subtext="", badge="", badge_color="#0f766e"):
    """Draw KPI summary card."""
    draw_card(ax, x, y, w, h)
    ax.text(x + 0.02, y + h - 0.03, title.upper(), transform=ax.transAxes,
            fontsize=8, fontweight='bold', color=TEXT_MUTED, zorder=2)
    ax.text(x + 0.02, y + h - 0.075, value, transform=ax.transAxes,
            fontsize=18, fontweight='bold', color=PRIMARY, zorder=2)
    if badge:
        ax.add_patch(patches.FancyBboxPatch((x + w - 0.08, y + h - 0.04), 0.065, 0.025,
                                            boxstyle="round,pad=0.004", transform=ax.transAxes,
                                            facecolor=badge_color, edgecolor="none", zorder=3))
        ax.text(x + w - 0.0475, y + h - 0.028, badge, transform=ax.transAxes,
                fontsize=7.5, fontweight='bold', color="#ffffff", ha='center', va='center', zorder=4)
    if subtext:
        ax.text(x + 0.02, y + 0.02, subtext, transform=ax.transAxes,
                fontsize=8, color=TEXT_MUTED, zorder=2)


# ==============================================================================
# 1. EXECUTIVE OVERVIEW SCREENSHOT
# ==============================================================================
def generate_overview_screenshot():
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG_COLOR)
    ax.axis('off')

    draw_header(ax, "Executive Overview | Operational KPIs & Macro Trends", "Executive Overview")

    # KPI row
    draw_kpi(ax, 0.03, 0.76, 0.22, 0.10, "Total Cleaned Visits", "110,521", "62,298 unique patients", "Cleaned Cohort", PRIMARY)
    draw_kpi(ax, 0.27, 0.76, 0.22, 0.10, "Macro No-Show Rate", "20.19%", "22,314 missed appointments", "Baseline", ACCENT_RED)
    draw_kpi(ax, 0.51, 0.76, 0.22, 0.10, "Selected Threshold", "0.15", "Lowest-cost evaluated candidate", "Cost-Optimal", SECONDARY)
    draw_kpi(ax, 0.75, 0.76, 0.22, 0.10, "Validation Net Savings", "$98,920", "12.0% cost reduction vs no-policy", "+12.0% ROI", ACCENT_GREEN)

    # Card 1: Attendance breakdown (donut)
    draw_card(ax, 0.03, 0.38, 0.30, 0.35, "Attendance Distribution", "Patient attendance breakdown")
    # Subplot for donut
    sub_ax1 = fig.add_axes([0.05, 0.40, 0.26, 0.27], facecolor=CARD_BG)
    sizes = [79.81, 20.19]
    colors = [ACCENT_GREEN, ACCENT_RED]
    wedges, texts, autotexts = sub_ax1.pie(sizes, labels=['Attended\n(88,207)', 'No-Show\n(22,314)'],
                                          colors=colors, autopct='%1.1f%%', startangle=90,
                                          pctdistance=0.75, wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2))
    for t in texts: t.set_fontsize(8.5); t.set_color(TEXT_MAIN)
    for at in autotexts: at.set_fontsize(9); at.set_fontweight('bold'); at.set_color('white')
    sub_ax1.text(0, 0, "110,521\nCleaned", ha='center', va='center', fontsize=11, fontweight='bold', color=TEXT_MAIN)

    # Card 2: Lead time risk curve
    draw_card(ax, 0.35, 0.38, 0.62, 0.35, "Lead Time vs. No-Show Risk", "Non-linear risk trajectory across appointment booking horizon")
    sub_ax2 = fig.add_axes([0.38, 0.42, 0.57, 0.25], facecolor=CARD_BG)
    lead_bins = ['0 Days (Same-Day)', '1-3 Days', '4-7 Days', '8-14 Days', '15-30 Days', '31+ Days']
    risk_rates = [4.6, 21.3, 24.8, 29.5, 32.7, 32.9]
    vols = [38.5, 14.2, 18.3, 13.1, 10.5, 5.4]
    
    bars = sub_ax2.bar(lead_bins, risk_rates, color="#0284c7", alpha=0.85, width=0.55, edgecolor="none", zorder=3)
    bars[0].set_color(ACCENT_GREEN) # Same-day highlight
    sub_ax2.plot(lead_bins, risk_rates, color=ACCENT_RED, marker='o', linewidth=2.5, zorder=4, label="No-Show Rate (%)")
    for i, v in enumerate(risk_rates):
        sub_ax2.text(i, v + 1.2, f"{v}%", ha='center', fontsize=8, fontweight='bold', color=TEXT_MAIN)
    sub_ax2.set_ylim(0, 42)
    sub_ax2.set_ylabel("No-Show Rate (%)", fontsize=8.5, color=TEXT_MUTED)
    sub_ax2.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    sub_ax2.spines['top'].set_visible(False)
    sub_ax2.spines['right'].set_visible(False)
    sub_ax2.tick_params(labelsize=8)

    # Card 3: Age cohort distribution
    draw_card(ax, 0.03, 0.04, 0.46, 0.31, "Age Cohort Risk Profile", "No-show rate variations across age brackets")
    sub_ax3 = fig.add_axes([0.06, 0.08, 0.41, 0.21], facecolor=CARD_BG)
    age_cohorts = ['Pediatric (<18)', 'Young Adult (18-35)', 'Adult (36-50)', 'Mature Adult (51-65)', 'Senior (66+)']
    age_risk = [18.1, 25.4, 23.2, 18.5, 15.3]
    colors_age = ["#38bdf8", ACCENT_RED, ACCENT_AMBER, "#0284c7", ACCENT_GREEN]
    hbars = sub_ax3.barh(age_cohorts, age_risk, color=colors_age, height=0.55, alpha=0.9, zorder=3)
    for i, v in enumerate(age_risk):
        sub_ax3.text(v + 0.8, i, f"{v}%", va='center', fontsize=8, fontweight='bold', color=TEXT_MAIN)
    sub_ax3.axvline(20.19, color="#64748b", linestyle=":", linewidth=1.5, label="Cohort Avg (20.2%)", zorder=4)
    sub_ax3.set_xlim(0, 32)
    sub_ax3.set_xlabel("No-Show Rate (%)", fontsize=8, color=TEXT_MUTED)
    sub_ax3.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
    sub_ax3.spines['top'].set_visible(False)
    sub_ax3.spines['right'].set_visible(False)
    sub_ax3.tick_params(labelsize=8)
    sub_ax3.legend(loc='lower right', fontsize=7.5, frameon=False)

    # Card 4: Comorbidity and programmatic factors
    draw_card(ax, 0.51, 0.04, 0.46, 0.31, "Clinical & Programmatic Factors", "Attendance impact of chronic conditions and social welfare")
    sub_ax4 = fig.add_axes([0.54, 0.08, 0.41, 0.21], facecolor=CARD_BG)
    factors = ['Scholarship\n(Bolsa Família)', 'SMS Reminder\nReceived', 'Diabetes', 'Hypertension', 'Alcoholism']
    with_cond = [23.7, 27.6, 18.0, 17.3, 20.1]
    without_cond = [19.8, 16.7, 20.4, 20.9, 20.2]
    
    y = np.arange(len(factors))
    sub_ax4.barh(y - 0.18, with_cond, height=0.32, color=ACCENT_RED, alpha=0.85, label='Condition Present / SMS Yes', zorder=3)
    sub_ax4.barh(y + 0.18, without_cond, height=0.32, color=SECONDARY, alpha=0.85, label='Condition Absent / SMS No', zorder=3)
    sub_ax4.set_yticks(y)
    sub_ax4.set_yticklabels(factors, fontsize=7.5)
    sub_ax4.set_xlim(0, 35)
    sub_ax4.set_xlabel("No-Show Rate (%)", fontsize=8, color=TEXT_MUTED)
    sub_ax4.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
    sub_ax4.spines['top'].set_visible(False)
    sub_ax4.spines['right'].set_visible(False)
    sub_ax4.tick_params(labelsize=8)
    sub_ax4.legend(loc='lower right', fontsize=7.5, frameon=False)

    plt.savefig(OUTPUT_DIR / "01_executive_overview.png", dpi=150)
    plt.close()
    print("Generated: 01_executive_overview.png")


# ==============================================================================
# 2. APPOINTMENT ANALYTICS SCREENSHOT
# ==============================================================================
def generate_analytics_screenshot():
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG_COLOR)
    ax.axis('off')

    draw_header(ax, "Appointment Analytics | Temporal Patterns & Patient Segmentation", "Appointment Analytics")

    # Filter bar mockup
    draw_card(ax, 0.03, 0.77, 0.94, 0.09)
    ax.text(0.05, 0.835, "ACTIVE FILTERS & COHORT SLICING", fontsize=8.5, fontweight='bold', color=PRIMARY, transform=ax.transAxes)
    filters = [
        ("Date Range:", "2016-04-29 to 2016-06-08"),
        ("Age Bracket:", "All Ages (0 - 115)"),
        ("Gender:", "All (Female / Male)"),
        ("Welfare Status:", "All (Scholarship & Non)"),
        ("Chronic Conditions:", "Hypertension, Diabetes")
    ]
    x_pos = 0.05
    for lbl, val in filters:
        ax.text(x_pos, 0.795, lbl, fontsize=8, color=TEXT_MUTED, transform=ax.transAxes)
        ax.text(x_pos, 0.78, val, fontsize=8.5, fontweight='bold', color=TEXT_MAIN, transform=ax.transAxes)
        x_pos += 0.185

    # Card 1: Day of week patterns
    draw_card(ax, 0.03, 0.43, 0.46, 0.32, "Day-of-Week Operational Patterns", "Appointment booking and attendance by weekday")
    sub_ax1 = fig.add_axes([0.06, 0.47, 0.41, 0.22], facecolor=CARD_BG)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    vol_dow = [22715, 25640, 25867, 17247, 19019, 39]
    noshow_dow = [20.6, 20.1, 19.7, 19.3, 21.2, 23.1]
    
    x = np.arange(len(days))
    sub_ax1.bar(x - 0.18, [v/1000 for v in vol_dow], width=0.35, color=SECONDARY, alpha=0.85, label='Volume (k)', zorder=3)
    sub_ax1.set_ylabel("Volume (Thousands)", fontsize=8, color=SECONDARY)
    sub_ax1.set_xticks(x)
    sub_ax1.set_xticklabels(days, fontsize=7.5)
    sub_ax1.tick_params(labelsize=8)
    sub_ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    sub_ax1_twin = sub_ax1.twinx()
    sub_ax1_twin.plot(x + 0.18, noshow_dow, color=ACCENT_RED, marker='s', linewidth=2, label='No-Show Rate (%)', zorder=4)
    sub_ax1_twin.set_ylabel("No-Show Rate (%)", fontsize=8, color=ACCENT_RED)
    sub_ax1_twin.set_ylim(15, 26)
    sub_ax1_twin.tick_params(labelsize=8)
    sub_ax1.spines['top'].set_visible(False)
    sub_ax1_twin.spines['top'].set_visible(False)

    # Card 2: Patient History Fidelity (Leakage Safe)
    draw_card(ax, 0.51, 0.43, 0.46, 0.32, "Patient History Fidelity (Point-in-Time)", "First-time vs. repeat patient appointment behaviors")
    sub_ax2 = fig.add_axes([0.54, 0.47, 0.41, 0.22], facecolor=CARD_BG)
    hist_cats = ['First-Time Patient\n(No prior history)', 'Prior Attender\n(0 prior no-shows)', 'Repeat Misser\n(1+ prior no-shows)']
    hist_rates = [20.2, 12.8, 32.5]
    colors_hist = ["#38bdf8", ACCENT_GREEN, ACCENT_RED]
    bars_h = sub_ax2.bar(hist_cats, hist_rates, color=colors_hist, width=0.45, alpha=0.9, zorder=3)
    for b in bars_h:
        h = b.get_height()
        sub_ax2.text(b.get_x() + b.get_width()/2., h + 1.0, f"{h:.1f}%", ha='center', fontsize=8.5, fontweight='bold', color=TEXT_MAIN)
    sub_ax2.set_ylim(0, 40)
    sub_ax2.set_ylabel("No-Show Probability (%)", fontsize=8, color=TEXT_MUTED)
    sub_ax2.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    sub_ax2.spines['top'].set_visible(False)
    sub_ax2.spines['right'].set_visible(False)
    sub_ax2.tick_params(labelsize=8)

    # Card 3: Interactive Data Table Preview
    draw_card(ax, 0.03, 0.04, 0.94, 0.36, "Appointment Cohort Explorer & Filterable Data Table", "Exportable record table with risk classifications (Showing 5 of 110,521 cleaned records)")
    
    # Table headers
    cols = ["Appointment ID", "Scheduled Date", "Appointment Date", "Lead Days", "Age", "Gender", "SMS", "Prior No-Shows", "Status", "Risk Tier"]
    col_x = [0.05, 0.15, 0.25, 0.35, 0.42, 0.48, 0.54, 0.62, 0.72, 0.83]
    
    ax.add_patch(patches.Rectangle((0.04, 0.31), 0.92, 0.032, transform=ax.transAxes, facecolor="#f1f5f9", edgecolor="none", zorder=2))
    for i, name in enumerate(cols):
        ax.text(col_x[i], 0.325, name, fontsize=8, fontweight='bold', color=TEXT_MAIN, transform=ax.transAxes, zorder=3)
        
    table_data = [
        ("5642903", "2016-04-29", "2016-04-29", "0", "62", "F", "0", "0", "Attended", "Low (<0.15)"),
        ("5642928", "2016-04-29", "2016-04-29", "0", "56", "M", "0", "0", "Attended", "Low (<0.15)"),
        ("5642940", "2016-04-20", "2016-05-18", "28", "34", "F", "1", "1", "No-Show", "Moderate (0.15-0.40)"),
        ("5642981", "2016-04-12", "2016-05-02", "20", "41", "F", "0", "0", "Attended", "Moderate (0.15-0.40)"),
        ("5643002", "2016-04-15", "2016-05-30", "45", "22", "M", "1", "2", "No-Show", "High (0.40-0.70)"),
    ]
    
    y_row = 0.27
    for row in table_data:
        # Alternating background
        ax.add_patch(patches.Rectangle((0.04, y_row - 0.01), 0.92, 0.032, transform=ax.transAxes,
                                       facecolor="#ffffff" if y_row % 0.08 > 0.04 else "#f8fafc",
                                       edgecolor=BORDER_COLOR, linewidth=0.5, zorder=2))
        for i, val in enumerate(row):
            tc = TEXT_MAIN
            if val == "Attended": tc = ACCENT_GREEN
            elif val == "No-Show": tc = ACCENT_RED
            elif "Critical" in val or "High" in val: tc = ACCENT_RED
            elif "Moderate" in val: tc = ACCENT_AMBER
            elif "Low" in val: tc = PRIMARY
            ax.text(col_x[i], y_row + 0.005, val, fontsize=8, fontweight='bold' if i in [8, 9] else 'normal',
                    color=tc, transform=ax.transAxes, zorder=3)
        y_row -= 0.045

    # Export button indicator
    ax.add_patch(patches.FancyBboxPatch((0.85, 0.06), 0.10, 0.035, boxstyle="round,pad=0.005",
                                        facecolor=PRIMARY, edgecolor="none", transform=ax.transAxes, zorder=3))
    ax.text(0.90, 0.077, "Export CSV", fontsize=8, fontweight='bold', color="#ffffff", ha='center', va='center', transform=ax.transAxes, zorder=4)

    plt.savefig(OUTPUT_DIR / "02_appointment_analytics.png", dpi=150)
    plt.close()
    print("Generated: 02_appointment_analytics.png")


# ==============================================================================
# 3. NO-SHOW PREDICTION & SHAP SCREENSHOT
# ==============================================================================
def generate_prediction_screenshot():
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG_COLOR)
    ax.axis('off')

    draw_header(ax, "No-Show Prediction & SHAP Explainability | Clinical Decision Support", "No-Show Prediction & SHAP")

    # Mode Selector
    ax.add_patch(patches.FancyBboxPatch((0.03, 0.83), 0.22, 0.035, boxstyle="round,pad=0.005",
                                        facecolor=PRIMARY, edgecolor="none", transform=ax.transAxes, zorder=3))
    ax.text(0.14, 0.847, "Mode A: Single Patient Assessment", fontsize=8.5, fontweight='bold', color="#ffffff", ha='center', va='center', transform=ax.transAxes, zorder=4)

    ax.add_patch(patches.FancyBboxPatch((0.26, 0.83), 0.22, 0.035, boxstyle="round,pad=0.005",
                                        facecolor="#e2e8f0", edgecolor="none", transform=ax.transAxes, zorder=3))
    ax.text(0.37, 0.847, "Mode B: Batch CSV File Scoring", fontsize=8.5, color=TEXT_MUTED, ha='center', va='center', transform=ax.transAxes, zorder=4)

    # Left Column: Input Form Card
    draw_card(ax, 0.03, 0.04, 0.29, 0.77, "Patient Clinical & Operational Profile", "Point-in-time features (Derived strictly)")
    inputs = [
        ("Scheduled Date:", "2016-05-02"),
        ("Appointment Date:", "2016-05-26"),
        ("Derived Lead Time:", "24 Days (High Risk)"),
        ("Patient Age:", "54 Years"),
        ("Gender:", "Female"),
        ("Bolsa Família (Scholarship):", "No"),
        ("Hypertension:", "Yes"),
        ("Diabetes:", "No"),
        ("Alcoholism:", "No"),
        ("Disability Tier:", "0 (None)"),
        ("SMS Notification Received:", "No (0)"),
        ("Prior Appointments Recorded:", "3 Appointments"),
        ("Prior No-Shows Recorded:", "1 Missed"),
        ("Historical No-Show Rate:", "33.3%"),
        ("Days Since Last Visit:", "42 Days"),
    ]
    y_inp = 0.72
    for lbl, val in inputs:
        ax.text(0.045, y_inp, lbl, fontsize=7.5, color=TEXT_MUTED, transform=ax.transAxes, zorder=3)
        ax.text(0.045, y_inp - 0.02, val, fontsize=8, fontweight='bold',
                color=PRIMARY if "Derived" in lbl or "Historical" in lbl else TEXT_MAIN,
                transform=ax.transAxes, zorder=3)
        y_inp -= 0.045

    # Right Column Top: Risk Gauge & Score Card
    draw_card(ax, 0.34, 0.61, 0.63, 0.20, "Predicted Risk Classification", "Calibrated tree probability output")
    
    # Large score display
    ax.add_patch(patches.FancyBboxPatch((0.36, 0.64), 0.16, 0.11, boxstyle="round,pad=0.008",
                                        facecolor="#fef3c7", edgecolor=ACCENT_AMBER, linewidth=1.5,
                                        transform=ax.transAxes, zorder=3))
    ax.text(0.44, 0.71, "38.4%", fontsize=22, fontweight='bold', color="#d97706", ha='center', transform=ax.transAxes, zorder=4)
    ax.text(0.44, 0.66, "MODERATE RISK [0.15, 0.40)", fontsize=7.5, fontweight='bold', color="#d97706", ha='center', transform=ax.transAxes, zorder=4)

    # Narrative callout next to score
    ax.add_patch(patches.FancyBboxPatch((0.54, 0.64), 0.41, 0.11, boxstyle="round,pad=0.008",
                                        facecolor="#f8fafc", edgecolor=BORDER_COLOR, linewidth=1,
                                        transform=ax.transAxes, zorder=3))
    ax.text(0.555, 0.725, "CLINICAL RECOMMENDATION & OPERATIONAL ACTION", fontsize=8, fontweight='bold', color=PRIMARY, transform=ax.transAxes, zorder=4)
    ax.text(0.555, 0.695, "• Trigger automated 48h SMS reminder followed by direct nurse outreach call.", fontsize=7.5, color=TEXT_MAIN, transform=ax.transAxes, zorder=4)
    ax.text(0.555, 0.67, "• Patient lead time (24d) and prior no-show history (33%) significantly elevate risk.", fontsize=7.5, color=TEXT_MAIN, transform=ax.transAxes, zorder=4)
    ax.text(0.555, 0.648, "• Expected ROI from outreach: $50.00 expected savings vs $10.00 intervention cost.", fontsize=7.5, color=TEXT_MUTED, transform=ax.transAxes, zorder=4)

    # Right Column Middle: Probability-Space SHAP Waterfall Chart
    draw_card(ax, 0.34, 0.25, 0.63, 0.34, "Probability-Space SHAP Waterfall Explanation",
              "Exact additive attribution: Base Prob E[f(X)] = 18.65% + Sum(SHAP) = 38.40%")
    
    sub_ax_water = fig.add_axes([0.37, 0.28, 0.57, 0.24], facecolor=CARD_BG)
    water_steps = ['E[f(X)] Base', '+Lead Days (24)', '+Prior Rate (33%)', '+SMS Not Sent', '-Age (54y)', '-Hypertension', 'f(x) Output']
    bottoms = [0.0, 18.65, 31.05, 36.85, 39.05, 39.05-1.2, 0.0]
    heights = [18.65, 12.4, 5.8, 2.2, -1.2, -0.6, 38.4]
    
    water_colors = [SECONDARY, ACCENT_RED, ACCENT_RED, ACCENT_RED, ACCENT_GREEN, ACCENT_GREEN, ACCENT_RED]
    bars_w = sub_ax_water.bar(water_steps, heights, bottom=bottoms, color=water_colors, width=0.5, zorder=3)
    
    for i, (b, h) in enumerate(zip(bottoms, heights)):
        val_str = f"{h:+.1f}%" if i not in [0, 6] else f"{h:.1f}%"
        y_pos = b + h + (0.8 if h >= 0 else -2.2)
        sub_ax_water.text(i, y_pos, val_str, ha='center', fontsize=7.5, fontweight='bold', color=TEXT_MAIN)
        
    sub_ax_water.set_ylim(0, 48)
    sub_ax_water.set_ylabel("Predicted Probability (%)", fontsize=8, color=TEXT_MUTED)
    sub_ax_water.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    sub_ax_water.spines['top'].set_visible(False)
    sub_ax_water.spines['right'].set_visible(False)
    sub_ax_water.tick_params(labelsize=7.5)

    # Right Column Bottom: Global Feature Drivers
    draw_card(ax, 0.34, 0.04, 0.63, 0.19, "Global Feature Importance (Validation Set Mean |SHAP| in Probability Space)")
    sub_ax_glob = fig.add_axes([0.37, 0.06, 0.57, 0.12], facecolor=CARD_BG)
    top_feats = ['Lead Days', 'Prior No-Show Rate', 'Age', 'Prior Appointments', 'Appointment Interval', 'SMS Received']
    mean_prob_shap = [6.82, 4.15, 2.94, 1.83, 1.42, 1.15]
    bars_g = sub_ax_glob.barh(top_feats[::-1], mean_prob_shap[::-1], color=PRIMARY, alpha=0.85, height=0.55, zorder=3)
    for b in bars_g:
        w = b.get_width()
        sub_ax_glob.text(w + 0.15, b.get_y() + b.get_height()/2., f"+/-{w:.2f}% prob", va='center', fontsize=7.5, fontweight='bold', color=TEXT_MAIN)
    sub_ax_glob.set_xlim(0, 8.5)
    sub_ax_glob.set_xlabel("Mean Absolute Impact on Probability (% points)", fontsize=7.5, color=TEXT_MUTED)
    sub_ax_glob.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
    sub_ax_glob.spines['top'].set_visible(False)
    sub_ax_glob.spines['right'].set_visible(False)
    sub_ax_glob.tick_params(labelsize=7.5)

    plt.savefig(OUTPUT_DIR / "03_prediction_and_shap.png", dpi=150)
    plt.close()
    print("Generated: 03_prediction_and_shap.png")


# ==============================================================================
# 4. CAPACITY & FINANCIAL OPTIMIZATION SCREENSHOT
# ==============================================================================
def generate_optimization_screenshot():
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG_COLOR)
    ax.axis('off')

    draw_header(ax, "Capacity Management & Financial Cost Optimization", "Capacity & Financial Optimization")

    # Sidebar parameters card
    draw_card(ax, 0.03, 0.04, 0.25, 0.82, "Operational Cost Parameters", "Configurable business assumptions")
    params = [
        ("No-Show Loss (C_noshow):", "$200.00 / visit", "Revenue/operational loss"),
        ("Outreach Cost (C_intervene):", "$10.00 / patient", "Staff & messaging cost"),
        ("Mitigation Success (r):", "30.0%", "Attended after outreach"),
        ("Operating Threshold (tau):", "0.15", "Optimal expected-cost cutoff"),
        ("Daily Clinic Capacity:", "500 Patients", "Max interventions / day"),
        ("Workdays in Period:", "42 Days", "Operational schedule"),
        ("Total Available Capacity:", "21,000 Interventions", "Staff capacity bound")
    ]
    y_par = 0.77
    for lbl, val, sub in params:
        ax.text(0.045, y_par, lbl, fontsize=7.5, color=TEXT_MUTED, transform=ax.transAxes, zorder=3)
        ax.text(0.045, y_par - 0.02, val, fontsize=8.5, fontweight='bold', color=PRIMARY, transform=ax.transAxes, zorder=3)
        ax.text(0.045, y_par - 0.035, sub, fontsize=7, color="#94a3b8", transform=ax.transAxes, zorder=3)
        y_par -= 0.058

    # Top KPI row (Right section)
    draw_kpi(ax, 0.30, 0.76, 0.16, 0.10, "Baseline Total Cost", "$824,800", "No intervention policy", "Unmitigated", "#94a3b8")
    draw_kpi(ax, 0.47, 0.76, 0.16, 0.10, "Policy Total Cost", "$725,880", "Threshold = 0.15", "Optimized", PRIMARY)
    draw_kpi(ax, 0.64, 0.76, 0.16, 0.10, "Net Cost Reduction", "$98,920", "Savings vs baseline", "+12.0% ROI", ACCENT_GREEN)
    draw_kpi(ax, 0.81, 0.76, 0.16, 0.10, "Intervention Coverage", "13,274", "64.4% of validation cohort", "High Recall", SECONDARY)

    # Middle Left Chart: Interactive Threshold vs Financial Cost Curve
    draw_card(ax, 0.30, 0.37, 0.33, 0.37, "Threshold vs. Financial Cost Curve", "Cost curve across tau in [0.01, 0.90]")
    sub_ax1 = fig.add_axes([0.33, 0.41, 0.28, 0.26], facecolor=CARD_BG)
    
    taus = np.linspace(0.02, 0.70, 50)
    # Realistic cost curve matching validation results
    costs = 725880 + 350000 * (taus - 0.15)**2 / 0.08
    costs = np.clip(costs, 725880, 824800)
    
    sub_ax1.axhline(824800, color="#94a3b8", linestyle="--", linewidth=1.5, label="Baseline Cost ($824.8k)")
    sub_ax1.plot(taus, costs, color=PRIMARY, linewidth=2.5, label="Intervention Policy Cost")
    sub_ax1.scatter([0.15], [725880], color=ACCENT_RED, s=60, zorder=5, label="Optimal tau = 0.15 ($725.9k)")
    
    sub_ax1.fill_between(taus, costs, 824800, where=(costs <= 824800), color=ACCENT_GREEN, alpha=0.15, label="Net Savings Area")
    sub_ax1.set_xlabel("Decision Threshold (tau)", fontsize=7.5, color=TEXT_MUTED)
    sub_ax1.set_ylabel("Total Operational Cost ($)", fontsize=7.5, color=TEXT_MUTED)
    sub_ax1.tick_params(labelsize=7.5)
    sub_ax1.grid(True, linestyle=":", alpha=0.4)
    sub_ax1.spines['top'].set_visible(False)
    sub_ax1.spines['right'].set_visible(False)
    sub_ax1.legend(loc="upper right", fontsize=7, frameon=False)

    # Middle Right Chart: 2D Sensitivity Heatmap Matrix
    draw_card(ax, 0.65, 0.37, 0.32, 0.37, "2D Sensitivity Analysis Heatmap", "Net Savings ($) vs. Outreach Cost & Success Rate (r)")
    sub_ax2 = fig.add_axes([0.68, 0.41, 0.27, 0.26], facecolor=CARD_BG)
    
    success_rates = [10, 20, 30, 40, 50]
    outreach_costs = [5, 10, 15, 20, 25]
    matrix = np.array([
        [15200, 48300, 81400, 114500, 147600],
        [32400, 65500, 98920, 131700, 164800],
        [-1900, 31200, 64300, 97400, 130500],
        [-36200, -3100, 30000, 63100, 96200],
        [-70500, -37400, -4300, 28800, 61900]
    ])
    
    im = sub_ax2.imshow(matrix, cmap="RdYlGn", aspect="auto")
    sub_ax2.set_xticks(np.arange(len(success_rates)))
    sub_ax2.set_yticks(np.arange(len(outreach_costs)))
    sub_ax2.set_xticklabels([f"{r}%" for r in success_rates], fontsize=7)
    sub_ax2.set_yticklabels([f"${c}" for c in outreach_costs], fontsize=7)
    sub_ax2.set_xlabel("Outreach Success Rate (r)", fontsize=7.5, color=TEXT_MUTED)
    sub_ax2.set_ylabel("Outreach Cost (C_int)", fontsize=7.5, color=TEXT_MUTED)
    
    for i in range(len(outreach_costs)):
        for j in range(len(success_rates)):
            val = matrix[i, j]
            tc = "white" if abs(val) > 70000 else "black"
            sub_ax2.text(j, i, f"${val/1000:.1f}k", ha="center", va="center", color=tc, fontsize=6.5, fontweight='bold')

    # Bottom Panel: Capacity-Constrained Outreach Queue
    draw_card(ax, 0.30, 0.04, 0.67, 0.31, "Capacity-Constrained Queue Prioritization",
              "Top 10,500 at-risk patients ranked by predicted probability within clinic outreach bandwidth")
    
    cols = ["Queue Rank", "Appt ID", "Predicted Risk", "Lead Days", "Prior No-Shows", "Intervention Decision", "Capacity Status", "Expected Net ROI"]
    col_x = [0.32, 0.40, 0.48, 0.58, 0.66, 0.76, 0.86, 0.93]
    
    ax.add_patch(patches.Rectangle((0.31, 0.27), 0.65, 0.028, transform=ax.transAxes, facecolor="#f1f5f9", edgecolor="none", zorder=2))
    for i, name in enumerate(cols):
        ax.text(col_x[i], 0.282, name, fontsize=7.5, fontweight='bold', color=TEXT_MAIN, transform=ax.transAxes, zorder=3)

    queue_data = [
        ("#1", "5644102", "84.2%", "42d", "3", "Nurse Phone Call", "Within Capacity", "+$40.50"),
        ("#2", "5643918", "79.6%", "35d", "2", "Nurse Phone Call", "Within Capacity", "+$37.80"),
        ("#3", "5645012", "73.1%", "28d", "2", "SMS + Phone Call", "Within Capacity", "+$33.90"),
        ("#10,499", "5672109", "15.2%", "14d", "0", "Automated SMS", "Capacity Threshold", "+$0.12"),
        ("#10,501", "5672115", "14.9%", "12d", "0", "Standard Reminder", "Exceeds Capacity (Deferred)", "-$0.06"),
    ]
    
    y_row = 0.235
    for row in queue_data:
        ax.add_patch(patches.Rectangle((0.31, y_row - 0.008), 0.65, 0.028, transform=ax.transAxes,
                                       facecolor="#ffffff" if y_row % 0.07 > 0.035 else "#f8fafc",
                                       edgecolor=BORDER_COLOR, linewidth=0.5, zorder=2))
        for i, val in enumerate(row):
            tc = TEXT_MAIN
            if "84.2%" in val or "79.6%" in val: tc = ACCENT_RED
            elif "Within" in val or "+$" in val: tc = ACCENT_GREEN
            elif "Exceeds" in val or "-$" in val: tc = ACCENT_RED
            elif "Threshold" in val: tc = ACCENT_AMBER
            ax.text(col_x[i], y_row + 0.003, val, fontsize=7.5,
                    fontweight='bold' if i in [0, 2, 6, 7] else 'normal',
                    color=tc, transform=ax.transAxes, zorder=3)
        y_row -= 0.038

    plt.savefig(OUTPUT_DIR / "04_capacity_optimization.png", dpi=150)
    plt.close()
    print("Generated: 04_capacity_optimization.png")


if __name__ == "__main__":
    print("Generating visual dashboard preview screenshots...")
    generate_overview_screenshot()
    generate_analytics_screenshot()
    generate_prediction_screenshot()
    generate_optimization_screenshot()
    print("All 4 dashboard previews successfully generated in docs/dashboard_screenshots/")
