"""
CarePredict - Historical Analytics & Visual Aggregations
Computes demographic, temporal, and patient-history aggregations for interactive visualization.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.config import COLORS

def get_age_bracket(age: int) -> str:
    """
    Categorizes age into mutually exclusive demographic cohorts.
    Intervals:
      - Pediatric: < 18
      - Young Adult: 18–35
      - Adult: 36–50
      - Mature Adult: 51–65
      - Senior: 66+
    """
    if pd.isna(age) or age < 0:
        raise ValueError(f"Invalid age: {age}. Age cannot be negative or null.")
    if age < 18:
        return "Pediatric (<18)"
    elif age <= 35:
        return "Young Adult (18-35)"
    elif age <= 50:
        return "Adult (36-50)"
    elif age <= 65:
        return "Mature Adult (51-65)"
    else:
        return "Senior (66+)"

def get_lead_time_bracket(lead_days: int) -> str:
    """
    Categorizes scheduling lead time into non-overlapping operational windows.
    Exact Intervals:
      - 0 Days: lead_days == 0 (Same-Day)
      - 1–3 Days: 1 <= lead_days <= 3
      - 4–7 Days: 4 <= lead_days <= 7
      - 8–14 Days: 8 <= lead_days <= 14
      - 15–30 Days: 15 <= lead_days <= 30
      - 31+ Days: lead_days >= 31
    Negative lead times are rejected explicitly.
    """
    if pd.isna(lead_days) or lead_days < 0:
        raise ValueError(f"Invalid negative or missing lead time: {lead_days}. Negative lead times cannot be categorized.")
    if lead_days == 0:
        return "0 Days (Same-Day)"
    elif lead_days <= 3:
        return "1-3 Days"
    elif lead_days <= 7:
        return "4-7 Days"
    elif lead_days <= 14:
        return "8-14 Days"
    elif lead_days <= 30:
        return "15-30 Days"
    else:
        return "31+ Days"

def plot_volume_trend(df: pd.DataFrame) -> go.Figure:
    """Renders daily appointment volume and no-show counts over time."""
    daily = df.groupby(df['appointment_day'].dt.date).agg(
        total=('no_show', 'count'),
        noshows=('no_show', 'sum')
    ).reset_index()
    daily['shows'] = daily['total'] - daily['noshows']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily['appointment_day'],
        y=daily['shows'],
        name='Attended',
        marker_color=COLORS['primary']
    ))
    fig.add_trace(go.Bar(
        x=daily['appointment_day'],
        y=daily['noshows'],
        name='No-Show',
        marker_color=COLORS['danger']
    ))

    fig.update_layout(
        barmode='stack',
        title="<b>Daily Appointment Volume & Outcomes</b>",
        xaxis=dict(title="Appointment Date", gridcolor="#f1f5f9"),
        yaxis=dict(title="Number of Appointments", gridcolor="#f1f5f9"),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320
    )
    return fig

def plot_attendance_donut(df: pd.DataFrame) -> go.Figure:
    """Renders clean attendance vs. no-show donut chart."""
    total = len(df)
    noshows = int(df['no_show'].sum())
    shows = total - noshows

    fig = go.Figure(data=[go.Pie(
        labels=['Attended', 'No-Show'],
        values=[shows, noshows],
        hole=0.62,
        marker=dict(colors=[COLORS['primary'], COLORS['danger']]),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])

    fig.update_layout(
        title="<b>Overall Attendance Distribution</b>",
        showlegend=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=50, b=20),
        height=320
    )
    return fig

def plot_lead_time_risk(df: pd.DataFrame) -> go.Figure:
    """Renders no-show rate curve across lead-time brackets."""
    temp_df = df.copy()
    temp_df['lead_bracket'] = temp_df['lead_days'].apply(get_lead_time_bracket)
    
    order = ["0 Days (Same-Day)", "1-3 Days", "4-7 Days", "8-14 Days", "15-30 Days", "31+ Days"]
    agg = temp_df.groupby('lead_bracket').agg(
        total=('no_show', 'count'),
        noshow_rate=('no_show', 'mean')
    ).reindex(order).reset_index()
    agg['noshow_pct'] = agg['noshow_rate'] * 100.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg['lead_bracket'],
        y=agg['noshow_pct'],
        marker_color=COLORS['teal'],
        text=[f"{v:.1f}%" for v in agg['noshow_pct']],
        textposition='outside'
    ))

    fig.update_layout(
        title="<b>No-Show Rate by Scheduling Lead Time</b>",
        xaxis=dict(title="Lead Time Horizon", gridcolor="#f1f5f9"),
        yaxis=dict(title="No-Show Rate (%)", gridcolor="#f1f5f9", range=[0, 40]),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320
    )
    return fig

def plot_age_distribution_risk(df: pd.DataFrame) -> go.Figure:
    """Renders no-show rate across age brackets."""
    temp_df = df.copy()
    temp_df['age_cohort'] = temp_df['age'].apply(get_age_bracket)
    
    order = ["Pediatric (<18)", "Young Adult (18-35)", "Adult (36-50)", "Mature Adult (51-65)", "Senior (66+)"]
    agg = temp_df.groupby('age_cohort').agg(
        noshow_pct=('no_show', lambda s: s.mean() * 100.0)
    ).reindex(order).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg['age_cohort'],
        y=agg['noshow_pct'],
        marker_color='#6366f1',
        text=[f"{v:.1f}%" for v in agg['noshow_pct']],
        textposition='outside'
    ))

    fig.update_layout(
        title="<b>No-Show Rate by Patient Age Cohort</b>",
        xaxis=dict(title="Age Demographic Cohort", gridcolor="#f1f5f9"),
        yaxis=dict(title="No-Show Rate (%)", gridcolor="#f1f5f9", range=[0, 32]),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320
    )
    return fig

def plot_weekday_risk(df: pd.DataFrame) -> go.Figure:
    """Renders no-show rate by appointment day of week."""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    temp_df = df.copy()
    temp_df['dow'] = temp_df['appointment_day'].dt.dayofweek
    temp_df = temp_df[temp_df['dow'] < 6]  # Exclude Sunday if 0 visits
    
    agg = temp_df.groupby('dow').agg(
        noshow_pct=('no_show', lambda s: s.mean() * 100.0)
    ).reset_index()
    agg['day_name'] = agg['dow'].apply(lambda d: days[d] if d < len(days) else "Sun")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg['day_name'],
        y=agg['noshow_pct'],
        marker_color='#0ea5e9',
        text=[f"{v:.1f}%" for v in agg['noshow_pct']],
        textposition='outside'
    ))

    fig.update_layout(
        title="<b>No-Show Rate by Appointment Weekday</b>",
        xaxis=dict(title="Weekday", gridcolor="#f1f5f9"),
        yaxis=dict(title="No-Show Rate (%)", gridcolor="#f1f5f9", range=[0, 30]),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320
    )
    return fig

def plot_comorbidity_comparison(df: pd.DataFrame) -> go.Figure:
    """Compares attendance rates for patients with and without specific conditions."""
    conditions = [
        ('hypertension', 'Hypertension'),
        ('diabetes', 'Diabetes'),
        ('alcoholism', 'Alcoholism'),
        ('scholarship', 'Bolsa Família')
    ]
    records = []
    for col, name in conditions:
        if col in df.columns:
            rate_with = float(df[df[col] == 1]['no_show'].mean() * 100.0)
            rate_without = float(df[df[col] == 0]['no_show'].mean() * 100.0)
            records.append({"Condition": name, "Status": "Diagnosed / Enrolled", "No-Show Rate": rate_with})
            records.append({"Condition": name, "Status": "Not Diagnosed", "No-Show Rate": rate_without})

    df_plot = pd.DataFrame(records)
    fig = px.bar(
        df_plot,
        x="Condition",
        y="No-Show Rate",
        color="Status",
        barmode="group",
        color_discrete_map={"Diagnosed / Enrolled": COLORS['danger'], "Not Diagnosed": COLORS['primary']},
        title="<b>No-Show Rates by Clinical & Socioeconomic Status</b>"
    )
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        yaxis=dict(title="No-Show Rate (%)", gridcolor="#f1f5f9"),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=320
    )
    return fig

