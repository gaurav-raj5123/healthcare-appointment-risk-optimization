"""
CarePredict - Reusable UI Components
KPI cards, badges, banners, callout boxes, and custom HTML widgets.
Includes strict output escaping using html.escape() to prevent HTML injection.
"""

import html
import streamlit as st
from app.config import CSS_PATH, RISK_TIERS

def inject_custom_css():
    """Injects the customized healthcare analytics CSS stylesheet."""
    if CSS_PATH.exists():
        with open(CSS_PATH, 'r', encoding='utf-8') as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def render_header(title: str, subtitle: str):
    """Renders a styled healthcare header banner with escaped inputs."""
    safe_title = html.escape(str(title))
    safe_subtitle = html.escape(str(subtitle))
    st.markdown(
        f"""
        <div class="header-container">
            <h1 class="header-title">{safe_title}</h1>
            <p class="header-subtitle">{safe_subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_kpi_card(label: str, value: str, subtext: str = ""):
    """Renders an executive KPI metric card with escaped inputs."""
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_subtext = html.escape(str(subtext)) if subtext else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{safe_label}</div>
            <div class="kpi-value">{safe_value}</div>
            <div class="kpi-subtext">{safe_subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_risk_badge(risk_name: str) -> str:
    """Returns safe HTML for an operational risk tier badge."""
    safe_name = html.escape(str(risk_name))
    if risk_name in RISK_TIERS:
        badge_class = RISK_TIERS[risk_name]["badge_class"]
    else:
        badge_class = "badge-unknown"
    return f'<span class="{badge_class}">{safe_name}</span>'

def render_callout(message: str, box_type: str = "notice", escape_text: bool = False):
    """
    Renders a styled notice or warning callout box.
    If escape_text is True, dynamic untrusted content is escaped via html.escape().
    """
    safe_message = html.escape(str(message)) if escape_text else str(message)
    css_class = "notice-box" if box_type == "notice" else "warning-box"
    st.markdown(f'<div class="{css_class}">{safe_message}</div>', unsafe_allow_html=True)

