"""
Test Suite 14: Streamlit AppTest Headless Integration Tests
Verifies that all 4 main pages of CarePredict render cleanly with zero unhandled exceptions:
  1. Executive Overview
  2. Appointment Analytics
  3. No-Show Prediction & Explainability
  4. Capacity & Financial Optimization
"""

from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_ENTRYPOINT = str(Path(__file__).resolve().parent.parent / "app" / "app.py")

def test_app_initial_render():
    at = AppTest.from_file(APP_ENTRYPOINT, default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"AppTest initial run threw exceptions: {[e.value for e in at.exception]}"

def test_app_page_navigation():
    pages = [
        "Executive Overview",
        "Appointment Analytics",
        "No-Show Prediction & Explainability",
        "Capacity & Financial Optimization"
    ]
    for page in pages:
        at = AppTest.from_file(APP_ENTRYPOINT, default_timeout=30)
        at.run()
        assert len(at.exception) == 0
        if page != "Executive Overview":
            at.sidebar.radio[0].set_value(page).run()
        assert len(at.exception) == 0, f"AppTest threw exceptions on page {page}: {[e.value for e in at.exception]}"
