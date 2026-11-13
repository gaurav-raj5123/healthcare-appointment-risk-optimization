"""
Test Suite 2: Lead-Time Bracket Boundary Tests
Verifies that lead-time categories use exact non-overlapping intervals:
  - 0 Days: lead_days = 0
  - 1–3 Days: 1 <= lead_days <= 3
  - 4–7 Days: 4 <= lead_days <= 7
  - 8–14 Days: 8 <= lead_days <= 14
  - 15–30 Days: 15 <= lead_days <= 30
  - 31+ Days: lead_days >= 31
And verifies that negative lead times are explicitly rejected.
"""

import pytest
from app.analytics import get_lead_time_bracket

def test_same_day_boundary():
    assert get_lead_time_bracket(0) == "0 Days (Same-Day)"

def test_1_to_3_days_boundary():
    assert get_lead_time_bracket(1) == "1-3 Days"
    assert get_lead_time_bracket(2) == "1-3 Days"
    assert get_lead_time_bracket(3) == "1-3 Days"

def test_4_to_7_days_boundary():
    assert get_lead_time_bracket(4) == "4-7 Days"
    assert get_lead_time_bracket(7) == "4-7 Days"

def test_8_to_14_days_boundary():
    assert get_lead_time_bracket(8) == "8-14 Days"
    assert get_lead_time_bracket(14) == "8-14 Days"

def test_15_to_30_days_boundary():
    assert get_lead_time_bracket(15) == "15-30 Days"
    assert get_lead_time_bracket(30) == "15-30 Days"

def test_31_plus_days_boundary():
    assert get_lead_time_bracket(31) == "31+ Days"
    assert get_lead_time_bracket(100) == "31+ Days"

def test_negative_lead_time_rejected():
    with pytest.raises(ValueError, match="Invalid negative or missing lead time"):
        get_lead_time_bracket(-1)
    with pytest.raises(ValueError, match="Invalid negative or missing lead time"):
        get_lead_time_bracket(-10)

def test_null_lead_time_rejected():
    with pytest.raises(ValueError):
        get_lead_time_bracket(None)

