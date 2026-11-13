"""
Test Suite 3: Risk Tier Boundary Tests
Verifies that operational risk tiers use exact mutually exclusive intervals:
  - Low Risk:      0.00 <= p < 0.15
  - Moderate Risk: 0.15 <= p < 0.40
  - High Risk:     0.40 <= p < 0.70
  - Critical Risk: 0.70 <= p <= 1.00
Tests boundary values: 0, 0.149999, 0.15, 0.399999, 0.40, 0.699999, 0.70, 1.0.
Verifies rejection of NaN, inf, negative, and values > 1.0.
"""

import math
import numpy as np
import pytest
from app.model_service import get_risk_tier

def test_low_risk_boundaries():
    assert get_risk_tier(0.0)["name"] == "Low Risk"
    assert get_risk_tier(0.05)["name"] == "Low Risk"
    assert get_risk_tier(0.149999)["name"] == "Low Risk"

def test_moderate_risk_boundaries():
    assert get_risk_tier(0.15)["name"] == "Moderate Risk"
    assert get_risk_tier(0.25)["name"] == "Moderate Risk"
    assert get_risk_tier(0.399999)["name"] == "Moderate Risk"

def test_high_risk_boundaries():
    assert get_risk_tier(0.40)["name"] == "High Risk"
    assert get_risk_tier(0.55)["name"] == "High Risk"
    assert get_risk_tier(0.699999)["name"] == "High Risk"

def test_critical_risk_boundaries():
    assert get_risk_tier(0.70)["name"] == "Critical Risk"
    assert get_risk_tier(0.85)["name"] == "Critical Risk"
    assert get_risk_tier(1.0)["name"] == "Critical Risk"

def test_invalid_negative_probability():
    with pytest.raises(ValueError, match="out of bounds"):
        get_risk_tier(-0.01)

def test_invalid_greater_than_one():
    with pytest.raises(ValueError, match="out of bounds"):
        get_risk_tier(1.001)

def test_invalid_nan():
    with pytest.raises(ValueError, match="finite numeric value"):
        get_risk_tier(float('nan'))
    with pytest.raises(ValueError, match="finite numeric value"):
        get_risk_tier(np.nan)

def test_invalid_inf():
    with pytest.raises(ValueError, match="finite numeric value"):
        get_risk_tier(float('inf'))
    with pytest.raises(ValueError, match="finite numeric value"):
        get_risk_tier(float('-inf'))

def test_invalid_none():
    with pytest.raises(ValueError):
        get_risk_tier(None)

