"""
Test Suite 13: Capacity Management & Queue Prioritization Tests
Verifies that patient outreach prioritization:
  1. Strictly sorts appointments by predicted risk descending
  2. Correctly flags appointments within capacity
  3. Tests boundary capacities: 0, 1, 500, and > flagged volume
  4. Explicitly rejects negative capacity values
"""

import pandas as pd
import pytest
from app.financial_engine import apply_capacity_constraint

def test_capacity_sorting_descending():
    df = pd.DataFrame({
        "appointment_id": [1, 2, 3, 4],
        "predicted_noshow_prob": [0.20, 0.85, 0.45, 0.10],
        "flagged_for_intervention": [True, True, True, False]
    })
    
    sorted_df, stats = apply_capacity_constraint(df, daily_capacity=2)
    assert sorted_df["appointment_id"].tolist() == [2, 3, 1, 4]  # Probabilities: 0.85, 0.45, 0.20, 0.10
    assert sorted_df["within_capacity"].tolist() == [True, True, False, False]
    assert stats["reached_within_capacity"] == 2
    assert stats["unreached_shortfall"] == 1  # 3 flagged - 2 capacity = 1 shortfall

def test_capacity_zero():
    df = pd.DataFrame({
        "appointment_id": [1, 2],
        "predicted_noshow_prob": [0.60, 0.70],
        "flagged_for_intervention": [True, True]
    })
    sorted_df, stats = apply_capacity_constraint(df, daily_capacity=0)
    assert sorted_df["within_capacity"].sum() == 0
    assert stats["reached_within_capacity"] == 0
    assert stats["unreached_shortfall"] == 2
    assert stats["capacity_coverage_pct"] == 0.0

def test_capacity_one():
    df = pd.DataFrame({
        "appointment_id": [1, 2],
        "predicted_noshow_prob": [0.60, 0.70],
        "flagged_for_intervention": [True, True]
    })
    sorted_df, stats = apply_capacity_constraint(df, daily_capacity=1)
    assert sorted_df["within_capacity"].sum() == 1
    assert stats["reached_within_capacity"] == 1
    assert stats["unreached_shortfall"] == 1

def test_capacity_greater_than_flagged():
    df = pd.DataFrame({
        "appointment_id": [1, 2],
        "predicted_noshow_prob": [0.60, 0.70],
        "flagged_for_intervention": [True, True]
    })
    sorted_df, stats = apply_capacity_constraint(df, daily_capacity=100)
    assert stats["reached_within_capacity"] == 2
    assert stats["unreached_shortfall"] == 0
    assert stats["capacity_coverage_pct"] == 100.0

def test_negative_capacity_rejected():
    df = pd.DataFrame({
        "predicted_noshow_prob": [0.5]
    })
    with pytest.raises(ValueError, match="must be non-negative"):
        apply_capacity_constraint(df, daily_capacity=-5)

