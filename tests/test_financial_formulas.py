"""
Test Suite 11: Financial Cost Modeling Formula Tests
Verifies that historical and prospective cost formulas strictly adhere to the business specifications:
  Total Cost = Flagged * C_intervene + [FN + (1 - r) * TP] * C_noshow
  Baseline Cost = Actual No-Shows * C_noshow
  Net Savings = Baseline Cost - Total Cost
"""

import numpy as np
from app.financial_engine import evaluate_historical_cost, evaluate_prospective_cost

def test_historical_cost_formula_exact_math():
    # Synthetic controlled cohort
    # 10 appointments: 4 actual no-shows (1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    # Predicted probabilities:
    # y_pred at 0.5 threshold -> [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    # TP = 3, FN = 1, FP = 1, TN = 5 -> Flagged = 4
    y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.4, 0.2, 0.1, 0.1, 0.1])
    
    c_int = 10.0
    c_noshow = 200.0
    r = 0.30

    res = evaluate_historical_cost(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.50,
        c_intervene=c_int,
        c_unfilled=c_noshow,
        effectiveness=r
    )

    assert res["tp"] == 3
    assert res["fn"] == 1
    assert res["fp"] == 1
    assert res["tn"] == 5
    assert res["flagged_count"] == 4

    expected_intervention_cost = 4 * 10.0  # $40
    # unprevented = FN + (1 - r) * TP = 1 + (0.70 * 3) = 1 + 2.1 = 3.1
    expected_unprevented_cost = (1 + 0.70 * 3) * 200.0  # 3.1 * 200 = $620
    expected_total_cost = 40.0 + 620.0  # $660

    assert abs(res["intervention_cost"] - expected_intervention_cost) < 1e-6
    assert abs(res["capacity_loss_cost"] - expected_unprevented_cost) < 1e-6
    assert abs(res["total_cost"] - expected_total_cost) < 1e-6

    expected_baseline_cost = 4 * 200.0  # $800
    expected_net_savings = 800.0 - 660.0  # $140
    assert abs(res["baseline_cost"] - expected_baseline_cost) < 1e-6
    assert abs(res["net_savings"] - expected_net_savings) < 1e-6

def test_prospective_cost_formula():
    y_prob = np.array([0.8, 0.6, 0.4, 0.2])  # 4 appointments, expected no-shows = 2.0
    res = evaluate_prospective_cost(
        y_prob=y_prob,
        threshold=0.50,
        c_intervene=10.0,
        c_unfilled=200.0,
        effectiveness=0.30
    )
    assert res["flagged_count"] == 2  # 0.8 and 0.6 >= 0.5
    assert res["total_appointments"] == 4
    assert abs(res["expected_noshows"] - 2.0) < 1e-6
