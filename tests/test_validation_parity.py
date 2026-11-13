"""
Test Suite 12: Validation Benchmark Parity Tests
Verifies that evaluating the unweighted XGBoost model predictions on the 20,539 validation cohort
matches the frozen Step 5 benchmark results to the dollar:
  - Baseline Cost: $824,800
  - Policy Cost: $725,880
  - Net Savings: $98,920
  - TP: 3,861 | FP: 9,413 | FN: 263 | TN: 7,002
  - Recall: 93.62% | Flagged: 13,274
"""

import pandas as pd
from app.financial_engine import evaluate_historical_cost
from app.config import VAL_PREDICTIONS_PATH

def test_validation_cohort_exact_cost_parity():
    val_preds = pd.read_csv(VAL_PREDICTIONS_PATH)
    assert len(val_preds) == 20539, f"Expected 20,539 validation visits, got {len(val_preds)}"

    y_true = val_preds['y_true'].values
    y_prob = val_preds['y_prob_unweighted'].values

    res = evaluate_historical_cost(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.15,
        c_intervene=10.0,
        c_unfilled=200.0,
        effectiveness=0.30
    )

    assert res["baseline_cost"] == 824800.0, f"Expected baseline $824,800, got {res['baseline_cost']}"
    assert res["total_cost"] == 725880.0, f"Expected policy cost $725,880, got {res['total_cost']}"
    assert res["net_savings"] == 98920.0, f"Expected net savings $98,920, got {res['net_savings']}"

    assert res["tp"] == 3861, f"Expected TP 3,861, got {res['tp']}"
    assert res["fp"] == 9413, f"Expected FP 9,413, got {res['fp']}"
    assert res["fn"] == 263, f"Expected FN 263, got {res['fn']}"
    assert res["tn"] == 7002, f"Expected TN 7,002, got {res['tn']}"
    assert res["flagged_count"] == 13274, f"Expected 13,274 flagged, got {res['flagged_count']}"

    recall_pct = round(res["recall"] * 100.0, 2)
    assert recall_pct == 93.62, f"Expected recall 93.62%, got {recall_pct}%"

