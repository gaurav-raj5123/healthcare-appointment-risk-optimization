"""
Step 5: Threshold Selection and Business Cost Analysis
Evaluates the unweighted XGBoost model on the validation set at:
  - Default Threshold: 0.50
  - Business-Selected Threshold: 0.15
Uses illustrative cost assumptions:
  - Unfilled appointment cost: $200 per missed appointment
  - Intervention cost: $10 per flagged/intervened patient (both TP and FP receive outreach)
  - Intervention effectiveness: 30% (illustrative assumption)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VAL_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'val_features.csv')
PREDS_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'val_predictions_xgb_unweighted.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'xgboost_unweighted.joblib')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'steps', '05_evaluation', 'figures')
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'step5_threshold_comparison.csv')
OUTPUT_JSON_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'step5_threshold_cost_analysis.json')

# Illustrative Business Assumptions
COST_UNFILLED_APPOINTMENT = 200.0  # $200 per missed slot
COST_INTERVENTION = 10.0            # $10 per flagged/intervened patient
INTERVENTION_EFFECTIVENESS = 0.30   # 30% of actual no-shows prevented if contacted

THRESHOLDS = [0.50, 0.15]

def load_validation_predictions():
    """
    Loads existing validation predictions or runs inference using the saved unweighted XGBoost model without retraining.
    Performs comprehensive data integrity and reliability validations.
    """
    if os.path.exists(PREDS_PATH):
        print(f"Loading existing validation predictions from: {PREDS_PATH}")
        df_preds = pd.read_csv(PREDS_PATH)
        y_val = df_preds['y_true'].values
        probs_val = df_preds['y_prob_unweighted'].values
    elif os.path.exists(MODEL_PATH) and os.path.exists(VAL_DATA_PATH):
        import joblib
        print(f"Loading unweighted XGBoost model from: {MODEL_PATH}")
        pipe_xgb = joblib.load(MODEL_PATH)
        df_val = pd.read_csv(VAL_DATA_PATH)
        meta_cols = ['appointment_id', 'patient_id', 'scheduled_day', 'appointment_day', 'sms_received', 'no_show']
        feature_cols = [c for c in df_val.columns if c not in meta_cols]
        X_val = df_val[feature_cols]
        y_val = df_val['no_show'].values
        print("Generating predictions on validation set (without retraining)...")
        probs_val = pipe_xgb.predict_proba(X_val)[:, 1]
    else:
        raise FileNotFoundError("Neither precomputed predictions nor the unweighted XGBoost model could be found!")
    
    # -------------------------------------------------------------
    # RELIABILITY & INTEGRITY CHECKS
    # -------------------------------------------------------------
    # 1. Matching lengths
    assert len(y_val) == len(probs_val), f"Length mismatch: y_val ({len(y_val)}) vs probs_val ({len(probs_val)})!"
    
    # 2. Binary labels
    assert set(np.unique(y_val)) <= {0, 1}, f"Validation labels must be strictly binary {0, 1}, found {set(np.unique(y_val))}!"
    
    # 3. Finite probabilities between 0 and 1
    assert np.all(np.isfinite(probs_val)), "Non-finite probability values detected in validation predictions!"
    assert np.all((probs_val >= 0.0) & (probs_val <= 1.0)), "Predicted probabilities must lie within [0.0, 1.0]!"
    
    # 4. Verify against raw validation dataset to ensure predictions are not stale
    if os.path.exists(VAL_DATA_PATH):
        df_val_check = pd.read_csv(VAL_DATA_PATH)
        assert len(df_val_check) == len(y_val), f"Mismatch between validation dataset rows ({len(df_val_check)}) and predictions ({len(y_val)})!"
        assert np.array_equal(y_val, df_val_check['no_show'].values), "Mismatch between saved prediction labels and validation ground-truth labels!"
        
    actual_noshows = int(np.sum(y_val == 1))
    actual_shows = int(np.sum(y_val == 0))
    print(f"Validation dataset size: {len(y_val):,} appointments")
    print(f"Actual No-Shows (y=1):   {actual_noshows:,} ({actual_noshows/len(y_val)*100:.2f}%)")
    print(f"Actual Attended (y=0):   {actual_shows:,} ({actual_shows/len(y_val)*100:.2f}%)")
    return y_val, probs_val

def compute_threshold_metrics(y_true, y_prob, threshold):
    """
    Computes classification and economic cost metrics for a given decision threshold.
    All calculations are derived programmatically from the confusion matrix.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    
    total_appointments = len(y_true)
    flagged_count = tp + fp
    pct_flagged = (flagged_count / total_appointments) * 100.0
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    # -------------------------------------------------------------
    # ECONOMIC COST CALCULATION
    # Total Cost = Flagged Patients * Intervention Cost + [FN + (1 - r) * TP] * Cost of Unfilled Appointment
    # Note: Intervention cost ($10) applies to all flagged patients (TP + FP).
    # -------------------------------------------------------------
    unprevented_noshows = fn + (1.0 - INTERVENTION_EFFECTIVENESS) * tp
    intervention_cost = flagged_count * COST_INTERVENTION
    capacity_loss_cost = unprevented_noshows * COST_UNFILLED_APPOINTMENT
    expected_cost = intervention_cost + capacity_loss_cost
    
    # Baseline Cost (No intervention: all no-shows occur without mitigation)
    actual_noshows = int(np.sum(y_true == 1))
    baseline_cost = actual_noshows * COST_UNFILLED_APPOINTMENT
    net_savings = baseline_cost - expected_cost
    
    cm = [[tn, fp], [fn, tp]]
    
    return {
        "threshold": threshold,
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "flagged_count": flagged_count,
        "pct_flagged": round(float(pct_flagged), 2),
        "intervention_cost": round(float(intervention_cost), 2),
        "unprevented_noshows": round(float(unprevented_noshows), 1),
        "capacity_loss_cost": round(float(capacity_loss_cost), 2),
        "expected_cost": round(float(expected_cost), 2),
        "baseline_cost": round(float(baseline_cost), 2),
        "net_savings": round(float(net_savings), 2),
        "confusion_matrix": cm
    }

def plot_confusion_matrices(results, y_val):
    """Renders side-by-side confusion matrix comparisons for threshold 0.50 vs 0.15."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    for idx, res in enumerate(results):
        t = res['threshold']
        cm = np.array(res['confusion_matrix'])
        
        tn, fp = cm[0]
        fn, tp = cm[1]
        
        labels = [
            [f"TN\n{tn:,}\n({tn/cm.sum()*100:.1f}%)", f"FP\n{fp:,}\n({fp/cm.sum()*100:.1f}%)"],
            [f"FN\n{fn:,}\n({fn/cm.sum()*100:.1f}%)", f"TP\n{tp:,}\n({tp/cm.sum()*100:.1f}%)"]
        ]
        
        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="Blues" if t == 0.50 else "YlGnBu",
            cbar=False,
            ax=axes[idx],
            annot_kws={"fontsize": 11, "fontweight": "bold"}
        )
        
        title_tag = "CONVENTIONAL DEFAULT" if t == 0.50 else "BUSINESS-SELECTED"
        axes[idx].set_title(
            f"Threshold = {t:.2f} ({title_tag})\n"
            f"Recall: {res['recall']*100:.2f}% | Precision: {res['precision']*100:.2f}%\n"
            f"Flagged: {res['flagged_count']:,} ({res['pct_flagged']:.1f}%) | Cost: ${res['expected_cost']:,.0f}",
            fontsize=11,
            fontweight='bold',
            pad=10
        )
        axes[idx].set_xlabel("Predicted Class (0=Show, 1=No-Show)", fontsize=10, fontweight='bold')
        axes[idx].set_ylabel("Actual Class (0=Show, 1=No-Show)", fontsize=10, fontweight='bold')
        axes[idx].set_xticklabels(['Show (0)', 'No-Show (1)'])
        axes[idx].set_yticklabels(['Show (0)', 'No-Show (1)'])
        
    plt.suptitle("Confusion Matrix Comparison on Validation Set (Unweighted XGBoost)", fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    
    save_fig_path = os.path.join(FIGURES_DIR, 'confusion_matrix_threshold_comparison.png')
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix figure to: {save_fig_path}")

def run_step5_evaluation():
    print("=" * 75)
    print("STEP 5: THRESHOLD SELECTION AND BUSINESS COST ANALYSIS")
    print("=" * 75)
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    y_val, probs_val = load_validation_predictions()
    
    results = []
    for t in THRESHOLDS:
        res = compute_threshold_metrics(y_val, probs_val, t)
        results.append(res)
        
    df_comp = pd.DataFrame(results)
    
    print("\n" + "-" * 75)
    print("THRESHOLD COMPARISON TABLE (Validation Set)")
    print("-" * 75)
    display_cols = [
        'threshold', 'recall', 'precision', 'f1_score', 'accuracy',
        'true_positives', 'false_positives', 'false_negatives', 'true_negatives',
        'flagged_count', 'pct_flagged', 'expected_cost', 'net_savings'
    ]
    print(df_comp[display_cols].to_string(index=False))
    
    baseline_cost = results[0]['baseline_cost']
    cost_50 = results[0]['expected_cost']
    cost_15 = results[1]['expected_cost']
    incremental_savings = cost_50 - cost_15
    pct_reduction = (incremental_savings / cost_50) * 100.0
    
    print("\n" + "-" * 75)
    print("BUSINESS COST SUMMARY (Illustrative Assumptions: FN=$200, FP=$10, Efficacy=30%)")
    print("-" * 75)
    print(f"No-Intervention Baseline Cost:              ${baseline_cost:,.2f}")
    print(f"Threshold 0.50 Expected Cost:               ${cost_50:,.2f} (Savings: ${baseline_cost - cost_50:,.2f})")
    print(f"Threshold 0.15 Expected Cost:               ${cost_15:,.2f} (Savings: ${baseline_cost - cost_15:,.2f})")
    print(f"Incremental Cost Reduction (0.15 vs 0.50):   ${incremental_savings:,.2f} ({pct_reduction:.2f}% lower cost)")
    
    # Save CSV
    df_comp.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nSaved threshold comparison CSV to: {OUTPUT_CSV_PATH}")
    
    # Save JSON Report
    payload = {
        "analysis_scope": "Validation partition only (May 20 - May 31, 2016; N=20,539)",
        "model_evaluated": "XGBoost (Unweighted)",
        "illustrative_assumptions": {
            "unfilled_appointment_cost": COST_UNFILLED_APPOINTMENT,
            "intervention_cost": COST_INTERVENTION,
            "intervention_effectiveness": INTERVENTION_EFFECTIVENESS,
            "interpretation": "Under the illustrative business assumptions, the theoretical decision threshold is approximately 0.167, assuming calibrated probabilities and uniform intervention effectiveness and costs. The empirical analysis identifies 0.15 as the lowest-cost candidate among the thresholds evaluated.",
            "validation_limitation": "The threshold and associated cost were selected and evaluated using the same validation dataset. Therefore, these figures are validation-based selection results under illustrative assumptions, not unbiased estimates of real-world savings."
        },
        "baseline_no_intervention_cost": baseline_cost,
        "threshold_evaluations": results,
        "incremental_savings_0_15_vs_0_50": incremental_savings,
        "pct_cost_reduction_0_15_vs_0_50": round(pct_reduction, 2)
    }
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved threshold cost analysis JSON to: {OUTPUT_JSON_PATH}")
    
    # Plot and save figures
    plot_confusion_matrices(results, y_val)
    
    print("\nSTEP 5 THRESHOLD SELECTION AND BUSINESS COST ANALYSIS COMPLETE.")
    return df_comp

if __name__ == '__main__':
    run_step5_evaluation()
