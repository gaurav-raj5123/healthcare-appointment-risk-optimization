"""
CarePredict - Financial & Operational Optimization Engine
Implements asymmetric healthcare cost modeling, threshold optimization,
capacity-constrained scheduling, and sensitivity analysis.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from app.config import (
    DEFAULT_THRESHOLD,
    DEFAULT_COST_INTERVENTION,
    DEFAULT_COST_UNFILLED,
    DEFAULT_OUTREACH_EFFECTIVENESS,
    COLORS
)

def evaluate_historical_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    c_intervene: float = DEFAULT_COST_INTERVENTION,
    c_unfilled: float = DEFAULT_COST_UNFILLED,
    effectiveness: float = DEFAULT_OUTREACH_EFFECTIVENESS
) -> dict:
    """
    Computes exact historical confusion matrix and cost metrics on validation data.
    Formula: Total Cost = Flagged * c_intervene + [FN + (1 - r) * TP] * c_unfilled
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    
    total_samples = len(y_true)
    flagged_count = tp + fp
    pct_flagged = (flagged_count / total_samples) * 100.0 if total_samples > 0 else 0.0
    
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    
    # Financial metrics
    intervention_cost = flagged_count * c_intervene
    unprevented_noshows = fn + (1.0 - effectiveness) * tp
    prevented_noshows = effectiveness * tp
    capacity_loss_cost = unprevented_noshows * c_unfilled
    
    total_cost = intervention_cost + capacity_loss_cost
    actual_noshows = int(np.sum(y_true == 1))
    baseline_cost = actual_noshows * c_unfilled
    net_savings = baseline_cost - total_cost
    pct_reduction = (net_savings / baseline_cost) * 100.0 if baseline_cost > 0 else 0.0
    
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "flagged_count": flagged_count,
        "pct_flagged": pct_flagged,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "actual_noshows": actual_noshows,
        "prevented_noshows": prevented_noshows,
        "unprevented_noshows": unprevented_noshows,
        "intervention_cost": intervention_cost,
        "capacity_loss_cost": capacity_loss_cost,
        "total_cost": total_cost,
        "baseline_cost": baseline_cost,
        "net_savings": net_savings,
        "pct_cost_reduction": pct_reduction
    }

def evaluate_prospective_cost(
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    c_intervene: float = DEFAULT_COST_INTERVENTION,
    c_unfilled: float = DEFAULT_COST_UNFILLED,
    effectiveness: float = DEFAULT_OUTREACH_EFFECTIVENESS
) -> dict:
    """
    Computes modeled expected costs for prospective cohorts without ground-truth labels.
    Uses expected value integration over predicted probabilities.
    """
    total_samples = len(y_prob)
    flagged_mask = y_prob >= threshold
    flagged_count = int(np.sum(flagged_mask))
    pct_flagged = (flagged_count / total_samples) * 100.0 if total_samples > 0 else 0.0
    
    expected_noshows = float(np.sum(y_prob))
    expected_flagged_noshows = float(np.sum(y_prob[flagged_mask]))  # Expected TP
    expected_unflagged_noshows = float(np.sum(y_prob[~flagged_mask]))  # Expected FN
    
    expected_prevented = effectiveness * expected_flagged_noshows
    expected_unprevented = expected_unflagged_noshows + (1.0 - effectiveness) * expected_flagged_noshows
    
    intervention_cost = flagged_count * c_intervene
    capacity_loss_cost = expected_unprevented * c_unfilled
    total_cost = intervention_cost + capacity_loss_cost
    
    baseline_cost = expected_noshows * c_unfilled
    net_savings = baseline_cost - total_cost
    pct_reduction = (net_savings / baseline_cost) * 100.0 if baseline_cost > 0 else 0.0
    
    return {
        "threshold": threshold,
        "total_appointments": total_samples,
        "flagged_count": flagged_count,
        "pct_flagged": pct_flagged,
        "expected_noshows": expected_noshows,
        "expected_prevented": expected_prevented,
        "expected_unprevented": expected_unprevented,
        "intervention_cost": intervention_cost,
        "capacity_loss_cost": capacity_loss_cost,
        "total_cost": total_cost,
        "baseline_cost": baseline_cost,
        "net_savings": net_savings,
        "pct_cost_reduction": pct_reduction
    }

def generate_threshold_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected_threshold: float,
    c_intervene: float = DEFAULT_COST_INTERVENTION,
    c_unfilled: float = DEFAULT_COST_UNFILLED,
    effectiveness: float = DEFAULT_OUTREACH_EFFECTIVENESS
) -> tuple[pd.DataFrame, go.Figure, float]:
    """
    Sweeps thresholds from 0.02 to 0.70 to generate an empirical cost-utility curve.
    Returns:
      - df_curve: Evaluation grid.
      - fig: Interactive Plotly figure.
      - lowest_cost_threshold: Threshold yielding minimum total cost.
    """
    thresholds = np.linspace(0.02, 0.70, 69)
    records = []
    
    for t in thresholds:
        res = evaluate_historical_cost(
            y_true, y_prob, threshold=t,
            c_intervene=c_intervene, c_unfilled=c_unfilled, effectiveness=effectiveness
        )
        records.append({
            "threshold": round(float(t), 3),
            "total_cost": res["total_cost"],
            "net_savings": res["net_savings"],
            "recall": res["recall"],
            "precision": res["precision"],
            "flagged_count": res["flagged_count"]
        })
        
    df_curve = pd.DataFrame(records)
    lowest_row = df_curve.loc[df_curve['total_cost'].idxmin()]
    lowest_threshold = float(lowest_row['threshold'])
    
    # Render Plotly Chart
    fig = go.Figure()

    # Total Cost line
    fig.add_trace(go.Scatter(
        x=df_curve['threshold'],
        y=df_curve['total_cost'],
        mode='lines',
        name='Total Expected Cost',
        line=dict(color=COLORS['primary'], width=3),
        hovertemplate="Threshold: %{x:.2f}<br>Total Cost: $%{y:,.0f}<extra></extra>"
    ))

    # Baseline cost horizontal dashed line
    baseline_val = float(np.sum(y_true == 1) * c_unfilled)
    fig.add_trace(go.Scatter(
        x=[0.02, 0.70],
        y=[baseline_val, baseline_val],
        mode='lines',
        name='Zero-Intervention Baseline',
        line=dict(color=COLORS['danger'], width=2, dash='dash'),
        hoverinfo='y'
    ))

    # Mark Lowest-Cost Threshold
    fig.add_trace(go.Scatter(
        x=[lowest_threshold],
        y=[lowest_row['total_cost']],
        mode='markers',
        name=f'Lowest-Cost Candidate ({lowest_threshold:.2f})',
        marker=dict(color=COLORS['success'], size=12, symbol='star'),
        hoverinfo='text',
        text=[f"Lowest Cost: ${lowest_row['total_cost']:,.0f} at threshold {lowest_threshold:.2f}"]
    ))

    # Mark Current Selected Threshold
    sel_row = df_curve.iloc[(df_curve['threshold'] - selected_threshold).abs().argmin()]
    fig.add_trace(go.Scatter(
        x=[selected_threshold],
        y=[sel_row['total_cost']],
        mode='markers',
        name=f'User Selected ({selected_threshold:.2f})',
        marker=dict(color='#8b5cf6', size=11, symbol='circle'),
        hoverinfo='text',
        text=[f"Selected: ${sel_row['total_cost']:,.0f} at threshold {selected_threshold:.2f}"]
    ))

    fig.update_layout(
        title=dict(
            text="<b>Expected Healthcare Cost vs. Decision Threshold</b>",
            font=dict(size=14, color=COLORS['text'])
        ),
        xaxis=dict(
            title="Classification Decision Threshold (Prob >= Threshold → Flag)",
            gridcolor="#f1f5f9"
        ),
        yaxis=dict(
            title="Total Healthcare Expense ($ USD)",
            gridcolor="#f1f5f9",
            tickprefix="$"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=40, t=60, b=40),
        height=380
    )

    return df_curve, fig, lowest_threshold

def apply_capacity_constraint(df_predictions: pd.DataFrame, daily_capacity: int) -> tuple[pd.DataFrame, dict]:
    """
    Ranks appointments by predicted no-show risk descending
    and prioritizes the top N within available outreach capacity.
    """
    if daily_capacity < 0:
        raise ValueError(f"daily_capacity must be non-negative (provided: {daily_capacity}).")

    df = df_predictions.copy()
    if 'predicted_noshow_prob' not in df.columns:
        raise ValueError("DataFrame must contain 'predicted_noshow_prob' column.")
    
    # If flagged column not provided, derive using default threshold
    if 'flagged_for_intervention' not in df.columns:
        df['flagged_for_intervention'] = df['predicted_noshow_prob'] >= DEFAULT_THRESHOLD

    # Sort strictly by risk descending
    df_sorted = df.sort_values('predicted_noshow_prob', ascending=False).reset_index(drop=True)
    
    # Priority rank
    df_sorted['priority_rank'] = df_sorted.index + 1
    if daily_capacity == 0:
        df_sorted['within_capacity'] = False
    else:
        df_sorted['within_capacity'] = df_sorted['priority_rank'] <= daily_capacity
    
    flagged_mask = df_sorted['flagged_for_intervention']
    flagged_total = int(np.sum(flagged_mask))
    within_cap_flagged = int(np.sum(flagged_mask & df_sorted['within_capacity']))
    shortfall = max(0, flagged_total - daily_capacity)
    coverage_pct = (within_cap_flagged / flagged_total * 100.0) if flagged_total > 0 else 100.0

    stats = {
        "daily_capacity": daily_capacity,
        "total_appointments": len(df_sorted),
        "total_flagged": flagged_total,
        "reached_within_capacity": within_cap_flagged,
        "unreached_shortfall": shortfall,
        "capacity_coverage_pct": coverage_pct
    }

    return df_sorted, stats

def compute_sensitivity_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    c_intervene: float = DEFAULT_COST_INTERVENTION
) -> pd.DataFrame:
    """
    Computes a 2D sensitivity table varying outreach effectiveness and unfilled slot cost.
    """
    effectiveness_range = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    unfilled_costs = [150.0, 200.0, 250.0, 300.0]
    
    matrix_data = []
    for eff in effectiveness_range:
        row = {"Effectiveness (r)": f"{int(eff*100)}%"}
        for c_unfilled in unfilled_costs:
            res = evaluate_historical_cost(
                y_true, y_prob, threshold=threshold,
                c_intervene=c_intervene, c_unfilled=c_unfilled, effectiveness=eff
            )
            row[f"${int(c_unfilled)} Slot Cost"] = f"${res['net_savings']:,.0f}"
        matrix_data.append(row)
        
    return pd.DataFrame(matrix_data)

