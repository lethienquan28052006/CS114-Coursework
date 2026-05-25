from __future__ import annotations

import numpy as np
import pandas as pd


DRIVER_RULES = {
    "frequency_drop": ("Attendance dropped significantly", "Attendance is stable"),
    "low_current_activity": ("Low current month activity", "Current activity is healthy"),
    "contract_remaining_ratio": ("Contract is close to renewal window", "Contract still has runway"),
    "is_contract_ending": ("Contract ending soon", "Contract is not ending soon"),
    "engagement_score": ("Low engagement score", "Strong engagement score"),
    "loyalty_score": ("Low loyalty signal", "Strong loyalty signal"),
    "is_new_customer": ("New customer still forming habits", "Established customer history"),
    "far_no_group": ("Far location and no group activity", "Convenience or group support present"),
    "no_group_low_activity": ("No group visits and low activity", "Social engagement supports retention"),
    "spending_per_month": ("Low service spend", "Service spend indicates commitment"),
}


def _contribution_scores(row: pd.Series) -> dict[str, float]:
    freq_drop = float(row.get("frequency_drop", 0))
    low_current = float(row.get("low_current_activity", 0))
    contract_ratio = float(row.get("contract_remaining_ratio", 1))
    ending = float(row.get("is_contract_ending", 0))
    engagement = float(row.get("engagement_score", 0))
    loyalty = float(row.get("loyalty_score", 0))
    new_customer = float(row.get("is_new_customer", 0))
    far_no_group = float(row.get("far_no_group", 0))
    no_group_low = float(row.get("no_group_low_activity", 0))
    spend_month = float(row.get("spending_per_month", 0))

    return {
        "frequency_drop": max(freq_drop, 0) * 0.35,
        "low_current_activity": low_current * 1.2,
        "contract_remaining_ratio": max(1 - contract_ratio, 0) * 0.9,
        "is_contract_ending": ending * 1.0,
        "engagement_score": -min(engagement / 8, 1.5),
        "loyalty_score": -min(loyalty / 40, 1.5),
        "is_new_customer": new_customer * 0.75,
        "far_no_group": far_no_group * 0.55,
        "no_group_low_activity": no_group_low * 0.65,
        "spending_per_month": -min(spend_month / 120, 1.0),
    }


def explain_row(row: pd.Series, top_n: int = 5) -> dict:
    scores = _contribution_scores(row)
    items = []
    for feature, score in scores.items():
        risk_text, protective_text = DRIVER_RULES.get(feature, (feature, feature))
        items.append(
            {
                "feature": feature,
                "value": float(row.get(feature, 0)),
                "contribution": float(score),
                "reason": risk_text if score >= 0 else protective_text,
                "direction": "increases churn risk" if score >= 0 else "decreases churn risk",
            }
        )
    sorted_items = sorted(items, key=lambda x: abs(x["contribution"]), reverse=True)
    positives = [x for x in sorted_items if x["contribution"] > 0][:top_n]
    negatives = [x for x in sorted_items if x["contribution"] < 0][:top_n]
    return {
        "top_positive": positives,
        "top_negative": negatives,
        "chart_items": sorted_items[:8],
    }


def batch_top_reason(row: pd.Series) -> str:
    explanation = explain_row(row, top_n=1)
    if explanation["top_positive"]:
        return explanation["top_positive"][0]["reason"]
    return "No strong risk driver detected"
