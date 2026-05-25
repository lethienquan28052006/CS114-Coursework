import numpy as np
import pandas as pd

from .config import DEFAULT_THRESHOLDS, FEATURE_COLUMNS, ORIGINAL_FEATURES


def validate_input_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in ORIGINAL_FEATURES if column not in df.columns]


def coerce_input_types(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for column in ORIGINAL_FEATURES:
        if column not in data.columns:
            data[column] = 0
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data[ORIGINAL_FEATURES].fillna(0)


def calculate_feature_thresholds(df: pd.DataFrame) -> dict:
    if "Avg_additional_charges_total" not in df.columns or df.empty:
        return DEFAULT_THRESHOLDS.copy()
    charges = pd.to_numeric(df["Avg_additional_charges_total"], errors="coerce").dropna()
    if charges.empty:
        return DEFAULT_THRESHOLDS.copy()
    return {
        "spending_q25": float(charges.quantile(0.25)),
        "spending_q75": float(charges.quantile(0.75)),
    }


def _col(df: pd.DataFrame, column: str, default: float = 0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def create_features(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    data = coerce_input_types(df)
    out = data.copy()
    eps = 1e-6
    thresholds = thresholds or DEFAULT_THRESHOLDS

    lifetime = _col(out, "Lifetime")
    freq_total = _col(out, "Avg_class_frequency_total")
    freq_current = _col(out, "Avg_class_frequency_current_month")
    contract_period = _col(out, "Contract_period")
    month_to_end = _col(out, "Month_to_end_contract")
    charges = _col(out, "Avg_additional_charges_total")
    near = _col(out, "Near_Location")
    partner = _col(out, "Partner")
    promo = _col(out, "Promo_friends")
    phone = _col(out, "Phone")
    group = _col(out, "Group_visits")

    out["frequency_drop"] = freq_total - freq_current
    out["frequency_ratio_current_total"] = freq_current / (freq_total + eps)
    out["low_current_activity"] = (freq_current < 1).astype(int)
    out["high_current_activity"] = (freq_current >= 3).astype(int)
    out["engagement_score"] = lifetime * freq_current
    out["total_engagement_score"] = lifetime * freq_total

    out["contract_remaining_ratio"] = month_to_end / (contract_period + eps)
    out["is_contract_ending"] = (month_to_end <= 1).astype(int)
    out["is_short_contract"] = (contract_period <= 1).astype(int)
    out["is_long_contract"] = (contract_period >= 12).astype(int)
    out["renewal_pressure_score"] = out["is_contract_ending"] * out["is_short_contract"]

    out["is_new_customer"] = (lifetime <= 1).astype(int)
    out["is_loyal_customer"] = (lifetime >= 6).astype(int)
    out["lifetime_per_contract"] = lifetime / (contract_period + eps)
    out["loyalty_score"] = lifetime * contract_period

    out["spending_per_month"] = charges / (lifetime + 1)
    out["high_spending"] = (charges >= thresholds.get("spending_q75", DEFAULT_THRESHOLDS["spending_q75"])).astype(int)
    out["low_spending"] = (charges <= thresholds.get("spending_q25", DEFAULT_THRESHOLDS["spending_q25"])).astype(int)

    out["convenience_score"] = near + partner + promo + phone
    out["social_commitment_score"] = group + promo + partner
    out["far_no_group"] = ((near == 0) & (group == 0)).astype(int)
    out["no_contact_info"] = (phone == 0).astype(int)

    out["new_low_activity"] = out["is_new_customer"] * out["low_current_activity"]
    out["ending_low_activity"] = out["is_contract_ending"] * out["low_current_activity"]
    out["far_low_activity"] = (1 - near) * out["low_current_activity"]
    out["short_contract_low_activity"] = out["is_short_contract"] * out["low_current_activity"]
    out["no_group_low_activity"] = (1 - group) * out["low_current_activity"]
    out["partner_long_contract"] = partner * out["is_long_contract"]
    out["promo_group"] = promo * group

    out["log_lifetime"] = np.log1p(lifetime.clip(lower=0))
    out["log_additional_charges"] = np.log1p(charges.clip(lower=0))
    out["log_frequency_current"] = np.log1p(freq_current.clip(lower=0))
    out["log_frequency_total"] = np.log1p(freq_total.clip(lower=0))

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    for column in FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = 0
    return out[FEATURE_COLUMNS]
