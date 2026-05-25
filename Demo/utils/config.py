from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_PATH = MODEL_DIR / "best_feature_engineering_model.pkl"

ORIGINAL_FEATURES = [
    "gender",
    "Near_Location",
    "Partner",
    "Promo_friends",
    "Phone",
    "Contract_period",
    "Group_visits",
    "Age",
    "Avg_additional_charges_total",
    "Month_to_end_contract",
    "Lifetime",
    "Avg_class_frequency_total",
    "Avg_class_frequency_current_month",
]

ENGINEERED_FEATURES = [
    "frequency_drop",
    "frequency_ratio_current_total",
    "low_current_activity",
    "high_current_activity",
    "engagement_score",
    "total_engagement_score",
    "contract_remaining_ratio",
    "is_contract_ending",
    "is_short_contract",
    "is_long_contract",
    "renewal_pressure_score",
    "is_new_customer",
    "is_loyal_customer",
    "lifetime_per_contract",
    "loyalty_score",
    "spending_per_month",
    "high_spending",
    "low_spending",
    "convenience_score",
    "social_commitment_score",
    "far_no_group",
    "no_contact_info",
    "new_low_activity",
    "ending_low_activity",
    "far_low_activity",
    "short_contract_low_activity",
    "no_group_low_activity",
    "partner_long_contract",
    "promo_group",
    "log_lifetime",
    "log_additional_charges",
    "log_frequency_current",
    "log_frequency_total",
]

FEATURE_COLUMNS = ORIGINAL_FEATURES + ENGINEERED_FEATURES

DEFAULT_THRESHOLDS = {
    "spending_q25": 68.0,
    "spending_q75": 180.0,
}


def ensure_directories() -> None:
    for path in [MODEL_DIR, UPLOAD_DIR, OUTPUT_DIR, REPORT_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)
