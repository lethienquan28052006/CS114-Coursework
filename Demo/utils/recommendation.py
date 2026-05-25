def generate_recommendations(features: dict, drivers: list[dict] | None = None) -> list[str]:
    driver_features = {driver.get("feature") for driver in (drivers or [])}
    recommendations: list[str] = []

    if features.get("frequency_drop", 0) > 0.5 or features.get("low_current_activity", 0) == 1 or {"frequency_drop", "low_current_activity"} & driver_features:
        recommendations.append("Send re-engagement campaign and offer a free personal training session.")

    if features.get("is_contract_ending", 0) == 1 or features.get("contract_remaining_ratio", 1) <= 0.2 or {"is_contract_ending", "contract_remaining_ratio"} & driver_features:
        recommendations.append("Offer renewal discount before the contract expires.")

    if features.get("engagement_score", 0) < 2:
        recommendations.append("Schedule trainer consultation to rebuild gym routine.")

    if features.get("Group_visits", 0) == 0 or features.get("no_group_low_activity", 0) == 1:
        recommendations.append("Invite the customer to group classes or community events.")

    if features.get("is_new_customer", 0) == 1:
        recommendations.append("Run onboarding follow-up and habit-building support.")

    if not recommendations:
        recommendations.append("Send personalized check-in and monitor activity trend.")

    return list(dict.fromkeys(recommendations))


def batch_recommendation(features: dict, drivers: list[dict] | None = None) -> str:
    return " | ".join(generate_recommendations(features, drivers)[:2])
