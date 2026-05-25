def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "LOW"
    if probability < 0.7:
        return "MEDIUM"
    return "HIGH"


def risk_badge_class(risk_level: str) -> str:
    return {
        "LOW": "risk-low",
        "MEDIUM": "risk-medium",
        "HIGH": "risk-high",
    }.get(risk_level, "risk-medium")
