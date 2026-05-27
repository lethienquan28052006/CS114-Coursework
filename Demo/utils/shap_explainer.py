from __future__ import annotations

import numpy as np
import pandas as pd


DRIVER_RULES = {
    "frequency_drop": ("Tần suất tập giảm đáng kể", "Tần suất tập ổn định"),
    "low_current_activity": ("Hoạt động trong tháng hiện tại thấp", "Hoạt động hiện tại ở mức tốt"),
    "contract_remaining_ratio": ("Hợp đồng sắp đến giai đoạn gia hạn", "Hợp đồng vẫn còn nhiều thời gian"),
    "is_contract_ending": ("Hợp đồng sắp hết hạn", "Hợp đồng chưa gần hết hạn"),
    "engagement_score": ("Điểm tương tác thấp", "Điểm tương tác tốt"),
    "loyalty_score": ("Tín hiệu gắn bó thấp", "Tín hiệu gắn bó tốt"),
    "is_new_customer": ("Khách hàng mới chưa hình thành thói quen", "Khách hàng đã có lịch sử gắn bó"),
    "far_no_group": ("Ở xa và không tham gia lớp nhóm", "Có lợi thế về vị trí hoặc hoạt động nhóm"),
    "no_group_low_activity": ("Không tham gia lớp nhóm và hoạt động thấp", "Tương tác xã hội hỗ trợ giữ chân"),
    "spending_per_month": ("Chi tiêu dịch vụ thấp", "Chi tiêu dịch vụ thể hiện mức cam kết"),
}


FEATURE_LABELS = {
    "frequency_drop": "Mức giảm tần suất tập",
    "low_current_activity": "Hoạt động tháng hiện tại thấp",
    "contract_remaining_ratio": "Tỷ lệ thời hạn hợp đồng còn lại",
    "is_contract_ending": "Hợp đồng sắp hết hạn",
    "engagement_score": "Điểm tương tác",
    "loyalty_score": "Điểm gắn bó",
    "is_new_customer": "Khách hàng mới",
    "far_no_group": "Ở xa và không tập nhóm",
    "no_group_low_activity": "Không tập nhóm và ít hoạt động",
    "spending_per_month": "Chi tiêu mỗi tháng",
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
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "value": float(row.get(feature, 0)),
                "contribution": float(score),
                "reason": risk_text if score >= 0 else protective_text,
                "direction": "làm tăng rủi ro rời bỏ" if score >= 0 else "làm giảm rủi ro rời bỏ",
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
    return "Không phát hiện yếu tố rủi ro nổi bật"
