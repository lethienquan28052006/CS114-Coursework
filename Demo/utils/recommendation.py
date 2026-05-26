def generate_recommendations(features: dict, drivers: list[dict] | None = None) -> list[str]:
    driver_features = {driver.get("feature") for driver in (drivers or [])}
    recommendations: list[str] = []

    if features.get("frequency_drop", 0) > 0.5 or features.get("low_current_activity", 0) == 1 or {"frequency_drop", "low_current_activity"} & driver_features:
        recommendations.append("Gửi chiến dịch tái tương tác và tặng một buổi tập cá nhân miễn phí.")

    if features.get("is_contract_ending", 0) == 1 or features.get("contract_remaining_ratio", 1) <= 0.2 or {"is_contract_ending", "contract_remaining_ratio"} & driver_features:
        recommendations.append("Đề xuất ưu đãi gia hạn trước khi hợp đồng hết hạn.")

    if features.get("engagement_score", 0) < 2:
        recommendations.append("Sắp xếp buổi tư vấn với huấn luyện viên để xây dựng lại thói quen tập luyện.")

    if features.get("Group_visits", 0) == 0 or features.get("no_group_low_activity", 0) == 1:
        recommendations.append("Mời khách hàng tham gia lớp nhóm hoặc các sự kiện cộng đồng.")

    if features.get("is_new_customer", 0) == 1:
        recommendations.append("Theo dõi sau khi đăng ký và hỗ trợ khách hàng hình thành thói quen tập luyện.")

    if not recommendations:
        recommendations.append("Gửi tin nhắn chăm sóc cá nhân hóa và theo dõi xu hướng hoạt động.")

    return list(dict.fromkeys(recommendations))


def batch_recommendation(features: dict, drivers: list[dict] | None = None) -> str:
    return " | ".join(generate_recommendations(features, drivers)[:2])
