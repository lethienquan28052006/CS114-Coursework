2. Pipeline tổng thể
Raw data
→ EDA
→ Train-test split
→ Preprocessing
→ Feature engineering
→ Imbalance handling
→ Train baseline models
→ Tune best models
→ Evaluate
→ SHAP explainability
→ Export final pipeline
3. Notebook 01 — EDA

Mục tiêu: hiểu data, target, phân phối, quan hệ feature với churn.

Nội dung nên có:

1. Load data
2. Check shape, columns, dtypes
3. Check missing values
4. Check duplicated rows
5. Target distribution: Churn = 0/1
6. Univariate analysis
7. Bivariate analysis với Churn
8. Correlation heatmap
9. Initial insights

Các biểu đồ nên làm:

- Countplot Churn
- Histogram Age, Lifetime, Avg_class_frequency_total
- Boxplot Lifetime vs Churn
- Boxplot Month_to_end_contract vs Churn
- Boxplot Avg_class_frequency_current_month vs Churn
- Countplot Contract_period vs Churn
- Correlation heatmap

Insight ban đầu từ ảnh bạn gửi:

- Data có 4000 dòng, không missing.
- Churn mean khoảng 0.27 → tỷ lệ churn khoảng 27%, có imbalance nhẹ/vừa.
- Lifetime lệch phải mạnh, nhiều khách mới.
- Month_to_end_contract tập trung nhiều ở 1 tháng → đây có thể là feature rất mạnh.
- Avg_class_frequency_current_month có vẻ là feature hành vi quan trọng.
- Contract_period có các mức rõ ràng, nhiều khả năng là 1, 6, 12 tháng.
4. Notebook 02 — Preprocessing baseline

Vì data này hầu hết đã encode dạng số 0/1, preprocessing không quá phức tạp.

Chia cột:

target_col = "Churn"

binary_cols = [
    "gender", "Near_Location", "Partner",
    "Promo_friends", "Phone", "Group_visits"
]

numeric_cols = [
    "Age",
    "Avg_additional_charges_total",
    "Month_to_end_contract",
    "Lifetime",
    "Avg_class_frequency_total",
    "Avg_class_frequency_current_month"
]

ordinal_cols = [
    "Contract_period"
]

Pipeline đề xuất:

- Không encode thêm nếu tất cả đã numeric.
- Scale numeric cho Logistic Regression, SVM, KNN.
- Không cần scale cho RandomForest, XGBoost, CatBoost, LightGBM.

Quan trọng: split trước rồi mới preprocess để tránh data leakage.

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
5. Notebook 03 — Data imbalance

Target churn khoảng 27%, không quá lệch nhưng vẫn nên thử.

Bạn nên so sánh 3 hướng:

1. Không xử lý imbalance
2. class_weight="balanced"
3. SMOTE

Không nên áp dụng SMOTE trước train-test split.

Pipeline đúng:

Train-test split
→ Fit preprocessing trên train
→ Apply SMOTE chỉ trên X_train
→ Train model
→ Evaluate trên X_test gốc

Models nên thử imbalance:

- Logistic Regression + class_weight
- RandomForest + class_weight
- XGBoost + scale_pos_weight
- CatBoost + auto_class_weights
- SMOTE + RandomForest/XGBoost/LightGBM
6. Notebook 04 — Training models

Baseline models nên có:

1. Logistic Regression
2. KNN
3. Decision Tree
4. Random Forest
5. Extra Trees
6. Gradient Boosting
7. XGBoost
8. LightGBM
9. CatBoost

Metrics nên dùng:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- ROC-AUC
- PR-AUC

Với churn, đừng chỉ nhìn Accuracy. Nên ưu tiên:

- Recall của class Churn = 1
- F1-score class Churn = 1
- ROC-AUC
- PR-AUC
7. Notebook 05 — Model tuning

Chỉ tune 2–3 model tốt nhất, không tune tất cả.

Tôi khuyên tune:

1. CatBoost
2. XGBoost
3. RandomForest hoặc LightGBM

Search strategy:

- RandomizedSearchCV trước
- Sau đó Optuna nếu muốn chuyên nghiệp hơn

CV:

StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

Scoring chính:

scoring = "f1"

hoặc tốt hơn:

scoring = "roc_auc"

Nếu mục tiêu là bắt churn tốt hơn, có thể dùng:

scoring = "recall"
8. Notebook 06 — Evaluation + SHAP

Phần này rất quan trọng để báo cáo đẹp.

Cần có:

1. Confusion matrix
2. Classification report
3. ROC curve
4. Precision-Recall curve
5. Feature importance
6. SHAP summary plot
7. SHAP dependence plot
8. Phân tích business insight

Các feature khả năng cao sẽ mạnh:

- Lifetime
- Month_to_end_contract
- Avg_class_frequency_current_month
- Avg_class_frequency_total
- Contract_period
- Age

Diễn giải business:

Khách có lifetime thấp, sắp hết hợp đồng, tần suất đi tập tháng hiện tại thấp → khả năng churn cao.
9. Notebook 07 — Final pipeline

Notebook cuối nên làm gọn:

1. Load data
2. Split data
3. Preprocessing
4. Train best model
5. Evaluate final
6. Save model
7. Save preprocessing pipeline
8. Test predict 1 khách hàng mẫu

Export:

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(preprocessor, "models/preprocessing_pipeline.pkl")
10. Feature engineering nên thử

Nên tạo các feature này:

df["contract_remaining_ratio"] = df["Month_to_end_contract"] / df["Contract_period"]

df["frequency_drop"] = (
    df["Avg_class_frequency_total"] 
    - df["Avg_class_frequency_current_month"]
)

df["engagement_score"] = (
    df["Avg_class_frequency_current_month"] 
    * df["Lifetime"]
)

df["spending_per_month"] = (
    df["Avg_additional_charges_total"] 
    / (df["Lifetime"] + 1)
)

df["is_new_customer"] = (df["Lifetime"] <= 1).astype(int)

df["is_contract_ending"] = (df["Month_to_end_contract"] <= 1).astype(int)

df["low_activity"] = (
    df["Avg_class_frequency_current_month"] < 1
).astype(int)

Nhưng nhớ: feature engineering phải tạo sau split trong pipeline hoặc tạo bằng transformer để tránh leakage nhẹ.

11. Thứ tự làm tốt nhất
01_eda
→ 02_baseline_models không feature engineering
→ 03_feature_engineering
→ 04_imbalance_handling
→ 05_model_tuning
→ 06_shap_analysis
→ 07_final_pipeline

Tôi đề xuất bạn đừng làm imbalance quá sớm. Hãy có baseline sạch trước, rồi mới chứng minh imbalance handling có cải thiện hay không.

12. Mục tiêu điểm số hợp lý

Với dataset này, nếu làm tốt:

Accuracy: 0.88–0.93
ROC-AUC: 0.90+
F1 churn class: 0.75–0.85

Mục tiêu báo cáo không phải chỉ “accuracy cao”, mà là:

Mô hình phát hiện khách hàng có nguy cơ rời bỏ và giải thích được nguyên nhân ch