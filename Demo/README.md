# Gym Customer Churn Prediction System

Production-style FastAPI web application for predicting gym customer churn.

## Features

- Single customer churn prediction
- Batch CSV prediction
- Feature engineering recreated in backend
- Risk level analysis
- SHAP-style churn driver explanation
- Rule-based retention recommendations
- Modern glassmorphism dashboard UI

## Project Structure

```text
Demo/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── best_feature_engineering_model.pkl
├── utils/
├── templates/
├── static/
├── uploads/
├── outputs/
└── reports/
```

## Setup

From the `Demo/` directory:

```bash
pip install -r requirements.txt
```

## Run App

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## CSV Batch Prediction

The CSV must include these original input columns:

- `gender`
- `Near_Location`
- `Partner`
- `Promo_friends`
- `Phone`
- `Contract_period`
- `Group_visits`
- `Age`
- `Avg_additional_charges_total`
- `Month_to_end_contract`
- `Lifetime`
- `Avg_class_frequency_total`
- `Avg_class_frequency_current_month`

The app adds:

- `churn_probability`
- `prediction`
- `risk_level`
- `top_reason`
- `recommendation`

## Model Notes

`models/best_feature_engineering_model.pkl` is loaded at startup. The backend recreates feature engineering before prediction, so the model receives the same feature layout used during training.

The loader supports either:

- a raw CatBoost-style model with `predict_proba`
- a saved artifact dictionary containing `model`, `pipeline`, thresholds, or feature metadata

## Screenshots

Add screenshots here after running the app locally:

- Home page
- Prediction result
- Batch prediction result
