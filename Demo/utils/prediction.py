from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_THRESHOLDS, MODEL_PATH
from .feature_engineering import create_features
from .recommendation import batch_recommendation, generate_recommendations
from .risk_analysis import get_risk_level
from .shap_explainer import batch_top_reason, explain_row


class ChurnPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        self.feature_columns = None
        self.load_model()

    def load_model(self) -> None:
        loaded = joblib.load(self.model_path)
        if isinstance(loaded, dict):
            self.model = loaded.get("model") or loaded.get("pipeline")
            self.thresholds = loaded.get("feature_thresholds") or self.thresholds
            self.feature_columns = loaded.get("feature_engineered_columns")
        else:
            self.model = loaded

        if self.model is None:
            raise ValueError("Model file does not contain a usable model object.")

    def _align_features(self, x_fe: pd.DataFrame) -> pd.DataFrame:
        expected = self.feature_columns
        if expected is None and hasattr(self.model, "feature_names_in_"):
            expected = list(self.model.feature_names_in_)
        if expected is None and hasattr(self.model, "named_steps"):
            final_model = self.model.named_steps.get("model")
            if final_model is not None and hasattr(final_model, "feature_names_in_"):
                expected = list(final_model.feature_names_in_)
        if expected:
            aligned = x_fe.copy()
            for column in expected:
                if column not in aligned.columns:
                    aligned[column] = 0
            return aligned[expected]
        return x_fe

    def _predict_proba(self, x_fe: pd.DataFrame) -> np.ndarray:
        x_fe = self._align_features(x_fe)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x_fe)[:, 1]
        scores = self.model.decision_function(x_fe)
        return 1 / (1 + np.exp(-scores))

    def predict_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        x_fe = create_features(df, thresholds=self.thresholds)
        probabilities = self._predict_proba(x_fe)
        predictions = (probabilities >= 0.5).astype(int)

        result = df.copy()
        result["churn_probability"] = probabilities
        result["prediction"] = np.where(predictions == 1, "CHURN", "NOT CHURN")
        result["risk_level"] = [get_risk_level(float(p)) for p in probabilities]

        reasons = []
        recommendations = []
        for idx in range(len(x_fe)):
            row = x_fe.iloc[idx]
            explanation = explain_row(row, top_n=3)
            reasons.append(batch_top_reason(row))
            recommendations.append(batch_recommendation(row.to_dict(), explanation["top_positive"]))

        result["top_reason"] = reasons
        result["recommendation"] = recommendations
        return result, x_fe

    def predict_single(self, payload: dict) -> dict:
        raw_df = pd.DataFrame([payload])
        result_df, x_fe = self.predict_dataframe(raw_df)
        row = result_df.iloc[0]
        fe_row = x_fe.iloc[0]
        explanation = explain_row(fe_row, top_n=5)
        recommendations = generate_recommendations(fe_row.to_dict(), explanation["top_positive"])
        probability = float(row["churn_probability"])

        return {
            "probability": probability,
            "probability_percent": round(probability * 100, 2),
            "prediction": row["prediction"],
            "risk_level": row["risk_level"],
            "top_reasons": explanation["top_positive"][:3],
            "protective_factors": explanation["top_negative"][:3],
            "chart_items": explanation["chart_items"],
            "recommendations": recommendations,
            "engineered_features": fe_row.to_dict(),
        }
