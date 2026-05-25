from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from utils.config import BASE_DIR, MODEL_PATH, ORIGINAL_FEATURES, OUTPUT_DIR, UPLOAD_DIR, ensure_directories
from utils.feature_engineering import validate_input_columns
from utils.prediction import ChurnPredictor
from utils.risk_analysis import risk_badge_class


ensure_directories()

app = FastAPI(
    title="Gym Customer Churn Prediction System",
    description="Production-style FastAPI app for AI churn prediction.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


try:
    predictor = ChurnPredictor(MODEL_PATH)
    MODEL_LOAD_ERROR = None
except Exception as exc:  # pragma: no cover - displayed in UI
    predictor = None
    MODEL_LOAD_ERROR = str(exc)


def _parse_form_value(value: str, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid numeric value for {field_name}.") from exc


def _ensure_model_ready() -> ChurnPredictor:
    if predictor is None:
        raise HTTPException(status_code=500, detail=f"Model could not be loaded: {MODEL_LOAD_ERROR}")
    return predictor


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "features": ORIGINAL_FEATURES,
            "model_error": MODEL_LOAD_ERROR,
        },
    )


@app.post("/predict")
async def predict(
    request: Request,
    gender: float = Form(...),
    Near_Location: float = Form(...),
    Partner: float = Form(...),
    Promo_friends: float = Form(...),
    Phone: float = Form(...),
    Contract_period: float = Form(...),
    Group_visits: float = Form(...),
    Age: float = Form(...),
    Avg_additional_charges_total: float = Form(...),
    Month_to_end_contract: float = Form(...),
    Lifetime: float = Form(...),
    Avg_class_frequency_total: float = Form(...),
    Avg_class_frequency_current_month: float = Form(...),
):
    model = _ensure_model_ready()
    payload = {
        "gender": gender,
        "Near_Location": Near_Location,
        "Partner": Partner,
        "Promo_friends": Promo_friends,
        "Phone": Phone,
        "Contract_period": Contract_period,
        "Group_visits": Group_visits,
        "Age": Age,
        "Avg_additional_charges_total": Avg_additional_charges_total,
        "Month_to_end_contract": Month_to_end_contract,
        "Lifetime": Lifetime,
        "Avg_class_frequency_total": Avg_class_frequency_total,
        "Avg_class_frequency_current_month": Avg_class_frequency_current_month,
    }
    result = model.predict_single(payload)
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "input_data": payload,
            "result": result,
            "risk_badge": risk_badge_class(result["risk_level"]),
        },
    )


@app.get("/batch")
async def batch_page(request: Request):
    return templates.TemplateResponse(
        "batch_upload.html",
        {
            "request": request,
            "required_columns": ORIGINAL_FEATURES,
            "model_error": MODEL_LOAD_ERROR,
        },
    )


@app.post("/batch_predict")
async def batch_predict(request: Request, file: UploadFile = File(...)):
    model = _ensure_model_ready()
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    upload_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    upload_path = UPLOAD_DIR / upload_name
    contents = await file.read()
    upload_path.write_bytes(contents)

    try:
        df = pd.read_csv(upload_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc

    missing = validate_input_columns(df)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    try:
        result_df, _ = model.predict_dataframe(df.copy())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    output_name = f"churn_predictions_{uuid.uuid4().hex}.csv"
    output_path = OUTPUT_DIR / output_name
    result_df.to_csv(output_path, index=False)

    summary = {
        "rows": len(result_df),
        "high_risk": int((result_df["risk_level"] == "HIGH").sum()),
        "medium_risk": int((result_df["risk_level"] == "MEDIUM").sum()),
        "low_risk": int((result_df["risk_level"] == "LOW").sum()),
        "avg_probability": round(float(result_df["churn_probability"].mean()) * 100, 2),
    }

    return templates.TemplateResponse(
        "batch_result.html",
        {
            "request": request,
            "summary": summary,
            "preview": result_df.head(20).to_dict(orient="records"),
            "filename": output_name,
        },
    )


@app.get("/download/{filename}")
async def download_file(filename: str):
    safe_name = Path(filename).name
    file_path = OUTPUT_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, media_type="text/csv", filename=safe_name)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": predictor is not None, "model_error": MODEL_LOAD_ERROR}
