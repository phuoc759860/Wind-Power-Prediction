import logging
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.input_manager import (
    list_input_files,
    add_input_file,
    remove_input_file,
    load_all_data_generic,
    view_input_data,
    edit_input_data,
    get_data_summary,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
METADATA_DIR = DATA_DIR / "metadata"

TURBINES = [f"TB{i:02d}" for i in range(1, 13)]
HORIZONS = ["10min", "30min", "1hour", "6hour", "24hour"]
MODELS = ["lightgbm", "xgboost"]
RATED_POWER = 2200

_loaded_models: Dict[str, dict] = {}
_availability: Dict[str, dict] = {}


def _load_all_models():
    global _loaded_models
    _loaded_models.clear()
    for fname in os.listdir(MODELS_DIR):
        if not fname.endswith("_model.joblib"):
            continue
        key = fname.replace("_model.joblib", "")
        if not any(key.startswith(t) for t in TURBINES + ["farm_total_power"]):
            continue
        model_path = MODELS_DIR / fname
        scaler_path = MODELS_DIR / f"{key}_scaler.joblib"
        features_path = MODELS_DIR / f"{key}_features.json"

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        features = json.load(open(features_path)) if features_path.exists() else []

        _loaded_models[key] = {
            "model": model,
            "scaler": scaler,
            "feature_cols": features,
        }
    logger.info(f"Loaded {len(_loaded_models)} models")


def _load_availability():
    global _availability
    path = METADATA_DIR / "availability_report.json"
    if path.exists():
        with open(path) as f:
            _availability = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all_models()
    _load_availability()
    yield


app = FastAPI(
    title="AMG Wind Power Forecasting API",
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": len(_loaded_models),
        "turbines": len(TURBINES),
    }


@app.get("/turbines")
def list_turbines():
    result = []
    for tb in TURBINES:
        info = _availability.get(f"{tb}_power", {})
        result.append({
            "id": tb,
            "rated_power_kw": RATED_POWER,
            "availability_pct": info.get("availability_pct", 0),
            "generating_hours": info.get("generating_hours", 0),
            "stopped_hours": info.get("stopped_hours", 0),
            "missing_hours": info.get("missing_hours", 0),
        })
    return result


@app.get("/models")
def list_models():
    grouped = {}
    for key in sorted(_loaded_models.keys()):
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            target, model_type = parts
        else:
            target, model_type = key, "unknown"
        turbine = "farm_total_power"
        for tb in TURBINES + ["farm_total_power"]:
            if target.startswith(tb):
                turbine = tb
                break
        if turbine not in grouped:
            grouped[turbine] = []
        grouped[turbine].append({
            "key": key,
            "model_type": model_type,
            "features": len(_loaded_models[key]["feature_cols"]),
        })
    return grouped


@app.get("/evaluations")
def get_evaluations():
    path = BASE_DIR / "outputs" / "forecasts" / "evaluation_metrics.csv"
    if not path.exists():
        raise HTTPException(404, "evaluation_metrics.csv not found")
    df = pd.read_csv(path)
    import math
    records = df.to_dict(orient="records")
    clean = []
    for r in records:
        clean.append({k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in r.items()})
    return clean


@app.get("/outputs/metrics")
def get_metrics():
    path = BASE_DIR / "outputs" / "forecasts" / "metrics.csv"
    if not path.exists():
        raise HTTPException(404, "metrics.csv not found")
    df = pd.read_csv(path)
    import math
    records = df.to_dict(orient="records")
    return [{k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in r.items()} for r in records]


@app.get("/outputs/power-forecast")
def get_power_forecast(limit: int = 100):
    path = BASE_DIR / "outputs" / "forecasts" / "power_forecast.csv"
    if not path.exists():
        raise HTTPException(404, "power_forecast.csv not found")
    df = pd.read_csv(path, nrows=limit)
    return df.to_dict(orient="records")


@app.get("/outputs/farm-forecast")
def get_farm_forecast(limit: int = 100):
    path = BASE_DIR / "outputs" / "forecasts" / "farm_forecast.csv"
    if not path.exists():
        raise HTTPException(404, "farm_forecast.csv not found")
    df = pd.read_csv(path, nrows=limit)
    return df.to_dict(orient="records")


@app.get("/outputs/ramp-alerts")
def get_ramp_alerts(limit: int = 100):
    path = BASE_DIR / "outputs" / "forecasts" / "ramp_alert.csv"
    if not path.exists():
        raise HTTPException(404, "ramp_alert.csv not found")
    df = pd.read_csv(path, nrows=limit)
    return df.to_dict(orient="records")


@app.get("/outputs/anomaly-alerts")
def get_anomaly_alerts(limit: int = 100):
    path = BASE_DIR / "outputs" / "forecasts" / "anomaly_alert.csv"
    if not path.exists():
        raise HTTPException(404, "anomaly_alert.csv not found")
    df = pd.read_csv(path, nrows=limit)
    return df.to_dict(orient="records")


@app.get("/outputs/failure-risk")
def get_failure_risk(limit: int = 100):
    path = BASE_DIR / "outputs" / "forecasts" / "failure_risk.csv"
    if not path.exists():
        raise HTTPException(404, "failure_risk.csv not found")
    df = pd.read_csv(path, nrows=limit)
    return df.to_dict(orient="records")


@app.get("/outputs/data-quality")
def get_data_quality():
    path = BASE_DIR / "outputs" / "forecasts" / "data_quality_report.csv"
    if not path.exists():
        raise HTTPException(404, "data_quality_report.csv not found")
    df = pd.read_csv(path).fillna("")
    import math
    records = df.to_dict(orient="records")
    return [{k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in r.items()} for r in records]


@app.get("/download/{filename}")
def download_file(filename: str):
    allowed = [
        "power_forecast.csv", "farm_forecast.csv", "metrics.csv",
        "evaluation_metrics.csv", "ramp_alert.csv", "anomaly_alert.csv",
        "failure_risk.csv", "data_quality_report.csv",
    ]
    if filename not in allowed:
        raise HTTPException(400, "File not allowed for download")
    path = BASE_DIR / "outputs" / "forecasts" / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} not found")
    from fastapi.responses import FileResponse
    return FileResponse(str(path), filename=filename, media_type="text/csv")


# ============================================================
# INPUT FILE MANAGEMENT ENDPOINTS
# ============================================================


@app.get("/inputs")
def list_inputs():
    return list_input_files()


@app.post("/inputs/upload")
async def upload_input(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(file.filename).suffix.lower()
    supported = {".xlsx", ".xls", ".csv", ".parquet", ".json"}
    if ext not in supported:
        raise HTTPException(400, f"Unsupported format '{ext}'. Supported: {', '.join(supported)}")

    raw_dir = BASE_DIR / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dest = raw_dir / file.filename
    if dest.exists():
        raise HTTPException(409, f"File '{file.filename}' already exists")

    content = await file.read()
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(413, "File exceeds 500 MB limit")

    with open(dest, "wb") as f:
        f.write(content)

    logger.info(f"Uploaded input file: {file.filename} ({len(content) / 1024 / 1024:.2f} MB)")
    return {
        "filename": file.filename,
        "size_mb": round(len(content) / 1024 / 1024, 3),
        "extension": ext,
        "status": "uploaded",
    }


@app.delete("/inputs/{filename}")
def delete_input(filename: str):
    return remove_input_file(filename)


@app.get("/inputs/data")
def get_input_data(
    filename: Optional[str] = Query(None),
    nrows: int = Query(100, ge=1, le=10000),
):
    df = view_input_data(filename=filename, nrows=nrows)
    return JSONResponse(content=json.loads(df.to_json(orient="records", date_format="iso")))


@app.get("/inputs/summary")
def input_summary():
    return get_data_summary()


class InputEditItem(BaseModel):
    condition_column: Optional[str] = Field(None, description="Column to filter on (default: PCTimeStamp)")
    condition_value: str = Field(..., description="Value to match for filtering")
    target_column: str = Field(..., description="Column to update")
    new_value: float = Field(..., description="New value to set")


class InputEditRequest(BaseModel):
    updates: List[InputEditItem]
    save_copy: bool = Field(False, description="Save edited data as a new file")
    output_filename: Optional[str] = Field(None, description="Output filename for edited data")


@app.put("/inputs/data")
def update_input_data(req: InputEditRequest):
    updates_dict = [u.model_dump() for u in req.updates]
    result = edit_input_data(
        updates=updates_dict,
        save_copy=req.save_copy,
        filename=req.output_filename,
    )
    if result["updates_failed"] > 0:
        logger.warning(f"Edit completed with {result['updates_failed']} failure(s): {result['errors']}")
    return result


class PredictInput(BaseModel):
    turbine_id: str = Field(..., examples=["TB01"])
    wind_speed: float = Field(..., ge=0, le=50, examples=[8.5])
    temperature: float = Field(..., ge=-30, le=60, examples=[22.0])
    frequency: float = Field(..., ge=45, le=55, examples=[50.0])
    power: float = Field(0, ge=0, le=RATED_POWER, examples=[1500],
                         description="Current power output (kW)")
    power_lag1: Optional[float] = Field(None, ge=-500, le=RATED_POWER)
    power_lag6: Optional[float] = Field(None, ge=-500, le=RATED_POWER)
    hour_of_day: Optional[int] = Field(None, ge=0, le=23)
    month: Optional[int] = Field(None, ge=1, le=12)
    model_type: Optional[str] = Field("lightgbm", examples=["lightgbm"])


class PredictionResult(BaseModel):
    turbine_id: str
    horizon: str
    model_type: str
    predicted_power_kw: float
    confidence_lower_kw: float
    confidence_upper_kw: float


class PredictResponse(BaseModel):
    turbine_id: str
    predictions: List[PredictionResult]


@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictInput):
    if not _loaded_models:
        raise HTTPException(503, "No models loaded")
    if data.turbine_id not in TURBINES:
        raise HTTPException(400, f"Invalid turbine: {data.turbine_id}")

    model_type = data.model_type or "lightgbm"
    predictions = []

    for horizon in HORIZONS:
        target = f"{data.turbine_id}_power_target_{horizon}"
        model_key = f"{target}_{model_type}"

        if model_key not in _loaded_models:
            predictions.append(PredictionResult(
                turbine_id=data.turbine_id, horizon=horizon, model_type=model_type,
                predicted_power_kw=0, confidence_lower_kw=0, confidence_upper_kw=0,
            ))
            continue

        info = _loaded_models[model_key]
        features = pd.DataFrame([{
            "TB01_wind_speed": data.wind_speed,
            "TB02_wind_speed": data.wind_speed,
            "TB03_wind_speed": data.wind_speed,
            "TB04_wind_speed": data.wind_speed,
            "TB05_wind_speed": data.wind_speed,
            "TB06_wind_speed": data.wind_speed,
            "TB07_wind_speed": data.wind_speed,
            "TB08_wind_speed": data.wind_speed,
            "TB09_wind_speed": data.wind_speed,
            "TB10_wind_speed": data.wind_speed,
            "TB11_wind_speed": data.wind_speed,
            "TB12_wind_speed": data.wind_speed,
            "TB01_power": data.power,
            "TB02_power": data.power,
            "TB03_power": data.power,
            "TB04_power": data.power,
            "TB05_power": data.power,
            "TB06_power": data.power,
            "TB07_power": data.power,
            "TB08_power": data.power,
            "TB09_power": data.power,
            "TB10_power": data.power,
            "TB11_power": data.power,
            "TB12_power": data.power,
            "farm_total_power": data.power * 12,
            "farm_avg_power": data.power,
            "farm_avg_wind_speed": data.wind_speed,
            "TB01_power_lag1": data.power_lag1 or 0,
            "TB01_power_lag6": data.power_lag6 or 0,
            "hour_of_day": data.hour_of_day or 12,
            "month": data.month or 6,
        }])

        valid_cols = [c for c in info["feature_cols"] if c in features.columns]
        X = features.reindex(columns=info["feature_cols"], fill_value=0)

        scaler = info.get("scaler")
        if scaler is not None:
            X = scaler.transform(X)

        pred = float(info["model"].predict(X)[0])
        pred = max(0, min(RATED_POWER, pred))

        sigma = pred * 0.08
        predictions.append(PredictionResult(
            turbine_id=data.turbine_id, horizon=horizon, model_type=model_type,
            predicted_power_kw=round(pred, 2),
            confidence_lower_kw=round(max(0, pred - 1.96 * sigma), 2),
            confidence_upper_kw=round(min(RATED_POWER, pred + 1.96 * sigma), 2),
        ))

    return PredictResponse(turbine_id=data.turbine_id, predictions=predictions)


class FarmPredictInput(BaseModel):
    wind_speed: float = Field(..., ge=0, le=50, examples=[8.5])
    temperature: float = Field(..., ge=-30, le=60, examples=[22.0])
    frequency: float = Field(..., ge=45, le=55, examples=[50.0])
    power: float = Field(0, ge=0, le=RATED_POWER * 12, examples=[20000])
    hour_of_day: Optional[int] = None
    month: Optional[int] = None
    model_type: Optional[str] = "lightgbm"


class FarmPredictResponse(BaseModel):
    predictions: List[PredictionResult]


@app.post("/predict/farm", response_model=FarmPredictResponse)
def predict_farm(data: FarmPredictInput):
    if not _loaded_models:
        raise HTTPException(503, "No models loaded")

    model_type = data.model_type or "lightgbm"
    predictions = []

    for horizon in HORIZONS:
        target = f"farm_total_power_target_{horizon}"
        model_key = f"{target}_{model_type}"

        if model_key not in _loaded_models:
            continue

        info = _loaded_models[model_key]
        features = pd.DataFrame([{
            "farm_total_power": data.power,
            "farm_avg_power": data.power / 12,
            "farm_avg_wind_speed": data.wind_speed,
            "hour_of_day": data.hour_of_day or 12,
            "month": data.month or 6,
        }])

        X = features.reindex(columns=info["feature_cols"], fill_value=0)
        scaler = info.get("scaler")
        if scaler is not None:
            X = scaler.transform(X)

        pred = float(info["model"].predict(X)[0])
        pred = max(0, min(RATED_POWER * 12, pred))

        sigma = pred * 0.08
        predictions.append(PredictionResult(
            turbine_id="FARM", horizon=horizon, model_type=model_type,
            predicted_power_kw=round(pred, 2),
            confidence_lower_kw=round(max(0, pred - 1.96 * sigma), 2),
            confidence_upper_kw=round(min(RATED_POWER * 12, pred + 1.96 * sigma), 2),
        ))

    return FarmPredictResponse(predictions=predictions)
