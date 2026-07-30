import logging
import os
import json
import time
import hashlib
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
LOGS_DIR = BASE_DIR / "logs"
os.makedirs(str(LOGS_DIR), exist_ok=True)

TURBINES = [f"TB{i:02d}" for i in range(1, 13)]
HORIZONS = ["10min", "30min", "1hour", "6hour", "24hour"]
HORIZON_MINUTES = {"10min": 10, "30min": 30, "1hour": 60, "6hour": 360, "24hour": 1440}
MODEL_TYPES = ["lightgbm", "xgboost"]
RATED_POWER = 2200
MODEL_VERSION = "2.0.0"

# ── Auth ──────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    key_file = BASE_DIR / "configs" / "api_key.txt"
    if key_file.exists():
        API_KEY = key_file.read_text(encoding="utf-8").strip()
if not API_KEY:
    API_KEY = "amg-wind-2024-dev"
    logger.warning("No API_KEY env or configs/api_key.txt found; using default dev key")

security_scheme = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)):
    if credentials is None:
        raise HTTPException(401, "Missing Authorization header (Bearer <api-key>)")
    if credentials.credentials != API_KEY:
        raise HTTPException(403, "Invalid API key")
    return credentials.credentials

# ── Rate limiter (in-memory sliding window) ───────────────────────────────────
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "60"))

_request_log: Dict[str, List[float]] = defaultdict(list)

def _rate_limit(client_ip: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SEC
    requests = _request_log[client_ip]
    requests[:] = [t for t in requests if t > window_start]
    if len(requests) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(429, f"Rate limit exceeded ({RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SEC}s)")
    requests.append(now)

# ── Audit log ─────────────────────────────────────────────────────────────────
_audit_logger = logging.getLogger("api_audit")
_handler = logging.FileHandler(str(LOGS_DIR / "api_audit.log"), encoding="utf-8")
_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
_audit_logger.addHandler(_handler)
_audit_logger.setLevel(logging.INFO)
_audit_logger.propagate = False

async def audit_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    client = request.client.host if request.client else "unknown"
    _audit_logger.info(
        f"{client} {request.method} {request.url.path} "
        f"{response.status_code} {elapsed*1000:.0f}ms"
    )
    response.headers["X-Request-Time-Ms"] = str(round(elapsed * 1000))
    return response

# ── State ─────────────────────────────────────────────────────────────────────
_model_registry: Dict[str, dict] = {}   # key -> metadata (always loaded)
_model_cache: Dict[str, dict] = {}      # key -> {model, scaler, feature_cols} (lazy)
_availability: Dict[str, dict] = {}
_residual_quantiles: Dict[str, dict] = {}
_ram_benchmark: dict = {}

def _scan_model_registry():
    global _model_registry
    _model_registry.clear()
    for fname in os.listdir(MODELS_DIR):
        if not fname.endswith("_model.joblib"):
            continue
        key = fname.replace("_model.joblib", "")
        if not any(key.startswith(t) for t in TURBINES + ["farm_total_power"]):
            continue

        features_path = MODELS_DIR / f"{key}_features.json"
        metadata_path = MODELS_DIR / f"{key}_metadata.json"

        feature_cols = []
        if features_path.exists():
            with open(features_path) as f:
                feature_cols = json.load(f)

        meta = {"key": key, "n_features": len(feature_cols), "feature_cols": feature_cols}
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta.update(json.load(f))

        _model_registry[key] = meta
    logger.info(f"Scanned {len(_model_registry)} models in registry (lazy-load)")


def _get_model(key: str) -> dict:
    if key in _model_cache:
        return _model_cache[key]

    if key not in _model_registry:
        raise HTTPException(404, f"Model '{key}' not found in registry")

    t0 = time.time()
    model_path = MODELS_DIR / f"{key}_model.joblib"
    scaler_path = MODELS_DIR / f"{key}_scaler.joblib"

    model = joblib.load(str(model_path))
    scaler = joblib.load(str(scaler_path)) if scaler_path.exists() else None

    load_time = time.time() - t0
    _ram_benchmark[key] = {"load_time_ms": round(load_time * 1000, 1)}
    logger.info(f"Lazy-loaded model '{key}' in {load_time*1000:.0f}ms")

    info = {
        "model": model,
        "scaler": scaler,
        "feature_cols": _model_registry[key].get("feature_cols", []),
    }
    _model_cache[key] = info
    return info


def _load_availability():
    global _availability
    path = METADATA_DIR / "availability_report.json"
    if path.exists():
        with open(path) as f:
            _availability = json.load(f)


def _load_residual_quantiles():
    global _residual_quantiles
    path = METADATA_DIR / "residual_quantiles.json"
    if path.exists():
        with open(path) as f:
            _residual_quantiles = json.load(f)
        logger.info(f"Loaded residual quantiles for {len(_residual_quantiles)} groups")


def _get_model_version(key: str) -> str:
    meta = _model_registry.get(key, {})
    return meta.get("git_commit", MODEL_VERSION)[:8] if meta.get("git_commit") else MODEL_VERSION


def _compute_ci(pred: float, turbine_id: str, horizon: str, model_type: str) -> tuple:
    qkey = f"{turbine_id}_power_{horizon}_{model_type}"
    qdata = _residual_quantiles.get(qkey)
    if qdata and "q_95" in qdata:
        q = qdata["q_95"]
        lo = max(0, pred - q)
        hi = min(RATED_POWER, pred + q)
    else:
        sigma = pred * 0.08
        lo = max(0, pred - 1.96 * sigma)
        hi = min(RATED_POWER, pred + 1.96 * sigma)
    return round(lo, 2), round(hi, 2)


# ── App ───────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    _scan_model_registry()
    _load_availability()
    _load_residual_quantiles()
    elapsed = time.time() - t0
    RAM_GB = None
    try:
        import psutil
        RAM_GB = psutil.Process().memory_info().rss / 1e9
    except ImportError:
        pass
    logger.info(f"Startup completed in {elapsed*1000:.0f}ms"
                + (f" | RSS: {RAM_GB:.2f}GB" if RAM_GB else ""))
    yield


app = FastAPI(
    title="AMG Wind Power Forecasting API",
    version=MODEL_VERSION,
    lifespan=lifespan,
)
app.middleware("http")(audit_middleware)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ── Pydantic models ───────────────────────────────────────────────────────────
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
    horizon_min: int
    model_type: str
    model_version: str
    timestamp_issue: str
    timestamp_target: str
    predicted_power_kw: float
    confidence_lower_kw: float
    confidence_upper_kw: float


class PredictResponse(BaseModel):
    turbine_id: str
    predictions: List[PredictionResult]


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


def _build_features(data: PredictInput) -> pd.DataFrame:
    return pd.DataFrame([{
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


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_in_registry": len(_model_registry),
        "models_loaded_in_ram": len(_model_cache),
        "turbines": len(TURBINES),
        "rate_limit": f"{RATE_LIMIT_REQUESTS}/{RATE_LIMIT_WINDOW_SEC}s",
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
    for key in sorted(_model_registry.keys()):
        parts = key.rsplit("_", 1)
        target, model_type = (parts[0], parts[1]) if len(parts) == 2 else (key, "unknown")
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
            "features": _model_registry[key]["n_features"],
            "loaded": key in _model_cache,
        })
    return grouped


# ── Protected endpoints ───────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse,
          dependencies=[Depends(verify_api_key)])
def predict(data: PredictInput, request: Request):
    _rate_limit(request.client.host if request.client else "unknown")
    if not _model_registry:
        raise HTTPException(503, "No models registered")
    if data.turbine_id not in TURBINES:
        raise HTTPException(400, f"Invalid turbine: {data.turbine_id}")

    model_type = data.model_type or "lightgbm"
    predictions = []
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for horizon in HORIZONS:
        target = f"{data.turbine_id}_power_target_{horizon}"
        model_key = f"{target}_{model_type}"
        horizon_min = HORIZON_MINUTES[horizon]
        target_dt = datetime.now(timezone.utc) + timedelta(minutes=horizon_min)
        timestamp_target = target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        if model_key not in _model_registry:
            predictions.append(PredictionResult(
                turbine_id=data.turbine_id, horizon=horizon, horizon_min=horizon_min,
                model_type=model_type, model_version=MODEL_VERSION,
                timestamp_issue=now_utc, timestamp_target=timestamp_target,
                predicted_power_kw=0, confidence_lower_kw=0, confidence_upper_kw=0,
            ))
            continue

        info = _get_model(model_key)
        features = _build_features(data)
        X = features.reindex(columns=info["feature_cols"], fill_value=0)

        scaler = info.get("scaler")
        if scaler is not None:
            X = scaler.transform(X)

        pred = float(info["model"].predict(X)[0])
        pred = max(0, min(RATED_POWER, pred))
        lo, hi = _compute_ci(pred, data.turbine_id, horizon, model_type)

        predictions.append(PredictionResult(
            turbine_id=data.turbine_id, horizon=horizon, horizon_min=horizon_min,
            model_type=model_type, model_version=_get_model_version(model_key),
            timestamp_issue=now_utc, timestamp_target=timestamp_target,
            predicted_power_kw=round(pred, 2),
            confidence_lower_kw=lo, confidence_upper_kw=hi,
        ))

    return PredictResponse(turbine_id=data.turbine_id, predictions=predictions)


@app.post("/predict/farm", response_model=FarmPredictResponse,
          dependencies=[Depends(verify_api_key)])
def predict_farm(data: FarmPredictInput, request: Request):
    _rate_limit(request.client.host if request.client else "unknown")
    if not _model_registry:
        raise HTTPException(503, "No models registered")

    model_type = data.model_type or "lightgbm"
    predictions = []
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for horizon in HORIZONS:
        target = f"farm_total_power_target_{horizon}"
        model_key = f"{target}_{model_type}"
        horizon_min = HORIZON_MINUTES[horizon]
        target_dt = datetime.now(timezone.utc) + timedelta(minutes=horizon_min)
        timestamp_target = target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        if model_key not in _model_registry:
            continue

        info = _get_model(model_key)
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
        lo, hi = _compute_ci(pred, "farm_total_power", horizon, model_type)

        predictions.append(PredictionResult(
            turbine_id="FARM", horizon=horizon, horizon_min=horizon_min,
            model_type=model_type, model_version=_get_model_version(model_key),
            timestamp_issue=now_utc, timestamp_target=timestamp_target,
            predicted_power_kw=round(pred, 2),
            confidence_lower_kw=lo, confidence_upper_kw=hi,
        ))

    return FarmPredictResponse(predictions=predictions)


# ── Output file endpoints ─────────────────────────────────────────────────────
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


@app.get("/outputs/temperature-warnings")
def get_temperature_warnings(limit: int = 100):
    path = BASE_DIR / "outputs" / "forecasts" / "temperature_warning.csv"
    if not path.exists():
        raise HTTPException(404, "temperature_warning.csv not found")
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
        "failure_risk.csv", "data_quality_report.csv", "temperature_warning.csv",
    ]
    if filename not in allowed:
        raise HTTPException(400, "File not allowed for download")
    path = BASE_DIR / "outputs" / "forecasts" / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} not found")
    return FileResponse(str(path), filename=filename, media_type="text/csv")


# ── Input file management ─────────────────────────────────────────────────────
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
