import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# API key must come from the environment (no default key exists in the codebase).
# Fail-closed: if absent, protected endpoints return 503.
os.environ.setdefault("API_KEY", "amg-wind-2024-test")

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.api import app

API_KEY = os.environ["API_KEY"]
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture(scope="module")
def client():
    from src.api import _scan_model_registry, _load_availability
    _scan_model_registry()
    _load_availability()
    return TestClient(app, raise_server_exceptions=False)


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "AMG Wind Farm" in r.text


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["models_in_registry"] >= 0
    assert data["turbines"] == 12


def test_turbines(client):
    r = client.get("/turbines")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 12
    for tb in data:
        assert "id" in tb
        assert "observed_availability_pct" in tb
        assert tb["observed_availability_pct"] > 0


def test_models(client):
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    keys = [k for k in data if k.startswith("TB")]
    assert len(keys) > 0


def test_predict_lightgbm(client):
    r = client.post("/predict", json={
        "turbine_id": "TB01",
        "wind_speed": 8.5,
        "temperature": 22.0,
        "frequency": 50.0,
        "power": 1500,
        "power_lag1": 1400,
        "power_lag6": 1300,
        "hour_of_day": 12,
        "month": 6,
        "model_type": "lightgbm",
    }, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["turbine_id"] == "TB01"
    assert len(data["predictions"]) == 5
    horizons = [p["horizon"] for p in data["predictions"]]
    assert "10min" in horizons
    assert "24hour" in horizons
    for p in data["predictions"]:
        assert 0 <= p["predicted_power_kw"] <= 2200
        assert p["confidence_lower_kw"] <= p["predicted_power_kw"]
        assert p["predicted_power_kw"] <= p["confidence_upper_kw"]


def test_predict_missing_auth(client):
    r = client.post("/predict", json={
        "turbine_id": "TB01",
        "wind_speed": 8.0,
        "temperature": 20.0,
        "frequency": 50.0,
        "power": 1000,
    })
    assert r.status_code == 401


def test_predict_invalid_key(client):
    r = client.post("/predict", json={
        "turbine_id": "TB01",
        "wind_speed": 8.0,
        "temperature": 20.0,
        "frequency": 50.0,
        "power": 1000,
    }, headers={"Authorization": "Bearer wrong-key"})
    assert r.status_code == 403


def test_predict_xgboost(client):
    r = client.post("/predict", json={
        "turbine_id": "TB05",
        "wind_speed": 10.0,
        "temperature": 18.0,
        "frequency": 50.02,
        "power": 1800,
        "model_type": "xgboost",
    }, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["turbine_id"] == "TB05"
    assert len(data["predictions"]) == 5


def test_predict_invalid_turbine(client):
    r = client.post("/predict", json={
        "turbine_id": "TB99",
        "wind_speed": 8.0,
        "temperature": 20.0,
        "frequency": 50.0,
        "power": 1000,
    }, headers=HEADERS)
    assert r.status_code == 400


def test_evaluations(client):
    r = client.get("/evaluations", headers=HEADERS)
    assert r.status_code in (200, 404)


def test_metrics(client):
    r = client.get("/outputs/metrics", headers=HEADERS)
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        data = r.json()
        assert len(data) > 0


def test_power_forecast(client):
    r = client.get("/outputs/power-forecast?limit=5", headers=HEADERS)
    assert r.status_code in (200, 404)


def test_farm_forecast(client):
    r = client.get("/outputs/farm-forecast?limit=5", headers=HEADERS)
    assert r.status_code in (200, 404)


def test_data_quality(client):
    r = client.get("/outputs/data-quality", headers=HEADERS)
    assert r.status_code in (200, 404)


def test_ramp_alerts(client):
    r = client.get("/outputs/ramp-alerts?limit=5", headers=HEADERS)
    assert r.status_code in (200, 404)


def test_alert_accuracy(client):
    r = client.get("/outputs/alert-accuracy?limit=5", headers=HEADERS)
    assert r.status_code in (200, 404)


def test_anomaly_accuracy(client):
    r = client.get("/outputs/anomaly-accuracy?limit=5", headers=HEADERS)
    assert r.status_code in (200, 404)


def test_failure_risk(client):
    r = client.get("/outputs/failure-risk?limit=5", headers=HEADERS)
    assert r.status_code in (200, 404)


def test_download(client):
    r = client.get("/download/metrics.csv", headers=HEADERS)
    assert r.status_code in (200, 404)


def test_download_blocked(client):
    r = client.get("/download/../../etc/passwd", headers=HEADERS)
    assert r.status_code in (400, 404)


# ============================================================
# INPUT MANAGEMENT ENDPOINT TESTS
# ============================================================


def test_input_list(client):
    r = client.get("/inputs", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_input_summary(client):
    r = client.get("/inputs/summary", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert "total_files" in data


def test_input_data(client):
    r = client.get("/inputs/data?nrows=5", headers=HEADERS)
    assert r.status_code in (200, 404)


def test_upload_and_delete_input(tmp_path, client):
    src = tmp_path / "temp_test.csv"
    pd.DataFrame({
        "PCTimeStamp": ["2025-01-01 00:00:00", "2025-01-01 00:10:00"],
        "wind_speed": [7.0, 8.0],
        "power": [1400, 1600],
    }).to_csv(src, index=False)

    with open(src, "rb") as f:
        r = client.post("/inputs/upload", files={"file": ("temp_test.csv", f, "text/csv")},
                        headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == "temp_test.csv"
    assert data["status"] == "uploaded"

    r = client.get("/inputs", headers=HEADERS)
    filenames = [f["filename"] for f in r.json()]
    assert "temp_test.csv" in filenames

    r = client.delete("/inputs/temp_test.csv", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["status"] == "removed"


def test_upload_unsupported_format(tmp_path, client):
    src = tmp_path / "test.txt"
    src.write_text("hello world")
    with open(src, "rb") as f:
        r = client.post("/inputs/upload", files={"file": ("test.txt", f, "text/plain")},
                        headers=HEADERS)
    assert r.status_code == 400
    assert "Unsupported" in r.text
