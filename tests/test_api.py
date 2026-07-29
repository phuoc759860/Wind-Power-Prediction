import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from src.api import app, _load_all_models, _load_availability, _loaded_models, _availability

_load_all_models()
_load_availability()

client = TestClient(app, raise_server_exceptions=False)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "AMG Wind Farm" in r.text


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] > 0
    assert data["turbines"] == 12


def test_turbines():
    r = client.get("/turbines")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 12
    for tb in data:
        assert "id" in tb
        assert "availability_pct" in tb
        assert 0 < tb["availability_pct"] < 100


def test_models():
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert "TB01" in data
    assert len(data["TB01"]) >= 2


def test_predict_lightgbm():
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
    })
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


def test_predict_xgboost():
    r = client.post("/predict", json={
        "turbine_id": "TB05",
        "wind_speed": 10.0,
        "temperature": 18.0,
        "frequency": 50.02,
        "power": 1800,
        "model_type": "xgboost",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["turbine_id"] == "TB05"
    assert len(data["predictions"]) == 5


def test_predict_invalid_turbine():
    r = client.post("/predict", json={
        "turbine_id": "TB99",
        "wind_speed": 8.0,
        "temperature": 20.0,
        "frequency": 50.0,
        "power": 1000,
    })
    assert r.status_code == 400


def test_evaluations():
    r = client.get("/evaluations")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 130


def test_metrics():
    r = client.get("/outputs/metrics")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 130
    assert "MAE" in data[0]
    assert "RMSE" in data[0]
    assert "skill_score" in data[0]


def test_power_forecast():
    r = client.get("/outputs/power-forecast?limit=5")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 5
    assert "timestamp_issue" in data[0]
    assert "y_pred" in data[0]
    assert "y_low" in data[0]
    assert "y_high" in data[0]


def test_farm_forecast():
    r = client.get("/outputs/farm-forecast?limit=5")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 5
    assert "farm_power_pred" in data[0]
    assert "farm_energy_pred" in data[0]


def test_data_quality():
    r = client.get("/outputs/data-quality")
    assert r.status_code == 200
    data = r.json()
    assert len(data) > 0
    assert "column" in data[0]
    assert "missing_rate" in data[0]


def test_ramp_alerts():
    r = client.get("/outputs/ramp-alerts?limit=5")
    assert r.status_code == 200
    data = r.json()
    assert len(data) <= 5
    assert "ramp_type" in data[0]
    assert "expected_change" in data[0]


def test_failure_risk():
    r = client.get("/outputs/failure-risk?limit=5")
    assert r.status_code == 200
    data = r.json()
    assert len(data) <= 5
    assert "failure_probability" in data[0]


def test_download():
    r = client.get("/download/metrics.csv")
    assert r.status_code == 200
    assert "text/csv" in r.headers.get("content-type", "")


def test_download_blocked():
    r = client.get("/download/../../etc/passwd")
    assert r.status_code in (400, 404)


# ============================================================
# INPUT MANAGEMENT ENDPOINT TESTS
# ============================================================


def test_input_list():
    r = client.get("/inputs")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_input_summary():
    r = client.get("/inputs/summary")
    assert r.status_code == 200
    data = r.json()
    assert "total_files" in data
    assert data["total_files"] >= 0


def test_input_data():
    r = client.get("/inputs/data?nrows=5")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_upload_and_delete_input(tmp_path):
    src = tmp_path / "temp_test.csv"
    pd.DataFrame({
        "PCTimeStamp": ["2025-01-01 00:00:00", "2025-01-01 00:10:00"],
        "wind_speed": [7.0, 8.0],
        "power": [1400, 1600],
    }).to_csv(src, index=False)

    with open(src, "rb") as f:
        r = client.post("/inputs/upload", files={"file": ("temp_test.csv", f, "text/csv")})
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == "temp_test.csv"
    assert data["status"] == "uploaded"

    r = client.get("/inputs")
    filenames = [f["filename"] for f in r.json()]
    assert "temp_test.csv" in filenames

    r = client.delete("/inputs/temp_test.csv")
    assert r.status_code == 200
    assert r.json()["status"] == "removed"


def test_edit_input_via_api():
    r = client.put("/inputs/data", json={
        "updates": [{
            "condition_column": "PCTimeStamp",
            "condition_value": "2025-01-01 00:00:00",
            "target_column": "power",
            "new_value": 999,
        }],
        "save_copy": False,
    })
    assert r.status_code == 200
    data = r.json()
    assert "updates_applied" in data


def test_upload_unsupported_format(tmp_path):
    src = tmp_path / "test.txt"
    src.write_text("hello world")
    with open(src, "rb") as f:
        r = client.post("/inputs/upload", files={"file": ("test.txt", f, "text/plain")})
    assert r.status_code == 400
    assert "Unsupported" in r.text
