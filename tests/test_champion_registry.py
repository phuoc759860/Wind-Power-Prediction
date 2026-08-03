"""Champion consistency test (reviewer: the API must serve the champion).

Loads the real champion_registry.json and drives the real FastAPI
/predict/champion endpoint, asserting the served selected_model matches the
registry entry for EVERY (level x horizon) cell — not just a hardcoded one.

    registry = json.load(open("champion_registry.json"))
    api = get_prediction("farm", "10min")
    assert api["selected_model"] == registry["farm"]["10min"]["model_key"]

Also verifies the registry is derivable from evaluation_metrics.csv (each
champion is the lowest-mean-RMSE deployable ML model per level x horizon) and
that every champion model artifact exists on disk.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# API key comes from the environment (fail-closed server). Must be set before
# src.api is ever imported.
os.environ.setdefault("API_KEY", "amg-champion-test")

import pandas as pd
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "champion_registry.json"
METRICS_PATH = REPO_ROOT / "outputs" / "forecasts" / "evaluation_metrics.csv"

LEVELS = ["turbine", "farm"]
HORIZONS = ["10min", "30min", "1hour", "6hour", "24hour"]
ML_ALGS = ["lightgbm", "xgboost"]


def _load_registry() -> dict:
    assert REGISTRY_PATH.exists(), "champion_registry.json missing"
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _champion_payload(level: str, horizon: str, entry: dict) -> dict:
    payload = {
        "level": level,
        "horizon": horizon,
        "wind_speed": 8.5,
        "temperature": 24.0,
        "frequency": 50.0,
        "power": 1200,
        "model_type": entry["model_key"].rsplit("_", 1)[1],
    }
    if level == "turbine":
        payload["turbine_id"] = "TB01"
    return payload


@pytest.fixture(scope="module")
def registry() -> dict:
    return _load_registry()


@pytest.fixture(scope="module")
def client() -> TestClient:
    import src.api as api
    from src.api import app

    # Populate caches exactly as the lifespan startup would (a TestClient used
    # outside a context manager does not run lifespan events).
    api._scan_model_registry()
    api._load_champion_registry()
    assert api._champion_registry, "champion registry not loaded by the API"
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['API_KEY']}"}


def test_champion_registry_schema(registry):
    for level in LEVELS:
        assert level in registry, f"registry missing level {level!r}"
        for horizon in HORIZONS:
            entry = registry[level].get(horizon)
            assert entry, f"{level}/{horizon}: missing registry entry"
            for field in ["model_key", "model_path", "feature_version",
                          "model_version", "run_id", "training_cutoff", "quality_flag"]:
                assert field in entry, f"{level}/{horizon}: missing field {field!r}"
            assert entry["quality_flag"] == "PASS"
            assert entry["model_path"].endswith(".joblib")


def test_champion_model_files_exist(registry):
    for level in LEVELS:
        for horizon in HORIZONS:
            path = REPO_ROOT / registry[level][horizon]["model_path"]
            assert path.exists(), f"{level}/{horizon}: missing champion model {path}"
            assert path.stat().st_size > 0, f"{level}/{horizon}: champion model is empty"


def test_api_serves_registry_champion_for_every_cell(registry, client, headers):
    """Reviewer: for every registry cell, the API must serve the registered
    champion — assert selected_model == registry[level][horizon]["model_key"]."""
    for level in LEVELS:
        for horizon in HORIZONS:
            entry = registry[level][horizon]
            r = client.post("/predict/champion",
                            json=_champion_payload(level, horizon, entry),
                            headers=headers)
            assert r.status_code == 200, f"{level}/{horizon}: {r.text}"
            api_response = r.json()

            assert api_response["selected_model"] == entry["model_key"], (
                f"{level}/{horizon}: API served {api_response['selected_model']!r} "
                f"but registry says {entry['model_key']!r}")
            assert api_response["prediction"] > 0, f"{level}/{horizon}: non-positive forecast"
            assert api_response["quality_flag"] == entry["quality_flag"]
            assert api_response["feature_version"] == entry["feature_version"]
            assert api_response["model_version"] == entry["model_version"]
            assert api_response["run_id"] == entry["run_id"]
            assert api_response["training_cutoff"] == entry["training_cutoff"]


def test_reviewer_example_farm_10min(registry, client, headers):
    """The reviewer's exact example:
        registry = json.load(open("champion_registry.json"))
        api = get_prediction()
        assert api["selected_model"] == registry["farm"]["10min"]["model_key"]
    """
    entry = registry["farm"]["10min"]
    r = client.post("/predict/champion",
                    json=_champion_payload("farm", "10min", entry),
                    headers=headers)
    assert r.status_code == 200, r.text
    api_response = r.json()
    assert api_response["selected_model"] == registry["farm"]["10min"]["model_key"]
    assert api_response["prediction"] > 0


def test_registry_champion_is_lowest_rmse_ml_model(registry):
    """The registry must be derivable from evaluation_metrics.csv: each champion
    is the deployable ML model (lightgbm/xgboost) with lowest mean RMSE for its
    level x horizon."""
    assert METRICS_PATH.exists(), "evaluation_metrics.csv missing"
    metrics = pd.read_csv(METRICS_PATH)
    ml = metrics[metrics["model"].isin(ML_ALGS)].copy()
    ml["level"] = ml["target"].astype(str).map(
        lambda t: "farm" if str(t).lower().startswith("farm") else "turbine")
    assert not ml.empty, "no ML rows in evaluation_metrics.csv"

    agg = ml.groupby(["horizon", "level", "model"], observed=True)["rmse"].mean().reset_index()
    best = agg.loc[agg.groupby(["horizon", "level"], observed=True)["rmse"].idxmin()]

    for level in LEVELS:
        for horizon in HORIZONS:
            row = best[(best["level"] == level) & (best["horizon"] == horizon)].iloc[0]
            expected_key = (
                (f"farm_total_power_target_{horizon}" if level == "farm"
                 else f"TB01_power_target_{horizon}")
                + f"_{row['model']}"
            )
            assert registry[level][horizon]["model_key"] == expected_key, (
                f"{level}/{horizon}: registry champion {registry[level][horizon]['model_key']!r} "
                f"does not match best-ML model {expected_key!r}")


def test_champion_targets_have_metric_rows(registry):
    """Every registry model_key must map to an actual row in evaluation_metrics.csv."""
    assert METRICS_PATH.exists(), "evaluation_metrics.csv missing"
    metrics = pd.read_csv(METRICS_PATH)
    for level in LEVELS:
        for horizon in HORIZONS:
            key = registry[level][horizon]["model_key"]
            target = key.rsplit("_", 1)[0]
            sub = metrics[(metrics["target"] == target) & (metrics["horizon"] == horizon)]
            assert not sub.empty, f"{level}/{horizon}: no metric rows for target {target!r}"
