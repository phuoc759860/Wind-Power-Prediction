"""True end-to-end integration test (reviewer: not just unit tests).

One test that drives the entire production chain on a small synthetic
dataset in a temp workspace:

    Raw CSV -> Cleaning (mapping/validation/preprocessing) -> Feature
    Engineering -> Train -> Prediction -> Evaluation -> Champion Registry
    -> FastAPI -> JSON Response

Everything runs through the real pipeline functions (src/load_data,
src/column_mapping, src/data_validation, src/preprocessing,
src/feature_engineering, src/split_time_series, src/train_power_model,
src/predict, src/evaluate, generate_outputs.build_champion_registry) and the
real FastAPI app (src/api), not through fakes or mocks. The API's module-level
MODELS_DIR / BASE_DIR are pointed at the temp workspace for the duration of
the test and restored afterwards.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# API key must come from the environment (fail-closed server). Set it before
# src.api is ever imported.
os.environ.setdefault("API_KEY", "amg-wind-e2e-test")

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


def _make_synthetic_raw_df(n_turbines: int = 3, days: int = 4, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic SCADA feed (complete, no gaps) in the raw schema
    the column-mapping step expects (PCTimeStamp + TBxx_... Avg. columns)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-05-01 00:00:00", periods=days * 24 * 6, freq="10min")
    rows = {"PCTimeStamp": ts}
    hour = ts.hour + ts.minute / 60.0
    for i in range(1, n_turbines + 1):
        tb = f"TB{i:02d}"
        ws = np.clip(4 + 6 * np.sin(2 * np.pi * (hour + i) / 24) + rng.normal(0, 0.6, len(ts)), 0, 25)
        p = np.zeros(len(ts))
        ramp = (ws >= 3) & (ws < 12)
        p[ramp] = 1500 * (ws[ramp] - 3) / 9.0
        p[ws >= 12] = 2200
        p = np.clip(p + rng.normal(0, 30, len(ts)), 0, 2200)
        temp = 24 + 4 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 0.5, len(ts))
        freq = 50 + rng.normal(0, 0.02, len(ts))
        rows[f"{tb}_Ambient WindSpeed Avg."] = np.round(ws, 3)
        rows[f"{tb}_Ambient Temp. Avg."] = np.round(temp, 2)
        rows[f"{tb}_Grid Production Power Avg."] = np.round(p, 1)
        rows[f"{tb}_Grid Production Frequency Avg."] = np.round(freq, 3)
    return pd.DataFrame(rows)


CONFIG_YAML = """
data:
  sampling_interval_minutes: 10
  timestamp_column: "PCTimeStamp"
  timezone: "UTC"
turbines:
  ids: ["TB01", "TB02", "TB03"]
  count: 3
  rated_power_kw: 2200
  cut_in_speed: 3.0
  cut_out_speed: 25.0
  rated_speed: 12.0
forecasting:
  horizons:
    - name: "10min"
      steps: 1
    - name: "1hour"
      steps: 6
features:
  version: "test-v1"
  lag_steps: [1, 2]
  rolling_windows: [3]
  rolling_stats: ["mean"]
  temporal:
    - hour_of_day
    - day_of_week
    - month
    - season
    - is_weekend
  power_change:
    - diff_1
    - diff_3
  ramp:
    threshold_pct_per_min: 0.5
training:
  split:
    train_ratio: 0.6
    validation_ratio: 0.2
    test_ratio: 0.2
  random_state: 42
  cv_folds: 3
  models:
    baseline: ["persistence"]
    ml: ["lightgbm"]
  tuning:
    enabled: false
models:
  lightgbm:
    n_estimators: 20
    max_depth: 3
    learning_rate: 0.1
    subsample: 0.8
    feature_fraction: 0.5
    objective: "regression"
    random_state: 42
    verbose: -1
"""


def test_end_to_end(tmp_path):
    import yaml

    # ---------------------------------------------------------------
    # 0. Workspace: temp raw dir + config, exactly like the real project
    # ---------------------------------------------------------------
    raw_dir = tmp_path / "data" / "raw"
    models_dir = tmp_path / "models"
    out_forecasts = tmp_path / "outputs" / "forecasts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_forecasts.mkdir(parents=True, exist_ok=True)

    _make_synthetic_raw_df().to_csv(raw_dir / "turbine_data.csv", index=False)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(CONFIG_YAML, encoding="utf-8")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    horizons = config["forecasting"]["horizons"]
    turbine_ids = config["turbines"]["ids"]

    # ---------------------------------------------------------------
    # 1. Raw CSV -> Cleaning (load + column mapping + validation)
    # ---------------------------------------------------------------
    from src.load_data import load_all_data
    from src.column_mapping import apply_column_mapping
    from src.data_validation import run_validation

    raw_data = load_all_data(str(raw_dir))
    assert len(raw_data) == len(_make_synthetic_raw_df())  # no rows lost on load

    mapped_data, mapping = apply_column_mapping(raw_data)
    assert "timestamp" in mapped_data.columns
    assert f"{turbine_ids[0]}_power" in mapped_data.columns
    assert f"{turbine_ids[0]}_wind_speed" in mapped_data.columns

    validation_results = run_validation(mapped_data)
    assert isinstance(validation_results, dict)

    # ---------------------------------------------------------------
    # 2. Cleaning -> preprocess_pipeline (flags, farm aggregates)
    # ---------------------------------------------------------------
    from src.preprocessing import preprocess_pipeline

    processed = preprocess_pipeline(mapped_data, config)
    for col in ["is_observed", "is_synthetic", "is_imputed", "is_simulated",
                "farm_total_power", "farm_avg_power", "farm_avg_wind_speed"]:
        assert col in processed.columns, f"preprocessing lost column {col}"

    # ---------------------------------------------------------------
    # 3. Feature Engineering (features + horizon targets)
    # ---------------------------------------------------------------
    from src.feature_engineering import build_feature_matrix, create_target_columns

    feature_data = build_feature_matrix(processed, config)
    feature_data = create_target_columns(feature_data, horizons)
    for tb in turbine_ids:
        for h in horizons:
            target = f"{tb}_power_target_{h['name']}"
            assert target in feature_data.columns
    assert "farm_total_power_target_10min" in feature_data.columns

    # ---------------------------------------------------------------
    # 4. Time-based split (train / val / test)
    # ---------------------------------------------------------------
    from src.split_time_series import split_by_time

    train_df, val_df, test_df = split_by_time(
        feature_data,
        train_ratio=config["training"]["split"]["train_ratio"],
        val_ratio=config["training"]["split"]["validation_ratio"],
        test_ratio=config["training"]["split"]["test_ratio"],
    )
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0

    # ---------------------------------------------------------------
    # 5. Train every target model through the real trainer
    # ---------------------------------------------------------------
    from src.train_power_model import train_power_models, save_models

    all_models = {}
    base_targets = [f"{tb}_power" for tb in turbine_ids] + ["farm_total_power"]
    for base_target in base_targets:
        for h in horizons:
            target = f"{base_target}_target_{h['name']}"
            results, models = train_power_models(train_df, val_df, target, config)
            assert models, f"no model trained for {target}"
            all_models.update(models)
    assert len(all_models) == len(base_targets) * len(horizons)
    assert "TB01_power_target_10min_lightgbm" in all_models
    assert "farm_total_power_target_10min_lightgbm" in all_models

    save_models(all_models, str(models_dir), config=config, seed=42, data_path=str(raw_dir))
    assert (models_dir / "TB01_power_target_10min_lightgbm_model.joblib").exists()
    assert (models_dir / "TB01_power_target_10min_lightgbm_features.json").exists()
    assert (models_dir / "TB01_power_target_10min_lightgbm_metadata.json").exists()

    # ---------------------------------------------------------------
    # 6. Prediction + evaluation on the test window (uses official mask)
    # ---------------------------------------------------------------
    from src.predict import predict_power
    from src.evaluate import evaluate_all_models

    predictions = predict_power(test_df, all_models, config)
    assert len(predictions) == len(all_models)
    assert predictions["TB01_power_target_10min_lightgbm"]["predictions"] is not None

    eval_df = evaluate_all_models(test_df, predictions, config)
    assert not eval_df.empty
    assert {"target", "model", "horizon", "rmse"}.issubset(eval_df.columns)
    assert "TB01_power_target_10min" in set(eval_df["target"])

    # ---------------------------------------------------------------
    # 7. Champion registry auto-generated from the evaluation metrics
    # ---------------------------------------------------------------
    import generate_outputs as go

    eval_df.to_csv(out_forecasts / "evaluation_metrics.csv", index=False)
    real_go_base, real_go_out = go.BASE, go.OUT
    go.BASE, go.OUT = tmp_path, out_forecasts
    try:
        registry = go.build_champion_registry(output_path=tmp_path / "champion_registry.json")
    finally:
        go.BASE, go.OUT = real_go_base, real_go_out
    assert registry.get("turbine", {}).get("10min", {}).get("model_key") == \
        "TB01_power_target_10min_lightgbm"
    assert registry.get("farm", {}).get("10min", {}).get("model_key") == \
        "farm_total_power_target_10min_lightgbm"

    # ---------------------------------------------------------------
    # 8. FastAPI -> JSON Response (real app, temp workspace)
    # ---------------------------------------------------------------
    import src.api as api
    from src.api import app

    real_models_dir, real_base_dir = api.MODELS_DIR, api.BASE_DIR
    try:
        api.MODELS_DIR = models_dir
        api.BASE_DIR = tmp_path
        api._scan_model_registry()
        api._load_champion_registry()
        assert "TB01_power_target_10min_lightgbm" in api._model_registry
        assert api._champion_registry.get("turbine", {}).get("10min")

        client = TestClient(app, raise_server_exceptions=False)
        headers = {"Authorization": f"Bearer {os.environ['API_KEY']}"}

        # 8a. /predict: trained turbine model -> real JSON forecast
        r = client.post("/predict", json={
            "turbine_id": "TB01",
            "wind_speed": 8.5,
            "temperature": 24.0,
            "frequency": 50.0,
            "power": 1200,
            "power_lag1": 1100,
            "power_lag6": 900,
            "hour_of_day": 12,
            "month": 5,
            "model_type": "lightgbm",
        }, headers=headers)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["turbine_id"] == "TB01"
        assert len(data["predictions"]) == 5  # API contract: all 5 horizons
        preds = {p["horizon"]: p for p in data["predictions"]}
        assert preds["10min"]["predicted_power_kw"] > 0  # real model, not placeholder
        assert preds["1hour"]["predicted_power_kw"] > 0
        for p in data["predictions"]:
            assert 0 <= p["predicted_power_kw"] <= 2200
            assert p["confidence_lower_kw"] <= p["predicted_power_kw"] <= p["confidence_upper_kw"]
            assert p["model_version"]
            assert p["model_type"] == "lightgbm"

        # 8b. /predict/champion: registry-driven champion -> JSON
        r = client.post("/predict/champion", json={
            "level": "turbine",
            "horizon": "10min",
            "turbine_id": "TB01",
            "wind_speed": 8.5,
            "temperature": 24.0,
            "frequency": 50.0,
            "power": 1200,
            "model_type": "lightgbm",
        }, headers=headers)
        assert r.status_code == 200, r.text
        champ = r.json()
        assert champ["selected_model"] == "TB01_power_target_10min_lightgbm"
        assert champ["prediction"] > 0
        assert champ["quality_flag"] == "PASS"
        assert champ["feature_version"] == "test-v1"
        assert champ["run_id"] == "unknown" or champ["run_id"]
        assert champ["training_cutoff"]
    finally:
        # Restore the real workspace + registry state so other test modules
        # (e.g. test_api.py) still see the production models.
        api.MODELS_DIR, api.BASE_DIR = real_models_dir, real_base_dir
        api._scan_model_registry()
        api._load_champion_registry()
