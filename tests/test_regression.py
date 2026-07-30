"""Model regression test — ensure new model versions don't degrade RMSE/R².

Loads the current model set, runs predictions on a known test set, and
compares metrics against a baseline stored in data/metadata/regression_baseline.json.
If no baseline exists, creates one.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import yaml

from src.feature_engineering import build_feature_matrix, create_target_columns
from src.split_time_series import split_by_time
from src.train_power_model import load_models
from src.predict import predict_with_model
from src.evaluate import compute_metrics


BASELINE_PATH = Path(__file__).resolve().parent.parent / "data" / "metadata" / "regression_baseline.json"
TOLERANCE_RMSE = 1.15  # allow 15% RMSE regression
TOLERANCE_R2 = 0.90    # allow 10% R² regression


@pytest.fixture(scope="module")
def test_data_and_models():
    base = Path(__file__).resolve().parent.parent
    with open(base / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)
    processed = pd.read_parquet(base / "data" / "processed" / "processed_data.parquet")
    feature_data = build_feature_matrix(processed, config)
    horizons = config.get("forecasting", {}).get("horizons", [])
    feature_data = create_target_columns(feature_data, horizons)
    split_cfg = config.get("training", {}).get("split", {})
    _, _, test_df = split_by_time(feature_data,
                                   train_ratio=split_cfg.get("train_ratio", 0.7),
                                   val_ratio=split_cfg.get("validation_ratio", 0.15),
                                   test_ratio=split_cfg.get("test_ratio", 0.15))
    models = load_models(str(base / "models"))
    return test_df, models


def _evaluate_model(test_df, models, target, model_key_prefix):
    """Return (rmse, r2) for the best available model of a target."""
    for suffix in ["lightgbm", "xgboost"]:
        key = f"{target}_{suffix}"
        if key in models:
            preds = predict_with_model(models[key], test_df)
            actual = test_df[target].values[:len(preds)]
            mask = ~(np.isnan(actual) | np.isnan(preds))
            if mask.sum() < 10:
                continue
            metrics = compute_metrics(actual[mask], preds[mask])
            return metrics.get("rmse", float("inf")), metrics.get("r2", float("-inf"))
    return None, None


def test_models_loaded(test_data_and_models):
    test_df, models = test_data_and_models
    assert len(models) > 0, "No models loaded"


def test_regression_vs_baseline(test_data_and_models):
    """Compare current model RMSE/R² against stored baseline."""
    test_df, models = test_data_and_models
    current = {}

    turbine_ids = [f"TB{i:02d}" for i in range(1, 13)]
    horizons = ["10min", "30min", "1hour", "6hour", "24hour"]
    for tid in turbine_ids[:3]:
        for h in horizons:
            target = f"{tid}_power_target_{h}"
            rmse, r2 = _evaluate_model(test_df, models, target, target)
            if rmse is not None:
                current[target] = {"rmse": round(rmse, 2), "r2": round(r2, 4)}

    if not BASELINE_PATH.exists():
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_PATH, "w") as f:
            json.dump(current, f, indent=2)
        pytest.skip("No baseline found — created one. Run again to check regression.")

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    failures = []
    for target, metrics in current.items():
        if target not in baseline:
            continue
        b_rmse, b_r2 = baseline[target]["rmse"], baseline[target]["r2"]
        c_rmse, c_r2 = metrics["rmse"], metrics["r2"]
        if c_rmse > b_rmse * TOLERANCE_RMSE:
            failures.append(f"{target}: RMSE {c_rmse} vs baseline {b_rmse} (>{TOLERANCE_RMSE}x)")
        if c_r2 < b_r2 * TOLERANCE_R2:
            failures.append(f"{target}: R² {c_r2} vs baseline {b_r2} (<{TOLERANCE_R2}x)")

    assert not failures, "Regression detected:\n" + "\n".join(failures)
