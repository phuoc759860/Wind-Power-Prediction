from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.feature_engineering import build_feature_matrix, create_target_columns
from src.split_time_series import split_by_time
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.train_baseline import (
    select_feature_columns,
    persistence_predictions,
)
from src.train_power_model import (
    prepare_features,
    train_xgboost,
    train_lightgbm,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
TARGET = "farm_total_power_target_10min"

logger = logging.getLogger("fair_experiment")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


def _named_rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _named_mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def _r2(actual: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    return 0.0 if ss_tot == 0 else float(1 - (ss_res / ss_tot))


def _save_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["experiment", "ablation", "model"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    df.to_csv(path, index=False)


def _build_split_matrix(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    shared_cols = [c for c in feature_cols if c in df.columns]
    X = df[shared_cols].reindex(columns=shared_cols, fill_value=0).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].to_numpy(dtype=float)


def _evaluate_model(model_name: str, feature_cols: list[str], train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    X_train, y_train = _build_split_matrix(train_df, feature_cols, TARGET)
    X_val, y_val = _build_split_matrix(val_df, feature_cols, TARGET)
    X_test, y_test = _build_split_matrix(test_df, feature_cols, TARGET)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(f"Insufficient data for {TARGET}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    if model_name == "persistence":
        pred = persistence_predictions(test_df, TARGET, 1)
        actual = test_df[TARGET].to_numpy(dtype=float)
        valid = ~(np.isnan(actual) | np.isnan(pred))
        actual = actual[valid]
        pred = pred[valid]
    elif model_name == "ridge":
        ridge_scaler = StandardScaler()
        X_train_r = ridge_scaler.fit_transform(X_train)
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_r, y_train)
        pred = model.predict(ridge_scaler.transform(X_test))
        actual = y_test
    elif model_name == "xgboost":
        model = train_xgboost(X_train_s, y_train, CONFIG)
        pred = model.predict(X_test_s)
        actual = y_test
    elif model_name == "lightgbm":
        model = train_lightgbm(X_train_s, y_train, CONFIG)
        pred = model.predict(X_test_s)
        actual = y_test
    else:
        raise ValueError(f"Unknown model {model_name}")

    valid = ~(np.isnan(actual) | np.isnan(pred))
    actual = actual[valid]
    pred = pred[valid]

    return {
        "model": model_name,
        "target": TARGET,
        "n_features": len(feature_cols),
        "mae": round(_named_mae(actual, pred), 4),
        "rmse": round(_named_rmse(actual, pred), 4),
        "r2": round(_r2(actual, pred), 4),
        "n_samples": int(len(actual)),
    }


def main() -> None:
    processed = pd.read_parquet(ROOT / "data" / "processed" / "processed_data.parquet")
    feature_data = build_feature_matrix(processed, CONFIG)
    feature_data = create_target_columns(feature_data, CONFIG["forecasting"]["horizons"])

    split_cfg = CONFIG["training"]["split"]
    train_df, val_df, test_df = split_by_time(
        feature_data,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["validation_ratio"],
        test_ratio=split_cfg["test_ratio"],
    )

    if TARGET not in feature_data.columns:
        raise ValueError(f"{TARGET} missing from feature matrix")

    feature150 = prepare_features(train_df, TARGET)[2]
    feature630 = select_feature_columns(train_df, TARGET)
    feature150 = list(feature150)
    feature630 = list(feature630)

    experiment_rows: list[dict] = []
    for experiment_name, feature_cols in [("feature150", feature150), ("feature630", feature630)]:
        for model_name in ["persistence", "ridge", "xgboost", "lightgbm"]:
            row = _evaluate_model(model_name, feature_cols, train_df, val_df, test_df)
            row["experiment"] = experiment_name
            experiment_rows.append(row)
            logger.info("%s %s -> RMSE=%.4f MAE=%.4f R2=%.4f samples=%d", experiment_name, model_name, row["rmse"], row["mae"], row["r2"], row["n_samples"])

    _save_results(ROOT / "outputs" / "cv_results.csv", experiment_rows)
    (ROOT / "experiments" / "feature150" / "results.csv").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments" / "feature630" / "results.csv").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for r in experiment_rows if r["experiment"] == "feature150"]).to_csv(ROOT / "experiments" / "feature150" / "results.csv", index=False)
    pd.DataFrame([r for r in experiment_rows if r["experiment"] == "feature630"]).to_csv(ROOT / "experiments" / "feature630" / "results.csv", index=False)

    ablation_rows: list[dict] = []
    for block_name, block_cols in {
        "all_features": feature150,
        "without_lag": [c for c in feature150 if "_lag" not in c],
        "without_rolling": [c for c in feature150 if "_roll" not in c],
        "without_interaction": [c for c in feature150 if "farm_avg" not in c and "per_wind" not in c],
        "without_weather": [c for c in feature150 if "wind_speed" not in c and "temperature" not in c],
    }.items():
        for model_name in ["persistence", "ridge", "xgboost", "lightgbm"]:
            row = _evaluate_model(model_name, block_cols, train_df, val_df, test_df)
            row["ablation"] = block_name
            ablation_rows.append(row)

    _save_results(ROOT / "outputs" / "ablation.csv", ablation_rows)
    pd.DataFrame(ablation_rows).to_csv(ROOT / "experiments" / "ablation" / "ablation.csv", index=False)

    summary = {
        "target": TARGET,
        "shared_train_val_test_split": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "feature150": len(feature150),
        "feature630": len(feature630),
        "results_path": str(ROOT / "outputs" / "cv_results.csv"),
        "ablation_path": str(ROOT / "outputs" / "ablation.csv"),
    }
    (ROOT / "experiments" / "feature150" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ROOT / "experiments" / "feature630" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
