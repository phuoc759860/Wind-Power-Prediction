import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


def _baseline_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict:
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[valid]
    predicted = predicted[valid]
    if len(actual) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n_samples": 0}
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "n_samples": int(len(actual)),
    }


def persistence_forecast(test_data: pd.DataFrame, target_col: str, horizon_steps: int) -> np.ndarray:
    return test_data[target_col].shift(horizon_steps).values


def evaluate_persistence(test_data: pd.DataFrame, target_col: str, horizon_steps: int) -> Dict:
    shifted = persistence_forecast(test_data, target_col, horizon_steps)
    actual = test_data[target_col].values
    return _baseline_metrics(actual, shifted)


def train_ridge_regression(train_data: pd.DataFrame, target_col: str,
                           config: dict, alpha: float = 1.0):
    feature_cols = [c for c in train_data.columns
                    if c not in [target_col, "timestamp", "data_split", "time_index"]
                    and not c.startswith("_target_")
                    and not any(p in c for p in ["_missing", "_status"])
                    and train_data[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    valid_cols = [c for c in feature_cols if c in train_data.columns]
    X = train_data[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = train_data[target_col].fillna(0)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_s, y)
    return model, scaler, valid_cols


def evaluate_ridge(model, scaler, feature_cols, test_data: pd.DataFrame, target_col: str) -> Dict:
    valid_cols = [c for c in feature_cols if c in test_data.columns]
    X = test_data[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_s = scaler.transform(X)
    pred = model.predict(X_s)
    actual = test_data[target_col].values
    return _baseline_metrics(actual, pred)


def train_baselines(train_data: pd.DataFrame, val_data: pd.DataFrame,
                    power_cols: list, config: dict) -> Dict:
    from tqdm import tqdm

    logger.info("Training baseline models...")
    horizons = config.get("forecasting", {}).get("horizons", [])
    results = {}

    ridge_models = {}
    for col in tqdm(power_cols, desc="Training Ridge models"):
        fcols = _select_walk_forward_features(train_data, col)
        X = train_data[fcols].fillna(0).replace([np.inf, -np.inf], 0)
        y = train_data[col].fillna(0)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_s, y)
        ridge_models[col] = (model, scaler, fcols)

    for horizon in tqdm(horizons, desc="Horizons"):
        steps = horizon["steps"]
        name = horizon["name"]

        for col in power_cols:
            p_metrics = evaluate_persistence(val_data, col, steps)
            results[f"{col}_{name}"] = {
                "model": "persistence",
                "horizon": name,
                "turbine": col.replace("_power", ""),
                **p_metrics,
            }

            ridge_model, ridge_scaler, ridge_features = ridge_models[col]
            r_metrics = evaluate_ridge(ridge_model, ridge_scaler, ridge_features, val_data, col)
            results[f"{col}_{name}_ridge"] = {
                "model": "ridge",
                "horizon": name,
                "turbine": col.replace("_power", ""),
                **r_metrics,
            }

    logger.info(f"Baseline training complete: {len(results)} evaluations")
    return results


def _select_walk_forward_features(df: pd.DataFrame, target_col: str):
    turbine_prefix = target_col.replace("_power", "")
    temporal_keys = ["hour", "day", "month", "dayofweek", "dayofyear",
                     "is_weekend", "season", "sin_", "cos_"]
    farm_keys = ["farm_avg"]
    selected = []
    for c in df.columns:
        if c == target_col:
            continue
        if c in ["timestamp", "data_split", "time_index"]:
            continue
        if c.startswith("_target_") or "_missing" in c or "_status" in c:
            continue
        if df[c].dtype not in [np.float64, np.float32, np.int64, np.int32]:
            continue
        if turbine_prefix and turbine_prefix in c:
            selected.append(c)
        elif any(k in c for k in temporal_keys):
            selected.append(c)
        elif any(k in c for k in farm_keys):
            selected.append(c)
    return selected


def walk_forward_baselines(df: pd.DataFrame, power_cols: list, config: dict,
                           n_folds: int = 5, val_size: float = 0.15) -> Dict:
    from src.split_time_series import walk_forward_split
    from tqdm import tqdm

    folds = walk_forward_split(df, n_folds=n_folds, val_size=val_size)
    logger.info(f"Walk-forward baseline evaluation: {len(folds)} folds")

    fold_results = {}
    horizons = config.get("forecasting", {}).get("horizons", [])

    total_iters = len(folds) * len(horizons) * len(power_cols)
    progress = tqdm(total=total_iters, desc="Walk-forward baselines")

    for fold_info in folds:
        fold_num = fold_info["fold"]
        train_fold = fold_info["train"].reset_index(drop=True)
        val_fold = fold_info["val"]

        ridge_models = {}
        for col in power_cols:
            fcols = _select_walk_forward_features(train_fold, col)
            X = train_fold[fcols].fillna(0).replace([np.inf, -np.inf], 0)
            y = train_fold[col].fillna(0)
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_s, y)
            ridge_models[col] = (model, scaler, fcols)

        for horizon in horizons:
            steps = horizon["steps"]
            name = horizon["name"]
            for col in power_cols:
                p_metrics = evaluate_persistence(val_fold, col, steps)
                key = f"fold_{fold_num}_{col}_{name}"
                fold_results[key] = {"model": "persistence", "horizon": name, "fold": fold_num, **p_metrics}

                ridge_model, ridge_scaler, ridge_features = ridge_models[col]
                r_metrics = evaluate_ridge(ridge_model, ridge_scaler, ridge_features, val_fold, col)
                key2 = f"fold_{fold_num}_{col}_{name}_ridge"
                fold_results[key2] = {"model": "ridge", "horizon": name, "fold": fold_num, **r_metrics}

                progress.update(1)

    progress.close()

    wf_summary = {}
    baseline_names = ["persistence", "ridge"]
    for model_name in baseline_names:
        for horizon in config.get("forecasting", {}).get("horizons", []):
            h = horizon["name"]
            vals = [v["rmse"] for k, v in fold_results.items()
                    if v["model"] == model_name and v["horizon"] == h and not np.isnan(v.get("rmse", np.nan))]
            r2_vals = [v["r2"] for k, v in fold_results.items()
                       if v["model"] == model_name and v["horizon"] == h and not np.isnan(v.get("r2", np.nan))]
            if vals:
                wf_summary[f"{model_name}_{h}"] = {
                    "model": model_name, "horizon": h,
                    "rmse_mean": round(float(np.mean(vals)), 2),
                    "rmse_std": round(float(np.std(vals)), 2),
                    "r2_mean": round(float(np.mean(r2_vals)), 4),
                    "r2_std": round(float(np.std(r2_vals)), 4),
                    "n_folds": len(vals),
                }

    return {"fold_details": fold_results, "summary": wf_summary}
