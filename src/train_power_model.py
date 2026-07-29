import logging
import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def prepare_features(df: pd.DataFrame, target_col: str, feature_cols: List[str] = None,
                     exclude_patterns: List[str] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = df.copy()

    if exclude_patterns is None:
        exclude_patterns = ["_missing", "_status", "data_split", "time_index"]

    if feature_cols is None:
        feature_cols = []
        for col in df.columns:
            if col == target_col or col == "timestamp" or col == "data_split":
                continue
            if "_target_" in col:
                continue
            if any(pat in col for pat in exclude_patterns):
                continue
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                feature_cols.append(col)

    valid_features = [c for c in feature_cols if c in df.columns]
    X = df[valid_features].copy()
    y = df[target_col].copy()

    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    if len(valid_features) > 150:
        from sklearn.feature_selection import f_regression, SelectKBest
        selector = SelectKBest(f_regression, k=150)
        X_arr = selector.fit_transform(X.values, y.values)
        selected_mask = selector.get_support()
        selected_features = [f for f, s in zip(valid_features, selected_mask) if s]
        X = pd.DataFrame(X_arr, columns=selected_features, index=X.index)
        valid_features = selected_features
        logger.info(f"  Feature selection: reduced to {len(valid_features)} features")

    return X, y, valid_features


def scale_features(X_train, X_val=None, X_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = None
    X_test_scaled = None

    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_val_scaled, X_test_scaled


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, config: dict = None) -> RandomForestRegressor:
    params = (config or {}).get("models", {}).get("random_forest", {})
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, config: dict = None):
    if not HAS_XGB:
        logger.warning("XGBoost not installed")
        return None
    params = (config or {}).get("models", {}).get("xgboost", {})
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray, config: dict = None):
    if not HAS_LGBM:
        logger.warning("LightGBM not installed")
        return None
    params = (config or {}).get("models", {}).get("lightgbm", {})
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_linear_reg(X_train: np.ndarray, y_train: np.ndarray, config: dict = None) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


MODEL_TRAINERS = {
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
    "lightgbm": train_lightgbm,
    "linear_regression": train_linear_reg,
}


def train_power_models(train_data: pd.DataFrame, val_data: pd.DataFrame,
                       target_col: str, config: dict) -> Tuple[Dict, Dict]:
    logger.info(f"Training power models for: {target_col}")

    X_train, y_train, feature_cols = prepare_features(train_data, target_col)
    X_val, y_val, _ = prepare_features(val_data, target_col, feature_cols)

    if len(X_train) == 0 or len(X_val) == 0:
        logger.warning(f"  No valid data for {target_col}, skipping")
        return {}, {}

    logger.info(f"  Features: {len(feature_cols)}, Train: {len(X_train)}, Val: {len(X_val)}")

    scaler, X_train_s, X_val_s, _ = scale_features(X_train, X_val)

    models_cfg = config.get("training", {}).get("models", {})
    ml_models = models_cfg.get("ml", ["random_forest", "xgboost", "lightgbm"])

    results = {}
    trained_models = {}

    for model_name in ml_models:
        if model_name not in MODEL_TRAINERS:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue

        logger.info(f"  Training {model_name}...")
        try:
            trainer = MODEL_TRAINERS[model_name]
            model = trainer(X_train_s, y_train.values, config)

            if model is None:
                continue

            train_pred = model.predict(X_train_s)
            val_pred = model.predict(X_val_s)

            train_mae = np.mean(np.abs(y_train.values - train_pred))
            train_rmse = np.sqrt(np.mean((y_train.values - train_pred) ** 2))
            val_mae = np.mean(np.abs(y_val.values - val_pred))
            val_rmse = np.sqrt(np.mean((y_val.values - val_pred) ** 2))

            ss_res = np.sum((y_val.values - val_pred) ** 2)
            ss_tot = np.sum((y_val.values - np.mean(y_val.values)) ** 2)
            val_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            result_key = f"{target_col}_{model_name}"
            results[result_key] = {
                "model": model_name,
                "target": target_col,
                "n_features": len(feature_cols),
                "train_mae": round(train_mae, 4),
                "train_rmse": round(train_rmse, 4),
                "val_mae": round(val_mae, 4),
                "val_rmse": round(val_rmse, 4),
                "val_r2": round(val_r2, 4),
            }

            model_key = f"{target_col}_{model_name}"
            trained_models[model_key] = {
                "model": model,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "model_name": model_name,
                "target": target_col,
            }

            logger.info(f"    {model_name}: Val MAE={val_mae:.2f}, RMSE={val_rmse:.2f}, R2={val_r2:.4f}")

        except Exception as e:
            logger.error(f"  Error training {model_name}: {e}")

    best_key = min(
        [k for k in results if results[k]["target"] == target_col],
        key=lambda k: results[k]["val_rmse"],
        default=None,
    )
    if best_key:
        logger.info(f"  Best model for {target_col}: {results[best_key]['model']} "
                     f"(RMSE={results[best_key]['val_rmse']:.2f})")

    return results, trained_models


def save_models(trained_models: Dict, output_dir: str):
    import joblib
    os.makedirs(output_dir, exist_ok=True)

    for model_key, model_info in trained_models.items():
        safe_name = model_key.replace("/", "_").replace(" ", "_")
        joblib.dump(model_info["model"], os.path.join(output_dir, f"{safe_name}_model.joblib"))
        joblib.dump(model_info["scaler"], os.path.join(output_dir, f"{safe_name}_scaler.joblib"))

        with open(os.path.join(output_dir, f"{safe_name}_features.json"), "w") as f:
            json.dump(model_info["feature_cols"], f)

    logger.info(f"Saved {len(trained_models)} models to {output_dir}")


def load_models(model_dir: str) -> Dict:
    import joblib
    loaded = {}

    for fname in os.listdir(model_dir):
        if fname.endswith("_model.joblib"):
            target = fname.replace("_model.joblib", "")
            model = joblib.load(os.path.join(model_dir, fname))
            scaler_path = os.path.join(model_dir, f"{target}_scaler.joblib")
            features_path = os.path.join(model_dir, f"{target}_features.json")

            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            features = json.load(open(features_path)) if os.path.exists(features_path) else []

            loaded[target] = {"model": model, "scaler": scaler, "feature_cols": features}

    logger.info(f"Loaded {len(loaded)} models from {model_dir}")
    return loaded
