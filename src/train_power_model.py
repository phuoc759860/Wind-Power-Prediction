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
        exclude_patterns = ["_missing", "_status", "data_split", "time_index",
                            "is_observed", "is_synthetic", "is_imputed", "is_simulated"]

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


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, config: dict = None,
                         X_val=None, y_val=None, feature_cols=None) -> RandomForestRegressor:
    params = (config or {}).get("models", {}).get("random_forest", {})
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_linear_reg(X_train: np.ndarray, y_train: np.ndarray, config: dict = None,
                      X_val=None, y_val=None, feature_cols=None) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def _model_trainers(config: dict) -> Dict:
    """Select trainers per config.

    When training.tuning.enabled is false (default) XGBoost/LightGBM are trained
    directly with the fixed hyperparameters from models.xgboost / models.lightgbm
    (fast, fully reproducible). When enabled, a TimeSeriesSplit grid search is
    run per target and the best parameters are used.
    """
    tuning = config.get("training", {}).get("tuning", {})
    if tuning.get("enabled", False):
        return dict(MODEL_TRAINERS)
    fast = {
        "xgboost": train_xgboost,
        "lightgbm": train_lightgbm,
        "random_forest": train_random_forest,
        "linear_regression": train_linear_reg,
    }
    return {name: fast.get(name, MODEL_TRAINERS.get(name)) for name in MODEL_TRAINERS}


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

    trainers = _model_trainers(config)

    for model_name in ml_models:
        if model_name not in trainers:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue

        logger.info(f"  Training {model_name}...")
        try:
            trainer = trainers[model_name]
            model = trainer(X_train_s, y_train.values, config,
                            X_val=X_val_s, y_val=y_val.values, feature_cols=feature_cols)

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


def _tune_xgboost(X_train, y_train, config, X_val=None, y_val=None, feature_cols=None):
    from sklearn.model_selection import TimeSeriesSplit
    import xgboost as xgb

    tscv = TimeSeriesSplit(n_splits=3)
    base_params = {
        "objective": "reg:squarederror",
        "random_state": config.get("training", {}).get("random_state", 42),
    }

    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.15],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
    }

    best_score = float("inf")
    best_params = None
    for n_est in param_grid["n_estimators"]:
        for md in param_grid["max_depth"][:2]:
            for lr in param_grid["learning_rate"][:2]:
                params = {**base_params, "n_estimators": n_est, "max_depth": md, "learning_rate": lr}
                scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                    y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                    model = xgb.XGBRegressor(**params)
                    model.fit(X_tr_fold, y_tr_fold)
                    pred = model.predict(X_val_fold)
                    scores.append(np.sqrt(np.mean((y_val_fold - pred) ** 2)))
                mean_rmse = np.mean(scores)
                if mean_rmse < best_score:
                    best_score = mean_rmse
                    best_params = params

    final_X = np.vstack([X_train, X_val]) if X_val is not None else X_train
    final_y = np.concatenate([y_train, y_val]) if y_val is not None else y_train
    if best_params is None:
        best_params = {**base_params, "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
    model = xgb.XGBRegressor(**best_params)
    model.fit(final_X, final_y)
    return model


def _tune_lightgbm(X_train, y_train, config, X_val=None, y_val=None, feature_cols=None):
    from sklearn.model_selection import TimeSeriesSplit
    import lightgbm as lgb

    tscv = TimeSeriesSplit(n_splits=3)
    base_params = {
        "objective": "regression",
        "random_state": config.get("training", {}).get("random_state", 42),
        "verbose": -1,
    }

    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.15],
        "subsample": [0.7, 0.8, 1.0],
        "feature_fraction": [0.5, 0.7, 1.0],
    }

    best_score = float("inf")
    best_params = None
    for n_est in param_grid["n_estimators"]:
        for md in param_grid["max_depth"][:2]:
            for lr in param_grid["learning_rate"][:2]:
                params = {**base_params, "n_estimators": n_est, "max_depth": md, "learning_rate": lr}
                scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                    y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_tr_fold, y_tr_fold)
                    pred = model.predict(X_val_fold)
                    scores.append(np.sqrt(np.mean((y_val_fold - pred) ** 2)))
                mean_rmse = np.mean(scores)
                if mean_rmse < best_score:
                    best_score = mean_rmse
                    best_params = params

    final_X = np.vstack([X_train, X_val]) if X_val is not None else X_train
    final_y = np.concatenate([y_train, y_val]) if y_val is not None else y_train
    if best_params is None:
        best_params = {**base_params, "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
    model = lgb.LGBMRegressor(**best_params)
    model.fit(final_X, final_y)
    return model


MODEL_TRAINERS = {
    "random_forest": train_random_forest,
    "xgboost": _tune_xgboost,
    "lightgbm": _tune_lightgbm,
    "linear_regression": train_linear_reg,
}


def train_xgboost(X_train, y_train, config, X_val=None, y_val=None, feature_cols=None):
    if not HAS_XGB:
        return None
    params = (config or {}).get("models", {}).get("xgboost", {})
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


def train_lightgbm(X_train, y_train, config, X_val=None, y_val=None, feature_cols=None):
    if not HAS_LGBM:
        return None
    params = (config or {}).get("models", {}).get("lightgbm", {})
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model


def walk_forward_ml(df: pd.DataFrame, target_col: str, config: dict,
                    n_folds: int = 5) -> List[Dict]:
    from src.split_time_series import walk_forward_split

    folds = walk_forward_split(df, n_folds=n_folds)
    fold_results = []

    for fold_info in folds:
        train_fold = fold_info["train"].reset_index(drop=True)
        val_fold = fold_info["val"]

        X_train, y_train, feature_cols = prepare_features(train_fold, target_col)
        X_val, y_val, _ = prepare_features(val_fold, target_col, feature_cols)

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        scaler, X_train_s, X_val_s, _ = scale_features(X_train, X_val)
        y_train_vals = y_train.values
        y_val_vals = y_val.values

        models_cfg = config.get("training", {}).get("models", {})
        ml_models = models_cfg.get("ml", ["xgboost", "lightgbm"])

        for model_name in ml_models:
            if model_name == "xgboost" and HAS_XGB:
                model = train_xgboost(X_train_s, y_train_vals, config)
            elif model_name == "lightgbm" and HAS_LGBM:
                model = train_lightgbm(X_train_s, y_train_vals, config)
            else:
                continue

            if model is None:
                continue

            val_pred = model.predict(X_val_s)
            mae = float(np.mean(np.abs(y_val_vals - val_pred)))
            rmse = float(np.sqrt(np.mean((y_val_vals - val_pred) ** 2)))
            r2 = float(1 - np.sum((y_val_vals - val_pred) ** 2) / max(np.sum((y_val_vals - np.mean(y_val_vals)) ** 2), 1e-10))

            fold_results.append({
                "fold": fold_info["fold"],
                "model": model_name,
                "target": target_col,
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
                "n_val": len(y_val_vals),
            })

    return fold_results


def walk_forward_all_ml(df: pd.DataFrame, config: dict, n_folds: int = 3,
                        save_path: str = None) -> pd.DataFrame:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    turbine_ids = config.get("turbines", {}).get("ids", [])
    base_target_cols = [f"{tid}_power" for tid in turbine_ids if f"{tid}_power" in df.columns]
    base_target_cols.append("farm_total_power")
    base_target_cols = [c for c in base_target_cols if c in df.columns]
    horizons = config.get("forecasting", {}).get("horizons", [])

    logger.info(f"Walk-forward ML: {len(base_target_cols)} targets x {len(horizons)} horizons x {n_folds} folds")
    all_results = []
    total_combos = len(base_target_cols) * len(horizons)

    for base_target in tqdm(base_target_cols, desc="WF-ML targets"):
        for horizon in horizons:
            h_name = horizon["name"]
            target = f"{base_target}_target_{h_name}"
            if target not in df.columns:
                continue
            try:
                fold_results = walk_forward_ml(df, target, config, n_folds=n_folds)
                all_results.extend(fold_results)
                logger.info(f"  {target}: {len(fold_results)} folds")
            except Exception as e:
                logger.warning(f"  WF failed for {target}: {e}")

    wf_df = pd.DataFrame(all_results)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        wf_df.to_csv(save_path, index=False)
        logger.info(f"  WF results saved to {save_path}")

    return wf_df


def save_models(trained_models: Dict, output_dir: str, config: Optional[dict] = None,
                seed: Optional[int] = None, data_path: Optional[str] = None):
    import joblib
    import hashlib
    import subprocess
    os.makedirs(output_dir, exist_ok=True)

    # Capture provenance once
    git_commit = None
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        git_commit = "unknown"

    config_hash = None
    if config:
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]

    data_hash = None
    if data_path and os.path.exists(data_path):
        def _hash_stream(path):
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)

        hasher = hashlib.sha256()
        if os.path.isdir(data_path):
            for root, _, files in os.walk(data_path):
                for fn in sorted(files):
                    hasher.update(fn.encode("utf-8"))
                    _hash_stream(os.path.join(root, fn))
        else:
            _hash_stream(data_path)
        data_hash = hasher.hexdigest()[:16]

    python_version = f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}"

    for model_key, model_info in trained_models.items():
        safe_name = model_key.replace("/", "_").replace(" ", "_")
        joblib.dump(model_info["model"], os.path.join(output_dir, f"{safe_name}_model.joblib"))
        joblib.dump(model_info["scaler"], os.path.join(output_dir, f"{safe_name}_scaler.joblib"))

        with open(os.path.join(output_dir, f"{safe_name}_features.json"), "w") as f:
            json.dump(model_info["feature_cols"], f)

        metadata = {
            "model_key": model_key,
            "model_type": model_info.get("model_name", type(model_info["model"]).__name__),
            "n_features": len(model_info["feature_cols"]),
            "seed": seed,
            "python_version": python_version,
            "git_commit": git_commit,
            "config_hash": config_hash,
            "source_data_hash": data_hash,
            "dependencies": {k: v for k, v in {
                "pandas": "pandas",
                "numpy": "numpy",
                "scikit-learn": "sklearn",
                "xgboost": "xgboost",
                "lightgbm": "lightgbm",
            }.items() if _mod_version(v)},
        }
        with open(os.path.join(output_dir, f"{safe_name}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    logger.info(f"Saved {len(trained_models)} models to {output_dir}")


def _mod_version(mod_name):
    try:
        mod = __import__(mod_name)
        return mod.__version__
    except Exception:
        return None


def load_models(model_dir: str) -> Dict:
    import joblib
    loaded = {}

    for fname in os.listdir(model_dir):
        if fname.endswith("_model.joblib"):
            model_key = fname.replace("_model.joblib", "")
            model = joblib.load(os.path.join(model_dir, fname))
            scaler_path = os.path.join(model_dir, f"{model_key}_scaler.joblib")
            features_path = os.path.join(model_dir, f"{model_key}_features.json")

            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            features = json.load(open(features_path)) if os.path.exists(features_path) else []

            # Saved keys are "<target>_<model_name>"; recover the actual target
            # column so loaded models behave like the in-memory trained ones.
            target_col = model_key
            for suffix in ("_random_forest", "_xgboost", "_lightgbm"):
                if model_key.endswith(suffix):
                    target_col = model_key[: -len(suffix)]
                    break

            loaded[model_key] = {"model": model, "scaler": scaler,
                                 "feature_cols": features, "target": target_col}

    logger.info(f"Loaded {len(loaded)} models from {model_dir}")
    return loaded
