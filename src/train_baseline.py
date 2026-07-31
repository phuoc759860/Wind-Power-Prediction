import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)

# Columns that MUST never be used as features for any forecast.
# Target columns look like {base}_target_{horizon} and contain the FUTURE
# value P(t+h); any model that consumes them is leaking the answer.
NON_FEATURE_MARKERS = ["_target_", "_missing", "_status", "_is_anomaly",
                       "_anomaly_score", "_is_stopped", "_failure_event"]
META_COLUMNS = ["timestamp", "PCTimeStamp", "data_split", "time_index", "data_audit"]


def base_power_col(target_col: str) -> str:
    """TB02_power_target_10min -> TB02_power ; farm_total_power_target_1hour -> farm_total_power."""
    if "_target_" in target_col:
        return target_col.split("_target_")[0]
    return target_col


def is_feature_column(col: str, target_col: str, dtype=None) -> bool:
    """Deterministic allow-list of columns that are safe as features at issue time t."""
    if col == target_col:
        return False
    if col in META_COLUMNS:
        return False
    if any(marker in col for marker in NON_FEATURE_MARKERS):
        return False
    if dtype is not None:
        try:
            if not pd.api.types.is_numeric_dtype(dtype):
                return False
        except Exception:
            return False
    return True


def select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """Return numeric feature columns available at issue time t (no future info)."""
    selected = []
    for c in df.columns:
        if is_feature_column(c, target_col, df[c].dtype):
            selected.append(c)
    return selected


def _as_matrix(df: pd.DataFrame, feature_cols: List[str]):
    valid_cols = [c for c in feature_cols if c in df.columns]
    X = df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
    return X, valid_cols


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


def persistence_predictions(data: pd.DataFrame, target_col: str,
                            horizon_steps: int) -> np.ndarray:
    """Persistence forecast for the aligned supervised frame.

    The row at issue time t holds target P(t+h) (created with shift(-h)).
    The persistence forecast issued at t is therefore the current value P(t),
    i.e. the base power column at the SAME row index t.
    """
    base_col = base_power_col(target_col)
    if base_col not in data.columns:
        logger.warning(f"Base column {base_col} not found for persistence of {target_col}")
        return np.full(len(data), np.nan)
    return data[base_col].to_numpy(dtype=float)


def evaluate_persistence(data: pd.DataFrame, target_col: str, horizon_steps: int) -> Dict:
    pred = persistence_predictions(data, target_col, horizon_steps)
    actual = data[target_col].to_numpy(dtype=float) if target_col in data.columns else np.full(len(data), np.nan)
    return _baseline_metrics(actual, pred)


def train_ridge(train_data: pd.DataFrame, target_col: str, config: dict = None,
                alpha: float = 1.0):
    """Train Ridge to predict target P(t+h) from features available at time t only."""
    feature_cols = select_feature_columns(train_data, target_col)
    if not feature_cols:
        raise ValueError(f"No features selected for target {target_col}")

    X, valid_cols = _as_matrix(train_data, feature_cols)
    y = pd.to_numeric(train_data[target_col], errors="coerce").to_numpy()

    mask = y_mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    if len(y) < 50:
        logger.warning(f"Ridge target {target_col}: only {len(y)} valid training rows")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_s, y)
    return model, scaler, valid_cols


def evaluate_ridge(model, scaler, feature_cols, data: pd.DataFrame, target_col: str) -> Dict:
    X, _ = _as_matrix(data, feature_cols)
    X_s = scaler.transform(X)
    pred = model.predict(X_s)
    actual = data[target_col].to_numpy(dtype=float) if target_col in data.columns else np.full(len(data), np.nan)
    return _baseline_metrics(actual, pred)


def ridge_predictions(model, scaler, feature_cols, data: pd.DataFrame) -> np.ndarray:
    X, _ = _as_matrix(data, feature_cols)
    X_s = scaler.transform(X)
    return model.predict(X_s)


def build_test_baseline_predictions(test_data: pd.DataFrame, ridge_models: Dict,
                                    power_cols: list, config: dict) -> Dict:
    """Persistence + ridge predictions on the test set, keyed by target column.

    Returns {target_col: {"persistence": np.array, "ridge": np.array}}.
    Persistence uses P(t) at the issue row; ridge uses the trained leakage-free
    model. Both align with the supervised frame so skill scores use identical
    samples (P0-03).
    """
    horizons = config.get("forecasting", {}).get("horizons", [])
    out = {}
    for h in horizons:
        name, steps = h["name"], h["steps"]
        for col in power_cols:
            target = f"{col}_target_{name}"
            if target not in test_data.columns:
                continue
            p = persistence_predictions(test_data, target, steps)
            entry = {"persistence": p, "ridge": np.full(len(p), np.nan)}
            if target in ridge_models:
                model, scaler, fcols = ridge_models[target]
                entry["ridge"] = ridge_predictions(model, scaler, fcols, test_data)
            out[target] = entry
    return out


def train_baselines(train_data: pd.DataFrame, val_data: pd.DataFrame,
                    power_cols: list, config: dict,
                    return_models: bool = False):
    """Train persistence + ridge baselines for every (turbine x horizon).

    Ridge is trained once per horizon target so it predicts P(t+h), never the
    current value P(t). No feature column contains '_target_'.
    """
    from tqdm import tqdm

    logger.info("Training baseline models (persistence + ridge, leakage-free)...")
    horizons = config.get("forecasting", {}).get("horizons", [])
    results = {}
    ridge_models: Dict[str, Tuple] = {}

    # Target columns per horizon, e.g. TB02_power_target_10min
    horizon_targets: Dict[str, List[str]] = {}
    for horizon in horizons:
        name, steps = horizon["name"], horizon["steps"]
        targets = [f"{col}_target_{name}" for col in power_cols]
        horizon_targets[name] = [(t, steps) for t in targets]

    for name, targets in tqdm(horizon_targets.items(), desc="Baseline horizons"):
        for target, steps in targets:
            if target not in train_data.columns or target not in val_data.columns:
                logger.warning(f"Target {target} missing from data, skipping baseline")
                continue
            try:
                model, scaler, fcols = train_ridge(train_data, target, config)
            except Exception as e:
                logger.error(f"Ridge training failed for {target}: {e}")
                continue

            ridge_models[target] = (model, scaler, fcols)

            p_metrics = evaluate_persistence(val_data, target, steps)
            r_metrics = evaluate_ridge(model, scaler, fcols, val_data, target)

            base = base_power_col(target)
            results[f"{target}_persistence"] = {
                "model": "persistence", "horizon": name,
                "turbine": base, "target": target,
                **p_metrics,
            }
            results[f"{target}_ridge"] = {
                "model": "ridge", "horizon": name,
                "turbine": base, "target": target,
                **r_metrics,
            }

    logger.info(f"Baseline training complete: {len(results)} evaluations, "
                f"{len(ridge_models)} ridge models")
    if return_models:
        return results, ridge_models
    return results


def _select_walk_forward_features(df: pd.DataFrame, target_col: str) -> List[str]:
    """Leakage-safe feature selection used by walk-forward baseline evaluation."""
    return select_feature_columns(df, target_col)


def walk_forward_baselines(df: pd.DataFrame, power_cols: list, config: dict,
                           n_folds: int = 5, val_size: float = 0.15) -> Dict:
    from src.split_time_series import walk_forward_split
    from tqdm import tqdm

    folds = walk_forward_split(df, n_folds=n_folds, val_size=val_size)
    logger.info(f"Walk-forward baseline evaluation: {len(folds)} folds")

    fold_results = []
    horizons = config.get("forecasting", {}).get("horizons", [])

    total_iters = len(folds) * len(horizons) * len(power_cols)
    progress = tqdm(total=total_iters, desc="Walk-forward baselines")

    for fold_info in folds:
        fold_num = fold_info["fold"]
        train_fold = fold_info["train"].reset_index(drop=True)
        val_fold = fold_info["val"].reset_index(drop=True)

        for horizon in horizons:
            steps = horizon["steps"]
            name = horizon["name"]
            for col in power_cols:
                target = f"{col}_target_{name}"
                if target not in train_fold.columns or target not in val_fold.columns:
                    continue
                # Persistence on the SAME validation slice
                p_metrics = evaluate_persistence(val_fold, target, steps)
                fold_results.append({"model": "persistence", "horizon": name,
                                     "fold": fold_num, "turbine": col, **p_metrics})

                # Ridge on the SAME validation slice
                try:
                    model, scaler, fcols = train_ridge(train_fold, target, config)
                    r_metrics = evaluate_ridge(model, scaler, fcols, val_fold, target)
                    fold_results.append({"model": "ridge", "horizon": name,
                                         "fold": fold_num, "turbine": col, **r_metrics})
                except Exception as e:
                    logger.error(f"Walk-forward ridge failed {target} fold {fold_num}: {e}")

                progress.update(1)

    progress.close()

    fold_df = pd.DataFrame(fold_results)

    wf_summary = {}
    for model_name in ["persistence", "ridge"]:
        for horizon in horizons:
            h = horizon["name"]
            sub = fold_df[(fold_df["model"] == model_name) & (fold_df["horizon"] == h)]
            sub = sub.dropna(subset=["rmse"])
            if sub.empty:
                continue
            wf_summary[f"{model_name}_{h}"] = {
                "model": model_name,
                "horizon": h,
                "rmse_mean": round(float(sub["rmse"].mean()), 2),
                "rmse_std": round(float(sub["rmse"].std()), 2),
                "r2_mean": round(float(sub["r2"].mean()), 4),
                "r2_std": round(float(sub["r2"].std()), 4),
                # n_folds is the TRUE number of walk-forward folds (time windows).
                "n_folds": int(fold_df["fold"].nunique()) if not fold_df.empty else 0,
                # n_evaluations = number of turbine x fold evaluations used in the mean.
                "n_evaluations": int(len(sub)),
            }

    return {"fold_details": fold_df.to_dict(orient="records"), "summary": wf_summary,
            "n_folds": int(fold_df["fold"].nunique()) if not fold_df.empty else 0}
