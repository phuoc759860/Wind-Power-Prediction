import json
import logging
import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def predict_with_model(model_info: dict, X: pd.DataFrame) -> np.ndarray:
    model = model_info["model"]
    scaler = model_info.get("scaler")
    feature_cols = model_info.get("feature_cols", [])

    valid_cols = [c for c in feature_cols if c in X.columns]
    X_pred = X[valid_cols].fillna(0)

    # Symmetric with prepare_features (train_power_model.py): feature
    # engineering can emit +/-inf (e.g. pct_change from a 0-power row); they
    # must be dropped to 0 before scaling/predicting or the model call raises.
    X_pred = X_pred.replace([np.inf, -np.inf], np.nan).fillna(0)

    if scaler is not None:
        X_pred = scaler.transform(X_pred.values)

    predictions = model.predict(X_pred)
    return predictions


def predict_power(test_data: pd.DataFrame, trained_models: Dict,
                  config: dict) -> Dict:
    predictions = {}

    for model_key, model_info in trained_models.items():
        logger.info(f"Generating predictions for: {model_key}")

        try:
            pred_values = predict_with_model(model_info, test_data)
            target = model_info.get("target", model_key)
            model_name = model_info.get("model_name", model_key)

            predictions[model_key] = {
                "model_name": model_name,
                "target": target,
                "predictions": pred_values,
            }
        except Exception as e:
            logger.error(f"Error predicting {model_key}: {e}")

    return predictions


def _parse_target(target: str):
    """Parse target column name into (turbine_id, horizon)."""
    if "_target_" not in target:
        return "unknown", "unknown"
    base_part, horizon = target.split("_target_", 1)
    if base_part.endswith("_power"):
        turbine_id = base_part[:-6]
    else:
        turbine_id = base_part
    return turbine_id, horizon


def create_forecast_output(test_data: pd.DataFrame, predictions: Dict,
                           timestamp_col: str = "timestamp") -> pd.DataFrame:
    n = len(test_data)
    timestamps = test_data[timestamp_col].values[:n] if timestamp_col in test_data.columns else np.arange(n)

    dfs = []
    for model_key, pred_info in predictions.items():
        pred_values = pred_info["predictions"][:n]
        target = pred_info.get("target", model_key)
        model_name = pred_info.get("model_name", "unknown")
        turbine_id, horizon = _parse_target(target)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "turbine_id": turbine_id,
            "horizon": horizon,
            "model": model_name,
            "predicted": pred_values,
        })

        if target in test_data.columns:
            actual = test_data[target].values[:n]
            df["actual"] = actual
            df["error"] = actual - pred_values

        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    sort_cols = [c for c in ["timestamp", "turbine_id", "horizon", "model"] if c in result.columns]
    result = result.sort_values(sort_cols).reset_index(drop=True)
    return result


def add_confidence_intervals(predictions_df: pd.DataFrame, trained_models: Dict,
                             confidence: float = 0.9) -> pd.DataFrame:
    df = predictions_df.copy()
    alpha = 1 - confidence
    rated = 2200

    def _conformal_bounds(group):
        residuals = group["error"].dropna().abs().values if "error" in group.columns else None
        if residuals is None or len(residuals) < 10:
            std = group["predicted"].std()
            if pd.isna(std) or std == 0:
                lo = group["predicted"] * 0.05
                hi = group["predicted"] * 0.15
            else:
                lo = group["predicted"] - 1.96 * std
                hi = group["predicted"] + 1.96 * std
            group["lower_bound"] = lo.clip(lower=0, upper=rated)
            group["upper_bound"] = hi.clip(lower=0, upper=rated)
            return group

        q_lo = np.quantile(residuals, alpha / 2)
        q_hi = np.quantile(residuals, 1 - alpha / 2)
        group["lower_bound"] = (group["predicted"] + q_lo).clip(lower=0, upper=rated)
        group["upper_bound"] = (group["predicted"] + q_hi).clip(lower=0, upper=rated)
        return group

    if "error" in df.columns:
        df = df.groupby(["turbine_id", "horizon", "model"], group_keys=False).apply(_conformal_bounds)
    else:
        grouped = df.groupby(["turbine_id", "horizon", "model"])["predicted"]
        stds = grouped.transform("std")
        df["lower_bound"] = (df["predicted"] - 1.96 * stds).clip(lower=0, upper=rated)
        df["upper_bound"] = (df["predicted"] + 1.96 * stds).clip(lower=0, upper=rated)

    return df


def save_forecasts(forecast_df: pd.DataFrame, output_dir: str,
                   filename: str = "forecasts.csv"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    forecast_df.to_csv(output_path, index=False)
    logger.info(f"Forecasts saved to {output_path} ({forecast_df.shape})")
    return output_path
