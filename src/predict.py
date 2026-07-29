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


def create_forecast_output(test_data: pd.DataFrame, predictions: Dict,
                           timestamp_col: str = "timestamp") -> pd.DataFrame:
    columns = {}

    if timestamp_col in test_data.columns:
        columns["timestamp"] = test_data[timestamp_col].values[:len(test_data)]

    for model_key, pred_info in predictions.items():
        pred_values = pred_info["predictions"]
        target = pred_info.get("target", model_key)
        model_name = pred_info.get("model_name", "unknown")

        columns[f"{model_key}_predicted"] = pred_values[:len(test_data)]
        columns[f"{model_key}_model"] = np.full(len(test_data), model_name, dtype=object)

        if target in test_data.columns:
            actual = test_data[target].values[:len(test_data)]
            columns[f"{model_key}_actual"] = actual
            columns[f"{model_key}_error"] = actual - pred_values[:len(test_data)]

    output = pd.DataFrame(columns)
    return output


def add_confidence_intervals(predictions_df: pd.DataFrame, trained_models: Dict,
                             confidence: float = 0.9) -> pd.DataFrame:
    df = predictions_df.copy()
    z = 1.645 if confidence == 0.9 else 2.576 if confidence == 0.99 else 1.96
    rated = 2200

    pred_cols = [c for c in df.columns if c.endswith("_predicted")]
    new_cols = {}
    for col in pred_cols:
        pred = df[col].dropna()
        if len(pred) > 10:
            std = pred.std()
            new_cols[col.replace("_predicted", "_lower_bound")] = (df[col] - z * std).clip(lower=0, upper=rated)
            new_cols[col.replace("_predicted", "_upper_bound")] = (df[col] + z * std).clip(lower=0, upper=rated)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def save_forecasts(forecast_df: pd.DataFrame, output_dir: str,
                   filename: str = "forecasts.csv"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    forecast_df.to_csv(output_path, index=False)
    logger.info(f"Forecasts saved to {output_path} ({forecast_df.shape})")
    return output_path
