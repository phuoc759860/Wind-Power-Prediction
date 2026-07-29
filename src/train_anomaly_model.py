import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def compute_residual_anomalies(df: pd.DataFrame, power_col: str,
                                threshold_std: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    actual = df[power_col].values

    wind_col = power_col.replace("_power", "_wind_speed")
    if wind_col in df.columns:
        from sklearn.linear_model import LinearRegression
        mask = df[wind_col].notna() & df[power_col].notna()
        if mask.sum() > 10:
            X = df.loc[mask, [wind_col]].values
            y = df.loc[mask, power_col].values
            model = LinearRegression().fit(X, y)
            expected = model.predict(df[wind_col].fillna(0).values.reshape(-1, 1))
            residual = actual - expected
        else:
            residual = actual - np.nanmean(actual)
    else:
        residual = actual - np.nanmean(actual)

    mean_r = np.nanmean(residual)
    std_r = np.nanstd(residual)

    if std_r > 0:
        df[f"{power_col}_anomaly_score"] = np.abs(residual - mean_r) / std_r
    else:
        df[f"{power_col}_anomaly_score"] = 0

    df[f"{power_col}_is_anomaly"] = (df[f"{power_col}_anomaly_score"] > threshold_std).astype(int)

    n_anomalies = df[f"{power_col}_is_anomaly"].sum()
    logger.info(f"  {power_col}: {n_anomalies} anomalies detected ({n_anomalies/len(df)*100:.2f}%)")

    return df


def train_isolation_forest(X_train: np.ndarray, contamination: float = 0.05) -> IsolationForest:
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )
    model.fit(X_train)
    return model


def detect_anomalies_isolation_forest(df: pd.DataFrame, feature_cols: List[str],
                                       contamination: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    valid_cols = [c for c in feature_cols if c in df.columns]

    if not valid_cols:
        return df

    X = df[valid_cols].fillna(0).values
    model = train_isolation_forest(X, contamination)

    predictions = model.predict(X)
    scores = model.decision_function(X)

    df["isolation_forest_anomaly"] = (predictions == -1).astype(int)
    df["isolation_forest_score"] = -scores

    n_anomalies = df["isolation_forest_anomaly"].sum()
    logger.info(f"  Isolation Forest: {n_anomalies} anomalies detected ({n_anomalies/len(df)*100:.2f}%)")

    return df


def run_anomaly_detection(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logger.info("Running anomaly detection...")

    anomaly_cfg = config.get("anomaly", {})
    threshold = anomaly_cfg.get("residual_threshold_std", 3.0)
    contamination = anomaly_cfg.get("contamination", 0.05)

    power_cols = [c for c in df.columns if c.endswith("_power") and "target" not in c and "lag" not in c]

    for col in power_cols:
        df = compute_residual_anomalies(df, col, threshold)

    wind_cols = [c for c in df.columns if c.endswith("_wind_speed") and "lag" not in c][:3]
    if wind_cols:
        df = detect_anomalies_isolation_forest(df, wind_cols, contamination)

    return df


def get_anomaly_summary(df: pd.DataFrame) -> Dict:
    summary = {}
    anomaly_cols = [c for c in df.columns if c.endswith("_is_anomaly")]
    for col in anomaly_cols:
        summary[col] = {
            "count": int(df[col].sum()),
            "pct": round(df[col].mean() * 100, 2),
        }
    return summary
