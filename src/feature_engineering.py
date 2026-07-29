import logging
import pandas as pd
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)


def create_lag_features(df: pd.DataFrame, columns: List[str], lag_steps: List[int]) -> pd.DataFrame:
    new_cols = {}
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lag_steps:
            new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int], stats: List[str] = None) -> pd.DataFrame:
    if stats is None:
        stats = ["mean", "std"]

    new_cols = {}
    for col in columns:
        if col not in df.columns:
            continue
        for w in windows:
            roller = df[col].shift(1).rolling(window=w, min_periods=1)
            if "mean" in stats:
                new_cols[f"{col}_roll{w}_mean"] = roller.mean()
            if "std" in stats:
                new_cols[f"{col}_roll{w}_std"] = roller.std()
            if "min" in stats:
                new_cols[f"{col}_roll{w}_min"] = roller.min()
            if "max" in stats:
                new_cols[f"{col}_roll{w}_max"] = roller.max()
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def create_change_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    new_cols = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        new_cols[f"{col}_diff1"] = s.diff(1)
        new_cols[f"{col}_diff3"] = s.diff(3)
        new_cols[f"{col}_diff6"] = s.diff(6)
        new_cols[f"{col}_pctchange1"] = s.pct_change(1, fill_method=None)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def create_temporal_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    if timestamp_col not in df.columns:
        return df

    ts = df[timestamp_col]
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month

    def get_season(m):
        if m in [3, 4, 5]:
            return 1
        elif m in [6, 7, 8]:
            return 2
        elif m in [9, 10, 11]:
            return 3
        else:
            return 4

    new_cols = {
        "hour_of_day": hour,
        "day_of_week": dow,
        "month": month,
        "day_of_year": ts.dt.dayofyear,
        "is_weekend": (dow >= 5).astype(int),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "season": month.apply(get_season),
    }
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def create_ramp_features(df: pd.DataFrame, power_cols: List[str], threshold_pct_per_min: float = 0.5) -> pd.DataFrame:
    rated_power = 2000
    new_cols = {}

    for col in power_cols:
        if col not in df.columns:
            continue
        power_diff = df[col].diff(1)
        ramp_rate = power_diff / 10
        new_cols[f"{col}_ramp"] = ramp_rate
        new_cols[f"{col}_ramp_abs"] = ramp_rate.abs()
        threshold = rated_power * threshold_pct_per_min / 100
        new_cols[f"{col}_ramp_up"] = (ramp_rate > threshold).astype(int)
        new_cols[f"{col}_ramp_down"] = (ramp_rate < -threshold).astype(int)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    power_cols = [c for c in df.columns if c.endswith("_power") and "lag" not in c and "roll" not in c and "diff" not in c and "ramp" not in c]
    wind_cols = [c for c in df.columns if c.endswith("_wind_speed") and "lag" not in c and "roll" not in c]

    new_cols = {}
    if power_cols and wind_cols:
        if "farm_avg_power" not in df.columns:
            new_cols["farm_avg_power"] = df[power_cols].mean(axis=1)
        if "farm_avg_wind" not in df.columns:
            new_cols["farm_avg_wind"] = df[wind_cols].mean(axis=1)

        for turbine in ["TB01", "TB02", "TB03", "TB04", "TB05", "TB06",
                        "TB07", "TB08", "TB09", "TB10", "TB11", "TB12"]:
            pcol = f"{turbine}_power"
            wcol = f"{turbine}_wind_speed"
            if pcol in df.columns and wcol in df.columns:
                new_cols[f"{turbine}_power_per_wind"] = df[pcol] / df[wcol].replace(0, np.nan)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def create_target_columns(df: pd.DataFrame, horizons: List[Dict], timestamp_col: str = "timestamp") -> pd.DataFrame:
    power_cols = [c for c in df.columns if c.endswith("_power") and "lag" not in c
                  and "roll" not in c and "diff" not in c and "ramp" not in c
                  and not c.startswith("farm_")]

    new_cols = {}
    for horizon in horizons:
        steps = horizon["steps"]
        name = horizon["name"]
        for col in power_cols:
            new_cols[f"{col}_target_{name}"] = df[col].shift(-steps)

    total_power = "farm_total_power"
    if total_power in df.columns:
        for horizon in horizons:
            steps = horizon["steps"]
            h_name = horizon["name"]
            new_cols[f"{total_power}_target_{h_name}"] = df[total_power].shift(-steps)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def build_feature_matrix(df: pd.DataFrame, config: dict, turbine_id: str = None) -> pd.DataFrame:
    logger.info("Building feature matrix...")

    if turbine_id:
        power_col = f"{turbine_id}_power"
        wind_col = f"{turbine_id}_wind_speed"
        temp_col = f"{turbine_id}_temperature"
        target_cols = [c for c in [power_col, wind_col, temp_col] if c in df.columns]
    else:
        target_cols = [c for c in df.columns if c.endswith("_power") and "target" not in c]
        target_cols += [c for c in df.columns if c.endswith("_wind_speed") and "lag" not in c][:3]
        target_cols += [c for c in df.columns if c.endswith("_temperature") and "lag" not in c][:3]

    feature_cfg = config.get("features", {})
    lag_steps = feature_cfg.get("lag_steps", [1, 2, 3, 6, 12, 144])
    rolling_windows = feature_cfg.get("rolling_windows", [6, 18, 36, 144])
    rolling_stats = feature_cfg.get("rolling_stats", ["mean", "std"])

    logger.info("Creating lag features...")
    df = create_lag_features(df, target_cols, lag_steps)

    logger.info("Creating rolling features...")
    df = create_rolling_features(df, target_cols, rolling_windows, rolling_stats)

    logger.info("Creating change features...")
    power_cols = [c for c in target_cols if "power" in c]
    df = create_change_features(df, power_cols)

    logger.info("Creating temporal features...")
    df = create_temporal_features(df)

    logger.info("Creating ramp features...")
    df = create_ramp_features(df, power_cols)

    logger.info("Creating interaction features...")
    df = create_interaction_features(df)

    logger.info(f"Feature matrix built: {df.shape[1]} columns")
    return df
