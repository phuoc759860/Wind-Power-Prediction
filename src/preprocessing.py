import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def handle_missing_values(df: pd.DataFrame, max_gap: int = 12) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in ["timestamp", "data_split"]
                    and not c.endswith("_missing") and not c.endswith("_status")
                    and "farm_" not in c]

    for col in feature_cols:
        if df[col].isnull().sum() == 0:
            continue

        null_mask = df[col].isnull()
        groups = (~null_mask).cumsum()
        gap_sizes = null_mask.groupby(groups).transform("sum")

        short_gap = null_mask & (gap_sizes <= max_gap)
        if short_gap.sum() > 0:
            df.loc[short_gap, col] = df.loc[short_gap, col].ffill(limit=max_gap)

        long_gap = null_mask & (gap_sizes > max_gap)
        if long_gap.sum() > 0:
            logger.warning(
                f"Column {col}: {long_gap.sum()} values in long gaps (>{max_gap} steps) left as NaN"
            )

    return df


def remove_duplicates(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[timestamp_col], keep="first")
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate timestamps")
    return df


def enforce_sampling_interval(
    df: pd.DataFrame, interval_minutes: int = 10, timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    if timestamp_col not in df.columns:
        return df

    df = df.set_index(timestamp_col)
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f"{interval_minutes}min")
    df = df.reindex(idx)
    df.index.name = timestamp_col
    df = df.reset_index()

    new_rows = len(df) - len(df.dropna(subset=[c for c in df.columns if c != timestamp_col]))
    logger.info(f"Reindexed to {interval_minutes}min intervals: {len(df)} rows ({new_rows} new empty rows)")
    return df


def clip_physically_implausible(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    power_cols = [c for c in df.columns if c.endswith("_power") and "target" not in c
                  and "lag" not in c and "roll" not in c and "diff" not in c
                  and "ramp" not in c]
    for col in power_cols:
        df[col] = df[col].clip(lower=0, upper=2200)

    wind_cols = [c for c in df.columns if c.endswith("_wind_speed") and "lag" not in c
                 and "roll" not in c]
    for col in wind_cols:
        df[col] = df[col].clip(lower=0, upper=60)

    temp_cols = [c for c in df.columns if c.endswith("_temperature") and "lag" not in c
                 and "roll" not in c]
    for col in temp_cols:
        df[col] = df[col].clip(lower=-10, upper=55)

    freq_cols = [c for c in df.columns if c.endswith("_frequency") and "lag" not in c
                 and "roll" not in c]
    for col in freq_cols:
        df[col] = df[col].clip(lower=47, upper=53)

    return df


def compute_farm_total_power(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    power_cols = [c for c in df.columns if c.endswith("_power")
                  and "target" not in c and "lag" not in c and "roll" not in c
                  and "diff" not in c and "ramp" not in c and "farm_" not in c]

    if power_cols:
        df["farm_total_power"] = df[power_cols].sum(axis=1)
        df["farm_avg_power"] = df[power_cols].mean(axis=1)
        df["farm_active_turbines"] = (df[power_cols] > 0).sum(axis=1)

    return df


def compute_farm_avg_wind(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    wind_cols = [c for c in df.columns if c.endswith("_wind_speed")
                 and "lag" not in c and "roll" not in c and "farm_" not in c]

    if wind_cols:
        df["farm_avg_wind_speed"] = df[wind_cols].mean(axis=1)
        df["farm_wind_std"] = df[wind_cols].std(axis=1)

    return df


def create_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base_cols = [c for c in df.columns if c not in ["timestamp", "data_split"]
                 and not c.endswith("_missing") and not c.endswith("_status")
                 and "farm_" not in c]
    for col in base_cols:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

    power_cols = [c for c in df.columns if c.endswith("_power") and "target" not in c
                  and "lag" not in c and "roll" not in c and "diff" not in c
                  and "ramp" not in c and "farm_" not in c]
    if power_cols:
        df["any_power_missing"] = df[power_cols].isnull().any(axis=1).astype(int)

    return df


def detect_operating_status(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    power_cols = [c for c in df.columns if c.endswith("_power") and "target" not in c
                  and "lag" not in c and "roll" not in c and "diff" not in c
                  and "ramp" not in c and "farm_" not in c]

    for col in power_cols:
        turbine = col.replace("_power", "")
        wind_col = f"{turbine}_wind_speed"

        df[f"{turbine}_status"] = "unknown"

        if wind_col in df.columns:
            df.loc[df[wind_col].isnull(), f"{turbine}_status"] = "no_data"
            df.loc[df[col].isnull() & df[wind_col].notna(), f"{turbine}_status"] = "stopped"
            df.loc[df[col] == 0, f"{turbine}_status"] = "stopped"
            df.loc[(df[col] > 0) & (df[col] < 50), f"{turbine}_status"] = "partial_load"
            df.loc[df[col] >= 50, f"{turbine}_status"] = "generating"
            df.loc[df[wind_col] > 25, f"{turbine}_status"] = "curtailed"

    return df


def preprocess_pipeline(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    interval = config.get("data", {}).get("sampling_interval_minutes", 10)

    logger.info("Step 1: Removing duplicates...")
    df = remove_duplicates(df)

    logger.info("Step 2: Clipping physically implausible values...")
    df = clip_physically_implausible(df)

    logger.info("Step 3: Enforcing sampling interval...")
    df = enforce_sampling_interval(df, interval)

    logger.info("Step 4: Handling missing values (max_gap=12 steps = 2 hours)...")
    df = handle_missing_values(df, max_gap=12)

    logger.info("Step 5: Computing farm-level aggregates...")
    df = compute_farm_total_power(df)
    df = compute_farm_avg_wind(df)

    logger.info("Step 6: Creating missing flags...")
    df = create_missing_flags(df)

    logger.info("Step 7: Detecting operating status...")
    df = detect_operating_status(df)

    total_nulls = df.isnull().sum().sum()
    logger.info(f"Preprocessing complete: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Remaining nulls: {total_nulls}")

    return df
