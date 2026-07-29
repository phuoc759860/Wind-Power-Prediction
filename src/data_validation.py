import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

VALID_VALUE_RANGES = {
    "wind_speed": (0, 60),
    "temperature": (-30, 55),
    "power": (0, 2500),
    "frequency": (45, 55),
}


def validate_timestamps(df: pd.DataFrame, timestamp_col: str = "timestamp") -> Dict:
    issues = {}

    if timestamp_col not in df.columns:
        issues["missing_timestamp_column"] = True
        return issues

    null_ts = df[timestamp_col].isnull().sum()
    issues["null_timestamps"] = int(null_ts)

    if len(df) > 1:
        diffs = df[timestamp_col].diff().dt.total_seconds() / 60
        expected_interval = 10
        unusual_intervals = diffs[(diffs != expected_interval) & (diffs.notna())]
        issues["unusual_intervals_count"] = int(len(unusual_intervals))

        if len(unusual_intervals) > 0:
            issues["interval_stats"] = {
                "min": float(diffs.min()),
                "max": float(diffs.max()),
                "mean": float(diffs.mean()),
            }

    duplicates = df[timestamp_col].duplicated().sum()
    issues["duplicate_timestamps"] = int(duplicates)

    return issues


def validate_value_ranges(df: pd.DataFrame, column_groups: Dict = None) -> Dict:
    issues = {}

    for col in df.columns:
        if col == "timestamp":
            continue

        measurement_type = None
        for mtype, (vmin, vmax) in VALID_VALUE_RANGES.items():
            if col.endswith(f"_{mtype}"):
                measurement_type = mtype
                break

        if measurement_type is None:
            continue

        vmin, vmax = VALID_VALUE_RANGES[measurement_type]
        col_data = df[col].dropna()

        if len(col_data) == 0:
            continue

        out_of_range = ((col_data < vmin) | (col_data > vmax)).sum()
        if out_of_range > 0:
            issues[col] = {
                "out_of_range_count": int(out_of_range),
                "out_of_range_pct": round(out_of_range / len(col_data) * 100, 2),
                "min_value": float(col_data.min()),
                "max_value": float(col_data.max()),
                "expected_range": [vmin, vmax],
            }

    return issues


def validate_missing_data(df: pd.DataFrame) -> Dict:
    issues = {}
    total_rows = len(df)

    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            issues[col] = {
                "null_count": int(null_count),
                "null_pct": round(null_count / total_rows * 100, 2),
            }

    return issues


def validate_power_consistency(df: pd.DataFrame) -> Dict:
    issues = {}
    power_cols = [c for c in df.columns if c.endswith("_power")]

    for col in power_cols:
        data = df[col].dropna()
        if len(data) == 0:
            continue

        negative_power = (data < 0).sum()
        if negative_power > 0:
            issues[f"{col}_negative"] = int(negative_power)

        turbine = col.replace("_power", "")
        wind_col = f"{turbine}_wind_speed"
        if wind_col in df.columns:
            mask = (df[wind_col] > 25) & (df[col] > 100)
            high_power_high_wind = mask.sum()
            if high_power_high_wind > 0:
                issues[f"{col}_high_wind_anomaly"] = int(high_power_high_wind)

    return issues


def run_validation(df: pd.DataFrame) -> Dict:
    logger.info("Starting data validation...")

    results = {
        "timestamp_issues": validate_timestamps(df),
        "value_range_issues": validate_value_ranges(df),
        "missing_data_issues": validate_missing_data(df),
        "power_consistency_issues": validate_power_consistency(df),
    }

    total_issues = sum(len(v) for v in results.values())
    logger.info(f"Validation complete: {total_issues} issue categories found")

    for category, issues in results.items():
        if issues:
            logger.warning(f"  {category}: {len(issues)} issues")

    return results
