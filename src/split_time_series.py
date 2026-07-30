import logging
import pandas as pd
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)


def split_by_time(df: pd.DataFrame, timestamp_col: str = "timestamp",
                  train_ratio: float = 0.7, val_ratio: float = 0.15,
                  test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    total = len(df)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    train["data_split"] = "train"
    val["data_split"] = "validation"
    test["data_split"] = "test"

    logger.info(f"Time-based split:")
    logger.info(f"  Train: {len(train)} rows ({train[timestamp_col].min()} to {train[timestamp_col].max()})")
    logger.info(f"  Val:   {len(val)} rows ({val[timestamp_col].min()} to {val[timestamp_col].max()})")
    logger.info(f"  Test:  {len(test)} rows ({test[timestamp_col].min()} to {test[timestamp_col].max()})")

    return train, val, test


def get_split_statistics(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                         timestamp_col: str = "timestamp", interval_minutes: int = 10) -> dict:
    stats = {}
    all_combined = pd.concat([train, val, test]).sort_values(timestamp_col).reset_index(drop=True)

    for name, split in [("train", train), ("validation", val), ("test", test), ("total", all_combined)]:
        if timestamp_col not in split.columns:
            continue
        ts = pd.to_datetime(split[timestamp_col])
        start, end = ts.min(), ts.max()
        expected_steps = int((end - start).total_seconds() / 60 / interval_minutes) + 1
        actual_steps = len(split)
        n_duplicates = split.duplicated(subset=[timestamp_col]).sum()

        stats[name] = {
            "rows": actual_steps,
            "timestamp_start": str(start),
            "timestamp_end": str(end),
            "expected_steps": expected_steps,
            "actual_steps": actual_steps,
            "step_diff": actual_steps - expected_steps,
            "n_duplicate_timestamps": int(n_duplicates),
            "n_missing_timestamps": max(0, expected_steps - actual_steps + n_duplicates),
            "timezone": str(ts.dt.tz) if ts.dt.tz is not None else "None (naive)",
        }

        power_cols = [c for c in split.columns if c.endswith("_power") and "target" not in c and "lag" not in c]
        if power_cols:
            stats[name]["avg_power"] = round(split[power_cols].mean().mean(), 2)
            stats[name]["total_energy_mwh"] = round(split[power_cols].mean().sum() * interval_minutes / 60000, 2)

    return stats


def add_time_index(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df["time_index"] = range(len(df))
    return df


def walk_forward_split(df: pd.DataFrame, n_folds: int = 5,
                       val_size: float = 0.15, timestamp_col: str = "timestamp"):
    df = df.copy().sort_values(timestamp_col).reset_index(drop=True)
    total = len(df)
    folds = []
    test_size = val_size
    train_start_ratio = 0.3
    step_ratio = 0.15

    for i in range(n_folds):
        train_end = int(total * (train_start_ratio + i * step_ratio))
        val_end = int(total * (train_start_ratio + step_ratio + i * step_ratio))
        test_end = min(int(total * (train_start_ratio + 2 * step_ratio + i * step_ratio)), total)

        train_end = max(train_end, 100)
        val_end = max(val_end, train_end + 1)
        test_end = max(test_end, val_end + 1)

        if test_end > total:
            break

        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:test_end].copy()

        if len(train) < 100 or len(val) < 10 or len(test) < 10:
            continue

        train["data_split"] = "train"
        val["data_split"] = "validation"
        test["data_split"] = "test"

        folds.append({
            "fold": i + 1,
            "train": train,
            "val": val,
            "test": test,
            "dates": {
                "train": f"{train[timestamp_col].min()} to {train[timestamp_col].max()}",
                "val": f"{val[timestamp_col].min()} to {val[timestamp_col].max()}",
                "test": f"{test[timestamp_col].min()} to {test[timestamp_col].max()}",
            }
        })

    logger.info(f"Walk-forward validation: {len(folds)} folds")
    return folds
