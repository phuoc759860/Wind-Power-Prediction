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


def get_split_statistics(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    stats = {}
    for name, split in [("train", train), ("validation", val), ("test", test)]:
        stats[name] = {
            "rows": len(split),
            "date_start": str(split["timestamp"].min()) if "timestamp" in split.columns else "N/A",
            "date_end": str(split["timestamp"].max()) if "timestamp" in split.columns else "N/A",
        }

        power_cols = [c for c in split.columns if c.endswith("_power") and "target" not in c and "lag" not in c]
        if power_cols:
            stats[name]["avg_power"] = round(split[power_cols].mean().mean(), 2)
            stats[name]["total_energy_mwh"] = round(split[power_cols].mean().sum() * 10 / 60000, 2)

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
    step = int(total * (1 - n_folds * test_size) / n_folds) if n_folds > 1 else int(total * 0.5)
    step = max(step, 100)

    for i in range(n_folds):
        train_end = int(total * (0.3 + i * 0.15))
        val_end = int(total * (0.45 + i * 0.15))
        test_end = min(int(total * (0.6 + i * 0.15)), total)

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
