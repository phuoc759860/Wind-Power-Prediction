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
    """Coverage statistics computed on the OBSERVED timestamps only.

    The raw source files have gaps, so expected_steps is derived from the
    observed min/max, and n_missing_timestamps reports the true holes.
    These numbers are NOT inflated by any padding that happened upstream.
    """
    stats = {}
    all_combined = pd.concat([train, val, test]).sort_values(timestamp_col).reset_index(drop=True)

    for name, split in [("train", train), ("validation", val), ("test", test), ("total", all_combined)]:
        if timestamp_col not in split.columns:
            continue
        ts = pd.to_datetime(split[timestamp_col])
        start, end = ts.min(), ts.max()
        n_rows = len(split)
        n_duplicates = int(split.duplicated(subset=[timestamp_col]).sum())
        ts_unique = ts.drop_duplicates()
        n_unique = len(ts_unique)

        expected_steps = int((end - start).total_seconds() / 60 / interval_minutes) + 1
        n_missing = max(0, expected_steps - n_unique)

        stats[name] = {
            "rows": n_rows,
            "timestamp_start": str(start),
            "timestamp_end": str(end),
            "expected_steps": expected_steps,
            "actual_steps": n_unique,
            "unique_timestamps": n_unique,
            "n_duplicate_timestamps": n_duplicates,
            "n_missing_timestamps": n_missing,
            "coverage_ratio": round(n_unique / expected_steps, 6) if expected_steps else None,
            "timezone": str(ts.dt.tz) if ts.dt.tz is not None else "None (naive)",
        }

        power_cols = [c for c in split.columns if c.endswith("_power") and "target" not in c and "lag" not in c]
        if power_cols:
            stats[name]["avg_power"] = round(split[power_cols].mean().mean(), 2)
            stats[name]["total_energy_mwh"] = round(split[power_cols].mean().sum() * interval_minutes / 60000, 2)

        # P0-05: rows must never be conflated with observed timestamps. After the
        # 10-min reindex the split contains synthetic (reindexed) rows and rows
        # whose values were forward-filled; report observed vs synthetic vs
        # imputed per split so coverage claims use observed counts only.
        if "is_observed" in split.columns:
            is_obs = split["is_observed"].fillna(0).astype(int)
            is_syn = split["is_synthetic"].fillna(0).astype(int) if "is_synthetic" in split.columns else (1 - is_obs)
            is_imp = split["is_imputed"].fillna(0).astype(int) if "is_imputed" in split.columns else pd.Series(0, index=split.index)
            stats[name]["n_observed_rows"] = int(is_obs.sum())
            stats[name]["n_synthetic_rows"] = int(is_syn.sum())
            stats[name]["n_imputed_rows"] = int(is_imp.sum())
            stats[name]["n_observed_not_imputed_rows"] = int(((is_obs == 1) & (is_imp == 0)).sum())
            stats[name]["observed_ratio"] = round(float(is_obs.mean()), 6) if len(split) else None

    return stats


def horizon_sample_counts(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                          config: dict) -> dict:
    """Per-horizon number of rows with a VALID (non-NaN) target P(t+h).

    A row whose target is NaN cannot be used to train or score that horizon.
    Rows close to the end of the data always have NaN targets for long horizons,
    so the effective sample count per split is horizon-dependent. This is the
    honest denominator that must drive reported metrics.
    """
    horizons = config.get("forecasting", {}).get("horizons", [])
    power_cols = [c for c in test.columns if c.endswith("_power") and "target" not in c and "lag" not in c]

    counts = {}
    for split_name, split in [("train", train), ("validation", val), ("test", test)]:
        per_split = {}
        for h in horizons:
            name = h["name"]
            targets = [f"{c}_target_{name}" for c in power_cols if f"{c}_target_{name}" in split.columns]
            if not targets:
                continue
            per_split[name] = {
                "n_rows": int(len(split)),
                "n_valid_targets": int(split[targets[0]].notna().sum()),
                "n_invalid_targets": int(split[targets[0]].isna().sum()),
                "ratio_valid": round(float(split[targets[0]].notna().mean()), 6),
            }
        counts[split_name] = per_split
    return counts


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

    logger.info(f"Walk-forward validation: {len(folds)} of {n_folds} requested folds produced")
    return folds
