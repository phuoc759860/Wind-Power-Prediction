from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_MASK_COLUMNS = (
    "observed_target",
    "target_imputed",
    "official_cutoff",
    "prediction_available",
    "feature_available",
)


def _ensure_bool_series(mask: pd.Series | np.ndarray | Iterable[bool], length: int) -> pd.Series:
    s = pd.Series(mask)
    if len(s) != length:
        raise ValueError(f"Official mask length mismatch: expected {length}, got {len(s)}")
    return s.astype(bool)


def build_official_mask(df: pd.DataFrame) -> pd.Series:
    """Build the single canonical evaluation mask.

    The reviewer requires all figures and metrics to use ONLY rows satisfying:
    observed_target & (~target_imputed) & (timestamp < official_cutoff) &
    prediction_available & feature_available.
    """
    if df is None or df.empty:
        raise ValueError("build_official_mask requires a non-empty DataFrame")

    missing = [c for c in REQUIRED_MASK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Official mask requires columns: {missing}")

    observed = df["observed_target"].fillna(0).astype(bool)
    target_imputed = df["target_imputed"].fillna(0).astype(bool)
    prediction_available = df["prediction_available"].fillna(0).astype(bool)
    feature_available = df["feature_available"].fillna(0).astype(bool)

    if "timestamp" in df.columns and "official_cutoff" in df.columns:
        cutoff = pd.to_datetime(df["official_cutoff"]).iloc[0]
        ts = pd.to_datetime(df["timestamp"])
        timestamp_ok = ts < cutoff
    else:
        timestamp_ok = pd.Series(True, index=df.index)

    mask = observed & (~target_imputed) & timestamp_ok & prediction_available & feature_available
    return _ensure_bool_series(mask, len(df))


def save_sample_trace(df: pd.DataFrame, mask: pd.Series | np.ndarray | Iterable[bool], output_path: str | Path = "sample_trace.csv") -> pd.DataFrame:
    """Persist the reviewer-visible row inclusion/exclusion trace.

    The saved file is intentionally small and deterministic so the reviewer can
    reproduce exactly why rows were kept or dropped from the official sample set.
    """
    official_mask = _ensure_bool_series(mask, len(df))
    trace = df.copy()
    trace["official_mask_keep"] = official_mask.astype(int)
    trace["mask_observed_target"] = trace["observed_target"].fillna(0).astype(bool)
    trace["mask_target_not_imputed"] = ~(trace["target_imputed"].fillna(0).astype(bool))
    trace["mask_before_cutoff"] = pd.to_datetime(trace["timestamp"]) < pd.to_datetime(trace["official_cutoff"])
    trace["mask_prediction_available"] = trace["prediction_available"].fillna(0).astype(bool)
    trace["mask_feature_available"] = trace["feature_available"].fillna(0).astype(bool)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace.loc[official_mask].to_csv(out_path, index=False)
    return trace.loc[official_mask].copy()
