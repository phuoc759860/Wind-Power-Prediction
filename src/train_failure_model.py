import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def detect_failure_events(df: pd.DataFrame, power_col: str,
                          stop_threshold: float = 5.0,
                          min_stop_duration: int = 3) -> pd.DataFrame:
    df = df.copy()
    turbine = power_col.replace("_power", "")

    df[f"{turbine}_is_stopped"] = (df[power_col] < stop_threshold).astype(int)

    stops = df[f"{turbine}_is_stopped"].values
    change_points = np.diff(stops, prepend=0)
    start_indices = np.where(change_points == 1)[0]
    end_indices = np.where(change_points == -1)[0]

    if len(end_indices) == 0 or (len(start_indices) > 0 and start_indices[-1] > end_indices[-1]):
        end_indices = np.append(end_indices, len(stops))

    df[f"{turbine}_failure_event"] = 0
    for start, end in zip(start_indices, end_indices):
        duration = end - start
        if duration >= min_stop_duration:
            df.iloc[start:end, df.columns.get_loc(f"{turbine}_failure_event")] = 1

    n_events = 0
    in_event = False
    for val in df[f"{turbine}_failure_event"]:
        if val == 1 and not in_event:
            n_events += 1
            in_event = True
        elif val == 0:
            in_event = False

    logger.info(f"  {turbine}: {n_events} extended stop events detected")
    return df


def compute_availability(df: pd.DataFrame, power_col: str) -> Dict:
    turbine = power_col.replace("_power", "")
    total = len(df)
    generating = (df[power_col] > 5).sum()
    stopped = (df[power_col] <= 5).sum()
    missing = df[power_col].isnull().sum()

    availability = generating / (generating + stopped) * 100 if (generating + stopped) > 0 else 0

    return {
        "turbine": turbine,
        "total_hours": round(total * 10 / 60, 1),
        "generating_hours": round(generating * 10 / 60, 1),
        "stopped_hours": round(stopped * 10 / 60, 1),
        "missing_hours": round(missing * 10 / 60, 1),
        "availability_pct": round(availability, 2),
    }


def run_failure_analysis(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, Dict]:
    logger.info("Running failure/availability analysis...")

    power_cols = [c for c in df.columns if c.endswith("_power") and "target" not in c
                  and "lag" not in c and "roll" not in c and "diff" not in c
                  and "ramp" not in c and "farm_" not in c]

    availability_results = {}
    for col in power_cols:
        avail = compute_availability(df, col)
        availability_results[col] = avail

    return df, availability_results


def get_failure_summary(df: pd.DataFrame) -> Dict:
    summary = {}
    event_cols = [c for c in df.columns if c.endswith("_failure_event")]
    for col in event_cols:
        turbine = col.replace("_failure_event", "")
        events = df[col].sum()
        summary[turbine] = {
            "failure_events": int(events),
            "failure_hours": round(events * 10 / 60, 1),
        }
    return summary
