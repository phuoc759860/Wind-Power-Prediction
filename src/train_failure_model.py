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


def _classify_states(df: pd.DataFrame, power_col: str) -> pd.DataFrame:
    df = df.copy()
    turbine = power_col.replace("_power", "")
    wind_col = f"{turbine}_wind_speed"
    status_col = f"{turbine}_status"
    has_status = status_col in df.columns

    def classify_row(row):
        p = row[power_col]
        if pd.isna(p):
            return "missing/unknown"
        if p > 5:
            return "generating"
        if has_status and row.get(status_col) == "curtailed":
            return "curtailed"
        if has_status and row.get(status_col) == "no_data":
            return "communication_loss"
        w = row.get(wind_col)
        if pd.notna(w) and w > 3:
            return "standby"
        return "stopped"

    df[f"{turbine}_state"] = df.apply(classify_row, axis=1)
    return df


def compute_availability(df: pd.DataFrame, power_col: str) -> Dict:
    turbine = power_col.replace("_power", "")
    df = _classify_states(df, power_col)
    state_col = f"{turbine}_state"

    total = len(df)
    state_counts = df[state_col].value_counts()
    minutes_per_sample = 10

    generating = state_counts.get("generating", 0)
    stopped = state_counts.get("stopped", 0)
    curtailed = state_counts.get("curtailed", 0)
    standby = state_counts.get("standby", 0)
    communication_loss = state_counts.get("communication_loss", 0)
    missing = state_counts.get("missing/unknown", 0)

    observed = generating + stopped + curtailed + standby
    observed_availability = generating / observed * 100 if observed > 0 else 0
    calendar_availability = generating / total * 100 if total > 0 else 0
    data_coverage = observed / total * 100 if total > 0 else 0

    return {
        "turbine": turbine,
        "total_hours": round(total * minutes_per_sample / 60, 1),
        "generating_hours": round(generating * minutes_per_sample / 60, 1),
        "stopped_hours": round(stopped * minutes_per_sample / 60, 1),
        "curtailed_hours": round(curtailed * minutes_per_sample / 60, 1),
        "standby_hours": round(standby * minutes_per_sample / 60, 1),
        "comm_loss_hours": round(communication_loss * minutes_per_sample / 60, 1),
        "missing_hours": round(missing * minutes_per_sample / 60, 1),
        "observed_availability_pct": round(observed_availability, 2),
        "calendar_availability_pct": round(calendar_availability, 2),
        "data_coverage_pct": round(data_coverage, 2),
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
