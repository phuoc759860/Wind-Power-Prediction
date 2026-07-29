import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

ORIGINAL_COLUMNS = {
    "PCTimeStamp": "timestamp",
}

COLUMN_PATTERNS = {
    "Ambient WindSpeed Avg.": "wind_speed",
    "Ambient Temp. Avg.": "temperature",
    "Grid Production Power Avg.": "power",
    "Grid Production Frequency Avg.": "frequency",
}


def extract_turbine_id(col_name: str) -> str:
    if col_name.startswith("TB") and "_" in col_name:
        return col_name.split("_")[0]
    return ""


def extract_measurement_type(col_name: str) -> str:
    for pattern, mtype in COLUMN_PATTERNS.items():
        if pattern in col_name:
            return mtype
    return "unknown"


def extract_numeric_id(col_name: str) -> int:
    parts = col_name.split("(")
    if len(parts) > 1:
        num_str = parts[-1].replace(")", "").strip()
        try:
            return int(num_str)
        except ValueError:
            pass
    return 0


def build_column_mapping(columns: List[str]) -> Dict[str, str]:
    mapping = {}
    for col in columns:
        if col in ORIGINAL_COLUMNS:
            mapping[col] = ORIGINAL_COLUMNS[col]
            continue

        turbine = extract_turbine_id(col)
        mtype = extract_measurement_type(col)

        if turbine and mtype != "unknown":
            std_name = f"{turbine}_{mtype}"
            if std_name in mapping:
                existing_id = extract_numeric_id(col)
                existing_col = [k for k, v in mapping.items() if v == std_name]
                if existing_col:
                    old_id = extract_numeric_id(existing_col[0])
                    if existing_id > old_id:
                        mapping[col] = std_name
            else:
                mapping[col] = std_name
        else:
            safe_name = col.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")
            mapping[col] = safe_name.lower()

    return mapping


def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, str] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if mapping is None:
        mapping = build_column_mapping(df.columns)

    logger.info(f"Column mapping created: {len(mapping)} columns mapped")

    rename_map = {orig: std for orig, std in mapping.items() if orig != std}
    df = df.rename(columns=rename_map)

    return df, mapping


def get_turbine_columns(df: pd.DataFrame, turbine_id: str, measurement: str) -> List[str]:
    col_name = f"{turbine_id}_{measurement}"
    return [col_name] if col_name in df.columns else []


def get_all_turbine_measurements(df: pd.DataFrame, measurement: str) -> Dict[str, str]:
    result = {}
    for col in df.columns:
        if col.endswith(f"_{measurement}"):
            turbine_id = col.replace(f"_{measurement}", "")
            result[turbine_id] = col
    return result


def get_power_columns(df: pd.DataFrame) -> Dict[str, str]:
    return get_all_turbine_measurements(df, "power")


def get_wind_speed_columns(df: pd.DataFrame) -> Dict[str, str]:
    return get_all_turbine_measurements(df, "wind_speed")


def get_temperature_columns(df: pd.DataFrame) -> Dict[str, str]:
    return get_all_turbine_measurements(df, "temperature")


def create_data_dictionary(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for orig_col, std_name in mapping.items():
        dtype = str(df[orig_col].dtype) if orig_col in df.columns else "unknown"
        null_count = int(df[orig_col].isnull().sum()) if orig_col in df.columns else 0
        null_pct = round(df[orig_col].isnull().mean() * 100, 2) if orig_col in df.columns else 0

        turbine = extract_turbine_id(orig_col)
        mtype = extract_measurement_type(orig_col)

        unit_map = {
            "wind_speed": "m/s",
            "temperature": "°C",
            "power": "kW",
            "frequency": "Hz",
        }

        rows.append({
            "original_column": orig_col,
            "standardized_name": std_name,
            "turbine_id": turbine,
            "measurement_type": mtype,
            "unit": unit_map.get(mtype, "unknown"),
            "data_type": dtype,
            "null_count": null_count,
            "null_pct": null_pct,
            "role": "target" if mtype == "power" else ("feature" if mtype in ["wind_speed", "temperature"] else "auxiliary"),
        })

    return pd.DataFrame(rows)
