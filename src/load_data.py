import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    import yaml
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_single_file(filepath: str, timestamp_col: str = "PCTimeStamp") -> pd.DataFrame:
    from src.input_manager import load_single_file_generic
    return load_single_file_generic(filepath, timestamp_col)


def load_all_data(raw_dir: str, timestamp_col: str = "PCTimeStamp") -> pd.DataFrame:
    from src.input_manager import load_all_data_generic
    return load_all_data_generic(raw_dir, timestamp_col)


def get_data_info(df: pd.DataFrame) -> dict:
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "null_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }
    if "PCTimeStamp" in df.columns:
        info["date_range"] = {
            "start": str(df["PCTimeStamp"].min()),
            "end": str(df["PCTimeStamp"].max()),
        }
    return info


def save_processed_data(df: pd.DataFrame, output_dir: str, filename: str = "processed_data.parquet"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    if filename.endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    elif filename.endswith(".csv"):
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)

    logger.info(f"Saved processed data to {output_path}")
    return output_path


def load_processed_data(processed_dir: str, filename: str = "processed_data.parquet") -> pd.DataFrame:
    file_path = os.path.join(processed_dir, filename)

    if filename.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif filename.endswith(".csv"):
        return pd.read_csv(file_path, parse_dates=["PCTimeStamp"])
    else:
        return pd.read_parquet(file_path)
