import os
import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv", ".parquet", ".json"}
MAX_UPLOAD_SIZE_MB = 500


def load_config(config_path: str = None) -> dict:
    import yaml
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_raw_dir() -> Path:
    config = load_config()
    raw_dir = config.get("data", {}).get("raw_dir", "data/raw")
    return Path(__file__).parent.parent / raw_dir


def list_input_files(raw_dir: Optional[str] = None) -> List[dict]:
    raw_path = Path(raw_dir) if raw_dir else _get_raw_dir()
    if not raw_path.exists():
        return []

    files = []
    for f in sorted(raw_path.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            stats = f.stat()
            size_mb = stats.st_size / (1024 * 1024)
            files.append({
                "filename": f.name,
                "extension": f.suffix.lower(),
                "size_mb": round(size_mb, 3),
                "modified": pd.Timestamp.fromtimestamp(stats.st_mtime).isoformat(),
            })
    return files


def add_input_file(
    source_path: str,
    target_name: Optional[str] = None,
    raw_dir: Optional[str] = None,
) -> dict:
    raw_path = Path(raw_dir) if raw_dir else _get_raw_dir()
    raw_path.mkdir(parents=True, exist_ok=True)

    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    ext = src.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    size_mb = src.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise ValueError(
            f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_UPLOAD_SIZE_MB} MB"
        )

    _validate_input_file(str(src), ext)

    dst_name = target_name or src.name
    dst_path = raw_path / dst_name

    if dst_path.exists():
        raise FileExistsError(f"File '{dst_name}' already exists in {raw_path}")

    shutil.copy2(str(src), str(dst_path))
    logger.info(f"Added input file: {dst_name} ({size_mb:.2f} MB)")

    return {
        "filename": dst_name,
        "path": str(dst_path),
        "size_mb": round(size_mb, 3),
        "extension": ext,
    }


def remove_input_file(
    filename: str,
    raw_dir: Optional[str] = None,
) -> dict:
    raw_path = Path(raw_dir) if raw_dir else _get_raw_dir()
    file_path = raw_path / filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    if not file_path.is_file():
        raise IsADirectoryError(f"Not a file: {filename}")

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{ext}'. Cannot remove.")

    size_mb = file_path.stat().st_size / (1024 * 1024)
    file_path.unlink()
    logger.info(f"Removed input file: {filename}")

    return {
        "filename": filename,
        "size_mb": round(size_mb, 3),
        "status": "removed",
    }


def load_single_file_generic(
    filepath: str,
    timestamp_col: str = "PCTimeStamp",
) -> pd.DataFrame:
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".xlsx":
        df = pd.read_excel(filepath, engine="openpyxl")
    elif ext == ".xls":
        df = pd.read_excel(filepath, engine="xlrd")
    elif ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext == ".parquet":
        df = pd.read_parquet(filepath)
    elif ext == ".json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)
    else:
        logger.warning(f"Timestamp column '{timestamp_col}' not found in {path.name}")

    logger.info(f"Loaded {path.name}: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_all_data_generic(
    raw_dir: Optional[str] = None,
    timestamp_col: Optional[str] = None,
) -> pd.DataFrame:
    raw_path = Path(raw_dir) if raw_dir else _get_raw_dir()
    config = load_config()
    if timestamp_col is None:
        timestamp_col = config.get("data", {}).get("timestamp_column", "PCTimeStamp")

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    cache_path = raw_path.parent / "processed" / "combined_raw.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    input_files = list_input_files(str(raw_path))
    if not input_files:
        raise FileNotFoundError(f"No supported input files found in {raw_path}")

    raw_times = [Path(raw_path / f["filename"]).stat().st_mtime for f in input_files]
    newest_raw = max(raw_times) if raw_times else 0

    if cache_path.exists() and cache_path.stat().st_mtime >= newest_raw:
        logger.info(f"Loading cached raw data from {cache_path}")
        combined = pd.read_parquet(cache_path)
        logger.info(f"Loaded cached dataset: {len(combined)} rows, {len(combined.columns)} columns")
        if timestamp_col in combined.columns:
            logger.info(f"Date range: {combined[timestamp_col].min()} to {combined[timestamp_col].max()}")
        return combined

    logger.info(f"Found {len(input_files)} input files in {raw_path}")
    logger.info(f"Loading {len(input_files)} files... (may take 1-2 minutes for large Excel files)")

    dfs = []
    for f_info in tqdm(input_files, desc="Loading data files", unit="file"):
        filepath = str(raw_path / f_info["filename"])
        logger.info(f"  Reading: {f_info['filename']} ({f_info['size_mb']:.1f} MB)")
        try:
            df = load_single_file_generic(filepath, timestamp_col)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {f_info['filename']}: {e}")

    if not dfs:
        raise ValueError("No data files loaded successfully")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=[timestamp_col])
    combined = combined.sort_values(timestamp_col).reset_index(drop=True)

    logger.info(
        f"Combined dataset: {len(combined)} rows, {len(combined.columns)} columns"
    )
    logger.info(
        f"Date range: {combined[timestamp_col].min()} to {combined[timestamp_col].max()}"
    )

    logger.info(f"Caching combined raw data to {cache_path}")
    combined.to_parquet(cache_path, index=False)

    return combined


def view_input_data(
    filename: Optional[str] = None,
    raw_dir: Optional[str] = None,
    nrows: int = 100,
    timestamp_col: Optional[str] = None,
) -> pd.DataFrame:
    raw_path = Path(raw_dir) if raw_dir else _get_raw_dir()
    config = load_config()
    if timestamp_col is None:
        timestamp_col = config.get("data", {}).get("timestamp_column", "PCTimeStamp")

    if filename:
        filepath = raw_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        return load_single_file_generic(str(filepath), timestamp_col).head(nrows)
    else:
        return load_all_data_generic(str(raw_path), timestamp_col).head(nrows)


def _validate_input_file(filepath: str, ext: str):
    path = Path(filepath)
    size_mb = path.stat().st_size / (1024 * 1024)

    if ext == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                if len(data) == 0:
                    raise ValueError("JSON file contains an empty array")
            elif isinstance(data, dict):
                pass
            else:
                raise ValueError("JSON file must contain an object or array")

    if ext in (".xlsx", ".xls"):
        try:
            test_df = pd.read_excel(filepath, nrows=1, engine="openpyxl" if ext == ".xlsx" else "xlrd")
            if test_df.empty:
                logger.warning(f"Excel file appears to have no rows: {path.name}")
        except Exception as e:
            raise ValueError(f"Invalid Excel file '{path.name}': {e}")

    if ext == ".csv":
        try:
            test_df = pd.read_csv(filepath, nrows=1)
            if test_df.empty:
                logger.warning(f"CSV file appears to have no rows: {path.name}")
        except Exception as e:
            raise ValueError(f"Invalid CSV file '{path.name}': {e}")

    if ext == ".parquet":
        try:
            test_df = pd.read_parquet(filepath)
            if test_df.empty:
                logger.warning(f"Parquet file appears to have no rows: {path.name}")
        except Exception as e:
            raise ValueError(f"Invalid Parquet file '{path.name}': {e}")

    logger.info(f"Validation passed: {path.name} ({size_mb:.2f} MB)")


def edit_input_data(
    updates: List[Dict[str, Union[str, float]]],
    filename: Optional[str] = None,
    raw_dir: Optional[str] = None,
    save_copy: bool = False,
    timestamp_col: Optional[str] = None,
) -> dict:
    raw_path = Path(raw_dir) if raw_dir else _get_raw_dir()
    config = load_config()
    if timestamp_col is None:
        timestamp_col = config.get("data", {}).get("timestamp_column", "PCTimeStamp")

    df = load_all_data_generic(str(raw_path), timestamp_col)

    edit_log = {
        "total_rows_before": len(df),
        "updates_applied": 0,
        "updates_failed": 0,
        "errors": [],
        "modified_columns": set(),
        "file_saved": None,
    }

    for update in updates:
        try:
            condition_col = update.get("condition_column", timestamp_col)
            condition_value = update.get("condition_value")
            target_column = update.get("target_column")
            new_value = update.get("new_value")

            if not condition_value or not target_column:
                edit_log["updates_failed"] += 1
                edit_log["errors"].append(f"Missing condition_value or target_column in update: {update}")
                continue

            if target_column not in df.columns:
                edit_log["updates_failed"] += 1
                edit_log["errors"].append(f"Column '{target_column}' not found in data")
                continue

            if condition_col not in df.columns:
                edit_log["updates_failed"] += 1
                edit_log["errors"].append(f"Condition column '{condition_col}' not found")
                continue

            if condition_col == timestamp_col:
                condition_value = pd.to_datetime(condition_value, errors="coerce")
                if pd.isna(condition_value):
                    edit_log["updates_failed"] += 1
                    edit_log["errors"].append(f"Invalid timestamp condition value: {condition_value}")
                    continue

            mask = df[condition_col] == condition_value
            count = mask.sum()

            if count == 0:
                edit_log["updates_failed"] += 1
                edit_log["errors"].append(
                    f"No rows matched condition: {condition_col}={condition_value}"
                )
                continue

            df.loc[mask, target_column] = new_value
            edit_log["updates_applied"] += 1
            edit_log["modified_columns"].add(target_column)
            logger.info(f"Edited {count} row(s): set {target_column} = {new_value} where {condition_col}={condition_value}")

        except Exception as e:
            edit_log["updates_failed"] += 1
            edit_log["errors"].append(str(e))

    edit_log["modified_columns"] = list(edit_log["modified_columns"])
    edit_log["total_rows_after"] = len(df)

    if save_copy or filename:
        out_path = raw_path / (filename or "edited_data.csv")
        if out_path.suffix.lower() == ".csv":
            df.to_csv(str(out_path), index=False)
        elif out_path.suffix.lower() == ".parquet":
            df.to_parquet(str(out_path), index=False)
        elif out_path.suffix.lower() in (".xlsx", ".xls"):
            df.to_excel(str(out_path), index=False, engine="openpyxl")
        else:
            out_path = out_path.with_suffix(".csv")
            df.to_csv(str(out_path), index=False)
        edit_log["file_saved"] = str(out_path)
        logger.info(f"Edited data saved to: {out_path}")

    return edit_log


def get_data_summary(raw_dir: Optional[str] = None) -> dict:
    raw_path = Path(raw_dir) if raw_dir else _get_raw_dir()

    files = list_input_files(str(raw_path))
    total_size_mb = sum(f["size_mb"] for f in files)

    summary = {
        "total_files": len(files),
        "total_size_mb": round(total_size_mb, 3),
        "files": files,
    }

    if files:
        try:
            df = load_all_data_generic(str(raw_path))
            summary["data_shape"] = list(df.shape)
            summary["columns"] = list(df.columns)
            summary["dtypes"] = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
            summary["missing_cells"] = int(df.isnull().sum().sum())
        except Exception as e:
            summary["load_error"] = str(e)

    return summary
