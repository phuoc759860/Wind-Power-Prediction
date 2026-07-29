import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from src.input_manager import (
    list_input_files,
    add_input_file,
    remove_input_file,
    load_single_file_generic,
    load_all_data_generic,
    view_input_data,
    edit_input_data,
    get_data_summary,
    SUPPORTED_EXTENSIONS,
)


@pytest.fixture
def temp_raw_dir(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    return raw


@pytest.fixture
def sample_csv(temp_raw_dir):
    path = temp_raw_dir / "test_data.csv"
    df = pd.DataFrame({
        "PCTimeStamp": pd.date_range("2024-01-01", periods=5, freq="10min"),
        "wind_speed": [5.0, 6.0, 7.0, 8.0, 9.0],
        "power": [1000, 1200, 1400, 1600, 1800],
    })
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_excel(temp_raw_dir):
    path = temp_raw_dir / "test_data.xlsx"
    df = pd.DataFrame({
        "PCTimeStamp": pd.date_range("2024-01-02", periods=3, freq="10min"),
        "wind_speed": [4.0, 5.0, 6.0],
        "power": [800, 1000, 1200],
    })
    df.to_excel(path, index=False, engine="openpyxl")
    return path


@pytest.fixture
def sample_json(temp_raw_dir):
    path = temp_raw_dir / "test_data.json"
    data = [
        {"PCTimeStamp": "2024-01-01 00:00:00", "wind_speed": 7.5, "power": 1500},
        {"PCTimeStamp": "2024-01-01 00:10:00", "wind_speed": 8.0, "power": 1600},
    ]
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestListInputFiles:
    def test_empty_directory(self, temp_raw_dir):
        files = list_input_files(str(temp_raw_dir))
        assert files == []

    def test_lists_csv_files(self, temp_raw_dir, sample_csv):
        files = list_input_files(str(temp_raw_dir))
        assert len(files) == 1
        assert files[0]["filename"] == "test_data.csv"
        assert files[0]["extension"] == ".csv"

    def test_lists_multiple_formats(self, temp_raw_dir, sample_csv, sample_excel):
        files = list_input_files(str(temp_raw_dir))
        assert len(files) == 2
        exts = {f["extension"] for f in files}
        assert exts == {".csv", ".xlsx"}


class TestAddInputFile:
    def test_adds_csv_file(self, temp_raw_dir, tmp_path):
        src = tmp_path / "source.csv"
        pd.DataFrame({"PCTimeStamp": ["2024-01-01"], "wind_speed": [5.0]}).to_csv(src, index=False)

        result = add_input_file(str(src), raw_dir=str(temp_raw_dir))
        assert result["filename"] == "source.csv"
        assert result["extension"] == ".csv"
        assert (temp_raw_dir / "source.csv").exists()

    def test_adds_with_custom_name(self, temp_raw_dir, tmp_path):
        src = tmp_path / "source.csv"
        pd.DataFrame({"PCTimeStamp": ["2024-01-01"], "wind_speed": [5.0]}).to_csv(src, index=False)

        result = add_input_file(str(src), target_name="custom.csv", raw_dir=str(temp_raw_dir))
        assert result["filename"] == "custom.csv"

    def test_raises_on_duplicate(self, temp_raw_dir, tmp_path):
        src = tmp_path / "dup.csv"
        pd.DataFrame({"PCTimeStamp": ["2024-01-01"], "wind_speed": [5.0]}).to_csv(src, index=False)

        add_input_file(str(src), raw_dir=str(temp_raw_dir))
        with pytest.raises(FileExistsError):
            add_input_file(str(src), raw_dir=str(temp_raw_dir))

    def test_raises_on_unsupported_format(self, temp_raw_dir, tmp_path):
        src = tmp_path / "test.txt"
        src.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            add_input_file(str(src), raw_dir=str(temp_raw_dir))

    def test_raises_on_missing_source(self, temp_raw_dir):
        with pytest.raises(FileNotFoundError):
            add_input_file("/nonexistent/file.csv", raw_dir=str(temp_raw_dir))


class TestRemoveInputFile:
    def test_removes_file(self, temp_raw_dir, sample_csv):
        result = remove_input_file("test_data.csv", raw_dir=str(temp_raw_dir))
        assert result["status"] == "removed"
        assert not (temp_raw_dir / "test_data.csv").exists()

    def test_raises_on_missing(self, temp_raw_dir):
        with pytest.raises(FileNotFoundError):
            remove_input_file("nonexistent.csv", raw_dir=str(temp_raw_dir))


class TestLoadSingleFileGeneric:
    def test_loads_csv(self, temp_raw_dir, sample_csv):
        df = load_single_file_generic(str(sample_csv))
        assert len(df) == 5
        assert "PCTimeStamp" in df.columns
        assert "wind_speed" in df.columns

    def test_loads_excel(self, temp_raw_dir, sample_excel):
        df = load_single_file_generic(str(sample_excel))
        assert len(df) == 3
        assert "PCTimeStamp" in df.columns

    def test_loads_json(self, temp_raw_dir, sample_json):
        df = load_single_file_generic(str(sample_json))
        assert len(df) == 2
        assert "wind_speed" in df.columns


class TestLoadAllDataGeneric:
    def test_loads_all(self, temp_raw_dir, sample_csv, sample_excel):
        df = load_all_data_generic(str(temp_raw_dir), timestamp_col="PCTimeStamp")
        assert len(df) == 8
        assert "PCTimeStamp" in df.columns

    def test_empty_directory_raises(self, temp_raw_dir):
        with pytest.raises(FileNotFoundError):
            load_all_data_generic(str(temp_raw_dir))


class TestViewInputData:
    def test_views_specific_file(self, temp_raw_dir, sample_csv):
        df = view_input_data("test_data.csv", raw_dir=str(temp_raw_dir))
        assert len(df) == 5


class TestEditInputData:
    def test_edits_value_by_timestamp(self, temp_raw_dir, sample_csv):
        df_before = pd.read_csv(sample_csv)
        original_power = df_before.loc[0, "power"]

        result = edit_input_data(
            updates=[{
                "condition_column": "PCTimeStamp",
                "condition_value": "2024-01-01 00:00:00",
                "target_column": "power",
                "new_value": 9999,
            }],
            raw_dir=str(temp_raw_dir),
        )
        assert result["updates_applied"] == 1
        assert result["updates_failed"] == 0

    def test_edit_fails_on_bad_column(self, temp_raw_dir, sample_csv):
        result = edit_input_data(
            updates=[{
                "condition_column": "PCTimeStamp",
                "condition_value": "2024-01-01 00:00:00",
                "target_column": "nonexistent_column",
                "new_value": 100,
            }],
            raw_dir=str(temp_raw_dir),
        )
        assert result["updates_applied"] == 0
        assert result["updates_failed"] >= 1

    def test_saves_copy(self, temp_raw_dir, sample_csv):
        result = edit_input_data(
            updates=[{
                "condition_column": "PCTimeStamp",
                "condition_value": "2024-01-01 00:00:00",
                "target_column": "power",
                "new_value": 5555,
            }],
            raw_dir=str(temp_raw_dir),
            save_copy=True,
            filename="edited_copy.csv",
        )
        assert result["updates_applied"] == 1
        assert result["file_saved"] is not None
        assert Path(result["file_saved"]).exists()


class TestGetDataSummary:
    def test_empty_directory(self, temp_raw_dir):
        summary = get_data_summary(str(temp_raw_dir))
        assert summary["total_files"] == 0

    def test_with_files(self, temp_raw_dir, sample_csv, sample_excel):
        summary = get_data_summary(str(temp_raw_dir))
        assert summary["total_files"] == 2
        assert summary["total_size_mb"] > 0
        assert "data_shape" in summary
        assert summary["data_shape"][1] >= 3



