import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataLoading:
    def test_load_single_file(self):
        from src.load_data import load_single_file
        raw_dir = Path(__file__).parent.parent / "data" / "raw"
        files = list(raw_dir.glob("*.xlsx"))
        if files:
            df = load_single_file(str(files[0]))
            assert len(df) > 0
            assert "PCTimeStamp" in df.columns

    def test_column_mapping(self):
        from src.column_mapping import build_column_mapping
        columns = [
            "PCTimeStamp",
            "TB01_Ambient WindSpeed Avg. (1)",
            "TB01_Ambient Temp. Avg. (13)",
            "TB01_Grid Production Power Avg. (25)",
            "TB01_Grid Production Frequency Avg. (37)",
        ]
        mapping = build_column_mapping(columns)
        assert mapping["PCTimeStamp"] == "timestamp"
        assert "TB01_wind_speed" in mapping.values()
        assert "TB01_power" in mapping.values()


class TestDataValidation:
    def test_validate_timestamps(self):
        from src.data_validation import validate_timestamps
        df = pd.DataFrame({
            "timestamp": pd.date_range("2021-01-01", periods=100, freq="10min"),
            "value": np.random.randn(100),
        })
        issues = validate_timestamps(df)
        assert issues.get("duplicate_timestamps", 0) == 0

    def test_validate_value_ranges(self):
        from src.data_validation import validate_value_ranges
        df = pd.DataFrame({
            "TB01_wind_speed": [5, 10, -1, 100],
            "TB01_power": [1000, 1500, 800, 1200],
        })
        issues = validate_value_ranges(df)
        assert "TB01_wind_speed" in issues


class TestFeatureEngineering:
    def test_create_lag_features(self):
        from src.feature_engineering import create_lag_features
        df = pd.DataFrame({"col1": range(20)})
        result = create_lag_features(df, ["col1"], [1, 3])
        assert "col1_lag1" in result.columns
        assert "col1_lag3" in result.columns
        assert result["col1_lag1"].iloc[0] != result["col1_lag1"].iloc[0]

    def test_create_temporal_features(self):
        from src.feature_engineering import create_temporal_features
        df = pd.DataFrame({"timestamp": pd.date_range("2021-01-01", periods=48, freq="10min")})
        result = create_temporal_features(df)
        assert "hour_of_day" in result.columns
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "hour_sin" in result.columns

    def test_create_rolling_features(self):
        from src.feature_engineering import create_rolling_features
        df = pd.DataFrame({"col1": np.random.randn(100)})
        result = create_rolling_features(df, ["col1"], [6], ["mean", "std"])
        assert "col1_roll6_mean" in result.columns
        assert "col1_roll6_std" in result.columns


class TestSplitTimeSeries:
    def test_split_by_time(self):
        from src.split_time_series import split_by_time
        df = pd.DataFrame({
            "timestamp": pd.date_range("2021-01-01", periods=1000, freq="10min"),
            "value": np.random.randn(1000),
        })
        train, val, test = split_by_time(df)
        assert len(train) > len(val)
        assert len(train) > len(test)
        assert train["timestamp"].max() < val["timestamp"].min()
        assert val["timestamp"].max() < test["timestamp"].min()


class TestMetrics:
    def test_compute_metrics(self):
        from src.evaluate import compute_metrics
        actual = np.array([100, 200, 300, 400, 500])
        predicted = np.array([110, 190, 310, 390, 510])
        metrics = compute_metrics(actual, predicted)
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert 0 < metrics["r2"] <= 1

    def test_skill_score(self):
        from src.evaluate import compute_skill_score
        ss = compute_skill_score(100, 150)
        assert 0 < ss < 1


class TestPreprocessing:
    def test_clip_physically_implausible(self):
        from src.preprocessing import clip_physically_implausible
        df = pd.DataFrame({
            "TB01_power": [-100, 500, 3000, 1000],
            "TB01_wind_speed": [-5, 10, 70, 15],
        })
        result = clip_physically_implausible(df)
        assert result["TB01_power"].min() >= 0
        assert result["TB01_power"].max() <= 2500
        assert result["TB01_wind_speed"].min() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
