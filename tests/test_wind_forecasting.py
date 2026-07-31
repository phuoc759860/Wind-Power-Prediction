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


class TestNoLookaheadLeakage:
    """
    Doc 4.9: Proves that every feature at time t depends only on data ≤ t.
    Uses a monotonic value series so any lookahead is trivially detectable.
    """

    N = 500

    def _make_monotonic(self):
        """Series where value at i = i*10. Any future-referencing feature will have value > current."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=self.N, freq="10min"),
            "TB01_power": np.arange(self.N, dtype=float) * 10,
            "TB01_wind_speed": np.arange(self.N, dtype=float) * 0.5,
            "TB01_temperature": 25 + np.sin(np.arange(self.N) * 0.1) * 3,
        })
        return df

    def test_lag_uses_only_past(self):
        """lag(t) = value(t - lag_steps), never references t+1."""
        from src.feature_engineering import create_lag_features

        df = self._make_monotonic()
        df = create_lag_features(df, ["TB01_power"], [1, 3, 6])

        # At row i, lag1 should equal power at i-1
        for i in range(7, self.N):
            expected = df["TB01_power"].iloc[i - 1]
            assert df["TB01_power_lag1"].iloc[i] == pytest.approx(expected, abs=1e-6), \
                f"Row {i}: lag1={df['TB01_power_lag1'].iloc[i]} != power[{i-1}]={expected}"

        for lag in [3, 6]:
            col = f"TB01_power_lag{lag}"
            for i in range(lag + 1, self.N):
                expected = df["TB01_power"].iloc[i - lag]
                assert df[col].iloc[i] == pytest.approx(expected, abs=1e-6), \
                    f"Row {i}: {col}={df[col].iloc[i]} != power[{i-lag}]={expected}"

    def test_rolling_after_shift_uses_only_past(self):
        """rolling at t is computed on shift(1), so it uses only rows < t."""
        from src.feature_engineering import create_rolling_features

        df = self._make_monotonic()
        df = create_rolling_features(df, ["TB01_power"], [6], ["mean", "std"])

        col = "TB01_power_roll6_mean"
        # At row 6, the rolling window is [0,1,2,3,4,5] (after shift(1))
        # Mean should be mean of power[0]..power[5] = (0+10+20+30+40+50)/6 = 150/6 = 25
        expected_r6 = np.mean([df["TB01_power"].iloc[j] for j in range(0, 6)])
        assert df[col].iloc[6] == pytest.approx(expected_r6, abs=1e-6), \
            f"Row 6 roll6_mean={df[col].iloc[6]} != expected={expected_r6}"

        # Verify row i never uses power[i] in its rolling mean
        for i in range(7, self.N):
            window_start = max(0, i - 6)
            expected = np.mean([df["TB01_power"].iloc[j] for j in range(window_start, i)])
            assert df[col].iloc[i] == pytest.approx(expected, abs=1e-4), \
                f"Row {i}: roll6_mean={df[col].iloc[i]} uses power up to i={i} (value={df['TB01_power'].iloc[i]})"

    def test_diff_uses_only_current_and_past(self):
        """diff(t) = power(t) - power(t-1). Uses t and t-1, never t+1."""
        from src.feature_engineering import create_change_features

        df = self._make_monotonic()
        df = create_change_features(df, ["TB01_power"])

        for i in range(2, self.N):
            expected = df["TB01_power"].iloc[i] - df["TB01_power"].iloc[i - 1]
            assert df["TB01_power_diff1"].iloc[i] == pytest.approx(expected, abs=1e-6), \
                f"Row {i}: diff1={df['TB01_power_diff1'].iloc[i]} != power[i]-power[i-1]={expected}"

    def test_ramp_uses_only_current_and_past(self):
        """ramp = diff(power)/10, uses only t and t-1. Verify value, not flag."""
        from src.feature_engineering import create_ramp_features

        df = self._make_monotonic()
        df = create_ramp_features(df, ["TB01_power"])

        for i in range(2, self.N):
            expected_rate = (df["TB01_power"].iloc[i] - df["TB01_power"].iloc[i - 1]) / 10.0
            assert df["TB01_power_ramp"].iloc[i] == pytest.approx(expected_rate, abs=1e-6), \
                f"Row {i}: ramp rate incorrect"

        # Verify no lookahead: ramp at row 0 should be NaN (no prior value)
        assert np.isnan(df["TB01_power_ramp"].iloc[0]), "Row 0 ramp should be NaN (no diff available)"

    def test_target_is_future(self):
        """target(t) = power(t + horizon_steps) — correct: this is the dependent variable."""
        from src.feature_engineering import create_target_columns

        df = self._make_monotonic()
        horizons = [{"name": "10min", "steps": 1}, {"name": "30min", "steps": 3}]
        df = create_target_columns(df, horizons)

        assert "TB01_power_target_10min" in df.columns
        for i in range(self.N - 1):
            expected = df["TB01_power"].iloc[i + 1]
            assert df["TB01_power_target_10min"].iloc[i] == pytest.approx(expected, abs=1e-6), \
                f"Row {i}: target_10min={df['TB01_power_target_10min'].iloc[i]} != power[{i+1}]={expected}"

    def test_preprocessing_ffill_no_lookahead(self):
        """handle_missing_values with ffill never uses future data."""
        from src.preprocessing import handle_missing_values

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="10min"),
            "TB01_power": [100.0, np.nan, np.nan, 400.0, 500.0, np.nan, 700.0, 800.0, 900.0, 1000.0],
        })
        # After ffill(limit=12): rows 1,2 filled from row 0 → 100; row 5 filled from row 4 → 500
        result = handle_missing_values(df, max_gap=12)
        filled = result["TB01_power"]
        assert filled.iloc[1] == pytest.approx(100.0), "Row 1 should be forward-filled from row 0"
        assert filled.iloc[2] == pytest.approx(100.0), "Row 2 should be forward-filled from row 0"
        assert filled.iloc[5] == pytest.approx(500.0), "Row 5 filled from past row 4 (500.0), no lookahead"

    def test_preprocessing_ffill_respects_long_gap(self):
        """Gaps larger than max_gap remain NaN (not filled from distant future)."""
        from src.preprocessing import handle_missing_values

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=15, freq="10min"),
            "TB01_power": [100.0] + [np.nan] * 13 + [900.0],
        })
        # 13 consecutive NaN with max_gap=12 → one value is beyond limit
        result = handle_missing_values(df, max_gap=12)
        filled = result["TB01_power"]
        # First 12 NaN rows get filled from row 0
        for i in range(1, 13):
            assert filled.iloc[i] == pytest.approx(100.0), f"Row {i} should be forward-filled (gap ≤ 12)"
        # The 13th NaN (row 13) is beyond limit, stays NaN
        assert np.isnan(filled.iloc[13]), "Row 13 has gap > 12, should stay NaN"


class TestComplianceMatrix:
    """Doc 4.16: Verify the traceability/compliance matrix exists and covers all 4.x requirements."""

    REQUIRED_IDS = {f"4.{i}" for i in range(1, 11)}

    def test_matrix_file_exists(self):
        path = Path(__file__).parent.parent / "configs" / "compliance_matrix.csv"
        assert path.exists(), "compliance_matrix.csv must exist"

    def test_all_requirements_covered(self):
        path = Path(__file__).parent.parent / "configs" / "compliance_matrix.csv"
        df = pd.read_csv(path, dtype=str)
        import re
        covered = set()
        for val in df["requirement_id"].dropna():
            m = re.match(r"(\d+\.\d+)", str(val).strip())
            if m:
                covered.add(m.group(1))
        missing = self.REQUIRED_IDS - covered
        assert not missing, f"Missing requirements in matrix: {sorted(missing)}"

    def test_each_row_has_status(self):
        path = Path(__file__).parent.parent / "configs" / "compliance_matrix.csv"
        df = pd.read_csv(path, dtype=str)
        invalid = df[df["status"].isna() | (df["status"].str.strip() == "")]
        assert len(invalid) == 0, f"Requirements missing status: {list(invalid['requirement_id'])}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
