import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_split_statistics_schema():
    from src.split_time_series import get_split_statistics

    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=12, freq="10min"),
        "TB01_power": np.arange(12, dtype=float),
    })
    train = df.iloc[:8].copy()
    val = df.iloc[8:10].copy()
    test = df.iloc[10:].copy()

    stats = get_split_statistics(train, val, test, timestamp_col="timestamp", interval_minutes=10)
    assert set(stats.keys()) == {"train", "validation", "test", "total"}
    assert stats["train"]["expected_steps"] == 8
    assert stats["train"]["actual_steps"] == 8
    assert stats["train"]["n_duplicate_timestamps"] == 0
    assert stats["train"]["n_missing_timestamps"] == 0
    assert stats["train"]["timezone"] == "None (naive)"
    assert "avg_power" in stats["train"]


def test_requirements_txt_pinned_versions():
    repo_root = Path(__file__).parent.parent
    req_path = repo_root / "requirements.txt"
    lines = [line.strip() for line in req_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]

    assert lines, "requirements.txt should contain at least one dependency"
    for line in lines:
        assert "==" in line, f"Dependency must be pinned with exact version: {line}"
        assert ">=" not in line and "<=" not in line and ">" not in line and "<" not in line and "~=" not in line, f"Loose version specifier found: {line}"


def test_coverage_calibration_schema():
    from src.evaluate import evaluate_coverage_calibration

    test_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="10min"),
        "TB01_power_target_10min": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    })
    predictions = {
        "TB01_power_target_10min_xgboost": {
            "model_name": "xgboost",
            "target": "TB01_power_target_10min",
            "predictions": np.array([11.0, 19.0, 29.0, 41.0, 52.0]),
        }
    }
    config = {
        "forecasting": {
            "horizons": [{"name": "10min", "steps": 1}]
        }
    }

    df = evaluate_coverage_calibration(test_df, predictions, config)
    assert not df.empty
    for col in ["target", "model", "nominal_confidence", "empirical_coverage", "calibration_error", "n_samples"]:
        assert col in df.columns
    assert df["empirical_coverage"].between(0.0, 1.0).all()


def test_readme_output_consistency():
    repo_root = Path(__file__).parent.parent
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert "coverage_calibration.csv" in readme
    assert "power_forecast.csv" in readme
    assert "farm_forecast.csv" in readme
    assert "y_low" in readme
    assert "y_high" in readme
    assert "max_error" in readme


def test_turbine_availability_and_data_coverage():
    from src.train_failure_model import compute_availability, detect_failure_events, run_failure_analysis

    df = pd.DataFrame({
        "TB01_power": [1000.0, 0.0, 0.0, 0.0, 50.0, 60.0],
        "TB01_wind_speed": [4.0, 4.0, 1.0, 2.0, 5.0, 5.0],
        "TB01_status": ["ok", "ok", "curtailed", "no_data", "ok", "ok"],
    })

    availability = compute_availability(df, "TB01_power")
    assert availability["turbine"] == "TB01"
    assert availability["observed_availability_pct"] == 60.0
    assert availability["calendar_availability_pct"] == 50.0
    assert availability["data_coverage_pct"] == 83.33

    failure_df = pd.DataFrame({"TB01_power": [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0]})
    events = detect_failure_events(failure_df, "TB01_power", stop_threshold=5.0, min_stop_duration=3)
    assert int(events["TB01_failure_event"].sum()) == 6
    assert int(events["TB01_failure_event"].iloc[0]) == 1
    assert int(events["TB01_failure_event"].iloc[4]) == 1

    _, availability_results = run_failure_analysis(df, {})
    assert "TB01_power" in availability_results


def test_model_training_random_forest():
    from src.train_power_model import train_power_models

    train_df = pd.DataFrame({
        "feature1": np.arange(20.0),
        "feature2": np.arange(20.0) * 0.5,
        "TB01_power_target_10min": np.arange(20.0) * 2.0 + 5.0,
    })
    val_df = train_df.copy()
    config = {"training": {"models": {"ml": ["random_forest"]}}}

    results, trained_models = train_power_models(train_df, val_df, "TB01_power_target_10min", config)
    assert results
    assert trained_models
    key = "TB01_power_target_10min_random_forest"
    assert key in trained_models
    assert trained_models[key]["model_name"] == "random_forest"
    assert results[key]["val_rmse"] >= 0.0


@pytest.mark.skipif(False, reason="Requires xgboost and lightgbm to be installed")
def test_hyperparameter_tuning_timeseries_split():
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    from src.train_power_model import _tune_xgboost, _tune_lightgbm

    X = np.vstack([np.arange(25.0), np.arange(25.0) * 0.5]).T
    y = X[:, 0] * 0.3 + X[:, 1] * 0.1 + 1.0
    config = {"training": {"random_state": 42}}

    xgb_model = _tune_xgboost(X, y, config)
    assert hasattr(xgb_model, "predict")

    lgb_model = _tune_lightgbm(X, y, config)
    assert hasattr(lgb_model, "predict")


def test_data_quality_report_generation(tmp_path, monkeypatch):
    import generate_outputs as go

    sample = pd.DataFrame({
        "TB01_power": [1000.0, np.nan, 2200.0, -50.0],
        "TB01_wind_speed": [5.0, 6.0, np.nan, 70.0],
        "TB01_temperature": [25.0, 50.0, -20.0, 15.0],
        "TB01_frequency": [49.5, 52.5, 48.0, 54.0],
    })

    temp_out = tmp_path / "out"
    temp_base = tmp_path / "project"
    monkeypatch.setattr(go, "OUT", temp_out)
    monkeypatch.setattr(go, "BASE", temp_base)
    monkeypatch.setattr(go.pd, "read_parquet", lambda path: sample)

    temp_out.mkdir(parents=True, exist_ok=True)
    go.generate_data_quality_report()
    output_file = temp_out / "data_quality_report.csv"
    assert output_file.exists()
    report = pd.read_csv(output_file)
    assert "column" in report.columns
    assert "missing_rate_pct" in report.columns
    assert "invalid_values" in report.columns
    assert "definition" in report.columns
    assert "data_source" in report.columns


def test_power_and_farm_forecast_quality_labels(tmp_path, monkeypatch):
    import generate_outputs as go

    class DummyModel:
        def predict(self, X):
            return np.arange(len(X), dtype=float) * 10.0 + 50.0

    models = {
        "TB01_power_target_10min_lightgbm": {"model": DummyModel(), "scaler": None, "feature_cols": []},
        "TB01_power_target_6hour_lightgbm": {"model": DummyModel(), "scaler": None, "feature_cols": []},
        "farm_total_power_target_10min_lightgbm": {"model": DummyModel(), "scaler": None, "feature_cols": []},
        "farm_total_power_target_6hour_lightgbm": {"model": DummyModel(), "scaler": None, "feature_cols": []},
    }

    test_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=6, freq="10min"),
        "TB01_power_target_10min": np.arange(6, dtype=float) * 5.0 + 10.0,
        "TB01_power_target_6hour": np.arange(6, dtype=float) * 7.0 + 20.0,
        "farm_total_power_target_10min": np.arange(6, dtype=float) * 20.0 + 100.0,
        "farm_total_power_target_6hour": np.arange(6, dtype=float) * 25.0 + 200.0,
    })

    monkeypatch.setattr(go, "OUT", tmp_path / "out")
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)

    go.generate_power_forecast(test_df, models)
    power_df = pd.read_csv(tmp_path / "out" / "power_forecast.csv")
    assert "forecast_quality" in power_df.columns
    assert "production" in power_df["forecast_quality"].unique()
    assert "reference_only" in power_df["forecast_quality"].unique()
    assert power_df["y_low"].notna().all()
    assert power_df["y_high"].notna().all()

    go.generate_farm_forecast(test_df, models)
    farm_df = pd.read_csv(tmp_path / "out" / "farm_forecast.csv")
    assert "forecast_quality" in farm_df.columns
    assert "farm_power_low" in farm_df.columns
    assert "farm_power_high" in farm_df.columns
    assert "production" in farm_df["forecast_quality"].unique()


def test_alert_accuracy_evaluation():
    from src.evaluate import evaluate_alert_accuracy

    test_data = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=12, freq="10min"),
        "TB01_power": np.array([100.0, 1200.0, 1200.0, 100.0, 100.0, 1500.0, 1500.0, 100.0, 100.0, 1200.0, 1200.0, 100.0]),
        "TB01_wind_speed": np.full(12, 5.0),
    })
    predictions = {
        "TB01_power_target_10min_xgboost": {
            "model_name": "xgboost",
            "target": "TB01_power_target_10min",
            "predictions": np.array([100.0, 1200.0, 1200.0, 100.0, 100.0, 1500.0, 1500.0, 100.0, 100.0, 1200.0, 1200.0, 100.0]),
        }
    }

    results = evaluate_alert_accuracy(test_data, predictions, ramp_threshold=0.5)
    assert isinstance(results, dict)
    assert len(results) == 1
    values = next(iter(results.values()))
    assert "precision" in values
    assert "recall" in values
    assert "false_alarm_rate" in values
    assert values["precision"] >= 0.0
    assert values["recall"] >= 0.0
    assert values["n_actual_events"] >= 1
    assert values["n_predicted_events"] >= 1


def test_tb12_analysis_reports_findings():
    from src.evaluate import analyze_tb12

    test_data = pd.DataFrame({
        "TB12_power": [0.0, 0.0, 0.0, 10.0, 12.0, 11.0, 12.0, 11.0, 50.0, 60.0, 70.0],
        "TB12_wind_speed": [2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        "TB09_wind_speed": [3.0] * 11,
        "TB11_wind_speed": [4.0] * 11,
    })
    results_df = pd.DataFrame([
        {"target": "TB12_power_target_10min", "model": "xgboost", "horizon": "10min", "r2": 0.5},
    ])

    analysis = analyze_tb12(test_data, results_df)
    assert "findings" in analysis
    assert isinstance(analysis["findings"], list)
    assert "mean_power_TB12" in analysis
    assert "frozen_data_ratio" in analysis
    assert analysis["mean_power_TB12"] == round(float(np.nanmean(test_data["TB12_power"])), 1)


def test_report_contains_pipeline_architecture():
    import generate_report as gr

    builder = gr.ReportBuilder()
    builder.story = []
    builder.build_methodology()

    texts = [getattr(item, "text", "") for item in builder.story]
    assert any("Pipeline Architecture" in t for t in texts)
    assert any("Feature Engineering" in t for t in texts)


def test_compliance_matrix_schema():
    repo_root = Path(__file__).parent.parent
    matrix = pd.read_csv(repo_root / "configs" / "compliance_matrix.csv", dtype=str, keep_default_na=False)

    expected_columns = {
        "requirement_id", "title", "status", "implementation_files",
        "tests", "notes", "test_result", "last_run_date",
    }
    assert expected_columns.issubset(set(matrix.columns))
    assert matrix["requirement_id"].str.match(r"^\d+\.\d+$").all()
    assert matrix["test_result"].isin(["PASS", "FAIL", "N/A", "N/A (no tests)", "N/A (document-only)"]).all()
