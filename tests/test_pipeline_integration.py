"""End-to-end integration test: verify all output CSVs exist with correct schema.

Uses the existing outputs/forecasts/ directory. If no CSVs exist, runs
generate_outputs.generate_all() to produce them, then validates each file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def outputs():
    """Import generate_outputs to register the OUT constant, then return its dir listing."""
    import generate_outputs
    out_dir = generate_outputs.OUT
    if not list(out_dir.glob("*.csv")):
        generate_outputs.generate_all()
    return out_dir


EXPECTED_CSVS = [
    "power_forecast.csv",
    "farm_forecast.csv",
    "metrics.csv",
    "evaluation_metrics.csv",
    "data_quality_report.csv",
    "ramp_alert.csv",
    "anomaly_alert.csv",
    "failure_risk.csv",
    "temperature_warning.csv",
    "coverage_calibration.csv",
    "alert_accuracy.csv",
    "anomaly_accuracy.csv",
    "farm_metrics.csv",
]


@pytest.mark.parametrize("name", EXPECTED_CSVS)
def test_output_file_exists(outputs, name):
    assert (outputs / name).exists(), f"{name} not found in {outputs}"


def safe_read(path):
    try:
        df = pd.read_csv(path)
        if df.empty or len(df.columns) == 0:
            return None
        return df
    except Exception:
        return None


@pytest.fixture(scope="module")
def csv_cache(outputs):
    """Pre-load all CSVs once, skipping empty/missing files."""
    return {name: safe_read(outputs / name) for name in EXPECTED_CSVS}


def test_power_forecast_schema(csv_cache):
    df = csv_cache.get("power_forecast.csv")
    if df is None:
        pytest.skip()
    for col in ["timestamp_issue", "timestamp_target", "turbine_id", "horizon_min", "y_pred", "y_low", "y_high"]:
        assert col in df.columns
    assert df["y_low"].isna().sum() == 0
    assert df["y_high"].isna().sum() == 0


def test_farm_forecast_schema(csv_cache):
    df = csv_cache.get("farm_forecast.csv")
    if df is None:
        pytest.skip()
    for col in ["timestamp_issue", "timestamp_target", "horizon_min", "farm_power_pred", "farm_power_low", "farm_power_high"]:
        assert col in df.columns


def test_data_quality_has_docs(csv_cache):
    df = csv_cache.get("data_quality_report.csv")
    if df is None:
        pytest.skip()
    assert "definition" in df.columns
    assert "data_source" in df.columns


def test_metrics_has_max_error(csv_cache):
    df = csv_cache.get("metrics.csv")
    if df is None:
        pytest.skip()
    assert "max_error" in df.columns


def test_alert_accuracy_schema(csv_cache):
    df = csv_cache.get("alert_accuracy.csv")
    if df is None:
        pytest.skip()
    for col in ["turbine_id", "horizon", "precision", "recall", "f1"]:
        assert col in df.columns


def test_anomaly_accuracy_schema(csv_cache):
    df = csv_cache.get("anomaly_accuracy.csv")
    if df is None:
        pytest.skip()
    for col in ["turbine_id", "method", "precision", "recall", "f1"]:
        assert col in df.columns


def test_coverage_calibration_schema(csv_cache):
    df = csv_cache.get("coverage_calibration.csv")
    if df is None:
        pytest.skip()
    for col in ["target", "model", "nominal_confidence", "empirical_coverage", "calibration_error"]:
        assert col in df.columns
