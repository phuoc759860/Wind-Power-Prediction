import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_timestamp_audit_csv_coverage_math():
    from src.audit import timestamp_audit_csv

    ts = pd.Series(pd.date_range("2026-01-01", periods=6, freq="10min"))
    ts = pd.concat([ts, ts.iloc[:2]])  # 2 duplicates removed inside audit
    df = timestamp_audit_csv(ts, interval_minutes=10)
    assert not df.empty
    overall = df[df["scope"] == "overall"].iloc[0]
    assert overall["n_rows"] == 6
    assert overall["expected_steps_10min"] == 6
    assert overall["n_missing_timestamps"] == 0
    assert overall["coverage_ratio"] == 1.0


def test_ridge_feature_evidence_flags_leakage():
    from src.audit import ridge_feature_evidence

    clean = ("clean", (None, None, ["TB01_power", "TB01_wind_speed"]))
    leaked = ("leaky", (None, None, ["TB01_power", "TB01_power_target_10min"]))
    ev = ridge_feature_evidence(dict([clean, leaked]), {"turbines": {"ids": ["TB01"]}})
    clean_row = ev[ev["target"] == "clean"].iloc[0]
    leak_row = ev[ev["target"] == "leaky"].iloc[0]
    assert clean_row["leakage_free"]
    assert not leak_row["leakage_free"]
    assert leak_row["leaked_future_columns"] == "TB01_power_target_10min"


def test_rated_power_for_target_farm_uses_26400():
    from src.evaluate import _rated_power_for_target

    assert _rated_power_for_target("TB01_power_target_10min") == 2200
    assert _rated_power_for_target("farm_total_power_target_24hour") == 26400


def test_append_baseline_rows_adds_persistence_and_ridge():
    from src.evaluate import append_baseline_rows

    test_data = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="10min"),
        "TB01_power_target_10min": np.array([10.0, 20.0, 30.0, 40.0,
                                             50.0, 60.0, 70.0, 80.0]),
    })
    baseline_predictions = {
        "TB01_power_target_10min": np.array([11.0, 21.0, 31.0, 41.0,
                                             51.0, 61.0, 71.0, 81.0]),
    }
    ridge_predictions = {
        "TB01_power_target_10min": np.array([12.0, 22.0, 32.0, 42.0,
                                             52.0, 62.0, 72.0, 82.0]),
    }
    base = pd.DataFrame({
        "target": ["TB01_power_target_10min"],
        "model": ["xgboost"],
        "horizon": ["10min"],
        "rmse": [5.0],
    })
    config = {"forecasting": {"horizons": [{"name": "10min", "steps": 1}]}}

    df = append_baseline_rows(base, test_data, baseline_predictions,
                              ridge_predictions, config)
    models = df["model"].tolist()
    assert "persistence" in models
    assert "ridge" in models
    persist = df[df["model"] == "persistence"].iloc[0]
    ridge = df[df["model"] == "ridge"].iloc[0]
    assert persist["n_samples"] == 8
    assert ridge["n_samples"] == 8
    # persistence errors are all exactly 1 kW here
    assert persist["mae"] == pytest.approx(1.0, abs=1e-4)


def test_leakage_assertions_pass_on_clean_ridge():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from src.audit import leakage_assertions

    n = 60
    power = (np.sin(np.arange(n) / 5.0) * 300 + 1200).round(1)
    feature_data = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="10min"),
        "TB01_power": power,
    })
    # target P(t+1) placed with shift(-1) exactly like create_target_columns
    feature_data["TB01_power_target_10min"] = power

    X = feature_data[["TB01_power"]].values[:-1]
    y = power[1:]
    model = Ridge(alpha=1.0).fit(X, y)
    scaler = StandardScaler().fit(X)
    config = {"turbines": {"ids": ["TB01"]},
              "forecasting": {"horizons": [{"name": "10min", "steps": 1}]}}

    ridge_models = {"TB01_power_target_10min": (model, scaler, ["TB01_power"])}
    la = leakage_assertions(feature_data, ridge_models, config)
    row = la.iloc[0]
    assert row["assert_target_not_in_X"]
    assert row["assert_no_future_features"]
    assert row["assert_timestamp_alignment"]
    assert row["n_timestamp_mismatches_checked"] == 0
    assert row["assert_not_identical_to_target"]
    assert row["all_passed"]


def test_generate_change_log_produces_valid_docx(tmp_path):
    import scripts.generate_change_log as gcl

    out = tmp_path / "change_log.docx"
    gcl.OUT_DIR = tmp_path
    gcl.main()
    assert out.exists()
    with zipfile.ZipFile(out) as z:
        assert "word/document.xml" in z.namelist()
        content = z.read("word/document.xml").decode("utf-8")
        assert "P0-01" in content
        assert "Nguyên nhân gốc" in content


def test_screening_summary_json_structure(tmp_path, monkeypatch):
    import json
    import generate_outputs as go

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"a": [1, 2]}).to_csv(out_dir / "ramp_alert.csv", index=False)
    monkeypatch.setattr(go, "OUT", out_dir)
    monkeypatch.setattr(go, "BASE", tmp_path / "base")

    go.generate_screening_summary()
    summary = json.loads((tmp_path / "base" / "data" / "metadata" / "alert_screening_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "SCREENING_ONLY"
    assert summary["confirmed_by_operator"] is False
    assert summary["file_counts"]["ramp_alert"]["n_rows"] == 2
