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


def test_farm_metrics_score_against_horizon_target():
    """P1-04: farm metrics must be scored on P(t+h), not the base farm_total_power."""
    from src.evaluate import compute_farm_level_metrics

    n = 50
    base = np.full(n, 1000.0)
    target = np.sin(np.arange(n) / 3.0) * 300 + 1200
    preds = target + np.random.default_rng(0).normal(0, 10, n)

    test_data = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="10min"),
        "farm_total_power": base,
        "farm_total_power_target_10min": target,
    })
    config = {"forecasting": {"horizons": [{"name": "10min", "steps": 1}]}}
    predictions = {"farm_total_power_target_10min_lgb": {
        "model_name": "lightgbm", "target": "farm_total_power_target_10min",
        "predictions": preds}}
    df = compute_farm_level_metrics(test_data, predictions, config)
    assert not df.empty
    assert df["horizon"].iloc[0] == "10min"
    assert df["r2"].iloc[0] > 0.9
    assert df["n_samples"].iloc[0] == n


def test_farm_bias_correction_fit_and_apply():
    from src.evaluate import fit_farm_bias_correction, apply_farm_bias_correction

    rng = np.random.default_rng(1)
    n = 80
    p = rng.uniform(0, 26400, n)
    actual = 1.2 * p + 500

    val_df = pd.DataFrame({"farm_total_power_target_6hour": actual})
    key = "farm_total_power_target_6hour_lgb"
    val_preds = {key: {"model_name": "lightgbm", "target": "farm_total_power_target_6hour",
                       "predictions": p}}
    config = {"forecasting": {"horizons": [{"name": "6hour", "steps": 36}]}}

    params = fit_farm_bias_correction(val_df, val_preds, config)
    assert key in params
    prm = params[key]
    assert prm["slope"] == pytest.approx(1.2, abs=0.05)
    assert prm["intercept"] == pytest.approx(500, abs=30)
    assert prm["val_mae_kw_corrected"] < prm["val_mae_kw_raw"]

    test_preds = {key: {"model_name": "lightgbm", "target": "farm_total_power_target_6hour",
                        "predictions": p},
                  "TB01_power_target_6hour_lgb": {"model_name": "lightgbm",
                                                  "target": "TB01_power_target_6hour",
                                                  "predictions": p}}
    corrected = apply_farm_bias_correction(test_preds, params)
    assert corrected[key]["bias_corrected"] is True
    np.testing.assert_allclose(corrected[key]["predictions"], 1.2 * p + 500, atol=0.5)
    np.testing.assert_allclose(corrected["TB01_power_target_6hour_lgb"]["predictions"], p)
    assert "bias_corrected" not in corrected["TB01_power_target_6hour_lgb"]


def test_farm_metrics_include_bias_corrected_columns():
    from src.evaluate import compute_farm_level_metrics

    rng = np.random.default_rng(3)
    n = 40
    p = rng.uniform(500, 20000, n)
    actual = 1.1 * p

    test_data = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="10min"),
        "farm_total_power_target_10min": actual,
    })
    key = "farm_total_power_target_10min_lgb"
    raw = {"predictions": p, "model_name": "lightgbm", "target": "farm_total_power_target_10min"}
    corrected = dict(raw, predictions=actual, kind="linear", slope=1.1,
                     intercept=0.0, scalar_offset_kw=0.0)
    config = {"forecasting": {"horizons": [{"name": "10min", "steps": 1}]}}
    df = compute_farm_level_metrics(test_data, {key: raw}, config,
                                    corrected_predictions={key: corrected})
    row = df.iloc[0]
    assert row["r2_corrected"] > 0.99
    assert abs(row["bias_corrected"]) < 1e-3
    assert row["correction_slope"] == 1.1


def test_farm_horizon_window_check_uses_identical_samples():
    from src.evaluate import farm_horizon_window_check

    rng = np.random.default_rng(2)
    n = 60
    t = np.sin(np.arange(n) / 2.0) * 300 + 1200
    t_6h = t.copy()
    t_24h = t.copy()
    t_24h[-24:] = np.nan  # shift(-144) trailing NaNs

    test_data = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="10min"),
        "farm_total_power_target_6hour": t_6h,
        "farm_total_power_target_24hour": t_24h,
    })
    noise = rng.normal(0, 20, n)
    predictions = {
        "farm_total_power_target_6hour_lgb": {
            "model_name": "lightgbm", "target": "farm_total_power_target_6hour",
            "predictions": t_6h + noise},
        "farm_total_power_target_24hour_lgb": {
            "model_name": "lightgbm", "target": "farm_total_power_target_24hour",
            "predictions": t_24h + noise},
    }
    config = {"forecasting": {"horizons": [{"name": "6hour", "steps": 36},
                                           {"name": "24hour", "steps": 144}]}}
    df = farm_horizon_window_check(test_data, predictions, config)
    row = df[(df["horizon_a"] == "6hour") & (df["horizon_b"] == "24hour")].iloc[0]
    assert bool(row["window_identical"]) is True
    assert row["n_common_samples"] == n - 24
    assert pd.notna(row["r2_a_on_common"])
    assert pd.notna(row["r2_b_on_common"])
