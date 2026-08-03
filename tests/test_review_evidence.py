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


def test_nmae_equals_mae_over_rated_power_percent():
    """P0-03: nMAE must equal MAE/P_rated*100 per row — turbines at 2200 kW,
    farm at 26400 kW — even when a farm target is iterated FIRST."""
    from src.evaluate import compute_metrics, evaluate_all_models

    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([110.0, 210.0, 290.0])
    assert compute_metrics(actual, predicted, 2200)["mae"] == pytest.approx(10.0)
    assert compute_metrics(actual, predicted, 2200)["nmae_pct"] == pytest.approx(10 / 2200 * 100, rel=1e-3)
    assert compute_metrics(actual, predicted, 2200)["nrmse_pct"] == pytest.approx(10 / 2200 * 100, rel=1e-3)
    assert compute_metrics(actual, predicted, 26400)["nmae_pct"] == pytest.approx(10 / 26400 * 100, rel=1e-3)

    test_data = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="10min"),
        "TB01_power_target_10min": actual,
        "farm_total_power_target_10min": actual,
    })
    predictions = {
        "farm_total_power_target_10min_xgb": {
            "model_name": "xgboost", "target": "farm_total_power_target_10min",
            "predictions": predicted},
        "TB01_power_target_10min_xgb": {
            "model_name": "xgboost", "target": "TB01_power_target_10min",
            "predictions": predicted},
    }
    config = {"forecasting": {"horizons": [{"name": "10min", "steps": 1}]}}
    df = evaluate_all_models(test_data, predictions, config)
    farm_row = df[df["target"] == "farm_total_power_target_10min"].iloc[0]
    tb_row = df[df["target"] == "TB01_power_target_10min"].iloc[0]
    assert farm_row["nmae_pct"] == pytest.approx(10 / 26400 * 100, rel=1e-3)
    assert tb_row["nmae_pct"] == pytest.approx(10 / 2200 * 100, rel=1e-3)
    assert tb_row["nrmse_pct"] == pytest.approx(10 / 2200 * 100, rel=1e-3)


def test_evaluation_metrics_csv_consistent_with_rated_power():
    """P0-03 (real data): every row of evaluation_metrics.csv satisfies
    nMAE = MAE/P_rated*100 (2200 kW turbines, 26400 kW farm)."""
    from src.evaluate import _rated_power_for_target

    base = Path(__file__).parent.parent
    df = pd.read_csv(base / "outputs" / "forecasts" / "evaluation_metrics.csv")
    assert not df.empty
    assert {"mae", "nmae_pct", "rmse", "nrmse_pct", "target"} <= set(df.columns)
    for _, row in df.iterrows():
        rp = _rated_power_for_target(str(row["target"]))
        assert row["nmae_pct"] == pytest.approx(row["mae"] / rp * 100, rel=1e-3)
        assert row["nrmse_pct"] == pytest.approx(row["rmse"] / rp * 100, rel=1e-3)


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


def test_leakage_assertions_audit_farm_models():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from src.audit import leakage_assertions

    n = 60
    power = (np.sin(np.arange(n) / 5.0) * 300 + 1200).round(1)
    feature_data = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="10min"),
        "TB01_power": power,
        "farm_total_power": power * 12,
    })
    feature_data["TB01_power_target_10min"] = power
    feature_data["farm_total_power_target_10min"] = power * 12

    def _fit(px):
        X = px.values[:-1]
        return Ridge(alpha=1.0).fit(X, px.values[1:]), StandardScaler().fit(X)

    m1, s1 = _fit(feature_data[["TB01_power"]])
    mf, sf = _fit(feature_data[["farm_total_power"]])
    ridge_models = {
        "TB01_power_target_10min": (m1, s1, ["TB01_power"]),
        "farm_total_power_target_10min": (mf, sf, ["farm_total_power"]),
    }
    config = {"turbines": {"ids": ["TB01"]},
              "forecasting": {"horizons": [{"name": "10min", "steps": 1}]}}

    la = leakage_assertions(feature_data, ridge_models, config)
    farm = la[la["turbine"] == "farm_total_power"]
    assert len(farm) == 1
    assert farm.iloc[0]["target_column"] == "farm_total_power_target_10min"
    assert farm.iloc[0]["assert_target_not_in_X"]
    assert farm.iloc[0]["all_passed"]

    la_turbine_only = leakage_assertions(feature_data, ridge_models, config,
                                         include_farm=False)
    assert "farm_total_power" not in la_turbine_only["turbine"].tolist()
    assert la_turbine_only.iloc[0]["turbine"] == "TB01"


class _DummyReg:
    def __init__(self, value=123.0):
        self._value = value

    def predict(self, X):
        return np.full(X.shape[0], self._value)


def _synthetic_audit_inputs(leak_model_key=None):
    turbines = [f"TB{i:02d}" for i in range(1, 13)]
    horizons = [{"name": "10min", "steps": 1}, {"name": "30min", "steps": 3},
                {"name": "1hour", "steps": 6}, {"name": "6hour", "steps": 36},
                {"name": "24hour", "steps": 144}]
    n = 120
    rng = np.random.default_rng(0)
    cols = {"timestamp": pd.date_range("2026-01-01", periods=n, freq="10min")}
    for tb in turbines:
        cols[f"{tb}_power"] = rng.normal(1200, 300, n).round(1)
    cols["farm_total_power"] = sum(cols[f"{tb}_power"] for tb in turbines)
    feature_data = pd.DataFrame(cols)
    base_targets = [f"{tb}_power" for tb in turbines] + ["farm_total_power"]
    for base in base_targets:
        for h in horizons:
            feature_data[f"{base}_target_{h['name']}"] = feature_data[base].shift(-h["steps"])

    ridge_models = {}
    ml_models = {}
    for base in base_targets:
        for h in horizons:
            target = f"{base}_target_{h['name']}"
            ridge_models[target] = (_DummyReg(), None, [base])
            for fam in ["xgboost", "lightgbm"]:
                key = f"{target}_{fam}"
                fcols = [base]
                if leak_model_key == key:
                    fcols.append(target)
                ml_models[key] = {"model": _DummyReg(), "scaler": None,
                                  "feature_cols": fcols}
    config = {"turbines": {"ids": turbines}, "forecasting": {"horizons": horizons}}
    return feature_data, ridge_models, config, ml_models


def test_leakage_audit_covers_all_195_models():
    from src.audit import run_leakage_audit

    feature_data, ridge_models, config, ml_models = _synthetic_audit_inputs()
    results = run_leakage_audit(feature_data, ridge_models, config, ml_models=ml_models)
    assert len(results) == 195
    assert (results["turbine"] == "farm_total_power").sum() == 15
    assert results["all_passed"].all()


def test_future_feature_detection_raises():
    from src.audit import run_leakage_audit

    feature_data, ridge_models, config, ml_models = _synthetic_audit_inputs(
        leak_model_key="farm_total_power_target_10min_lightgbm")
    with pytest.raises(RuntimeError, match="leakage assertions failed"):
        run_leakage_audit(feature_data, ridge_models, config, ml_models=ml_models)


def test_preprocessing_sets_provenance_flags():
    from src.preprocessing import preprocess_pipeline

    ts = pd.date_range("2026-01-01 00:00", periods=8, freq="10min")
    ts = ts.delete([2, 3])  # create a 20-min gap at indices 2-3
    df = pd.DataFrame({
        "timestamp": ts,
        "TB01_power": [100.0, 200.0, 500.0, 600.0, 700.0, 800.0],
    })
    config = {"data": {"sampling_interval_minutes": 10}}
    out = preprocess_pipeline(df, config, evaluation_cutoff=pd.Timestamp("2026-01-01 00:40"))

    assert "is_observed" in out.columns
    assert "is_synthetic" in out.columns
    assert "is_imputed" in out.columns
    assert "is_simulated" in out.columns

    # The two missing 10-min slots were inserted by reindexing -> synthetic, not observed.
    assert int(out["is_synthetic"].sum()) == 2
    assert int(out["is_observed"].sum()) == 6
    # ffill fills the gap -> those synthetic rows are also imputed.
    assert int(out["is_imputed"].sum()) >= 2

    # Rows at/after the cutoff are simulated.
    assert int(out["is_simulated"].sum()) == 4
    assert set(out.loc[out["is_simulated"] == 1, "timestamp"]).issubset(
        set(out.loc[out["timestamp"] >= "2026-01-01 00:40", "timestamp"]))

    # A row that is a genuine raw reading must never be flagged synthetic.
    raw_row = out.loc[out["TB01_power"] == 100.0].iloc[0]
    assert raw_row["is_observed"] == 1 and raw_row["is_synthetic"] == 0


def test_preprocessing_without_cutoff_has_no_simulated():
    from src.preprocessing import preprocess_pipeline

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="10min"),
        "TB01_power": np.arange(5, dtype=float),
    })
    out = preprocess_pipeline(df, {"data": {"sampling_interval_minutes": 10}})
    assert "is_simulated" in out.columns
    assert int(out["is_simulated"].sum()) == 0


def test_provenance_columns_excluded_from_ridge_features():
    from src.train_baseline import is_feature_column, select_feature_columns

    for col in ["is_observed", "is_synthetic", "is_imputed", "is_simulated"]:
        assert not is_feature_column(col, "TB01_power_target_10min", np.dtype("int64"))

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="10min"),
        "TB01_power": np.arange(5.0),
        "is_simulated": np.zeros(5, dtype=int),
        "TB01_power_target_10min": np.arange(5.0),
    })
    cols = select_feature_columns(df, "TB01_power_target_10min")
    assert "is_simulated" not in cols
    assert "TB01_power" in cols


def test_provenance_columns_excluded_from_ml_features():
    from src.train_power_model import prepare_features

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="10min"),
        "TB01_power": np.arange(10.0),
        "is_observed": np.ones(10, dtype=int),
        "is_simulated": np.zeros(10, dtype=int),
        "TB01_power_target_10min": np.arange(10.0) + 1.0,
    })
    X, y, cols = prepare_features(df, "TB01_power_target_10min")
    assert "is_observed" not in cols
    assert "is_simulated" not in cols
    assert "TB01_power" in cols


def test_official_test_window_truncates_simulated(tmp_path, monkeypatch):
    import generate_outputs as go

    full = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=6, freq="10min"),
        "is_simulated": [0, 0, 0, 0, 1, 1],
    })
    out = go._official_test_window(full)
    assert len(out) == 4
    assert (out["is_simulated"] == 0).all()

    # Fallback path without the is_simulated column uses config report_date.
    no_flag = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-01", periods=6, freq="D"),
        "TB01_power": np.arange(6.0),
    })
    monkeypatch.setattr(go, "load_config",
                        lambda: {"data": {"report_date": "2026-07-03"}})
    out2 = go._official_test_window(no_flag)
    assert out2["timestamp"].max() < pd.Timestamp("2026-07-03")


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


def test_report_model_counts_come_from_model_joblibs_not_cross_product():
    """P0-02: report counts read inventory (130 ML models), never the
    eval-table cross-product (65 targets x 4 models x 5 horizons = 1300)."""
    import json
    from pathlib import Path

    base = Path(__file__).parent.parent
    inv = json.load(open(base / "data" / "metadata" / "inventory_summary.json"))
    m = inv["counts"]["models"]
    assert m["ml_models"] == 130
    assert m["ml_models_complete"] == 130
    assert m["total_artifacts"] == 520
    assert m["baseline_evaluations"] == 10

    import generate_report
    b = generate_report.ReportBuilder()
    assert b.ml_models == 130
    assert b.model_count == 130
    assert b.model_artifacts_total == 520
    assert b.baseline_evaluations == 10
    assert "1300" not in f"{b.ml_models} {b.model_artifacts_total} {b.baseline_evaluations}"


def test_champions_are_data_driven_from_evaluation_csv():
    """P0-04: the champion-model-per-horizon-x-level table must be recomputed
    directly from evaluation_metrics.csv (min mean RMSE), one row per horizon x
    level cell, and must match an independent recomputation."""
    import generate_report

    b = generate_report.ReportBuilder()
    assert b.champions is not None and not b.champions.empty
    df = pd.read_csv(Path(__file__).parent.parent / "outputs" / "forecasts" / "evaluation_metrics.csv")
    df["level"] = df["target"].astype(str).map(
        lambda t: "farm" if t.lower().startswith("farm") else "turbine")
    agg = df.groupby(["horizon", "level", "model"]).agg(rmse=("rmse", "mean")).reset_index()
    expected = agg.loc[agg.groupby(["horizon", "level"])["rmse"].idxmin()]
    assert len(b.champions) == len(expected) == 10
    for _, r in b.champions.iterrows():
        sub = expected[(expected["horizon"] == r["horizon"]) & (expected["level"] == r["level"])]
        assert len(sub) == 1
        assert r["champion"] == sub.iloc[0]["model"]
        assert r["rmse"] == pytest.approx(sub.iloc[0]["rmse"], rel=1e-3)


def test_champion_table_acknowledges_baseline_wins():
    """P0-04: the champion table must not silently claim ML always wins —
    Ridge/persistence champion cells must exist where they beat the best ML
    model (the very observation the old Conclusions narrative ignored)."""
    import generate_report

    b = generate_report.ReportBuilder()
    baseline_cells = b.champions[b.champions["champion"].isin(["ridge", "persistence"])]
    assert len(baseline_cells) >= 3, "expected Ridge/persistence to champion several cells"
    for _, r in baseline_cells.iterrows():
        assert r["rmse"] < r["best_ml_rmse"], (
            f"{r['horizon']} {r['level']}: baseline champion must beat best ML in RMSE")
    ml_cells = b.champions[b.champions["champion"].isin(["lightgbm", "xgboost"])]
    # Data-driven: after excluding synthetic/imputed rows, baselines champion
    # most cells. The invariant that matters is that every ML champion cell is
    # genuinely the best ML model in its cell (never worse than best_ml_rmse),
    # and that ML cells still exist where ML beats ridge.
    assert len(ml_cells) >= 1, "expected at least one ML champion cell"
    for _, r in ml_cells.iterrows():
        assert r["rmse"] <= r["best_ml_rmse"]


def test_split_statistics_reports_observed_vs_synthetic():
    """P0-05: get_split_statistics must report observed vs synthetic vs imputed
    counts per split so coverage claims never conflate reindexed rows with
    observed timestamps."""
    from src.split_time_series import get_split_statistics

    ts = pd.date_range("2026-01-01", periods=10, freq="10min")
    df = pd.DataFrame({
        "timestamp": ts,
        "TB01_power": np.arange(10.0),
        "is_observed": [1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
        "is_synthetic": [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
        "is_imputed": [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    })
    stats = get_split_statistics(df.iloc[:4], df.iloc[4:7], df.iloc[7:10],
                                 timestamp_col="timestamp", interval_minutes=10)
    total = stats["total"]
    assert total["n_observed_rows"] == 7
    assert total["n_synthetic_rows"] == 3
    assert total["n_imputed_rows"] == 4
    assert total["n_observed_not_imputed_rows"] == 6
    assert total["observed_ratio"] == pytest.approx(0.7)


def test_evaluation_excludes_synthetic_imputed_target_rows():
    """P0-05: evaluate_all_models must exclude rows whose target is synthetic or
    imputed from the official metrics (n_samples drops to observed-not-imputed
    rows, and MAE/RMSE are computed on that subset only)."""
    from src.evaluate import evaluate_all_models

    n = 6
    ts = pd.date_range("2026-01-01", periods=n, freq="10min")
    actual = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    # Rows 1 and 4 are synthetic/imputed (target must not back official metrics).
    test_data = pd.DataFrame({
        "timestamp": ts,
        "TB01_power_target_10min": actual,
        "is_observed": [1, 1, 1, 0, 1, 1],
        "is_synthetic": [0, 0, 0, 1, 0, 0],
        "is_imputed": [0, 1, 0, 1, 0, 0],
    })
    preds = np.array([110.0, 205.0, 290.0, 395.0, 510.0, 590.0])
    predictions = {
        "TB01_power_target_10min_lightgbm": {
            "model_name": "lightgbm", "target": "TB01_power_target_10min",
            "predictions": preds},
    }
    config = {"forecasting": {"horizons": [{"name": "10min", "steps": 1}]}}
    out = evaluate_all_models(test_data, predictions, config)
    row = out.iloc[0]
    kept = np.array([0, 2, 4, 5])  # observed AND not imputed
    assert row["n_samples"] == len(kept) == 4
    expected_rmse = float(np.sqrt(np.mean((actual[kept] - preds[kept]) ** 2)))
    assert row["rmse"] == pytest.approx(expected_rmse, rel=1e-3)
    assert row["mae"] == pytest.approx(np.mean(np.abs(actual[kept] - preds[kept])), rel=1e-3)


def test_provenance_columns_carried_through_feature_engineering():
    """P0-05: is_observed/is_synthetic/is_imputed survive build_feature_matrix
    and create_target_columns so evaluation can filter on them."""
    from src.feature_engineering import build_feature_matrix, create_target_columns

    n = 20
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="10min"),
        "TB01_power": np.linspace(0, 1500, n),
        "TB01_wind_speed": np.linspace(1, 12, n),
        "TB01_temperature": np.linspace(15, 30, n),
        "is_observed": np.ones(n, dtype=int),
        "is_synthetic": np.zeros(n, dtype=int),
        "is_imputed": np.zeros(n, dtype=int),
    })
    config = {"features": {"lag_steps": [1, 2], "rolling_windows": [3], "rolling_stats": ["mean"]}}
    out = build_feature_matrix(df, config)
    out = create_target_columns(out, [{"name": "10min", "steps": 1}])
    for col in ["is_observed", "is_synthetic", "is_imputed"]:
        assert col in out.columns
    assert len(out) == n


def test_official_mask_is_canonical_and_reproducible():
    """Reviewer requirement: evaluation figures/metrics must use one canonical
    official sample mask instead of ad hoc row filters."""
    from evaluation.official_mask import build_official_mask, save_sample_trace

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=6, freq="10min"),
        "observed_target": [1, 1, 1, 1, 1, 1],
        "target_imputed": [0, 1, 0, 0, 0, 0],
        "official_cutoff": [pd.Timestamp("2026-01-01 01:00:00")] * 6,
        "prediction_available": [1, 1, 1, 0, 1, 1],
        "feature_available": [1, 1, 1, 1, 1, 0],
    })
    df.loc[:, "official_cutoff"] = pd.to_datetime(df["official_cutoff"])

    mask = build_official_mask(df)
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool
    assert mask.tolist() == [True, False, True, False, True, False]

    trace_path = Path("sample_trace.csv")
    if trace_path.exists():
        trace_path.unlink()
    save_sample_trace(df, mask, trace_path)
    assert trace_path.exists()
    trace_df = pd.read_csv(trace_path)
    assert list(trace_df.columns)[:6] == [
        "timestamp",
        "observed_target",
        "target_imputed",
        "official_cutoff",
        "prediction_available",
        "feature_available",
    ]
    assert len(trace_df) == 3


def test_official_mask_columns_derived_from_provenance_flags():
    """Reviewer Step: the pipeline never writes the five official-mask columns,
    so they must be derived from the provenance flags. An imputed row and a
    simulated (>= cutoff) row must be excluded by the derived official mask."""
    from evaluation.official_mask import (
        REQUIRED_MASK_COLUMNS,
        add_official_mask_columns,
        build_official_mask,
    )

    n = 4
    ts = pd.date_range("2026-01-01", periods=n, freq="10min")
    df = pd.DataFrame({
        "timestamp": ts,
        "TB01_power": np.linspace(0, 1200, n),
        "is_observed": [1, 1, 1, 1],
        "is_synthetic": [0, 0, 0, 0],
        "is_imputed": [0, 1, 0, 0],
        "is_simulated": [0, 0, 1, 1],
    })

    out = add_official_mask_columns(df)
    for col in REQUIRED_MASK_COLUMNS:
        assert col in out.columns

    # simulated rows (ts >= cutoff) set official_cutoff = earliest simulated ts
    cutoff = ts[2]
    assert pd.to_datetime(out["official_cutoff"].iloc[0]) == cutoff
    assert (out["observed_target"] == 1).all()
    assert out["target_imputed"].tolist() == [0, 1, 0, 0]

    mask = build_official_mask(out)
    assert mask.tolist() == [True, False, False, False]

    # explicit cutoff wins over the simulated-derived one
    out2 = add_official_mask_columns(df, evaluation_cutoff=ts[1])
    mask2 = build_official_mask(out2)
    assert mask2.tolist() == [True, False, False, False]


def test_official_mask_used_in_farm_metrics_and_evaluation():
    """Reviewer Step: every evaluation path (incl. farm metrics) must use the
    official mask, not ad-hoc filters. Imputed/simulated rows must be dropped
    from farm-level metrics rows."""
    from src.evaluate import compute_farm_level_metrics, _official_eval_mask

    n = 6
    ts = pd.date_range("2026-01-01", periods=n, freq="10min")
    test_df = pd.DataFrame({
        "timestamp": ts,
        "TB01_power": np.linspace(0, 1200, n),
        "TB02_power": np.linspace(0, 1000, n),
        "farm_total_power_target_10min": np.linspace(0, 2200, n),
        "is_observed": [1, 1, 1, 1, 1, 1],
        "is_synthetic": [0, 0, 0, 0, 0, 0],
        "is_imputed": [0, 1, 0, 0, 0, 0],
        "is_simulated": [0, 0, 0, 1, 1, 1],
    })
    farm_vals = np.linspace(100, 2100, n)
    predictions = {
        "farm_total_power_target_10min_lightgbm": {
            "model_name": "lightgbm",
            "target": "farm_total_power_target_10min",
            "predictions": farm_vals,
        }
    }
    config = {"rated_power": {"turbine": 1000.0, "farm": 26400.0}}

    mask = _official_eval_mask(test_df, n)
    assert mask.tolist() == [True, False, True, False, False, False]

    farm_df = compute_farm_level_metrics(test_df, predictions, config)
    assert not farm_df.empty
    row = farm_df.iloc[0]
    assert row["target"] == "farm_total_power_target_10min"
    assert row["n_samples"] == 2


def test_official_mask_derivation_rejects_missing_cutoff_cleanly():
    """If neither a simulated flag nor a cutoff is derivable, evaluation must not
    raise - it keeps all rows rather than silently dropping them."""
    from src.evaluate import _official_eval_mask

    n = 4
    test_df = pd.DataFrame({
        "TB01_power": np.linspace(0, 1200, n),
        "is_observed": [1, 1, 1, 1],
    })
    mask = _official_eval_mask(test_df, n)
    assert mask.tolist() == [True, True, True, True]


def test_no_date_after_report_date_in_headline_metrics():
    """P0-01/P0-03: no date after data.report_date appears anywhere in the
    headline metrics/forecast outputs. The official evaluation window is cut at
    evaluation_cutoff = min(report_date, raw_union_end); any timestamp at/after
    the report date in a shipped headline file would leak simulated rows into
    official claims."""
    import yaml
    import json

    base = Path(__file__).parent.parent
    config = yaml.safe_load(open(base / "configs" / "config.yaml", encoding="utf-8"))
    report_date = pd.Timestamp(config["data"]["report_date"])

    ew = json.load(open(base / "data" / "metadata" / "evaluation_window.json"))
    official_end = pd.Timestamp(ew["test_window_official_end"])
    assert official_end <= report_date, (
        f"test_window_official_end {official_end.date()} must be <= report_date {report_date.date()}")

    headline_files = [
        "power_forecast.csv", "farm_forecast.csv",
        "evaluation_metrics.csv", "farm_metrics.csv", "metrics.csv",
        "coverage_calibration.csv", "nwp_ablation.csv",
    ]
    for name in headline_files:
        path = base / "outputs" / "forecasts" / name
        assert path.exists(), f"headline file missing: {name}"
        df = pd.read_csv(path)
        for col in df.columns:
            if not df[col].dtype == object:
                continue
            sample = df[col].dropna().head(2000)
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() < max(1, len(sample) * 0.9):
                continue  # not a date-like column
            late = parsed[parsed > report_date]
            assert late.empty, (
                f"{name}:{col} contains {len(late)} timestamps after report_date "
                f"{report_date.date()} (e.g. {late.iloc[0]}) - official metrics "
                "must not include simulated rows")

    cov_path = base / "outputs" / "coverage.csv"
    assert cov_path.exists(), "coverage.csv missing from outputs/"
    cov_df = pd.read_csv(cov_path)
    for col in ["nominal", "coverage", "mean_width", "calibration_error"]:
        assert col in cov_df.columns, f"coverage.csv missing required column: {col}"


def test_report_figures_exist_and_have_content():
    """P3-02: every figure referenced by the report must exist, be a non-empty
    PNG, and cover the full horizon set — the report must fail loudly if any
    figure is missing rather than silently dropping it."""
    import generate_report as gr

    fig_dir = Path(gr.FIG_DIR)
    referenced = [
        "01_performance_heatmap.png",
        "02_horizon_decay.png",
        "03_best_model_scatter.png",
        "04_error_histogram.png",
        "06_radar_summary.png",
        "07_model_comparison.png",
        "08_horizon_comparison.png",
        "09_error_by_power_region.png",
        "10_error_by_season.png",
        "11_error_by_day_night.png",
        "12_residual_analysis.png",
        "13_error_by_wind_speed.png",
        "25_farm_bias_calibration.png",
    ]
    for name in referenced:
        path = fig_dir / name
        assert path.exists(), f"report references missing figure: {name}"
        assert path.stat().st_size > 0, f"figure is empty: {name}"
        assert path.read_bytes()[:8].startswith(b"\x89PNG"), f"not a PNG: {name}"

    # The report's _figure() must raise instead of silently dropping a missing PNG.
    builder = gr.ReportBuilder()
    try:
        builder._figure("__definitely_missing__.png", "should raise")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("_figure() should raise FileNotFoundError for a missing PNG")


def test_validate_outputs_passes_on_real_artifacts():
    """Reviewer: validate_outputs.py must assert evaluation_metrics.csv exists,
    contain >0 metric rows, and every figure must be a non-empty file."""
    import validate_outputs as vo

    metrics_path = vo._resolve_metrics_path()
    assert metrics_path.exists()
    metrics = pd.read_csv(metrics_path)
    assert len(metrics) > 0
    for col in vo.REQUIRED_EVAL_COLUMNS:
        assert col in metrics.columns

    figures = vo.validate_figures()
    assert len(figures) > 0
    assert all(f.stat().st_size > 0 for f in figures)

    vo.validate_coverage()
    vo.validate_champion_registry()
    report = vo.main()
    assert report["status"] == "PASS"
    assert (Path(vo.OUT) / "validation_report.json").exists()


def test_validate_outputs_raises_on_empty_figure(tmp_path, monkeypatch):
    import validate_outputs as vo

    empty_fig = tmp_path / "01_performance_heatmap.png"
    empty_fig.write_bytes(b"")
    monkeypatch.setattr(vo, "FIG_DIR", tmp_path)

    with pytest.raises(AssertionError, match="st_size == 0"):
        vo.validate_figures()


def test_validate_outputs_raises_on_missing_metrics(tmp_path, monkeypatch):
    import validate_outputs as vo

    monkeypatch.setattr(vo, "FORECAST_DIR", tmp_path)
    monkeypatch.setattr(vo, "OUT", tmp_path)

    with pytest.raises(AssertionError, match="Missing evaluation_metrics.csv"):
        vo.validate_metrics()


def test_validate_outputs_raises_on_empty_metrics(tmp_path, monkeypatch):
    import validate_outputs as vo

    (tmp_path / "evaluation_metrics.csv").write_text("target,model\n", encoding="utf-8")
    monkeypatch.setattr(vo, "FORECAST_DIR", tmp_path)
    monkeypatch.setattr(vo, "OUT", tmp_path)

    with pytest.raises(AssertionError, match="is empty"):
        vo.validate_metrics()


def _make_manifest_fixture(tmp_path):
    """Minimal base_dir with data/ and outputs/ trees for manifest tests."""
    from interval_calibration import save_run_manifest

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "processed_data.parquet").write_bytes(b"scada-bytes")

    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coverage.csv").write_text("nominal,coverage\n0.95,0.94\n", encoding="utf-8")

    config = {"seed": 42, "data": {"report_date": "2026-07-03"}}
    manifest_path = out_dir / "run_manifest.json"
    save_run_manifest(base_dir=tmp_path, config=config, output_path=manifest_path,
                      run_id="run-test-000000")
    return tmp_path, config, manifest_path


def test_build_run_manifest_contains_provenance_fields(tmp_path):
    from interval_calibration import build_run_manifest

    config = {"seed": 42}
    m = build_run_manifest(base_dir=tmp_path, config=config, run_id="run-test-000000")
    for key in ["run_id", "git_commit", "config_hash", "seed", "python",
                "packages", "data_hash", "output_hash", "timestamp"]:
        assert key in m, f"manifest missing {key}"
    assert m["run_id"] == "run-test-000000"
    assert m["seed"] == 42
    assert m["git_commit"] in ("unknown", m["git_commit"])
    assert isinstance(m["packages"], dict) and len(m["packages"]) > 0


def test_verify_run_manifest_passes_on_matching_state(tmp_path):
    from interval_calibration import verify_run_manifest

    base, config, manifest_path = _make_manifest_fixture(tmp_path)
    report = verify_run_manifest(manifest_path=manifest_path, base_dir=base,
                                 config=config)
    assert report["status"] == "PASS"
    assert all(c["match"] for c in report["checks"])


def test_verify_run_manifest_raises_on_tampered_data_hash(tmp_path):
    from interval_calibration import verify_run_manifest

    base, config, manifest_path = _make_manifest_fixture(tmp_path)
    (base / "data" / "processed" / "processed_data.parquet").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="data_hash"):
        verify_run_manifest(manifest_path=manifest_path, base_dir=base, config=config)


def test_verify_run_manifest_raises_on_tampered_seed(tmp_path):
    import json

    from interval_calibration import verify_run_manifest

    base, config, manifest_path = _make_manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="seed"):
        verify_run_manifest(manifest_path=manifest_path, base_dir=base, config=config)


def test_verify_run_manifest_raises_on_missing_manifest(tmp_path):
    from interval_calibration import verify_run_manifest

    with pytest.raises(RuntimeError, match="not found"):
        verify_run_manifest(manifest_path=tmp_path / "nope.json", base_dir=tmp_path)


def test_validate_report_passes_on_real_report():
    """Reviewer: validate_report.py must check every figure/CSV/metric, no
    missing tables, no blank figures — and must pass on the real artifacts."""
    import validate_report as vr

    figures = vr.validate_figures()
    assert len(figures) == len(vr._referenced_figures()) >= 10

    csvs = vr.validate_csvs()
    assert len(csvs) == len(vr.DOCUMENTED_CSVS) >= 15

    metrics = vr.validate_metrics()
    assert metrics["cells"] > 0

    assert vr.REPORT_PDF.exists()
    doc = vr.pymupdf.open(str(vr.REPORT_PDF))
    text = "\n".join(p.get_text() for p in doc)
    assert vr.validate_pdf_tables(doc, text)["sections"] >= 10
    assert len(vr.validate_pdf_figures(doc)) >= len(vr._referenced_figures())

    report = vr.main()
    assert report["status"] == "PASS"
    assert (Path(vr.BASE) / "outputs" / "report_validation.json").exists()


def test_validate_report_raises_on_missing_figure(tmp_path, monkeypatch):
    import validate_report as vr

    monkeypatch.setattr(vr, "FIG_DIR", tmp_path)
    with pytest.raises(AssertionError, match="missing figure"):
        vr.validate_figures()


def test_validate_report_raises_on_blank_figure(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image as PILImage

    import validate_report as vr

    for name in vr._referenced_figures():
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "01_performance_heatmap.png").write_bytes(
        PILImage.new("RGB", (12, 12), (255, 255, 255)).tobytes())

    monkeypatch.setattr(vr, "FIG_DIR", tmp_path)
    with pytest.raises(AssertionError, match="empty|blank|Not a PNG"):
        vr.validate_figures()


def test_validate_report_raises_on_missing_csv(tmp_path, monkeypatch):
    import validate_report as vr

    monkeypatch.setattr(vr, "DOCUMENTED_CSVS",
                        {"outputs/forecasts/__missing__.csv": (["a"], True)})
    with pytest.raises(AssertionError, match="missing CSV"):
        vr.validate_csvs()


def test_validate_report_raises_on_blank_csv(tmp_path, monkeypatch):
    import validate_report as vr

    blank_dir = tmp_path / "outputs" / "forecasts"
    blank_dir.mkdir(parents=True, exist_ok=True)
    blank = blank_dir / "blank.csv"
    blank.write_text("", encoding="utf-8")
    monkeypatch.setattr(vr, "BASE", tmp_path)
    monkeypatch.setattr(vr, "DOCUMENTED_CSVS",
                        {"outputs/forecasts/blank.csv": (["a"], True)})
    with pytest.raises(AssertionError, match="blank CSV"):
        vr.validate_csvs()


def test_validate_report_raises_on_missing_metric_cells(tmp_path, monkeypatch):
    import validate_report as vr

    partial = tmp_path / "evaluation_metrics.csv"
    partial.write_text(
        "target,model,horizon,mae,rmse,bias,r2,n_samples,skill_score,"
        "skill_vs_persistence,skill_vs_ridge\n"
        "TB01_power_target_10min,lightgbm,10min,1,2,3,0.9,10,0,0,0\n",
        encoding="utf-8")
    monkeypatch.setattr(vr, "CSV_DIR", tmp_path)
    with pytest.raises(AssertionError, match="missing metric cells"):
        vr.validate_metrics()
