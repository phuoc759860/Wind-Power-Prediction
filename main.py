import logging
import os
import sys
import time
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.load_data import load_all_data, save_processed_data, get_data_info
from src.column_mapping import apply_column_mapping, create_data_dictionary
from src.data_validation import run_validation
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import build_feature_matrix, create_target_columns
from src.split_time_series import split_by_time, get_split_statistics, walk_forward_split, horizon_sample_counts
from src.train_baseline import train_baselines, walk_forward_baselines, build_test_baseline_predictions
from src.train_power_model import train_power_models, save_models
from src.train_anomaly_model import run_anomaly_detection
from src.train_failure_model import run_failure_analysis
from src.evaluate import (evaluate_all_models, generate_evaluation_report,
                          plot_performance_heatmap, plot_horizon_decay,
                          plot_best_model_scatter, plot_error_histogram,
                          plot_farm_timeseries, plot_radar_summary,
                          compute_farm_level_metrics, analyze_tb12,
                          evaluate_alert_accuracy, evaluate_anomaly_detection,
                          plot_tb12_distribution, analyze_farm_bias,
                          plot_farm_bias_calibration, append_baseline_rows,
                          fit_farm_bias_correction, apply_farm_bias_correction,
                          farm_horizon_window_check)
from src.predict import predict_power, create_forecast_output, add_confidence_intervals, save_forecasts
from src.audit import (raw_file_manifest, raw_timestamp_union, timestamp_audit,
                       reindex_additions_report, leakage_audit, sample_trace,
                       horizon_valid_samples, ridge_feature_evidence,
                       leakage_assertions, write_sample_traces, write_checksums,
                       timestamp_audit_csv)
from src.inventory import generate_inventory
from src.nwp import load_nwp, build_stub_nwp, run_nwp_ablation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/wind_forecasting.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path=None):
    import yaml
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path=None, run_wf_ml=True, run_wf=True):
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("WIND POWER FORECASTING SYSTEM - AMG WIND FARM")
    logger.info("=" * 70)

    base_dir = Path(__file__).parent
    config = load_config(config_path)

    os.makedirs(base_dir / "data" / "processed", exist_ok=True)
    os.makedirs(base_dir / "data" / "metadata", exist_ok=True)
    os.makedirs(base_dir / "outputs" / "forecasts", exist_ok=True)
    os.makedirs(base_dir / "outputs" / "figures", exist_ok=True)
    os.makedirs(base_dir / "models", exist_ok=True)
    os.makedirs(base_dir / "logs", exist_ok=True)

    # ============================================================
    # STEP 1: Load Data
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: LOADING RAW DATA")
    logger.info("=" * 60)

    raw_dir = str(base_dir / "data" / "raw")
    raw_data = load_all_data(raw_dir)

    info = get_data_info(raw_data)
    logger.info(f"Raw data shape: {info['shape']}")
    logger.info(f"Date range: {info.get('date_range', 'N/A')}")

    raw_ts = None
    coverage = None
    reidx = None
    # Raw-file manifest (P1-01) + true raw coverage before any padding (P0-02)
    try:
        manifest = raw_file_manifest(raw_dir)
        manifest.to_csv(base_dir / "data" / "metadata" / "data_manifest.csv", index=False)
        logger.info(f"Raw manifest written: {len(manifest)} files (sha256, coverage windows)")

        raw_ts = raw_timestamp_union(raw_dir)
        coverage = timestamp_audit(raw_ts, interval_minutes=10)
        coverage["note"] = ("Coverage computed on the raw union of timestamps BEFORE any "
                            "reindexing/padding. n_missing_timestamps are true raw holes.")
        with open(base_dir / "data" / "metadata" / "raw_coverage_audit.json", "w") as f:
            json.dump(coverage, f, indent=2, default=str)
        ov = coverage["overall"]
        logger.info(f"RAW UNION coverage: {ov['n_rows']} unique timestamps, "
                    f"{ov['timestamp_start']} -> {ov['timestamp_end']}, "
                    f"expected {ov['expected_steps']}, missing {ov['n_missing_timestamps']} "
                    f"({ov['coverage_ratio']*100:.2f}%)")

        # P0-02 Step 1 evidence: flat timestamp_audit.csv + frozen checksums.txt
        timestamp_audit_csv(raw_ts, interval_minutes=10,
                            out_path=str(base_dir / "data" / "metadata" / "timestamp_audit.csv"))
        write_checksums(raw_dir, str(base_dir / "data" / "metadata" / "checksums.txt"))
        logger.info("timestamp_audit.csv + checksums.txt written (P0-02 evidence)")
    except Exception as e:
        logger.warning(f"Raw manifest/coverage audit failed: {e}")

    # P0-01: official evaluation window. Cut at min(report_date, raw_union_end).
    # Any row at/after this cutoff is flagged is_simulated=1 in preprocessing
    # and excluded from the official evaluation window below.
    report_date = config.get("data", {}).get("report_date")
    evaluation_cutoff = None
    if report_date:
        cutoff = pd.Timestamp(report_date)
        raw_end = None
        if raw_ts is not None and len(raw_ts) > 0:
            raw_end = pd.Timestamp(raw_ts.max())
            if raw_end < cutoff:
                cutoff = raw_end
        evaluation_cutoff = cutoff
        logger.info(f"P0-01 evaluation cutoff = min(report_date={report_date}, "
                    f"raw_union_end={raw_end if raw_ts is not None else 'N/A'}) = {evaluation_cutoff}")
    else:
        logger.warning("No data.report_date in config; evaluation window not cut (P0-01)")

    # ============================================================
    # STEP 2: Column Mapping
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: COLUMN MAPPING & STANDARDIZATION")
    logger.info("=" * 60)

    mapped_data, mapping = apply_column_mapping(raw_data)
    logger.info(f"Mapped {len(mapping)} columns")

    data_dict = create_data_dictionary(raw_data, mapping)
    data_dict.to_csv(base_dir / "data" / "metadata" / "data_dictionary.csv", index=False)
    logger.info("Data dictionary saved")

    with open(base_dir / "data" / "metadata" / "column_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    # ============================================================
    # STEP 3: Validation
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: DATA VALIDATION")
    logger.info("=" * 60)

    validation_results = run_validation(mapped_data)
    with open(base_dir / "data" / "metadata" / "validation_report.json", "w") as f:
        json.dump({k: str(v) for k, v in validation_results.items()}, f, indent=2)

    # ============================================================
    # STEP 3b: Data Audit (timestamp coverage, duplicates, timezone)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3b: DATA AUDIT — TIMESTAMP COVERAGE")
    logger.info("=" * 60)

    if "timestamp" in mapped_data.columns:
        ts = pd.to_datetime(mapped_data["timestamp"])
        logger.info(f"  Timezone: {ts.dt.tz if ts.dt.tz is not None else 'None (naive)'}")
        logger.info(f"  Range: {ts.min()} to {ts.max()}")
        logger.info(f"  Total raw timestamps: {len(ts)}")
        logger.info(f"  Duplicate timestamps: {ts.duplicated().sum()}")
        expected_steps = int((ts.max() - ts.min()).total_seconds() / 60 / 10) + 1
        logger.info(f"  Expected (10min interval): {expected_steps}")
        logger.info(f"  Actual: {len(ts)}")

        audit_data = {
            "timezone": str(ts.dt.tz) if ts.dt.tz is not None else "None (naive)",
            "start": str(ts.min()),
            "end": str(ts.max()),
            "total_raw_rows": len(ts),
            "expected_timestamps_10min": expected_steps,
            "duplicate_timestamps": int(ts.duplicated().sum()),
            "missing_timestamps_estimate": max(0, expected_steps - len(ts)),
        }
        with open(base_dir / "data" / "metadata" / "data_audit.json", "w") as f:
            json.dump(audit_data, f, indent=2, default=str)
        logger.info("  Data audit saved to data/metadata/data_audit.json")

    # ============================================================
    # STEP 4: Preprocessing
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: PREPROCESSING")
    logger.info("=" * 60)

    processed_data = preprocess_pipeline(mapped_data, config, evaluation_cutoff=evaluation_cutoff)
    save_processed_data(processed_data, str(base_dir / "data" / "processed"))

    # ============================================================
    # STEP 5: Feature Engineering
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: FEATURE ENGINEERING")
    logger.info("=" * 60)

    feature_data = build_feature_matrix(processed_data, config)

    horizons = config.get("forecasting", {}).get("horizons", [])
    feature_data = create_target_columns(feature_data, horizons)

    logger.info(f"Feature matrix shape: {feature_data.shape}")

    # ============================================================
    # STEP 6: Time-based Split
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: TIME-BASED SPLIT")
    logger.info("=" * 60)

    split_cfg = config.get("training", {}).get("split", {})
    train_df, val_df, test_df = split_by_time(
        feature_data,
        train_ratio=split_cfg.get("train_ratio", 0.7),
        val_ratio=split_cfg.get("validation_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
    )

    # P0-01: truncate the OFFICIAL test/evaluation window at the report cutoff.
    # Rows at/after the cutoff (the raw source extends beyond the report date)
    # are simulated / beyond-report and must not back official claims.
    simulated_test_df = pd.DataFrame()
    test_window_full = (str(test_df["timestamp"].min()), str(test_df["timestamp"].max()))
    if evaluation_cutoff is not None:
        simulated_test_df = test_df[test_df["timestamp"] >= evaluation_cutoff].copy()
        test_df = test_df[test_df["timestamp"] < evaluation_cutoff].copy()
        logger.info(f"P0-01 official test window cut at {evaluation_cutoff}: "
                    f"kept {len(test_df)} rows, excluded {len(simulated_test_df)} "
                    f"simulated/beyond-report rows")

    split_stats = get_split_statistics(train_df, val_df, test_df, interval_minutes=10)
    logger.info("Split statistics (detailed):")
    for split_name, s in split_stats.items():
        logger.info(f"  {split_name:12s}: {s['rows']:7d} rows  {s.get('timestamp_start',''):s} to {s.get('timestamp_end',''):s}")
        if "expected_steps" in s:
            logger.info(f"  {'':12s}  expected={s['expected_steps']}, unique={s['unique_timestamps']}, "
                        f"dups={s['n_duplicate_timestamps']}, missing={s['n_missing_timestamps']} "
                        f"(coverage {s['coverage_ratio']*100:.2f}%)")
        if "n_observed_rows" in s:
            logger.info(f"  {'':12s}  observed={s['n_observed_rows']}, synthetic={s['n_synthetic_rows']}, "
                        f"imputed={s['n_imputed_rows']}, observed-not-imputed={s['n_observed_not_imputed_rows']} "
                        f"(observed ratio {s.get('observed_ratio', 0)*100:.2f}%)")
    with open(base_dir / "data" / "metadata" / "split_statistics.json", "w") as f:
        json.dump(split_stats, f, indent=2, default=str)
    logger.info("  Split statistics saved to data/metadata/split_statistics.json")

    # P0-01 evidence: evaluation window definition + what was excluded.
    eval_window = {
        "policy": "evaluation_cutoff = min(data.report_date, raw_union_end); "
                  "rows >= cutoff are flagged is_simulated=1 and excluded from official evaluation",
        "report_date": str(report_date) if report_date else None,
        "raw_union_end": str(pd.Timestamp(raw_ts.max())) if raw_ts is not None and len(raw_ts) else None,
        "evaluation_cutoff": str(evaluation_cutoff) if evaluation_cutoff is not None else None,
        "test_window_full_start": test_window_full[0],
        "test_window_full_end": test_window_full[1],
        "test_window_official_start": str(test_df["timestamp"].min()),
        "test_window_official_end": str(test_df["timestamp"].max()),
        "n_test_rows_full": int(len(test_df) + len(simulated_test_df)),
        "n_test_rows_excluded_simulated": int(len(simulated_test_df)),
        "n_test_rows_official": int(len(test_df)),
    }
    with open(base_dir / "data" / "metadata" / "evaluation_window.json", "w") as f:
        json.dump(eval_window, f, indent=2, default=str)
    logger.info(f"  evaluation_window.json saved (official test window ends {eval_window['test_window_official_end']})")

    # Honest per-horizon valid-sample counts (rows whose target P(t+h) exists).
    h_samples = horizon_sample_counts(train_df, val_df, test_df, config)
    with open(base_dir / "data" / "metadata" / "horizon_sample_counts.json", "w") as f:
        json.dump(h_samples, f, indent=2, default=str)
    for split_name, per_h in h_samples.items():
        for h_name, hc in per_h.items():
            logger.info(f"  {split_name:10s} {h_name:6s}: valid targets {hc['n_valid_targets']}/{hc['n_rows']} "
                        f"({hc['ratio_valid']*100:.1f}%)")

    power_cols = [c for c in train_df.columns if c.endswith("_power")
                  and "target" not in c and "lag" not in c and "roll" not in c
                  and "diff" not in c and "ramp" not in c]
    logger.info(f"Power columns found: {len(power_cols)}")

    # Walk-forward validation for baselines
    wf_summary = {}
    if run_wf:
        logger.info("\n" + "-" * 50)
        logger.info("WALK-FORWARD VALIDATION (BASELINES)")
        logger.info("-" * 50)
        wf_results = walk_forward_baselines(feature_data, power_cols, config, n_folds=5)
        wf_summary = wf_results.get("summary", {})
        for k, v in sorted(wf_summary.items()):
            logger.info(f"  {v['model']:12s} {v['horizon']:6s}: RMSE={v['rmse_mean']:.1f} +/- {v['rmse_std']:.1f}  R2={v['r2_mean']:.4f} +/- {v['r2_std']:.4f}  (folds={v['n_folds']}, eval={v['n_evaluations']})")
        with open(base_dir / "data" / "metadata" / "walk_forward_summary.json", "w") as f:
            json.dump(wf_summary, f, indent=2, default=str)
    else:
        logger.info("SKIP: walk-forward baselines disabled (--skip-wf); reusing data/metadata/walk_forward_summary.json")

    # Walk-forward validation for ML models (full coverage: all turbines x horizons)
    # Optional: expensive (~30 min). Baseline walk-forward above is always run.
    if run_wf_ml:
        logger.info("\n" + "-" * 50)
        logger.info("WALK-FORWARD VALIDATION (ML MODELS - FULL)")
        logger.info("-" * 50)
        from src.train_power_model import walk_forward_all_ml
        wf_ml_df = walk_forward_all_ml(feature_data, config, n_folds=3,
                                       save_path=str(base_dir / "data" / "metadata" / "walk_forward_ml.csv"))

        if not wf_ml_df.empty:
            wf_summary_ml = []
            for (mdl, tgt), grp in wf_ml_df.groupby(["model", "target"]):
                wf_summary_ml.append({
                    "model": mdl, "target": tgt,
                    "rmse_mean": round(grp["rmse"].mean(), 2),
                    "rmse_std": round(grp["rmse"].std(), 2),
                    "r2_mean": round(grp["r2"].mean(), 4),
                    "r2_std": round(grp["r2"].std(), 4),
                    "n_folds": len(grp),
                })
            wf_summary_ml_df = pd.DataFrame(wf_summary_ml)
            wf_summary_ml_df.to_csv(base_dir / "data" / "metadata" / "walk_forward_ml_summary.csv", index=False)

            for _, r in wf_summary_ml_df.iterrows():
                logger.info(f"  {r['model']:12s} {str(r['target']):40s}: RMSE={r['rmse_mean']:.1f} +/- {r['rmse_std']:.1f}  R2={r['r2_mean']:.4f} +/- {r['r2_std']:.4f}  (n={r['n_folds']})")
            logger.info(f"  ML walk-forward: {len(wf_ml_df)} rows across {wf_ml_df['fold'].nunique()} folds")
    else:
        logger.info("SKIP: walk-forward ML validation disabled (--no-wf-ml). Baseline walk-forward was run above.")

    # ============================================================
    # STEP 7: Train Baselines
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: TRAINING BASELINE MODELS")
    logger.info("=" * 60)

    baseline_results, ridge_models = train_baselines(train_df, val_df, power_cols, config,
                                                     return_models=True)
    logger.info(f"Ridge models trained (leakage-free): {len(ridge_models)}")

    # P0-01 evidence: exact ridge feature columns + explicit leakage assertions.
    try:
        ridge_ev = ridge_feature_evidence(ridge_models, config)
        ridge_ev.to_csv(base_dir / "data" / "metadata" / "ridge_feature_columns.csv", index=False)
        n_ridge_leaks = int((~ridge_ev["leakage_free"]).sum()) if not ridge_ev.empty else 0
        logger.info(f"Ridge feature evidence: {len(ridge_ev)} models, "
                    f"{n_ridge_leaks} with target/future leaks")
        if n_ridge_leaks:
            raise RuntimeError(f"Ridge leakage detected: {n_ridge_leaks} models")
    except Exception as e:
        logger.error(f"Ridge feature evidence failed: {e}")
        raise

    # ============================================================
    # STEP 8: Train ML Models
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: TRAINING ML MODELS")
    logger.info("=" * 60)

    all_ml_results = {}
    all_trained_models = {}
    tuning_sink = {"records": [], "best_params": {}} if config.get("training", {}).get("tuning", {}).get("enabled", False) else None

    turbine_ids = config.get("turbines", {}).get("ids", [])
    base_target_cols = [f"{tid}_power" for tid in turbine_ids if f"{tid}_power" in train_df.columns]
    base_target_cols.append("farm_total_power")
    base_target_cols = [c for c in base_target_cols if c in train_df.columns]

    horizons = config.get("forecasting", {}).get("horizons", [])

    for base_target in base_target_cols:
        for horizon in horizons:
            h_name = horizon["name"]
            target = f"{base_target}_target_{h_name}"

            if target not in train_df.columns:
                logger.warning(f"Target {target} not found in training data, skipping")
                continue

            try:
                results, models = train_power_models(train_df, val_df, target, config,
                                                     tuning_sink=tuning_sink)
                all_ml_results.update(results)
                all_trained_models.update(models)
            except Exception as e:
                logger.error(f"Error training models for {target}: {e}")

    # P1-02: persist tuning evidence when hyperparameter tuning actually runs.
    if tuning_sink is not None:
        try:
            if tuning_sink["records"]:
                pd.DataFrame(tuning_sink["records"]).to_csv(
                    base_dir / "data" / "metadata" / "tuning_results.csv", index=False)
            else:
                pd.DataFrame(columns=["algorithm", "target", "n_estimators", "max_depth",
                                      "learning_rate", "fold", "cv_rmse"]).to_csv(
                    base_dir / "data" / "metadata" / "tuning_results.csv", index=False)
            with open(base_dir / "data" / "metadata" / "best_params.json", "w") as f:
                json.dump(tuning_sink["best_params"], f, indent=2, sort_keys=True)
            logger.info(f"Tuning evidence written: tuning_results.csv "
                        f"({len(tuning_sink['records'])} combos x folds), "
                        f"best_params.json ({len(tuning_sink['best_params'])} targets)")
        except Exception as e:
            logger.error(f"Tuning persistence failed: {e}")

    if all_trained_models:
        seed = config.get("training", {}).get("random_state", 42)
        raw_dir = str(base_dir / "data" / "raw")
        save_models(all_trained_models, str(base_dir / "models"),
                    config=config, seed=seed, data_path=raw_dir)

        # P0-01 guard: no trained model may use future/target columns.
        leak_df = leakage_audit(all_trained_models)
        leak_df.to_csv(base_dir / "data" / "metadata" / "leakage_audit.csv", index=False)
        n_leaks = int((~leak_df["leakage_free"]).sum()) if not leak_df.empty else 0
        if n_leaks:
            logger.error(f"LEAKAGE AUDIT FAILED: {n_leaks} models use future/target columns")
            raise RuntimeError(f"Leakage audit failed: {n_leaks} leaking models")
        logger.info(f"Leakage audit passed for {len(leak_df)} models (no target/future columns)")

    # ============================================================
    # STEP 9: Anomaly Detection
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 9: ANOMALY DETECTION")
    logger.info("=" * 60)

    test_with_anomaly = run_anomaly_detection(test_df, config)

    # ============================================================
    # STEP 10: Failure Analysis & Availability
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 10: FAILURE ANALYSIS & AVAILABILITY")
    logger.info("=" * 60)

    test_with_failure, availability_results = run_failure_analysis(test_df, config)

    for turbine, avail in availability_results.items():
        logger.info(f"  {turbine}: Observed availability={avail.get('observed_availability_pct')}% "
                    f"(calendar {avail.get('calendar_availability_pct')}%, coverage {avail.get('data_coverage_pct')}%)")

    with open(base_dir / "data" / "metadata" / "availability_report.json", "w") as f:
        json.dump(availability_results, f, indent=2, default=str)

    # ============================================================
    # STEP 11: Generate Predictions & Evaluate
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 11: GENERATING PREDICTIONS & EVALUATION")
    logger.info("=" * 60)

    predictions = predict_power(test_df, all_trained_models, config)

    # P1-04: fit farm bias correction on the VALIDATION split, apply to test.
    val_predictions = predict_power(val_df, all_trained_models, config)
    farm_bias_params = fit_farm_bias_correction(val_df, val_predictions, config)
    corrected_test_preds = apply_farm_bias_correction(predictions, farm_bias_params)
    for model_key, prm in farm_bias_params.items():
        logger.info(f"  Farm bias fit {model_key}: slope={prm['slope']} intercept={prm['intercept']} "
                    f"MAE {prm['val_mae_kw_raw']} -> {prm['val_mae_kw_corrected']} kW (val, n={prm['n_samples']})")

    # Same-sample persistence + ridge predictions on the TEST set (P0-03).
    test_baseline_preds = build_test_baseline_predictions(test_df, ridge_models, power_cols, config)
    persistence_preds = {t: v["persistence"] for t, v in test_baseline_preds.items()}
    ridge_preds = {t: v["ridge"] for t, v in test_baseline_preds.items()}

    results_df = evaluate_all_models(test_df, predictions, config,
                                     baseline_predictions=persistence_preds,
                                     ridge_predictions=ridge_preds)

    # P0-03: include persistence + ridge rows on the SAME test samples, then
    # persist evaluation_metrics.csv so generate_metrics()/the report use it.
    results_df = append_baseline_rows(results_df, test_df,
                                      persistence_preds, ridge_preds, config)
    results_df.to_csv(base_dir / "outputs" / "forecasts" / "evaluation_metrics.csv", index=False)
    logger.info(f"evaluation_metrics.csv written: {len(results_df)} rows "
                f"(incl. persistence + ridge baselines on same test samples)")

    if not results_df.empty:
        eval_report = generate_evaluation_report(results_df, str(base_dir / "outputs" / "figures"),
                                                  test_data=test_df, predictions=predictions,
                                                  baseline_results=baseline_results)

    # P0-01 evidence: explicit sample traces for TB02 at 10min / 1h / 24h.
    try:
        trace = sample_trace(feature_data, ridge_models, all_trained_models, config,
                             turbine="TB02", horizon="24hour", limit=500)
        trace.to_csv(base_dir / "outputs" / "forecasts" / "sample_trace_TB02_24hour.csv", index=False)
        logger.info(f"Sample trace written: {len(trace)} rows "
                    f"(issue t -> features at t -> target P(t+24h) -> model predictions)")
    except Exception as e:
        logger.warning(f"Sample trace failed: {e}")

    try:
        traces = write_sample_traces(feature_data, ridge_models, all_trained_models, config,
                                     cases=[("TB02", "10min"), ("TB02", "1hour"), ("TB02", "24hour"),
                                            ("TB01", "24hour"), ("TB05", "10min"), ("TB05", "24hour"),
                                            ("TB12", "10min"), ("TB12", "24hour")],
                                     out_dir=str(base_dir / "outputs" / "forecasts"), limit=500)
        for p in traces:
            logger.info(f"Sample trace written: {p}")
    except Exception as e:
        logger.warning(f"Additional sample traces failed: {e}")

    # P0-01 explicit assertion table (target not in X, alignment, not allclose).
    # Extended to every ML family (XGBoost/LightGBM) and every turbine (P1-01).
    try:
        leak_assert = leakage_assertions(feature_data, ridge_models, config)
        leak_assert.to_csv(base_dir / "data" / "metadata" / "leakage_audit_ridge.csv", index=False)
        n_fail = int((~leak_assert["all_passed"]).sum())
        logger.info(f"Ridge leakage assertions: {len(leak_assert)} cases, {n_fail} failed")
        if n_fail:
            raise RuntimeError(f"P0-01 assertion failed: {n_fail} (turbine, horizon) cases")
    except Exception as e:
        logger.error(f"Leakage assertions failed: {e}")
        raise

    try:
        leak_full = leakage_assertions(feature_data, ridge_models, config,
                                       ml_models=all_trained_models)
        leak_full.to_csv(base_dir / "data" / "metadata" / "leakage_audit_full.csv", index=False)
        n_fail_full = int((~leak_full["all_passed"]).sum())
        n_ml_rows = int((leak_full["model"] != "ridge").sum()) if not leak_full.empty else 0
        logger.info(f"Full leakage assertions (Ridge + ML): {len(leak_full)} cases "
                    f"({n_ml_rows} ML rows, {n_fail_full} failed)")
        if n_fail_full:
            raise RuntimeError(f"P1-01 assertion failed: {n_fail_full} (turbine, horizon, model) cases")
    except Exception as e:
        logger.error(f"Full leakage assertions failed: {e}")
        raise

    # Farm-level metrics (directly on summed farm power, not avg of turbine R²)
    logger.info("\n" + "-" * 50)
    logger.info("FARM-LEVEL METRICS (direct on farm total power)")
    logger.info("-" * 50)
    farm_metrics_df = compute_farm_level_metrics(test_df, predictions, config,
                                                 corrected_predictions=corrected_test_preds)
    if not farm_metrics_df.empty:
        for _, row in farm_metrics_df.iterrows():
            logger.info(f"  {row['horizon']:6s} {row['model']:10s} MAE={row['mae']:.1f} RMSE={row['rmse']:.1f} "
                        f"R2={row['r2']:.4f} | corrected R2={row['r2_corrected']:.4f}")
        farm_metrics_df.to_csv(base_dir / "outputs" / "forecasts" / "farm_metrics.csv", index=False)

    # Farm-model bias analysis (P1-04): direct farm model vs sum of turbines.
    try:
        farm_bias_df = analyze_farm_bias(test_df, predictions, config)
        if not farm_bias_df.empty:
            farm_bias_df.to_csv(base_dir / "outputs" / "forecasts" / "farm_bias.csv", index=False)
            plot_farm_bias_calibration(test_df, predictions, config,
                                       str(base_dir / "outputs" / "figures" / "25_farm_bias_calibration.png"))
            for _, row in farm_bias_df.iterrows():
                logger.info(f"  Farm bias {row['horizon']:6s}: bias={row['bias_kw']} kW "
                            f"({row['bias_pct_rated']}% rated), farm_vs_sum={row['farm_vs_sum_turbines_kw']} kW")
    except Exception as e:
        logger.warning(f"Farm bias analysis failed: {e}")

    # P1-04: same-window horizon comparison (24h vs 6h R2 artifact check).
    try:
        farm_window_df = farm_horizon_window_check(test_df, predictions, config)
        if not farm_window_df.empty:
            farm_window_df.to_csv(base_dir / "outputs" / "forecasts" / "farm_horizon_window_check.csv",
                                  index=False)
            for _, row in farm_window_df.iterrows():
                if row["horizon_a"] == "6hour" and row["horizon_b"] == "24hour":
                    logger.info(f"  Farm window check {row['horizon_a']} vs {row['horizon_b']}: "
                                f"n_common={row['n_common_samples']} R2_6h={row['r2_a_on_common']} "
                                f"R2_24h={row['r2_b_on_common']} (delta={row['r2_b_minus_a_on_common']})")
    except Exception as e:
        logger.warning(f"Farm horizon window check failed: {e}")

    # P1-05: NWP ingestion interface + SCADA-only vs SCADA+NWP ablation.
    # The shipped 6h/24h models are SCADA-only; this makes that explicit and
    # demonstrates the NWP ingestion path (real CSV if provided, stub otherwise).
    logger.info("\n" + "-" * 50)
    logger.info("NWP INGESTION + SCADA-ONLY vs SCADA+NWP ABLATION (P1-05)")
    logger.info("-" * 50)
    try:
        nwp_cfg = config.get("nwp", {})
        nwp_path = nwp_cfg.get("path", str(base_dir / "data" / "raw" / "nwp_forecast.csv"))
        stub_path = str(base_dir / "data" / "processed" / "nwp_stub_forecast.csv")
        nwp = load_nwp(nwp_path)
        nwp_source = "real_csv"
        if nwp is None:
            logger.info("  No NWP CSV provided; building deterministic STUB NWP "
                        "(perfect-forecast upper bound) to exercise the ingestion path")
            nwp = build_stub_nwp(feature_data, config, stub_path)
            nwp_source = "stub_synthetic"
        nwp_abl = run_nwp_ablation(train_df, val_df, test_df, config, nwp, nwp_source=nwp_source)
        if not nwp_abl.empty:
            nwp_abl.to_csv(base_dir / "outputs" / "forecasts" / "nwp_ablation.csv", index=False)
            for _, row in nwp_abl.iterrows():
                logger.info(f"  {row['feature_set']:15s} {row['target']} {row['horizon']:6s}: "
                            f"R2={row['r2']:.4f} RMSE={row['rmse_kw']:.1f} kW")
    except Exception as e:
        logger.warning(f"NWP ablation failed: {e}")

    # TB12 specific analysis
    logger.info("\n" + "-" * 50)
    logger.info("TB12 ANALYSIS")
    logger.info("-" * 50)
    tb12_analysis = analyze_tb12(test_df, results_df,
                                 split_data={"train": train_df, "val": val_df, "test": test_df})
    for key, val in tb12_analysis.items():
        logger.info(f"  {key}: {val}")
    with open(base_dir / "data" / "metadata" / "tb12_analysis.json", "w") as f:
        json.dump(tb12_analysis, f, indent=2, default=str)

    # Alert accuracy evaluation (per turbine × horizon)
    logger.info("\n" + "-" * 50)
    logger.info("ALERT ACCURACY (RAMP DETECTION)")
    logger.info("-" * 50)
    alert_acc = evaluate_alert_accuracy(test_df, predictions)
    for key, metrics in alert_acc.items():
        logger.info(f"  {metrics['turbine_id']:4s} {metrics['horizon']:6s} {metrics['model']:10s} "
                     f"Prec={metrics['precision']:.3f} Rec={metrics['recall']:.3f} "
                     f"F1={metrics['f1']:.3f} FAR={metrics['false_alarm_rate']:.3f}")
    if alert_acc:
        with open(base_dir / "data" / "metadata" / "alert_accuracy.json", "w") as f:
            json.dump(alert_acc, f, indent=2, default=str)

    # Anomaly detection accuracy evaluation
    logger.info("\n" + "-" * 50)
    logger.info("ANOMALY DETECTION ACCURACY")
    logger.info("-" * 50)
    anomaly_acc = evaluate_anomaly_detection(test_df)
    for tb, metrics in anomaly_acc.items():
        logger.info(f"  {tb}: Prec={metrics['precision']:.3f} Rec={metrics['recall']:.3f} "
                     f"F1={metrics['f1']:.3f} FAR={metrics['false_alarm_rate']:.3f} "
                     f"GT={metrics['n_gt_anomalies']} Detected={metrics['n_detected']}")
    if anomaly_acc:
        with open(base_dir / "data" / "metadata" / "anomaly_accuracy.json", "w") as f:
            json.dump(anomaly_acc, f, indent=2, default=str)

    # ============================================================
    # STEP 12: Create Forecast Output
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 12: CREATING FORECAST OUTPUT")
    logger.info("=" * 60)

    forecast_df = create_forecast_output(test_df, predictions)
    forecast_df = add_confidence_intervals(forecast_df, all_trained_models)
    save_forecasts(forecast_df, str(base_dir / "outputs" / "forecasts"))

    # ============================================================
    # STEP 13: Generate Summary Visualizations
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 13: GENERATING SUMMARY VISUALIZATIONS")
    logger.info("=" * 60)

    fig_dir = base_dir / "outputs" / "figures"

    plot_performance_heatmap(results_df, str(fig_dir / "01_performance_heatmap.png"))
    plot_horizon_decay(results_df, str(fig_dir / "02_horizon_decay.png"))
    plot_best_model_scatter(results_df, test_df, predictions, str(fig_dir / "03_best_model_scatter.png"))
    plot_error_histogram(results_df, test_df, predictions, str(fig_dir / "04_error_histogram.png"))
    plot_farm_timeseries(test_df, predictions, str(fig_dir / "05_farm_timeseries.png"))
    plot_radar_summary(results_df, str(fig_dir / "06_radar_summary.png"))

    plot_tb12_distribution(test_df, str(fig_dir / "14_tb12_distribution.png"))

    logger.info(f"Summary visualizations saved to {fig_dir}")

    from generate_outputs import generate_figures
    try:
        generate_figures()
        logger.info("Additional evaluation figures generated")
    except Exception as e:
        logger.warning(f"Could not generate additional figures: {e}")

    # ============================================================
    # STEP 14: Generate all doc Section 15 output files
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 14: GENERATING OUTPUT FILES (DOC SECTION 15)")
    logger.info("=" * 60)

    from generate_outputs import (
        generate_power_forecast, generate_farm_forecast, generate_metrics,
        generate_data_quality_report, generate_ramp_alert, generate_anomaly_alert,
        generate_failure_risk, generate_temperature_warning,
        generate_screening_summary, generate_alert_accuracy,
        generate_anomaly_accuracy,
    )
    generate_power_forecast(test_df, all_trained_models)
    generate_farm_forecast(test_df, all_trained_models)
    generate_metrics()
    generate_data_quality_report()
    generate_ramp_alert(test_df)
    generate_anomaly_alert(test_df)
    generate_failure_risk(test_df)
    generate_temperature_warning(test_df)
    generate_screening_summary()
    generate_alert_accuracy(test_df, all_trained_models)
    generate_anomaly_accuracy(test_df)

    # ============================================================
    # STEP 15: Provenance — reindex additions + auto inventory (P0-02, P1-01)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 15: PROVENANCE & INVENTORY")
    logger.info("=" * 60)

    try:
        reidx = reindex_additions_report(processed_data, raw_ts)
        with open(base_dir / "data" / "metadata" / "reindex_additions.json", "w") as f:
            json.dump(reidx, f, indent=2, default=str)
        logger.info(f"  Reindex additions: {reidx['n_synthetic_rows_reindexed']} synthetic rows "
                    f"({reidx['synthetic_ratio_pct']}% of processed) added by 10-min resampling")
    except Exception as e:
        logger.warning(f"  Reindex report failed: {e}")

    try:
        generate_inventory(base_dir)
    except Exception as e:
        logger.warning(f"  Inventory generation failed: {e}")

    # ============================================================
    # SUMMARY
    # ============================================================
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {elapsed:.1f} seconds")
    logger.info(f"Models trained: {len(all_trained_models)}")
    logger.info(f"Predictions generated: {len(predictions)}")
    logger.info(f"Training hardware: {os.cpu_count()} cores")

    # Log data split info
    logger.info(f"Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()} ({len(train_df)} samples)")
    logger.info(f"Val period:   {val_df['timestamp'].min()} to {val_df['timestamp'].max()} ({len(val_df)} samples)")
    logger.info(f"Test period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()} ({len(test_df)} samples)")

    # Forecast strategy note
    logger.info("Forecast strategy: Direct multi-horizon (separate model per target-horizon pair)")
    logger.info("Note: 6h and 24h models use only historical SCADA data (no NWP)")
    logger.info("      These are time-series baselines, not complete day-ahead systems")
    logger.info("      NWP integration recommended for 6h+ horizon improvement")

    # Report-relevant facts (P0-02)
    logger.info("-" * 50)
    logger.info("AUDIT FACTS FOR REPORT (version 2.1.0):")
    if raw_ts is not None:
        logger.info(f"  Raw union end: {raw_ts.max()}  (report reference date: {report_date or 'N/A'})")
    if evaluation_cutoff is not None:
        logger.info(f"  Official evaluation cutoff: {evaluation_cutoff} (test window ends at this)")
    logger.info(f"  Test split end (official): {test_df['timestamp'].max()}")
    if coverage is not None:
        logger.info(f"  Raw union missing timestamps: {coverage['overall']['n_missing_timestamps']} "
                    f"({(1 - coverage['overall']['coverage_ratio'])*100:.2f}%)")
    if reidx is not None:
        logger.info(f"  Synthetic reindexed rows: {reidx['n_synthetic_rows_reindexed']} "
                    f"({reidx['synthetic_ratio_pct']}%)")
    if not results_df.empty:
        for horizon in ["10min", "24hour"]:
            sub = results_df[results_df["horizon"] == horizon]
            if sub.empty:
                continue
            best = sub.loc[sub["r2"].idxmax()]
            logger.info(f"  Best {horizon} model: {best['model']} R2={best['r2']:.4f} "
                        f"RMSE={best['rmse']:.1f} kW  skill_vs_persistence={best['skill_vs_persistence']:.4f} "
                        f"(n={best['n_samples']})")

    if not results_df.empty:
        logger.info("\nModel Performance Summary:")
        agg_cols = ["mae", "rmse", "nrmse_pct", "bias", "r2"]
        available = [c for c in agg_cols if c in results_df.columns]
        summary = results_df.groupby("model")[available].mean().round(4)
        logger.info(f"\n{summary.to_string()}")

        logger.info("\nWalk-Forward Validation Summary (Baselines):")
        for k, v in sorted(wf_summary.items()):
            logger.info(f"  {v['model']:12s} {v['horizon']:6s}: RMSE={v['rmse_mean']:.1f}±{v['rmse_std']:.1f}")

        logger.info("(evaluation_metrics.csv written by generate_metrics() in STEP 14)")

    return results_df, forecast_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMG Wind Power Forecasting Pipeline")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: configs/config.yaml)")
    parser.add_argument("--run-all", action="store_true", default=True,
                        help="Run the full pipeline (default: True)")
    parser.add_argument("--no-wf-ml", action="store_true",
                        help="Skip the expensive walk-forward validation for ML models (~30 min). "
                             "Baseline walk-forward, training, evaluation and all audit outputs still run.")
    parser.add_argument("--skip-wf", action="store_true",
                        help="Skip baseline walk-forward validation (reuse walk_forward_summary.json). "
                             "Intended for quick evidence regeneration after a prior full run.")
    args = parser.parse_args()

    if args.run_all:
        results, forecasts = main(args.config, run_wf_ml=not args.no_wf_ml, run_wf=not args.skip_wf)
