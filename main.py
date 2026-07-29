import logging
import os
import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.load_data import load_all_data, save_processed_data, get_data_info
from src.column_mapping import apply_column_mapping, create_data_dictionary
from src.data_validation import run_validation
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import build_feature_matrix, create_target_columns
from src.split_time_series import split_by_time, get_split_statistics, walk_forward_split
from src.train_baseline import train_baselines, walk_forward_baselines
from src.train_power_model import train_power_models, save_models
from src.train_anomaly_model import run_anomaly_detection
from src.train_failure_model import run_failure_analysis
from src.evaluate import (evaluate_all_models, generate_evaluation_report,
                          plot_performance_heatmap, plot_horizon_decay,
                          plot_best_model_scatter, plot_error_histogram,
                          plot_farm_timeseries, plot_radar_summary,
                          compute_farm_level_metrics, analyze_tb12,
                          evaluate_alert_accuracy, plot_tb12_distribution)
from src.predict import predict_power, create_forecast_output, add_confidence_intervals, save_forecasts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/wind_forecasting.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def load_config():
    import yaml
    config_path = Path(__file__).parent / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("WIND POWER FORECASTING SYSTEM - AMG WIND FARM")
    logger.info("=" * 70)

    base_dir = Path(__file__).parent
    config = load_config()

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
    # STEP 4: Preprocessing
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: PREPROCESSING")
    logger.info("=" * 60)

    processed_data = preprocess_pipeline(mapped_data, config)
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

    split_stats = get_split_statistics(train_df, val_df, test_df)
    logger.info(f"Split statistics: {json.dumps(split_stats, indent=2, default=str)}")

    power_cols = [c for c in train_df.columns if c.endswith("_power")
                  and "target" not in c and "lag" not in c and "roll" not in c
                  and "diff" not in c and "ramp" not in c]
    logger.info(f"Power columns found: {len(power_cols)}")

    # Walk-forward validation for baselines (fast)
    logger.info("\n" + "-" * 50)
    logger.info("WALK-FORWARD VALIDATION (BASELINES)")
    logger.info("-" * 50)
    wf_results = walk_forward_baselines(feature_data, power_cols, config, n_folds=5)
    wf_summary = wf_results.get("summary", {})
    for k, v in sorted(wf_summary.items()):
        logger.info(f"  {v['model']:12s} {v['horizon']:6s}: RMSE={v['rmse_mean']:.1f} +/- {v['rmse_std']:.1f}  R2={v['r2_mean']:.4f} +/- {v['r2_std']:.4f}  (n={v['n_folds']})")
    with open(base_dir / "data" / "metadata" / "walk_forward_summary.json", "w") as f:
        json.dump(wf_summary, f, indent=2, default=str)

    # ============================================================
    # STEP 7: Train Baselines
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: TRAINING BASELINE MODELS")
    logger.info("=" * 60)

    baseline_results = train_baselines(train_df, val_df, power_cols, config)

    # ============================================================
    # STEP 8: Train ML Models
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: TRAINING ML MODELS")
    logger.info("=" * 60)

    all_ml_results = {}
    all_trained_models = {}

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
                results, models = train_power_models(train_df, val_df, target, config)
                all_ml_results.update(results)
                all_trained_models.update(models)
            except Exception as e:
                logger.error(f"Error training models for {target}: {e}")

    if all_trained_models:
        save_models(all_trained_models, str(base_dir / "models"))

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
        logger.info(f"  {turbine}: Availability={avail['availability_pct']}%")

    with open(base_dir / "data" / "metadata" / "availability_report.json", "w") as f:
        json.dump(availability_results, f, indent=2, default=str)

    # ============================================================
    # STEP 11: Generate Predictions & Evaluate
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 11: GENERATING PREDICTIONS & EVALUATION")
    logger.info("=" * 60)

    predictions = predict_power(test_df, all_trained_models, config)

    results_df = evaluate_all_models(test_df, predictions, baseline_results, config)

    if not results_df.empty:
        eval_report = generate_evaluation_report(results_df, str(base_dir / "outputs" / "figures"),
                                                  test_data=test_df, predictions=predictions,
                                                  baseline_results=baseline_results)

    # Farm-level metrics (directly on summed farm power, not avg of turbine R²)
    logger.info("\n" + "-" * 50)
    logger.info("FARM-LEVEL METRICS (direct on farm total power)")
    logger.info("-" * 50)
    farm_metrics_df = compute_farm_level_metrics(test_df, predictions, config)
    if not farm_metrics_df.empty:
        for _, row in farm_metrics_df.iterrows():
            logger.info(f"  {row['horizon']:6s} {row['model']:10s} MAE={row['mae']:.1f} RMSE={row['rmse']:.1f} R2={row['r2']:.4f}")
        farm_metrics_df.to_csv(base_dir / "outputs" / "forecasts" / "farm_metrics.csv", index=False)

    # TB12 specific analysis
    logger.info("\n" + "-" * 50)
    logger.info("TB12 ANALYSIS")
    logger.info("-" * 50)
    tb12_analysis = analyze_tb12(test_df, results_df)
    for key, val in tb12_analysis.items():
        logger.info(f"  {key}: {val}")
    with open(base_dir / "data" / "metadata" / "tb12_analysis.json", "w") as f:
        json.dump(tb12_analysis, f, indent=2, default=str)

    # Alert accuracy evaluation
    logger.info("\n" + "-" * 50)
    logger.info("ALERT ACCURACY (RAMP DETECTION)")
    logger.info("-" * 50)
    alert_acc = evaluate_alert_accuracy(test_df, predictions)
    for model_name, metrics in alert_acc.items():
        logger.info(f"  {model_name}: Precision={metrics['precision']:.3f} Recall={metrics['recall']:.3f} F1={metrics['f1']:.3f} FAR={metrics['false_alarm_rate']:.3f}")
    if alert_acc:
        with open(base_dir / "data" / "metadata" / "alert_accuracy.json", "w") as f:
            json.dump(alert_acc, f, indent=2, default=str)

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

    logger.info(f"7 summary visualizations saved to {fig_dir}")

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
    )
    generate_power_forecast(test_df, all_trained_models)
    generate_farm_forecast(test_df, all_trained_models)
    generate_metrics()
    generate_data_quality_report()
    generate_ramp_alert(test_df)
    generate_anomaly_alert(test_df)
    generate_failure_risk(test_df)
    generate_temperature_warning(test_df)

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

    if not results_df.empty:
        logger.info("\nModel Performance Summary:")
        agg_cols = ["mae", "rmse", "nrmse_pct", "bias", "r2"]
        available = [c for c in agg_cols if c in results_df.columns]
        summary = results_df.groupby("model")[available].mean().round(4)
        logger.info(f"\n{summary.to_string()}")

        logger.info("\nWalk-Forward Validation Summary (Baselines):")
        for k, v in sorted(wf_summary.items()):
            logger.info(f"  {v['model']:12s} {v['horizon']:6s}: RMSE={v['rmse_mean']:.1f}±{v['rmse_std']:.1f}")

        results_df.to_csv(base_dir / "outputs" / "forecasts" / "evaluation_metrics.csv", index=False)

    return results_df, forecast_df


if __name__ == "__main__":
    results, forecasts = main()
