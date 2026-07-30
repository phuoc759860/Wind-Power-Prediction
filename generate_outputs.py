import logging
import math
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs" / "forecasts"
OUT.mkdir(parents=True, exist_ok=True)

RATED_POWER = 2200
HORIZON_MAP = {"10min": 10, "30min": 30, "1hour": 60, "6hour": 360, "24hour": 1440}
HORIZON_NAMES = ["10min", "30min", "1hour", "6hour", "24hour"]
TURBINES = [f"TB{i:02d}" for i in range(1, 13)]
MODEL_VERSION = "2.0.0"
TIMESTAMP_NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_config():
    with open(BASE / "configs" / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def get_test_data():
    processed = pd.read_parquet(BASE / "data" / "processed" / "processed_data.parquet")
    config = load_config()
    from src.feature_engineering import build_feature_matrix, create_target_columns
    from src.split_time_series import split_by_time

    feature_data = build_feature_matrix(processed, config)
    horizons = config.get("forecasting", {}).get("horizons", [])
    feature_data = create_target_columns(feature_data, horizons)
    split_cfg = config.get("training", {}).get("split", {})
    _, _, test_df = split_by_time(
        feature_data,
        train_ratio=split_cfg.get("train_ratio", 0.7),
        val_ratio=split_cfg.get("validation_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
    )
    return test_df


def load_models():
    from src.train_power_model import load_models
    return load_models(str(BASE / "models"))


def _daily_agg(df, date_col, value_cols):
    df = df.copy()
    df["_date"] = pd.to_datetime(df[date_col]).dt.normalize()
    agg = {c: "mean" for c in value_cols}
    return df.groupby("_date").agg(agg).reset_index()


def _conformal_bounds_for_group(predicted, actual, confidence=0.9, rated=2200):
    residuals = np.abs(actual - predicted)
    alpha = 1 - confidence
    if len(residuals) >= 10:
        q_lo = np.quantile(residuals, alpha / 2)
        q_hi = np.quantile(residuals, 1 - alpha / 2)
        lower = np.maximum(0, predicted + q_lo)
        upper = np.minimum(rated, predicted + q_hi)
    else:
        std = np.std(predicted) if len(predicted) > 1 else predicted.std() if hasattr(predicted, 'std') else 0
        if pd.isna(std) or std == 0:
            lower = np.maximum(0, predicted * 0.95)
            upper = np.minimum(rated, predicted * 1.15)
        else:
            lower = np.maximum(0, predicted - 1.96 * std)
            upper = np.minimum(rated, predicted + 1.96 * std)
    return lower, upper


def generate_power_forecast(test_df, models):
    """Generate power_forecast.csv — daily aggregated per doc section 15 Table 9 Row 1"""
    logger.info("Generating power_forecast.csv (daily avg) ...")

    ts_col = "timestamp"
    if ts_col not in test_df.columns:
        ts_col = test_df.columns[0]

    raw_records = []
    for tb in TURBINES:
        for horizon in HORIZON_NAMES:
            for mdl_name in ["lightgbm", "xgboost"]:
                target = f"{tb}_power_target_{horizon}"
                model_key = f"{target}_{mdl_name}"
                if model_key not in models:
                    continue

                from src.predict import predict_with_model
                preds = predict_with_model(models[model_key], test_df)
                n = len(preds)
                actuals = test_df[target].values[:n] if target in test_df.columns else np.full(n, np.nan)

                lo, hi = _conformal_bounds_for_group(preds, actuals, confidence=0.9)

                for i in range(n):
                    raw_records.append({
                        ts_col: test_df[ts_col].values[i] if i < len(test_df) else pd.NaT,
                        "turbine_id": tb,
                        "horizon_min": HORIZON_MAP[horizon],
                        "model_version": f"{MODEL_VERSION}_{mdl_name}",
                        "forecast_quality": "reference_only" if HORIZON_MAP[horizon] >= 360 else "production",
                        "y_pred": preds[i],
                        "y_low": lo[i],
                        "y_high": hi[i],
                    })

    raw = pd.DataFrame(raw_records)
    raw["_date"] = pd.to_datetime(raw[ts_col]).dt.normalize()
    daily = raw.groupby(["_date", "turbine_id", "horizon_min", "model_version", "forecast_quality"], dropna=False).agg(
        y_pred=("y_pred", "mean"),
        y_low=("y_low", "mean"),
        y_high=("y_high", "mean"),
    ).reset_index()

    rows = []
    for _, r in daily.iterrows():
        rows.append({
            "timestamp_issue": str(r["_date"].date()),
            "timestamp_target": str(r["_date"].date()),
            "turbine_id": r["turbine_id"],
            "horizon_min": r["horizon_min"],
            "y_pred": round(r["y_pred"], 2),
            "y_low": round(max(0, r["y_low"]), 2),
            "y_high": round(min(RATED_POWER, r["y_high"]), 2),
            "model_version": r["model_version"],
            "forecast_quality": r["forecast_quality"],
        })

    df = pd.DataFrame(rows).sort_values(["timestamp_issue", "turbine_id", "horizon_min", "model_version"]).reset_index(drop=True)
    df.to_csv(OUT / "power_forecast.csv", index=False)
    logger.info(f"  power_forecast.csv: {df.shape[0]} rows (daily)")


def generate_farm_forecast(test_df, models):
    """Generate farm_forecast.csv — daily aggregated per doc section 15 Table 9 Row 2"""
    logger.info("Generating farm_forecast.csv (daily avg) ...")

    ts_col = "timestamp"
    if ts_col not in test_df.columns:
        ts_col = test_df.columns[0]

    raw_records = []
    for horizon in HORIZON_NAMES:
        for mdl_name in ["lightgbm", "xgboost"]:
            target = f"farm_total_power_target_{horizon}"
            model_key = f"{target}_{mdl_name}"
            if model_key not in models:
                continue

            from src.predict import predict_with_model
            preds = predict_with_model(models[model_key], test_df)
            n = len(preds)
            actuals = test_df[target].values[:n] if target in test_df.columns else np.full(n, np.nan)

            lo, hi = _conformal_bounds_for_group(preds, actuals, confidence=0.9)
            dt_minutes = HORIZON_MAP[horizon]

            for i in range(n):
                raw_records.append({
                    ts_col: test_df[ts_col].values[i] if i < len(test_df) else pd.NaT,
                    "horizon_min": dt_minutes,
                    "model_version": f"{MODEL_VERSION}_{mdl_name}",
                    "forecast_quality": "reference_only" if dt_minutes >= 360 else "production",
                    "farm_power_pred": preds[i],
                    "y_low": lo[i],
                    "y_high": hi[i],
                })

    raw = pd.DataFrame(raw_records)
    raw["_date"] = pd.to_datetime(raw[ts_col]).dt.normalize()
    daily = raw.groupby(["_date", "horizon_min", "model_version", "forecast_quality"], dropna=False).agg(
        farm_power_pred=("farm_power_pred", "mean"),
        y_low=("y_low", "mean"),
        y_high=("y_high", "mean"),
    ).reset_index()

    rows = []
    for _, r in daily.iterrows():
        farm_power = round(r["farm_power_pred"], 2)
        farm_energy = round(farm_power * r["horizon_min"] / 60.0, 2)
        rows.append({
            "timestamp_issue": str(r["_date"].date()),
            "timestamp_target": str(r["_date"].date()),
            "horizon_min": r["horizon_min"],
            "farm_power_pred": farm_power,
            "farm_power_low": round(max(0, r["y_low"]), 2),
            "farm_power_high": round(min(RATED_POWER, r["y_high"]), 2),
            "farm_energy_pred": farm_energy,
            "forecast_quality": r["forecast_quality"],
        })

    df = pd.DataFrame(rows).sort_values(["timestamp_issue", "horizon_min"]).reset_index(drop=True)
    df.to_csv(OUT / "farm_forecast.csv", index=False)
    logger.info(f"  farm_forecast.csv: {df.shape[0]} rows (daily)")


def generate_metrics():
    """Generate metrics.csv per doc section 15 Table 9 Row 6"""
    logger.info("Generating metrics.csv ...")

    eval_path = OUT / "evaluation_metrics.csv"
    if not eval_path.exists():
        logger.warning("  evaluation_metrics.csv not found, skipping metrics.csv")
        return

    df = pd.read_csv(eval_path)

    def parse_turbine(target):
        for tb in TURBINES:
            if target.startswith(tb):
                return tb
        return "farm"

    records = []
    for _, row in df.iterrows():
        turbine = parse_turbine(row.get("target", ""))
        records.append({
            "model": row.get("model", ""),
            "turbine_id": turbine,
            "horizon": row.get("horizon", ""),
            "MAE": round(row.get("mae", 0), 4),
            "RMSE": round(row.get("rmse", 0), 4),
            "nMAE": round(row.get("nmae_pct", 0), 4),
            "nRMSE": round(row.get("nrmse_pct", 0), 4),
            "Bias": round(row.get("bias", 0), 4),
            "R2": round(row.get("r2", 0), 4),
            "max_error": round(row.get("max_error", 0), 4),
            "skill_score": round(row.get("skill_score", 0), 4),
        })

    out = pd.DataFrame(records)
    out.to_csv(OUT / "metrics.csv", index=False)
    logger.info(f"  metrics.csv: {out.shape[0]} rows")


def generate_data_quality_report():
    """Generate data_quality_report.csv per doc section 15 Table 9 Row 7"""
    logger.info("Generating data_quality_report.csv ...")

    processed = pd.read_parquet(BASE / "data" / "processed" / "processed_data.parquet")

    physical_cols = []
    for tb in TURBINES:
        for suffix in ["_power", "_wind_speed", "_temperature", "_frequency"]:
            col = f"{tb}{suffix}"
            if col in processed.columns:
                physical_cols.append(col)

    physical_cols = [c for c in physical_cols if "target" not in c and "lag" not in c
                     and "roll" not in c and "diff" not in c and "ramp" not in c]

    unit_map = {"_power": "kW", "_wind_speed": "m/s", "_temperature": "degC", "_frequency": "Hz"}

    records = []
    for col in physical_cols:
        s = processed[col]
        total = len(s)
        missing = int(s.isna().sum())
        missing_rate = round(missing / total * 100, 2) if total > 0 else 0

        valid = s.dropna()
        invalid = 0
        if len(valid) > 0:
            if "_power" in col:
                invalid = int(((valid < 0) | (valid > 2200)).sum())
            elif "_wind_speed" in col:
                invalid = int(((valid < 0) | (valid > 60)).sum())
            elif "_temperature" in col:
                invalid = int(((valid < -10) | (valid > 55)).sum())
            elif "_frequency" in col:
                invalid = int(((valid < 47) | (valid > 53)).sum())

        min_val = round(float(valid.min()), 2) if len(valid) > 0 else ""
        max_val = round(float(valid.max()), 2) if len(valid) > 0 else ""

        unit = "check"
        for k, v in unit_map.items():
            if k in col:
                unit = v
                break

        if missing_rate > 50:
            remarks = "High missing rate - investigate"
        elif missing_rate > 20:
            remarks = "Significant missing data"
        elif missing_rate > 10:
            remarks = "Moderate missing data"
        elif missing_rate > 0:
            remarks = "Minor gaps"
        else:
            remarks = "Complete"

        records.append({
            "column": col,
            "missing_rate_pct": missing_rate,
            "invalid_values": invalid,
            "min": min_val,
            "max": max_val,
            "unit": unit,
            "remarks": remarks,
        })

    out = pd.DataFrame(records)
    out.to_csv(OUT / "data_quality_report.csv", index=False)
    logger.info(f"  data_quality_report.csv: {out.shape[0]} physical columns")


def generate_ramp_alert(test_df):
    """Generate ramp_alert.csv per doc section 15 Table 9 Row 3"""
    logger.info("Generating ramp_alert.csv ...")

    ts_col = "timestamp"
    if ts_col not in test_df.columns:
        ts_col = test_df.columns[0]

    dt_min = 10
    dt_hours = dt_min / 60.0
    ramp_threshold_mw_per_min = 0.1

    rows = []
    for tb in TURBINES:
        pwr_col = f"{tb}_power"
        if pwr_col not in test_df.columns:
            continue

        power = test_df[pwr_col].values
        timestamps = test_df[ts_col].values

        for i in range(1, len(power)):
            if np.isnan(power[i]) or np.isnan(power[i - 1]):
                continue
            delta_p = (power[i] - power[i - 1]) / 1000.0
            rate = delta_p / dt_min

            if abs(rate) > ramp_threshold_mw_per_min:
                ramp_type = "ramp_up" if rate > 0 else "ramp_down"
                rows.append({
                    "timestamp": str(timestamps[i]),
                    "ramp_type": ramp_type,
                    "expected_change": round(delta_p, 4),
                    "probability": round(min(1.0, abs(rate) / 0.5), 4),
                    "threshold": ramp_threshold_mw_per_min,
                    "affected_turbines": tb,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ramp_alert.csv", index=False)
    logger.info(f"  ramp_alert.csv: {df.shape[0]} rows")


def generate_anomaly_alert(test_df):
    """Generate anomaly_alert.csv per doc section 15 Table 9 Row 4"""
    logger.info("Generating anomaly_alert.csv ...")

    ts_col = "timestamp"
    if ts_col not in test_df.columns:
        ts_col = test_df.columns[0]

    rows = []
    for tb in TURBINES:
        pwr_col = f"{tb}_power"
        ws_col = f"{tb}_wind_speed"
        if pwr_col not in test_df.columns or ws_col not in test_df.columns:
            continue

        power = test_df[pwr_col].values
        ws = test_df[ws_col].values
        timestamps = test_df[ts_col].values

        valid = ~(np.isnan(power) | np.isnan(ws))
        if valid.sum() < 100:
            continue
        p_valid = power[valid]
        mean_p, std_p = np.mean(p_valid), np.std(p_valid)

        for i in range(len(power)):
            if np.isnan(power[i]) or np.isnan(ws[i]):
                continue
            z_score = abs(power[i] - mean_p) / (std_p + 1e-6)
            if z_score > 3.0:
                anomaly_score = round(float(z_score), 4)
                if power[i] < 0:
                    evidence = "Negative power output (possible motoring)"
                elif power[i] > RATED_POWER:
                    evidence = "Power exceeds rated capacity"
                elif ws[i] < 3 and power[i] > 500:
                    evidence = "High power at low wind speed"
                elif ws[i] > 15 and power[i] < 200:
                    evidence = "Low power at high wind speed"
                else:
                    evidence = f"Statistical anomaly (z={anomaly_score:.2f})"

                rows.append({
                    "timestamp": str(timestamps[i]),
                    "turbine_id": tb,
                    "anomaly_score": anomaly_score,
                    "suspected_component": "power_output",
                    "evidence": evidence,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "anomaly_alert.csv", index=False)
    logger.info(f"  anomaly_alert.csv: {df.shape[0]} rows")


def generate_temperature_warning(test_df):
    """Generate temperature_warning.csv — alerts for out-of-range or rapid temp changes"""
    logger.info("Generating temperature_warning.csv ...")

    ts_col = "timestamp"
    if ts_col not in test_df.columns:
        ts_col = test_df.columns[0]

    TEMP_HIGH_WARN = 45
    TEMP_HIGH_CRIT = 50
    TEMP_LOW_WARN = -10
    TEMP_LOW_CRIT = -15
    RAPID_CHANGE_DELTA = 5
    SUSTAINED_HIGH = 40
    SUSTAINED_WINDOW = 3

    rows = []
    for tb in TURBINES:
        temp_col = f"{tb}_temperature"
        if temp_col not in test_df.columns:
            continue

        temps = test_df[temp_col].values
        timestamps = test_df[ts_col].values
        sustained_count = 0

        for i in range(len(temps)):
            if np.isnan(temps[i]):
                sustained_count = 0
                continue

            ts = str(timestamps[i])
            temp_val = round(float(temps[i]), 2)

            if temps[i] >= TEMP_HIGH_CRIT:
                rows.append({"timestamp": ts, "turbine_id": tb, "temperature": temp_val,
                             "warning_type": "high_temperature", "severity": "critical",
                             "message": f"Critical high temperature: {temp_val}°C (≥{TEMP_HIGH_CRIT}°C)"})
            elif temps[i] >= TEMP_HIGH_WARN:
                rows.append({"timestamp": ts, "turbine_id": tb, "temperature": temp_val,
                             "warning_type": "high_temperature", "severity": "warning",
                             "message": f"High temperature: {temp_val}°C (≥{TEMP_HIGH_WARN}°C)"})

            if temps[i] <= TEMP_LOW_CRIT:
                rows.append({"timestamp": ts, "turbine_id": tb, "temperature": temp_val,
                             "warning_type": "low_temperature", "severity": "critical",
                             "message": f"Critical low temperature: {temp_val}°C (≤{TEMP_LOW_CRIT}°C)"})
            elif temps[i] <= TEMP_LOW_WARN:
                rows.append({"timestamp": ts, "turbine_id": tb, "temperature": temp_val,
                             "warning_type": "low_temperature", "severity": "warning",
                             "message": f"Low temperature: {temp_val}°C (≤{TEMP_LOW_WARN}°C)"})

            if i > 0 and not np.isnan(temps[i - 1]):
                delta = abs(temps[i] - temps[i - 1])
                if delta >= RAPID_CHANGE_DELTA:
                    rows.append({"timestamp": ts, "turbine_id": tb, "temperature": temp_val,
                                 "warning_type": "rapid_temp_change", "severity": "warning",
                                 "message": f"Rapid temperature change: {delta:.1f}°C in 10min (≥{RAPID_CHANGE_DELTA}°C)"})

            if temps[i] >= SUSTAINED_HIGH:
                sustained_count += 1
            else:
                sustained_count = 0

            if sustained_count == SUSTAINED_WINDOW:
                rows.append({"timestamp": ts, "turbine_id": tb, "temperature": temp_val,
                             "warning_type": "sustained_high_temp", "severity": "warning",
                             "message": f"Sustained high temperature: {temp_val}°C for {SUSTAINED_WINDOW} consecutive readings (≥{SUSTAINED_HIGH}°C)"})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "temperature_warning.csv", index=False)
    logger.info(f"  temperature_warning.csv: {df.shape[0]} rows")


def generate_figures():
    """Generate figures to outputs/figures/ — runs evaluation if needed"""
    logger.info("Generating figures ...")

    fig_dir = BASE / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    eval_path = OUT / "evaluation_metrics.csv"
    if eval_path.exists():
        results_df = pd.read_csv(eval_path)
        test_df = get_test_data()
    else:
        logger.info("  evaluation_metrics.csv not found, running evaluation ...")
        test_df = get_test_data()
        models = load_models()

        from src.evaluate import evaluate_all_models, compute_farm_level_metrics
        from src.predict import predict_power
        config = load_config()

        predictions = {}
        for tb in TURBINES:
            for horizon in HORIZON_NAMES:
                for mdl_name in ["lightgbm", "xgboost"]:
                    target = f"{tb}_power_target_{horizon}"
                    model_key = f"{target}_{mdl_name}"
                    if model_key in models:
                        from src.predict import predict_with_model
                        preds = predict_with_model(models[model_key], test_df)
                        predictions[model_key] = {"predictions": preds, "model_name": mdl_name, "target": target}

                farm_target = f"farm_total_power_target_{horizon}"
                farm_key = f"{farm_target}_{mdl_name}"
                if farm_key in models:
                    from src.predict import predict_with_model
                    preds = predict_with_model(models[farm_key], test_df)
                    predictions[farm_key] = {"predictions": preds, "model_name": mdl_name, "target": farm_target}

        results_df = evaluate_all_models(test_df, predictions, {}, config)
        if not results_df.empty:
            results_df.to_csv(OUT / "evaluation_metrics.csv", index=False)
            farm_metrics_df = compute_farm_level_metrics(test_df, predictions, config)
            if not farm_metrics_df.empty:
                farm_metrics_df.to_csv(OUT / "farm_metrics.csv", index=False)
            generate_metrics()
            from src.evaluate import evaluate_coverage_calibration
            evaluate_coverage_calibration(test_df, predictions, config, output_dir=str(OUT))
            logger.info(f"  evaluation_metrics.csv saved ({len(results_df)} rows)")

    from src.evaluate import (
        plot_performance_heatmap, plot_horizon_decay, plot_radar_summary,
        plot_tb12_distribution, plot_model_comparison, plot_horizon_comparison,
    )

    plot_performance_heatmap(results_df, str(fig_dir / "01_performance_heatmap.png"))
    plot_horizon_decay(results_df, str(fig_dir / "02_horizon_decay.png"))
    plot_radar_summary(results_df, str(fig_dir / "06_radar_summary.png"))
    plot_tb12_distribution(test_df, str(fig_dir / "14_tb12_distribution.png"))
    plot_model_comparison(results_df, str(fig_dir / "07_model_comparison.png"))
    plot_horizon_comparison(results_df, str(fig_dir / "08_horizon_comparison.png"))

    agg_cols = ["mae", "rmse", "nmae_pct", "nrmse_pct", "bias", "r2"]
    available = [c for c in agg_cols if c in results_df.columns]
    summary = results_df.groupby("model")[available].mean().round(4)
    summary.to_csv(fig_dir / "model_summary.csv")

    logger.info(f"  {len(list(fig_dir.glob('*.png')))} figures saved to {fig_dir}")


def convert_csv_to_xlsx():
    """Convert all CSVs in outputs/forecasts/ to XLSX in outputs/xlsx/"""
    logger.info("Converting CSVs to XLSX ...")
    from convert_to_xlsx import convert_all
    results = convert_all()
    for name, rows in results:
        logger.info(f"  {name}: {rows:,} rows -> XLSX")


def generate_failure_risk(test_df):
    """Generate failure_risk.csv per doc section 15 Table 9 Row 5"""
    logger.info("Generating failure_risk.csv ...")

    ts_col = "timestamp"
    if ts_col not in test_df.columns:
        ts_col = test_df.columns[0]

    rows = []
    for tb in TURBINES:
        pwr_col = f"{tb}_power"
        if pwr_col not in test_df.columns:
            continue

        power = test_df[pwr_col].values
        timestamps = test_df[ts_col].values

        valid_mask = ~np.isnan(power)
        if valid_mask.sum() < 100:
            continue
        p_valid = power[valid_mask]
        mean_p = np.mean(p_valid)

        stop_count = 0
        for i in range(len(power)):
            if np.isnan(power[i]):
                continue
            if power[i] < 10 and mean_p > 500:
                stop_count += 1
                if stop_count > 3:
                    risk_score = round(min(0.85, 0.3 + stop_count * 0.05), 4)
                    rows.append({
                        "timestamp": str(timestamps[i]),
                        "turbine_id": tb,
                        "component": "general",
                        "horizon": "24hour",
                        "stop_risk_score": risk_score,
                        "method": "heuristic (stop-count based, not validated against failure labels)",
                        "recommended_action": "Inspect turbine - repeated stops detected",
                    })
            else:
                stop_count = 0

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "failure_risk.csv", index=False)
    logger.info(f"  failure_risk.csv: {df.shape[0]} rows")


def generate_all():
    test_df = get_test_data()
    logger.info(f"Test data: {test_df.shape}")

    models = load_models()
    logger.info(f"Models loaded: {len(models)}")

    generate_power_forecast(test_df, models)
    generate_farm_forecast(test_df, models)
    generate_metrics()
    generate_data_quality_report()
    generate_ramp_alert(test_df)
    generate_anomaly_alert(test_df)
    generate_failure_risk(test_df)
    generate_temperature_warning(test_df)
    generate_coverage_calibration(test_df, models)


def generate_coverage_calibration(test_df, models):
    """Evaluate coverage probability and calibration of conformal CIs."""
    logger.info("Generating coverage_calibration.csv ...")
    config = load_config()
    predictions = {}
    for tb in TURBINES:
        for horizon in HORIZON_NAMES:
            for mdl_name in ["lightgbm", "xgboost"]:
                target = f"{tb}_power_target_{horizon}"
                model_key = f"{target}_{mdl_name}"
                if model_key in models:
                    from src.predict import predict_with_model
                    preds = predict_with_model(models[model_key], test_df)
                    predictions[model_key] = {"predictions": preds, "model_name": mdl_name, "target": target}

            farm_target = f"farm_total_power_target_{horizon}"
            farm_key = f"{farm_target}_{mdl_name}"
            if farm_key in models:
                from src.predict import predict_with_model
                preds = predict_with_model(models[farm_key], test_df)
                predictions[farm_key] = {"predictions": preds, "model_name": mdl_name, "target": farm_target}

    from src.evaluate import evaluate_coverage_calibration
    df = evaluate_coverage_calibration(test_df, predictions, config, output_dir=str(OUT))
    if not df.empty:
        logger.info(f"  coverage_calibration.csv: {df.shape[0]} rows")
    else:
        logger.warning("  coverage_calibration.csv: no data")


def main():
    logger.info("=" * 60)
    logger.info("Generating all required output files (Doc Section 15)")
    logger.info("=" * 60)

    generate_all()

    logger.info("=" * 60)
    logger.info("All output files generated in outputs/forecasts/")
    logger.info("=" * 60)

    for f in sorted(OUT.glob("*.csv")):
        logger.info(f"  {f.name}: {f.stat().st_size:,} bytes")

    generate_figures()
    convert_csv_to_xlsx()


if __name__ == "__main__":
    main()
