# AMG Wind Power Forecasting System

A complete machine learning pipeline for multi-horizon wind power forecasting at AMG Wind Farm (12 turbines, 26.4 MW total capacity). Includes data processing, model training, evaluation, REST API, and web dashboard.

## Project Structure

```
wind_forecasting/
├── main.py                    # Full 13-step pipeline orchestrator
├── generate_outputs.py        # Standalone output file generator (doc Section 15)
├── run_all.bat                # One-click pipeline launcher (Windows)
├── run_api.bat                # Windows launcher for API server
├── requirements.txt           # Python dependencies
├── configs/
│   └── config.yaml            # All project settings (horizons, models, features, etc.)
├── src/
│   ├── api.py                 # FastAPI REST server + dashboard
│   ├── load_data.py           # Raw Excel data loader
│   ├── column_mapping.py      # SCADA column name standardization
│   ├── data_validation.py     # Data quality checks
│   ├── preprocessing.py       # Data cleaning/transformation
│   ├── feature_engineering.py # Feature creation (lags, rolling, temporal, interactions)
│   ├── split_time_series.py   # Time-based train/val/test splitting
│   ├── train_baseline.py      # Persistence + linear regression baselines
│   ├── train_power_model.py   # XGBoost & LightGBM model training
│   ├── train_anomaly_model.py # Isolation forest anomaly detection
│   ├── train_failure_model.py # Failure risk analysis
│   ├── predict.py             # Model inference with confidence intervals
│   └── evaluate.py            # Metrics, skill scores, summary plots
├── tests/
│   ├── test_api.py            # 16 API endpoint tests
│   └── test_wind_forecasting.py
├── data/
│   ├── raw/                   # 11 SCADA Excel files (2021-2026)
│   ├── processed/             # Processed parquet file
│   └── metadata/              # Validation reports, data dictionary, availability
├── models/                    # 426 trained model artifacts
├── outputs/
│   ├── forecasts/             # 9 output CSV files per doc Section 15
│   └── figures/               # 8 visualization PNGs
├── static/
│   └── index.html             # Web dashboard frontend
└── logs/
    └── wind_forecasting.log
```

## Requirements

- **Python 3.10+** (tested on 3.10, 3.11)
- **OS**: Windows 10/11, Linux, macOS
- **RAM**: 16 GB minimum (32 GB recommended for training)
- **Disk**: 2 GB free space

## Quick Start

### 1. Create Environment (Conda recommended)

```bash
# Conda
conda create -n wind_forecast python=3.11
conda activate wind_forecast

# Or venv
python -m venv venv
# Windows: venv\Scripts\activate
# Linux:   source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
⏱ **~2-5 min** (depends on internet speed)

### 3. Generate Output Files (Fast — no training needed)

If the zip includes pre-trained models, you can skip training and go straight to output generation:

```bash
python generate_outputs.py
```
⏱ **~3-5 min** (loads processed data → feature engineering → model inference → 7 CSV files)

Outputs appear in `outputs/forecasts/`.

### 4. Run the Full Pipeline (Train Everything)

```bash
# Windows
run_all.bat

# Or manually
python main.py
```

⏱ **~10-15 min** (first run loads 11 Excel files)
⏱ **~6-10 min** (subsequent runs use cached parquet)

This executes all 13 steps: load data → column mapping → validation → preprocessing → feature engineering → time split → train baselines (persistence + ridge) → train ML models (XGBoost + LightGBM) → anomaly detection → failure analysis → generate predictions → create forecast output → summary visualizations.

Also performs walk-forward validation (5 folds) for baseline models.

### 5. Start the Web Dashboard

```bash
# Windows
run_api.bat

# Or directly
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```
⏱ starts in ~3s

Open http://localhost:8000 in your browser. API docs at http://localhost:8000/docs.

### 6. Run Tests

```bash
# Full test suite
python -m pytest tests/ -v

# API tests only
python -m pytest tests/test_api.py -v
```
⏱ **~2-5 min**

## Configuration

All settings are in `configs/config.yaml`:

| Section | Key Settings |
|---------|-------------|
| Farm | AMG Wind Farm, 12 turbines (TB01-TB12), 2200 kW each |
| Turbine specs | Cut-in 3 m/s, Cut-out 25 m/s, Rated speed 12 m/s |
| Data | 10-min SCADA sampling, timezone Asia/Ho_Chi_Minh |
| Horizons | 10min, 30min, 1hour, 6hour, 24hour |
| ML models | XGBoost (150 trees, depth 6) + LightGBM (150 trees, depth 6) |
| Features | Lags [1,2,3,6,12,144], rolling [6,18,36,144], temporal, interactions |
| Split | 70% train / 15% validation / 15% test (time-based) |
| Evaluation | MAE, nMAE, RMSE, nRMSE, Bias, R2, Skill Score, Max Error |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web dashboard |
| GET | `/health` | Server status, model count |
| GET | `/turbines` | 12 turbines with availability data |
| GET | `/models` | All loaded models grouped by turbine |
| GET | `/evaluations` | 130 evaluation metric rows |
| POST | `/predict` | Single turbine multi-horizon forecast |
| POST | `/predict/farm` | Farm-wide power forecast |
| GET | `/outputs/metrics` | Model metrics (MAE, RMSE, R2, skill score) |
| GET | `/outputs/power-forecast` | Per-turbine predictions with CI |
| GET | `/outputs/farm-forecast` | Farm-level predictions |
| GET | `/outputs/ramp-alerts` | Ramp event detection |
| GET | `/outputs/anomaly-alerts` | Anomaly detection results |
| GET | `/outputs/failure-risk` | Turbine failure risk |
| GET | `/outputs/data-quality` | Data quality report |
| GET | `/download/{filename}` | Download output CSV files |

## Output Files (Doc Section 15 Compliance)

Generated in `outputs/forecasts/`:

| File | Columns | Rows | Description |
|------|---------|------|-------------|
| `power_forecast.csv` | timestamp_issue, timestamp_target, turbine_id, horizon_min, y_pred, y_low, y_high, model_version | 5.6M | Per-turbine power forecasts with 95% CI |
| `farm_forecast.csv` | timestamp_issue, timestamp_target, horizon_min, farm_power_pred, farm_energy_pred | 468K | Aggregated farm power + energy |
| `metrics.csv` | model, turbine_id, horizon, MAE, nMAE, RMSE, nRMSE, Bias, R2, skill_score | 130 | Model performance metrics |
| `evaluation_metrics.csv` | target, model, horizon, mae, nmae_pct, rmse, nrmse_pct, bias, r2, max_error, skill_score, n_samples | 130 | Detailed evaluation metrics |
| `farm_metrics.csv` | horizon, model, mae, rmse, nmae_pct, nrmse_pct, bias, r2 | 10 | Farm-level metrics (direct on summed power) |
| `data_quality_report.csv` | column, missing_rate, invalid_count, min, max, unit_status, remarks | 115 | Column-level data quality |
| `ramp_alert.csv` | timestamp, ramp_type, expected_change, probability, threshold, affected_turbines | 376 | Ramp events detected |
| `failure_risk.csv` | timestamp, turbine_id, component, horizon, failure_probability, recommended_action | 42K | Turbine failure risk |
| `anomaly_alert.csv` | timestamp, turbine_id, anomaly_score, suspected_component, evidence | 0+ | Statistical anomalies (z>3.0) |

## Testing

```bash
# Run all 16 API tests
python -m pytest tests/test_api.py -v

# Run full test suite
python -m pytest tests/ -v
```

## Models

426 artifacts in `models/`, named as:

```
{turbine}_power_target_{horizon}_{algorithm}_{artifact}.{ext}
```

- **Turbines**: TB01-TB12 + farm_total_power (13 groups)
- **Horizons**: 10min, 30min, 1hour, 6hour, 24hour
- **Algorithms**: xgboost, lightgbm
- **Artifacts**: _model.joblib, _scaler.joblib, _features.json

## Data

11 semi-annual SCADA Excel files in `data/raw/`, covering January 2021 to July 2026. Each file contains 49 columns across 12 turbines: wind speed, temperature, power output, and frequency at 10-minute intervals.

Actual row counts per split are computed by the pipeline and saved to `data/metadata/split_statistics.json`. Run `python main.py` to regenerate them from the current data files rather than relying on hardcoded estimates.

## Key Formulas (from Vietnamese Doc)

**Ramp Rate:**
```
Ramp Rate = (P(t+Δt) - P(t)) / Δt    [MW/min]
```

**Forecast Error:**
```
E_target = F_target - O_target
```

**Skill Score:**
```
Skill Score = 1 - (RMSE_model / RMSE_baseline)
```

**NRMSE:**
```
nRMSE = (RMSE / P_rated) × 100%
```

## License

Internal project for AMG Wind Farm internship.
