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

Also performs walk-forward validation (5 chronological folds) for baseline models.

### 5. Start the Web Dashboard

```bash
# Windows
run_api.bat

# Or directly (use 'py -3.13' if your default 'python' has no deps installed)
py -3.13 -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```
⏱ starts in ~3s

First set `API_KEY` (server is fail-closed without it):
```bash
set API_KEY=your-secret-key
```
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

## Compliance Matrix

All 16 requirements (4.1–4.16) are mapped to implementation files, tests, and evidence in `configs/compliance_matrix.csv`.

## Output Files (Doc Section 15 Compliance)

Generated in `outputs/forecasts/`:

| File | Columns | Rows | Description |
|------|---------|------|-------------|
| `power_forecast.csv` | timestamp_issue, timestamp_target, turbine_id, horizon_min, y_pred, y_low, y_high, model_version, forecast_quality | 5.6M | Per-turbine power forecasts with conformal CI |
| `farm_forecast.csv` | timestamp_issue, timestamp_target, horizon_min, farm_power_pred, farm_power_low, farm_power_high, farm_energy_pred, forecast_quality | 468K | Aggregated farm power + energy + CI |
| `metrics.csv` | model, turbine_id, horizon, MAE, nMAE, RMSE, nRMSE, Bias, R2, max_error, skill_score | 130 | Model performance metrics |
| `evaluation_metrics.csv` | target, model, horizon, mae, nmae_pct, rmse, nrmse_pct, bias, r2, max_error, skill_score, n_samples | 130 | Detailed evaluation metrics |
| `farm_metrics.csv` | target, model, horizon, mae, rmse, nmae_pct, nrmse_pct, bias, r2, max_error, n_samples, n_at_capacity, n_zero_power, mae_corrected, rmse_corrected, bias_corrected, r2_corrected, correction_kind, correction_slope, correction_intercept, correction_scalar_kw | 10 | Farm-level metrics: raw vs bias-corrected (P1-04), scored on P(t+h) target |
| `farm_bias.csv` | horizon, n_samples, actual_mean_kw, farm_model_mean_kw, bias_kw, bias_pct_rated, mae_kw, farm_vs_sum_turbines_kw | 5 | Farm direct-model vs sum-of-turbines bias (P1-04) |
| `farm_horizon_window_check.csv` | horizon_a, horizon_b, n_common_samples, window_identical, window_start, window_end, r2_a_on_common, r2_b_on_common, r2_b_minus_a_on_common, n_at_capacity_a_common, n_at_capacity_b_common, n_zero_power_a_common, n_zero_power_b_common | 20 | Same-window/mask horizon R2 comparison (P1-04) |
| `data_quality_report.csv` | column, missing_rate_pct, invalid_values, min, max, unit, remarks, definition, data_source | 48+ | Column-level data quality with formula and source documented |
| `ramp_alert.csv` | timestamp, ramp_type, expected_change, probability, threshold, affected_turbines | 376 | Ramp events detected |
| `failure_risk.csv` | timestamp, turbine_id, component, horizon, stop_risk_score, method, recommended_action | 42K | Turbine stop risk score |
| `anomaly_alert.csv` | timestamp, turbine_id, anomaly_score, suspected_component, evidence | 0+ | Statistical anomalies (z>3.0) |
| `temperature_warning.csv` | timestamp, turbine_id, temperature, warning_type, severity, message | 0+ | Temperature threshold alerts |
| `coverage_calibration.csv` | target, model, horizon, nominal_confidence, empirical_coverage, mean_interval_width, calibration_error, n_samples | 220 | Conformal CI coverage calibration |

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

## Reproduce From Raw Data (Revision)

Everything in the revised report is regenerated from raw SCADA files — nothing is
hand-entered. Run these on a clean machine to reproduce every table and figure.

### 1. Environment (clean machine)

```bash
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### 2. Full pipeline (from raw data, leakage-free)

```bash
.venv\Scripts\python main.py --no-wf-ml --skip-wf
```

- `--no-wf-ml` skips the ~30 min ML walk-forward (baseline walk-forward still
  runs). Remove it for the full validation set.
- `--skip-wf` reuses the existing `walk_forward_summary.json` for quick
  regeneration. For a fully clean run use `.venv\Scripts\python main.py --no-wf-ml`.
- `training.tuning.enabled` in `configs/config.yaml` is `false` by default, so
  XGBoost/LightGBM train with fixed hyperparameters (fast, reproducible). Set it
  to `true` for the TimeSeriesSplit grid search per target (~3-4h).

Pipeline order: load raw xlsx -> manifest/checksums/timestamp audit -> mapping ->
validation -> preprocessing -> features + targets -> time split -> walk-forward
baselines -> ridge (leakage-free) -> LightGBM/XGBoost -> anomaly -> availability
-> evaluation (same-sample persistence + ridge baselines, skill scores) -> sample
traces + leakage assertions -> forecasts -> figures -> output files -> inventory.

### 3. Tests

```bash
.venv\Scripts\python -m pytest tests\ -q
```

### 4. Report (auto-generated from outputs)

```bash
.venv\Scripts\python generate_report.py
```

Output: `outputs/AMG_Wind_Power_Forecasting_Report_Revised.pdf`.

### 5. API + benchmark

```bash
set API_KEY=your-secret-key
run_api.bat            # starts uvicorn WITHOUT --reload (P2-01)
# in another shell:
.venv\Scripts\python scripts\api_benchmark.py
```

- Without `API_KEY` the server is FAIL-CLOSED (protected endpoints -> 503).
- `src/api.py` (P2-01) pre-warms models in a background thread at startup
  (`PREWARM_MODELS=all` default; `0`/`N` to disable/limit), uses one lock per
  model so concurrent cold requests never double-load an artifact, and returns
  503 if a single load exceeds `MODEL_LOAD_TIMEOUT` (default 30s) instead of
  hanging — eliminating the unbounded cold-load tail.
- Full benchmark (boots the server itself, cold lazy-load vs pre-warmed, measures
  startup time, RAM, cold-vs-warm p50/p95/p99, plus a check of the reviewer's
  "~96s max latency" claim):

```bash
.venv\Scripts\python scripts\benchmark_api.py
```

`scripts/benchmark_api.py` writes `06_test_reports/api_benchmark.csv` (delivery
tree) and `outputs/forecasts/api_benchmark.csv` (startup ms, registry scan,
per-endpoint p50/p95/p99, cold first-request latency, RAM baseline/peak,
fail-closed security rows).

### 6. Change log + submission package

```bash
.venv\Scripts\python scripts\generate_change_log.py   # data/metadata/change_log.docx
.venv\Scripts\python scripts\package_submission.py    # AMG_Wind_Forecasting_Revision/
```

### Key outputs (evidence)

| Evidence file | Covers |
|---|---|
| `data/metadata/timestamp_audit.csv` | A02 — raw-union coverage overall + per year/month |
| `data/metadata/checksums.txt`, `data_manifest.csv` | A02 — frozen raw files, sha256 |
| `data/metadata/leakage_audit.csv`, `leakage_audit_ridge.csv`, `ridge_feature_columns.csv` | A01/A03 — no target/future feature in X |
| `outputs/forecasts/sample_trace_TB02_*` (10min/1h/24h) | A01/A03 — issue t -> features -> target P(t+h) -> preds |
| `outputs/forecasts/evaluation_metrics.csv` | A04 — persistence + ridge rows, skill_vs_persistence, skill_vs_ridge, n_samples |
| `data/metadata/split_statistics.json`, `horizon_sample_counts.json` | A05 — exact train/val/test ranges + valid-sample counts |
| `data/metadata/inventory_summary.json` | A06 — unified model/artifact/API/test counts |
| `data/metadata/tb12_analysis.json` (per_split) | A07 — TB12 missing/stopped/frozen breakdown |
| `data/metadata/availability_report.json` | A08 — observed / calendar / data-coverage availability |
| `outputs/forecasts/farm_metrics.csv`, `farm_bias.csv`, `farm_horizon_window_check.csv`, `25_farm_bias_calibration.png` | A09 — farm-level bias + correction + same-window horizon check |
| `data/metadata/alert_screening_summary.json` + alert CSVs | A10 — heuristic screening, not confirmed faults |
| `outputs/forecasts/api_benchmark.csv` | A11 — startup + cold-vs-warm p50/p95/p99 latency, RAM, fail-closed security, cold-load tail probe |
| `logs/wind_forecasting.log`, `run_all.bat` | A12/A13 — reproducible run log |

### Notes on data reality

- Raw SCADA union: 2021-01-01 -> 2026-12-07, ~93.7% 10-min coverage (true raw
  holes kept visible in `timestamp_audit.csv`; gaps are reindexed, and the
  synthetic rows are counted in `reindex_additions.json`).
- File naming vs actual coverage: the last source file is named
  `01.2026-07.2026.xlsx` but its rows actually extend to `2026-12-07 23:50`
  (29 503 rows, a full calendar year). The report therefore states the actual
  observed coverage (-> 07/12/2026) and explains the misnaming; the raw files
  are frozen and never edited.
- **Data-entry flag for the SCADA export owner:** the file name
  `01.2026-07.2026.xlsx` does not match its content (extends ~5 months beyond
  the label, into December 2026). This is a supplier-side mislabel, not a
  pipeline bug. The pipeline already defends against it (evaluation_cutoff
  truncation, `is_simulated` flag, README + Section 3.1 disclosure), but every
  submission cycle the reviewer sees a raw extent that overshoots the report
  date. Please rename/re-split the export at the source (e.g. produce
  `01.2026-07.2026.xlsx` containing only through `2026-07-31`) so the raw
  union ends at the intended boundary.
- Official test window (from `data/metadata/evaluation_window.json`
  `test_window_official_end`, cutoff-truncated at `evaluation_cutoff =
  min(report_date, raw_union_end)`): 2026-01-16 -> 2026-07-31 (28 232 rows).
  The raw test split extends to 2026-12-07 (46 808 rows) but the 18 576 rows
  at/after the report date are flagged `is_simulated=1` and excluded from all
  official metrics.
- TB12 has ~44% missing power (per-split detail in `tb12_analysis.json`).
- 6h/24h models use historical SCADA only (no NWP); day-ahead requires NWP.
- Ramp/anomaly/temperature/failure outputs are heuristic screening flags; they
  are NOT confirmed events until verified against O&M ground truth.

## License

Internal project for AMG Wind Farm internship.
