# README_REPRODUCE — AMG Wind Power Forecasting (Revision)

Reproduce every number in the revised report from raw SCADA files. Nothing is
hand-entered: all tables/figures in the report are regenerated from the outputs
below.

## 1. Environment (clean machine)

- Python 3.12 (64-bit). Create a virtual environment:

```
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

If your environment already has the deps, skip the install.

## 2. Full pipeline (from raw data, leakage-free)

```
python main.py --no-wf-ml --skip-wf
```

- `--no-wf-ml` skips the ~30 min ML walk-forward (baseline walk-forward still
  runs). Remove it for the full validation set.
- `--skip-wf` reuses the existing `walk_forward_summary.json` for quick
  regeneration. For a fully clean run use `python main.py --no-wf-ml` only.
- Hyperparameter tuning: `training.tuning.enabled` in `configs/config.yaml` is
  `false` by default, so XGBoost/LightGBM train with the fixed hyperparameters
  from `models.xgboost`/`models.lightgbm` (fast, reproducible). Set it to `true`
  to run the TimeSeriesSplit grid search per target (~3-4h).

The pipeline runs: load raw xlsx -> manifest/checksums/timestamp audit ->
mapping -> validation -> preprocessing -> features + targets -> time split ->
walk-forward baselines -> ridge (leakage-free) -> LightGBM/XGBoost -> anomaly ->
availability -> evaluation (same-sample persistence + ridge baselines, skill
scores) -> sample traces + leakage assertions -> forecasts -> figures -> all
output files -> provenance/inventory.

## 3. Tests

```
.venv\Scripts\python -m pytest tests\ -q > outputs\forecasts\pytest_report.txt 2>&1
```

## 4. Report (auto-generated from outputs)

```
.venv\Scripts\python generate_report.py
```

Output: `outputs/AMG_Wind_Power_Forecasting_Report_Revised.pdf`.

## 5. API + benchmark

```
set API_KEY=your-secret-key
run_api.bat            # starts uvicorn WITHOUT --reload (P2-01)
# in another shell:
.venv\Scripts\python scripts\api_benchmark.py
```

- Without `API_KEY` the server is FAIL-CLOSED (protected endpoints -> 503).
- Benchmark writes `outputs/forecasts/api_benchmark.csv` (p50/p95/max latency +
  fail-closed checks).

## 6. Change log + submission package

```
.venv\Scripts\python scripts\generate_change_log.py   # data/metadata/change_log.docx
.venv\Scripts\python scripts\package_submission.py    # AMG_Wind_Forecasting_Revision/
```

## Key outputs

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
| `outputs/forecasts/farm_metrics.csv`, `farm_bias.csv`, `25_farm_bias_calibration.png` | A09 — farm-level bias + calibration |
| `data/metadata/alert_screening_summary.json` + alert CSVs | A10 — heuristic screening, not confirmed faults |
| `outputs/forecasts/api_benchmark.csv` | A11 — latency p50/p95 + fail-closed security |
| `logs/wind_forecasting.log`, `run_all.bat` | A12/A13 — reproducible run log |

## Notes on data reality

- Raw SCADA union: 2021-01-01 -> 2026-12-07, ~93.7% 10-min coverage (true raw
  holes kept visible in `timestamp_audit.csv`; gaps are reindexed, and the
  synthetic rows are counted in `reindex_additions.json`).
- File naming vs actual coverage: the last source file is named
  `01.2026-07.2026.xlsx` but its rows actually extend to `2026-12-07 23:50`
  (29 503 rows, a full calendar year). The report therefore states the actual
  observed coverage (-> 07/12/2026) and explains the misnaming; the raw files
  are frozen and never edited.
- Test window: 2026-01-16 -> 2026-12-07 (46 808 rows).
- TB12 has ~44% missing power (per-split detail in `tb12_analysis.json`).
- 6h/24h models use historical SCADA only (no NWP); day-ahead requires NWP.
- Ramp/anomaly/temperature/failure outputs are heuristic screening flags; they
  are NOT confirmed events until verified against O&M ground truth.
