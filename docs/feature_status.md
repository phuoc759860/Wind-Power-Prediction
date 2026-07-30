# Feature Status & Roadmap

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 4.1 | Split Statistics & Pipeline Architecture | **Implemented** | `get_split_statistics()` reports expected/actual steps, duplicates, missing |
| 4.2 | Turbine Availability & Data Coverage | **Implemented** | 3 availability definitions via `compute_availability()` |
| 4.3 | Time Series Split & Validation | **Implemented** | `split_by_time` + `walk_forward_split` with TimeSeriesSplit |
| 4.4 | Model Training (incl. Walk-Forward) | **Implemented** | XGBoost + LightGBM per turbine × horizon; walk-forward cross-validation |
| 4.5 | Evaluation Metrics | **Implemented** | MAE, RMSE, nRMSE%, R², max_error, skill score |
| 4.6 | Data Quality Report | **Implemented** | Per-column missing_rate, invalid_count, unit_status, definition, data_source |
| 4.7 | Architecture Diagram | **Document-only** | Pipeline diagram in generated report |
| 4.8 | NWP Forecast Quality Caveat | **Implemented** | `forecast_quality` = production (≤1h) / reference_only (6h, 24h) |
| 4.9 | No Feature Leakage | **Implemented** | 7 unit tests verify no lookahead in lags, rolling, diff, ramp, ffill |
| 4.10 | Hyperparameter Tuning | **Implemented** | `_tune_xgboost` / `_tune_lightgbm` with TimeSeriesSplit grid search |
| 4.11 | Reproducibility | **Implemented** | Per-model metadata: seed, git_commit, config_hash, data_hash, dependencies |
| 4.12 | Confidence Intervals (Conformal) | **Implemented** | Residual-quantile per-group CI; coverage calibration evaluated |
| 4.13 | Alert Accuracy Evaluation | **Implemented** | `evaluate_alert_accuracy()` / `evaluate_anomaly_detection()` → CSV + API |
| 4.14 | TB12 Turbine Analysis | **Implemented** | neighbor correlation, wind-speed comparison, stopped rate, frozen data detection |
| 4.15 | Output File Consistency | **Implemented** | 13 CSVs with consistent schema; download whitelist |
| 4.16 | Compliance / Traceability Matrix | **Implemented** | Maps all 16 requirements to files, tests, evidence |
| 5.1 | NWP (Numerical Weather Prediction) Integration | **Planned** | Would improve 6h+ horizon accuracy; requires external NWP data feed |
| 5.2 | Drift Detection & Auto-Retraining | **Planned** | Monitor prediction error drift; trigger retraining when degradation detected |
| 5.3 | SHAP Explainability Dashboard | **Implemented** | Waterfall charts in UI via SHAP |
| 5.4 | Multi-Farm Aggregation | **Planned** | Extend turbine-level models to support multiple wind farm sites |
| 5.5 | Real-Time Data Streaming | **Planned** | Replace periodic batch inference with streaming Kafka/MQTT ingestion |
