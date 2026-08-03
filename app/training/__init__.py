"""Training domain package (thin re-export shims).

Reviewer Step 12: training/feature/model code is reachable under a stable
app/training namespace. Implementations remain in the src/* modules; this
package only re-exports them so imports never break.
"""

from __future__ import annotations

from src.column_mapping import (  # noqa: F401
    apply_column_mapping,
    build_column_mapping,
    create_data_dictionary,
    extract_measurement_type,
    extract_numeric_id,
    extract_turbine_id,
    get_all_turbine_measurements,
    get_power_columns,
    get_temperature_columns,
    get_turbine_columns,
    get_wind_speed_columns,
)
from src.data_validation import (  # noqa: F401
    run_validation,
    validate_missing_data,
    validate_power_consistency,
    validate_timestamps,
    validate_value_ranges,
)
from src.feature_engineering import (  # noqa: F401
    build_feature_matrix,
    create_change_features,
    create_interaction_features,
    create_lag_features,
    create_ramp_features,
    create_rolling_features,
    create_target_columns,
    create_temporal_features,
)
from src.load_data import (  # noqa: F401
    get_data_info,
    load_all_data,
    load_config,
    load_processed_data,
    load_single_file,
    save_processed_data,
)
from src.nwp import (  # noqa: F401
    add_nwp_features,
    build_stub_nwp,
    load_nwp,
    run_nwp_ablation,
)
from src.preprocessing import (  # noqa: F401
    clip_physically_implausible,
    compute_farm_avg_wind,
    compute_farm_total_power,
    create_missing_flags,
    detect_operating_status,
    enforce_sampling_interval,
    handle_missing_values,
    preprocess_pipeline,
    remove_duplicates,
)
from src.predict import (  # noqa: F401
    add_confidence_intervals,
    create_forecast_output,
    predict_power,
    predict_with_model,
    save_forecasts,
)
from src.train_anomaly_model import (  # noqa: F401
    compute_residual_anomalies,
    detect_anomalies_isolation_forest,
    get_anomaly_summary,
    run_anomaly_detection,
    train_isolation_forest,
)
from src.train_baseline import (  # noqa: F401
    base_power_col,
    build_test_baseline_predictions,
    evaluate_persistence,
    evaluate_ridge,
    is_feature_column,
    persistence_predictions,
    ridge_predictions,
    select_feature_columns,
    train_baselines,
    train_ridge,
    walk_forward_baselines,
)
from src.train_failure_model import (  # noqa: F401
    compute_availability,
    detect_failure_events,
    get_failure_summary,
    run_failure_analysis,
)
from src.train_power_model import (  # noqa: F401
    load_models,
    prepare_features,
    save_models,
    scale_features,
    train_lightgbm,
    train_linear_reg,
    train_power_models,
    train_random_forest,
    train_xgboost,
    walk_forward_all_ml,
    walk_forward_ml,
)

__all__ = [
    # column mapping
    "apply_column_mapping",
    "build_column_mapping",
    "create_data_dictionary",
    "extract_measurement_type",
    "extract_numeric_id",
    "extract_turbine_id",
    "get_all_turbine_measurements",
    "get_power_columns",
    "get_temperature_columns",
    "get_turbine_columns",
    "get_wind_speed_columns",
    # data validation
    "run_validation",
    "validate_missing_data",
    "validate_power_consistency",
    "validate_timestamps",
    "validate_value_ranges",
    # feature engineering
    "build_feature_matrix",
    "create_change_features",
    "create_interaction_features",
    "create_lag_features",
    "create_ramp_features",
    "create_rolling_features",
    "create_target_columns",
    "create_temporal_features",
    # load data
    "get_data_info",
    "load_all_data",
    "load_config",
    "load_processed_data",
    "load_single_file",
    "save_processed_data",
    # nwp
    "add_nwp_features",
    "build_stub_nwp",
    "load_nwp",
    "run_nwp_ablation",
    # preprocessing
    "clip_physically_implausible",
    "compute_farm_avg_wind",
    "compute_farm_total_power",
    "create_missing_flags",
    "detect_operating_status",
    "enforce_sampling_interval",
    "handle_missing_values",
    "preprocess_pipeline",
    "remove_duplicates",
    # predict
    "add_confidence_intervals",
    "create_forecast_output",
    "predict_power",
    "predict_with_model",
    "save_forecasts",
    # anomaly
    "compute_residual_anomalies",
    "detect_anomalies_isolation_forest",
    "get_anomaly_summary",
    "run_anomaly_detection",
    "train_isolation_forest",
    # baselines
    "base_power_col",
    "build_test_baseline_predictions",
    "evaluate_persistence",
    "evaluate_ridge",
    "is_feature_column",
    "persistence_predictions",
    "ridge_predictions",
    "select_feature_columns",
    "train_baselines",
    "train_ridge",
    "walk_forward_baselines",
    # failure
    "compute_availability",
    "detect_failure_events",
    "get_failure_summary",
    "run_failure_analysis",
    # power models
    "load_models",
    "prepare_features",
    "save_models",
    "scale_features",
    "train_lightgbm",
    "train_linear_reg",
    "train_power_models",
    "train_random_forest",
    "train_xgboost",
    "walk_forward_all_ml",
    "walk_forward_ml",
]
