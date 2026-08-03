"""Step 12: app/* package restructure — thin re-export shims must expose the
same public API as their canonical src/* implementations without breaking
imports. Each domain package is exercised with a smoke call.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest


def _smoke_df():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="10min"),
        "TB01_power": [100.0, 200.0, 300.0, 400.0, 500.0],
        "TB01_wind_speed": [8.0, 9.0, 10.0, 11.0, 12.0],
    })


def test_app_evaluation_exports_official_mask_and_audit():
    from app.evaluation import (  # noqa: F401
        build_official_mask,
        save_sample_trace,
        leakage_audit,
        audit_model,
        split_by_time,
        evaluate_all_models,
    )
    assert callable(build_official_mask)
    assert callable(leakage_audit)


def test_app_evaluation_smoke_build_official_mask():
    from app.evaluation import build_official_mask
    df = _smoke_df()
    df["observed_target"] = 1
    df["target_imputed"] = 0
    df["official_cutoff"] = pd.Timestamp("2026-01-01 01:00:00")
    df["prediction_available"] = 1
    df["feature_available"] = 1
    mask = build_official_mask(df)
    assert mask.dtype == bool
    assert mask.tolist() == [True, True, True, True, True]


def test_app_evaluation_official_mask_submodule():
    from app.evaluation.official_mask import REQUIRED_MASK_COLUMNS, build_official_mask
    assert "observed_target" in REQUIRED_MASK_COLUMNS
    assert callable(build_official_mask)


def test_app_evaluation_leakage_audit_submodule():
    from app.evaluation.leakage_audit import audit_model, build_full_leakage_audit
    assert callable(audit_model)


def test_app_training_exports_models_and_features():
    from app.training import (  # noqa: F401
        build_feature_matrix,
        load_models,
        predict_power,
        preprocess_pipeline,
        train_power_models,
        train_ridge,
        load_config,
    )
    assert callable(build_feature_matrix)
    assert callable(train_power_models)


def test_app_api_exports_app_and_endpoints():
    from app.api import app, predict, predict_farm, predict_champion  # noqa: F401
    assert app is not None
    assert callable(predict)


def test_app_visualization_exports_plot_functions():
    from app.visualization import (  # noqa: F401
        plot_performance_heatmap,
        plot_best_model_scatter,
        plot_coverage_calibration_curve,
    )
    assert callable(plot_performance_heatmap)


def test_src_imports_still_resolve():
    """The shims must not shadow or replace canonical src/* imports."""
    import src.evaluate  # noqa: F401
    import src.api  # noqa: F401
    from src.evaluate import evaluate_all_models
    assert callable(evaluate_all_models)
