"""Visualization domain package (thin re-export shims).

Reviewer Step 12: plotting/visualization code is reachable under a stable
app/visualization namespace. Implementations remain in src/evaluate.py; this
package only re-exports them so imports never break.
"""

from __future__ import annotations

from src.evaluate import (  # noqa: F401
    plot_actual_vs_predicted,
    plot_alert_accuracy_dashboard,
    plot_baseline_comparison,
    plot_best_model_scatter,
    plot_conformal_forecast_timeseries,
    plot_coverage_calibration_curve,
    plot_data_quality_bars,
    plot_error_by_day_night,
    plot_error_by_power_region,
    plot_error_by_season,
    plot_error_by_wind_speed,
    plot_error_histogram,
    plot_failure_risk_heatmap,
    plot_farm_bias_calibration,
    plot_farm_forecast_summary,
    plot_farm_metrics_overview,
    plot_farm_timeseries,
    plot_forecast_quality_distribution,
    plot_horizon_comparison,
    plot_horizon_decay,
    plot_model_comparison,
    plot_model_metrics_by_turbine,
    plot_performance_heatmap,
    plot_radar_summary,
    plot_ramp_alert_timeline,
    plot_residual_analysis,
    plot_tb12_distribution,
)

__all__ = [
    "plot_actual_vs_predicted",
    "plot_alert_accuracy_dashboard",
    "plot_baseline_comparison",
    "plot_best_model_scatter",
    "plot_conformal_forecast_timeseries",
    "plot_coverage_calibration_curve",
    "plot_data_quality_bars",
    "plot_error_by_day_night",
    "plot_error_by_power_region",
    "plot_error_by_season",
    "plot_error_by_wind_speed",
    "plot_error_histogram",
    "plot_failure_risk_heatmap",
    "plot_farm_bias_calibration",
    "plot_farm_forecast_summary",
    "plot_farm_metrics_overview",
    "plot_farm_timeseries",
    "plot_forecast_quality_distribution",
    "plot_horizon_comparison",
    "plot_horizon_decay",
    "plot_model_comparison",
    "plot_model_metrics_by_turbine",
    "plot_performance_heatmap",
    "plot_radar_summary",
    "plot_ramp_alert_timeline",
    "plot_residual_analysis",
    "plot_tb12_distribution",
]
