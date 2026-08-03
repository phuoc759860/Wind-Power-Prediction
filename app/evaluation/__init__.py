"""Evaluation domain package (thin re-export shims).

Reviewer Step 12: evaluation code is reachable under a stable app/evaluation
namespace. Implementations remain in src/evaluate.py, src/audit.py,
src/split_time_series.py and the dedicated evaluation/ + app/validation/
packages; this package only re-exports them so imports never break.
"""

from __future__ import annotations

from src.audit import (  # noqa: F401
    file_sha256,
    horizon_valid_samples,
    leakage_assertions,
    leakage_audit,
    raw_file_manifest,
    raw_timestamp_union,
    reindex_additions_report,
    ridge_feature_evidence,
    sample_trace,
    timestamp_audit,
    timestamp_audit_csv,
    write_checksums,
    write_sample_traces,
)
from src.evaluate import (  # noqa: F401
    append_baseline_rows,
    apply_farm_bias_correction,
    analyze_farm_bias,
    analyze_tb12,
    compute_and_save_residual_quantiles,
    compute_farm_level_metrics,
    compute_metrics,
    compute_skill_score,
    evaluate_alert_accuracy,
    evaluate_anomaly_detection,
    evaluate_all_models,
    evaluate_coverage_calibration,
    farm_horizon_window_check,
    fit_farm_bias_correction,
    generate_evaluation_report,
)
from src.split_time_series import (  # noqa: F401
    add_time_index,
    get_split_statistics,
    horizon_sample_counts,
    split_by_time,
    walk_forward_split,
)
from evaluation.official_mask import build_official_mask, save_sample_trace  # noqa: F401
from app.validation.leakage_audit import (  # noqa: F401
    audit_model,
    build_full_leakage_audit,
)

__all__ = [
    # audit
    "file_sha256",
    "horizon_valid_samples",
    "leakage_assertions",
    "leakage_audit",
    "raw_file_manifest",
    "raw_timestamp_union",
    "reindex_additions_report",
    "ridge_feature_evidence",
    "sample_trace",
    "timestamp_audit",
    "timestamp_audit_csv",
    "write_checksums",
    "write_sample_traces",
    # evaluate
    "append_baseline_rows",
    "apply_farm_bias_correction",
    "analyze_farm_bias",
    "analyze_tb12",
    "compute_and_save_residual_quantiles",
    "compute_farm_level_metrics",
    "compute_metrics",
    "compute_skill_score",
    "evaluate_alert_accuracy",
    "evaluate_anomaly_detection",
    "evaluate_all_models",
    "evaluate_coverage_calibration",
    "farm_horizon_window_check",
    "fit_farm_bias_correction",
    "generate_evaluation_report",
    # splits
    "add_time_index",
    "get_split_statistics",
    "horizon_sample_counts",
    "split_by_time",
    "walk_forward_split",
    # masks
    "build_official_mask",
    "save_sample_trace",
    # leakage audit
    "audit_model",
    "build_full_leakage_audit",
]
