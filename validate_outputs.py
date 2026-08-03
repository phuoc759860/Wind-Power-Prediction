import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
FORECAST_DIR = OUT / "forecasts"
FIG_DIR = OUT / "figures"
META_DIR = BASE / "data" / "metadata"

REQUIRED_EVAL_COLUMNS = ["target", "model", "horizon", "mae", "rmse", "r2", "n_samples"]
REQUIRED_COVERAGE_COLUMNS = ["nominal", "coverage", "mean_width", "calibration_error"]

REQUIRED_OUTPUT_FILES = [
    "coverage.csv",
    "run_manifest.json",
    FORECAST_DIR / "coverage_calibration.csv",
    FORECAST_DIR / "data_quality_report.csv",
    FORECAST_DIR / "farm_forecast.csv",
    FORECAST_DIR / "nwp_ablation.csv",
    FORECAST_DIR / "power_forecast.csv",
]

REQUIRED_METADATA_FILES = [
    "checksums.txt",
    "data_manifest.csv",
    "evaluation_window.json",
    "horizon_sample_counts.json",
    "inventory_summary.json",
    "leakage_audit.csv",
    "leakage_audit_full.csv",
    "raw_coverage_audit.json",
    "reindex_additions.json",
    "ridge_feature_columns.csv",
    "split_statistics.json",
    "timestamp_audit.csv",
    "walk_forward_summary.json",
]


def _resolve_metrics_path() -> Path:
    """evaluation_metrics.csv lives under outputs/forecasts/; the reviewer
    shorthand outputs/evaluation_metrics.csv is accepted too."""
    for cand in (OUT / "evaluation_metrics.csv", FORECAST_DIR / "evaluation_metrics.csv"):
        if cand.exists():
            return cand
    return FORECAST_DIR / "evaluation_metrics.csv"


def validate_metrics() -> Path:
    # Reviewer check 1: the metrics file must exist.
    path = _resolve_metrics_path()
    assert os.path.exists(str(path)), (
        f"Missing evaluation_metrics.csv (searched {OUT / 'evaluation_metrics.csv'} "
        f"and {FORECAST_DIR / 'evaluation_metrics.csv'})")

    metrics = pd.read_csv(path)
    # Reviewer check 2: at least one metric row with the expected schema.
    assert len(metrics) > 0, f"{path.name} is empty (len(metrics) == 0)"
    for col in REQUIRED_EVAL_COLUMNS:
        assert col in metrics.columns, f"{path.name} missing required column {col!r}"
    assert metrics["n_samples"].notna().all(), f"{path.name} contains NaN n_samples"
    assert (metrics["n_samples"] > 0).all(), f"{path.name} contains zero-sample metric rows"

    logger.info(f"  {path.name}: {len(metrics):,} metric rows x {metrics.shape[1]} cols OK")
    return path


def validate_figures() -> list:
    figures = sorted(FIG_DIR.glob("*.png"))
    assert len(figures) > 0, f"No figures found in {FIG_DIR}"

    # Reviewer check 3: every figure must be a non-empty file.
    for fig in figures:
        assert fig.stat().st_size > 0, f"Figure {fig.name} is empty (st_size == 0)"

    logger.info(f"  figures: {len(figures)} non-empty PNGs OK")
    return figures


def validate_coverage() -> None:
    cov_path = OUT / "coverage.csv"
    assert cov_path.exists(), f"Missing {cov_path.relative_to(BASE)}"
    cov = pd.read_csv(cov_path)
    assert len(cov) > 0, f"{cov_path.name} is empty"
    for col in REQUIRED_COVERAGE_COLUMNS:
        assert col in cov.columns, f"{cov_path.name} missing required column {col!r}"
    logger.info(f"  coverage.csv: {len(cov)} coverage rows OK")


def validate_champion_registry() -> None:
    reg_path = BASE / "champion_registry.json"
    assert reg_path.exists(), "Missing champion_registry.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    assert len(reg) > 0, "champion_registry.json is empty"
    logger.info(f"  champion_registry.json: {len(reg)} champions OK")


def validate_required_files() -> None:
    for rel in REQUIRED_OUTPUT_FILES:
        path = rel if isinstance(rel, Path) else OUT / rel
        assert path.exists(), f"Missing required output: {path.relative_to(BASE)}"
        assert path.stat().st_size > 0, f"Required output is empty: {path.relative_to(BASE)}"
    logger.info(f"  outputs: {len(REQUIRED_OUTPUT_FILES)} required files non-empty OK")


def validate_metadata() -> None:
    missing = [name for name in REQUIRED_METADATA_FILES if not (META_DIR / name).exists()]
    assert not missing, f"Missing required metadata files: {missing}"
    logger.info(f"  metadata: {len(REQUIRED_METADATA_FILES)} files OK")


def main() -> dict:
    logger.info("=" * 60)
    logger.info("VALIDATING PIPELINE OUTPUTS (fail-closed)")
    logger.info("=" * 60)

    checks = {}

    metrics_path = validate_metrics()
    metrics = pd.read_csv(metrics_path)
    checks["evaluation_metrics.csv"] = {
        "exists": True,
        "rows": int(len(metrics)),
        "columns": [c for c in REQUIRED_EVAL_COLUMNS if c in metrics.columns],
    }

    figures = validate_figures()
    checks["figures"] = {
        "count": len(figures),
        "all_non_empty": all(f.stat().st_size > 0 for f in figures),
    }

    validate_coverage()
    checks["coverage.csv"] = {"exists": True, "rows": int(len(pd.read_csv(OUT / "coverage.csv")))}

    validate_champion_registry()
    checks["champion_registry.json"] = {"exists": True}

    validate_required_files()
    validate_metadata()

    report = {
        "validated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS",
        "checks": checks,
    }
    out_path = OUT / "validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  validation_report.json written: {out_path}")

    logger.info("=" * 60)
    logger.info("OUTPUT VALIDATION PASSED")
    logger.info("=" * 60)
    return report


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
