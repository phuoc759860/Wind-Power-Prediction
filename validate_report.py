import io
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import pymupdf
except ImportError:  # pragma: no cover - pymupdf is pinned in requirements
    pymupdf = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
CSV_DIR = BASE / "outputs" / "forecasts"
FIG_DIR = BASE / "outputs" / "figures"
REPORT_SRC = BASE / "generate_report.py"
REPORT_PDF = BASE / "outputs" / "AMG_Wind_Power_Forecasting_Report_Revised.pdf"

H1_PATTERN = re.compile(r'self\.story\.append\(Paragraph\("([^"]+)", s\["SectionH1"\]\)')
FIG_PATTERN = re.compile(r'_(?:figure|fig)\(\s*["\']([^"\']+\.png)["\']')
IMG_PATTERN = re.compile(r'Image\(str\((?:FIG_DIR|self\.fig_dir)\s*/\s*["\']([^"\']+\.png)["\']')

# ---------------------------------------------------------------------------
# Check 2 — every CSV the report documents, with the schema it claims in the
# Section 7 output-files table. `False` second flag = alert/event files that
# may legitimately hold zero rows (no events) but MUST still carry the header.
# ---------------------------------------------------------------------------
DOCUMENTED_CSVS = {
    "outputs/forecasts/power_forecast.csv": (
        ["timestamp_issue", "timestamp_target", "turbine_id", "horizon_min",
         "y_pred", "y_low", "y_high", "forecast_quality"], True),
    "outputs/forecasts/farm_forecast.csv": (
        ["timestamp_issue", "timestamp_target", "horizon_min", "farm_power_pred",
         "farm_power_low", "farm_power_high", "farm_energy_pred", "forecast_quality"], True),
    "outputs/forecasts/evaluation_metrics.csv": (
        ["target", "model", "horizon", "mae", "nmae_pct", "rmse", "nrmse_pct",
         "bias", "r2", "max_error", "skill_score", "skill_vs_persistence",
         "skill_vs_ridge", "n_samples"], True),
    "outputs/forecasts/farm_bias.csv": (
        ["horizon", "n_samples", "actual_mean_kw", "farm_model_mean_kw", "bias_kw",
         "bias_pct_rated", "mae_kw", "farm_vs_sum_turbines_kw"], True),
    "outputs/forecasts/sample_trace_TB02_24hour.csv": (["timestamp"], True),
    "outputs/forecasts/metrics.csv": (["model", "turbine_id", "horizon", "RMSE", "R2"], True),
    "outputs/forecasts/farm_metrics.csv": (["target", "model", "horizon", "mae", "rmse", "r2", "n_samples"], True),
    "outputs/forecasts/data_quality_report.csv": (
        ["column", "missing_rate_pct", "invalid_values", "unit", "definition", "data_source"], True),
    "outputs/forecasts/ramp_alert.csv": (
        ["timestamp", "ramp_type", "expected_change", "probability", "threshold", "affected_turbines"], False),
    "outputs/forecasts/failure_risk.csv": (
        ["timestamp", "turbine_id", "component", "horizon", "stop_risk_score", "recommended_action"], True),
    "outputs/forecasts/anomaly_alert.csv": (
        ["timestamp", "turbine_id", "anomaly_score", "suspected_component", "evidence"], False),
    "outputs/forecasts/temperature_warning.csv": (
        ["timestamp", "turbine_id", "temperature", "warning_type", "severity", "message"], False),
    "outputs/coverage.csv": (["nominal", "coverage", "mean_width", "calibration_error"], True),
    "outputs/forecasts/coverage_calibration.csv": (
        ["target", "model", "horizon", "nominal_confidence", "empirical_coverage",
         "mean_interval_width", "calibration_error", "n_samples", "scope"], True),
    "outputs/forecasts/alert_accuracy.csv": (
        ["turbine_id", "horizon", "model", "tp", "fp", "fn", "tn",
         "precision", "recall", "f1", "false_alarm_ratio", "false_alarm_rate",
         "specificity", "balanced_accuracy", "verification_status"], True),
    "outputs/forecasts/anomaly_accuracy.csv": (
        ["turbine_id", "method", "tp", "fp", "fn", "tn",
         "precision", "recall", "f1", "false_alarm_ratio", "false_alarm_rate",
         "verification_status"], True),
    "outputs/forecasts/farm_horizon_window_check.csv": (
        ["horizon_a", "horizon_b", "n_common_samples", "window_identical",
         "r2_a_on_common", "r2_b_on_common", "r2_b_minus_a_on_common"], True),
    "outputs/forecasts/nwp_ablation.csv": (
        ["target", "horizon", "feature_set", "n_features", "r2", "rmse_kw", "n_samples"], True),
}

# Core metric columns that must be complete (no NaN) in evaluation_metrics.csv
CORE_METRIC_COLUMNS = ["mae", "rmse", "bias", "r2", "n_samples", "skill_score",
                       "skill_vs_persistence", "skill_vs_ridge"]

# Data phrases that must appear in the PDF so the report tables are actually
# populated from real artifacts rather than empty/missing.
PDF_TABLE_MARKERS = [
    "Empirical coverage",
    "Nominal confidence",
    "calibration_error",
    "Requirements Traceability Matrix",
    "API Test Report",
]


def _norm(text: str) -> str:
    return " ".join(text.split())


def _referenced_figures() -> list:
    if not REPORT_SRC.exists():
        raise RuntimeError(f"Report source not found: {REPORT_SRC}")
    src = REPORT_SRC.read_text(encoding="utf-8")
    return sorted(set(FIG_PATTERN.findall(src)) | set(IMG_PATTERN.findall(src)))


def _section_headings() -> list:
    if not REPORT_SRC.exists():
        return []
    return [_norm(h) for h in H1_PATTERN.findall(REPORT_SRC.read_text(encoding="utf-8"))]


def _pixel_variance(data: bytes) -> float:
    """Grayscale pixel standard deviation — ~0 means a blank/uniform image."""
    import numpy as np
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(data)).convert("L")
    arr = np.asarray(img, dtype="float64")
    if arr.size == 0:
        return 0.0
    return float(arr.std())


# ---------------------------------------------------------------------------
# Check 1 — every figure the report references exists and is non-blank
# ---------------------------------------------------------------------------
def validate_figures() -> list:
    figures = _referenced_figures()
    assert len(figures) > 0, "report references no figures"

    checked = []
    for name in figures:
        path = FIG_DIR / name
        assert path.exists(), f"Report references missing figure: {name}"
        assert path.stat().st_size > 0, f"Figure is empty (0 bytes): {name}"
        assert path.read_bytes()[:8].startswith(b"\x89PNG"), f"Not a PNG file: {name}"
        var = _pixel_variance(path.read_bytes())
        assert var > 1e-3, f"Figure is blank (uniform pixels, std={var:.6g}): {name}"
        checked.append({"figure": name, "size": path.stat().st_size, "pixel_std": round(var, 4)})

    logger.info(f"  figures: {len(checked)} referenced PNGs exist and are non-blank OK")
    return checked


# ---------------------------------------------------------------------------
# Check 2 — every CSV the report documents exists with its claimed schema
# ---------------------------------------------------------------------------
def validate_csvs() -> list:
    checked = []
    for rel, (required, must_have_rows) in DOCUMENTED_CSVS.items():
        path = BASE / rel
        assert path.exists(), f"Report documents missing CSV: {rel}"
        assert path.stat().st_size > 0, f"Report documents a blank CSV: {rel}"

        df = pd.read_csv(path)
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"{rel}: missing documented columns {missing}"
        if must_have_rows:
            assert len(df) > 0, f"{rel}: empty CSV but report claims data rows"
        checked.append({"csv": rel, "rows": int(len(df))})

    logger.info(f"  csvs: {len(checked)} documented CSVs exist with the claimed schema OK")
    return checked


# ---------------------------------------------------------------------------
# Check 3 — every evaluation metric exists and is complete
# ---------------------------------------------------------------------------
def validate_metrics() -> dict:
    eval_path = CSV_DIR / "evaluation_metrics.csv"
    assert eval_path.exists(), "evaluation_metrics.csv missing"

    metrics = pd.read_csv(eval_path)
    assert len(metrics) > 0, "evaluation_metrics.csv is empty"
    for col in CORE_METRIC_COLUMNS:
        assert col in metrics.columns, f"evaluation_metrics.csv missing metric column {col!r}"
        assert metrics[col].notna().all(), f"metric column {col!r} contains NaN"
    assert (metrics["n_samples"] > 0).all(), "some metric rows have n_samples == 0"

    # Completeness: the (target, model) grid must cover every target and model
    # the config declares — a silently missing model/horizon must never pass.
    # Each target already encodes its horizon (e.g. TB01_power_target_10min),
    # and the horizon column must be consistent with that suffix.
    per = metrics.groupby("target")["horizon"].nunique()
    assert (per == 1).all(), "a target maps to multiple horizons in evaluation_metrics.csv"

    import yaml
    cfg = yaml.safe_load((BASE / "configs" / "config.yaml").read_text(encoding="utf-8"))
    turbines = cfg.get("turbines", {}).get("ids", [])
    horizons = [h["name"] for h in cfg.get("forecasting", {}).get("horizons", [])]
    expected_targets = ([f"{tb}_power_target_{h}" for tb in turbines for h in horizons]
                        + [f"farm_total_power_target_{h}" for h in horizons])
    expected_models = set(cfg.get("training", {}).get("models", {}).get("ml", []))
    expected_models |= {"ridge", "persistence"}

    present = set(map(tuple, metrics[["target", "model"]].itertuples(index=False)))
    expected = set((t, m) for t in expected_targets for m in expected_models)
    missing_cells = sorted(expected - present)
    assert not missing_cells, f"missing metric cells: {missing_cells[:10]}"

    for rel in ["outputs/forecasts/metrics.csv",
                "outputs/forecasts/farm_metrics.csv",
                "outputs/forecasts/coverage_calibration.csv"]:
        path = BASE / rel
        assert path.exists(), f"missing metric file {rel}"
        assert len(pd.read_csv(path)) > 0, f"metric file {rel} is empty"

    cov = pd.read_csv(BASE / "outputs/forecasts/coverage_calibration.csv")
    assert cov["empirical_coverage"].between(0.0, 1.0).all(), "coverage out of [0, 1]"

    logger.info(f"  metrics: {len(metrics):,} rows, full {len(expected):,}-cell "
                f"(target x model) grid, no NaN OK")
    return {"evaluation_rows": int(len(metrics)), "cells": len(expected)}

# ---------------------------------------------------------------------------
# Check 4 — no missing tables in the PDF (every section heading + data markers)
# ---------------------------------------------------------------------------
def validate_pdf_tables(doc, text: str) -> dict:
    headings = _section_headings()
    assert headings, "no SectionH1 headings found in report source"
    norm_text = _norm(text)

    missing_sections = [h for h in headings if _norm(h) not in norm_text]
    assert not missing_sections, f"PDF missing sections: {missing_sections}"

    missing_markers = [m for m in PDF_TABLE_MARKERS if _norm(m) not in norm_text]
    assert not missing_markers, f"PDF missing table content: {missing_markers}"

    assert doc.page_count >= 30, f"suspiciously short report: {doc.page_count} pages"

    logger.info(f"  tables: {len(headings)} sections present across "
                f"{doc.page_count} pages, data markers found OK")
    return {"pages": doc.page_count, "sections": len(headings)}


# ---------------------------------------------------------------------------
# Check 5 — no blank figures embedded in the PDF
# ---------------------------------------------------------------------------
def validate_pdf_figures(doc) -> list:
    assert pymupdf is not None, "pymupdf not installed; cannot inspect PDF"

    referenced = _referenced_figures()
    embedded = []
    for page_no in range(doc.page_count):
        for img in doc.get_page_images(page_no, full=True):
            xref = img[0]
            info = doc.extract_image(xref)
            raw = info["image"]
            assert len(raw) > 0, f"PDF embeds a zero-byte image (page {page_no + 1}, xref {xref})"
            var = _pixel_variance(raw)
            assert var > 1e-3, f"PDF embeds a blank image (page {page_no + 1}, xref {xref}, std={var:.6g})"
            embedded.append({"xref": xref, "page": page_no + 1, "pixel_std": round(var, 4)})

    assert len(embedded) >= len(referenced), (
        f"PDF embeds {len(embedded)} images but the report references "
        f"{len(referenced)} figures")

    logger.info(f"  pdf-figures: {len(embedded)} embedded images, all non-blank OK")
    return embedded


def main() -> dict:
    logger.info("=" * 60)
    logger.info("VALIDATING PDF REPORT (fail-closed)")
    logger.info("=" * 60)

    checks = {}
    checks["figures"] = validate_figures()

    checks["csvs"] = validate_csvs()

    checks["metrics"] = validate_metrics()

    assert REPORT_PDF.exists(), f"Report PDF not found: {REPORT_PDF}"
    assert pymupdf is not None, "pymupdf not installed; cannot inspect PDF"
    doc = pymupdf.open(str(REPORT_PDF))
    text = "\n".join(page.get_text() for page in doc)

    checks["tables"] = validate_pdf_tables(doc, text)
    checks["pdf_figures"] = validate_pdf_figures(doc)
    checks["pdf"] = {"pages": doc.page_count, "text_chars": len(text)}

    report = {
        "validated_at": datetime.now().isoformat(timespec="seconds"),
        "report_pdf": str(REPORT_PDF),
        "status": "PASS",
        "checks": checks,
    }
    out_path = BASE / "outputs" / "report_validation.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    logger.info(f"  report_validation.json written: {out_path}")
    logger.info("=" * 60)
    logger.info("REPORT VALIDATION PASSED")
    logger.info("=" * 60)
    return report


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
