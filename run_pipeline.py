#!/usr/bin/env python
"""Single-command end-to-end pipeline for clean-environment reproduction.

Usage (from a fresh venv with ``pip install -r requirements.txt``)::

    python run_pipeline.py

What it regenerates
-------------------
1. Full training / evaluation / audit pipeline (``main.py --no-wf-ml``)
2. Technical PDF report (``generate_report.py``)
3. Fail-closed output + report validators
4. Final ``outputs/run_manifest.json`` snapshot

Optional flags::

    python run_pipeline.py --full-wf      # also run ML walk-forward (~+30 min)
    python run_pipeline.py --skip-report  # training/audit only
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent

REQUIRED_AFTER = [
    "data/metadata/leakage_audit_full.csv",
    "outputs/run_manifest.json",
    "outputs/coverage.csv",
    "outputs/forecasts/evaluation_metrics.csv",
    "outputs/AMG_Wind_Power_Forecasting_Report_Revised.pdf",
    "champion_registry.json",
]


def _setup_logging() -> logging.Logger:
    (BASE / "logs").mkdir(parents=True, exist_ok=True)
    log_path = BASE / "logs" / "run_pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger("run_pipeline")


def _run(cmd: list[str], logger: logging.Logger, step: str) -> None:
    logger.info("=" * 70)
    logger.info("STEP: %s", step)
    logger.info("CMD : %s", " ".join(cmd))
    logger.info("=" * 70)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(BASE))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        logger.error("FAIL  %s (exit=%s, %.1fs)", step, proc.returncode, elapsed)
        raise SystemExit(proc.returncode)
    logger.info("PASS  %s (%.1fs)", step, elapsed)


def _verify_artifacts(logger: logging.Logger) -> None:
    logger.info("-" * 70)
    logger.info("Verifying regenerated artifacts ...")
    missing = []
    for rel in REQUIRED_AFTER:
        path = BASE / rel
        ok = path.exists() and path.stat().st_size > 0
        status = "PASS" if ok else "FAIL"
        size = path.stat().st_size if path.exists() else 0
        logger.info("  [%s] %s (%s bytes)", status, rel, size)
        if not ok:
            missing.append(rel)

    models = list((BASE / "models").glob("*_model.joblib"))
    figs = list((BASE / "outputs" / "figures").glob("*.png"))
    logger.info("  models/*.joblib : %d", len(models))
    logger.info("  figures/*.png   : %d", len(figs))
    if len(models) < 100:
        missing.append(f"models (found {len(models)}, expected >= 100)")
    if len(figs) < 8:
        missing.append(f"figures (found {len(figs)}, expected >= 8)")

    if missing:
        logger.error("Missing / empty artifacts: %s", missing)
        raise SystemExit(2)
    logger.info("All required artifacts present.")


def main() -> int:
    parser = argparse.ArgumentParser(description="AMG Wind clean-run pipeline")
    parser.add_argument("--full-wf", action="store_true",
                        help="Also run ML walk-forward validation (~+30 min)")
    parser.add_argument("--skip-wf", action="store_true",
                        help="Skip baseline walk-forward (reuse prior walk_forward_summary.json)")
    parser.add_argument("--skip-report", action="store_true",
                        help="Skip PDF generation and report validation")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip fail-closed validators")
    args = parser.parse_args()

    logger = _setup_logging()
    started = datetime.now(timezone.utc).isoformat()
    logger.info("AMG Wind Forecasting — clean pipeline start %s", started)
    logger.info("Python: %s", sys.version.replace("\n", " "))
    logger.info("CWD   : %s", BASE)

    # Ensure directories exist before main.py configures its FileHandler.
    for d in ["logs", "models", "outputs", "outputs/forecasts", "outputs/figures",
              "data/processed", "data/metadata"]:
        (BASE / d).mkdir(parents=True, exist_ok=True)

    py = sys.executable
    main_cmd = [py, "main.py"]
    if not args.full_wf:
        main_cmd.append("--no-wf-ml")
    if args.skip_wf:
        main_cmd.append("--skip-wf")

    _run(main_cmd, logger, "Train + evaluate + audit (main.py)")

    if not args.skip_report:
        _run([py, "generate_report.py"], logger, "Generate PDF report")

    if not args.skip_validate:
        _run([py, "validate_outputs.py"], logger, "Validate outputs (fail-closed)")
        if not args.skip_report:
            _run([py, "validate_report.py"], logger, "Validate PDF report (fail-closed)")

    _verify_artifacts(logger)
    logger.info("PIPELINE COMPLETE at %s", datetime.now(timezone.utc).isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
