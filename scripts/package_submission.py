"""Package the final submission folder per Latest_Requirements.docx §8.

Creates AMG_Wind_Forecasting_Revision/ with folders 01_Source_Code …
10_Reproduction, then optionally zips it. Pure stdlib.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEST = BASE_DIR / "AMG_Wind_Forecasting_Revision"

METADATA_FILES = [
    "data_manifest.csv", "checksums.txt", "column_mapping.json",
    "data_dictionary.csv", "raw_coverage_audit.json", "reindex_additions.json",
    "split_statistics.json", "horizon_sample_counts.json",
    "inventory_summary.json", "tb12_analysis.json", "availability_report.json",
    "alert_screening_summary.json", "alert_accuracy.json", "anomaly_accuracy.json",
    "walk_forward_summary.json", "leakage_audit.csv", "validation_report.json",
    "change_log.docx",
]

AUDIT_OUTPUTS = [
    "leakage_audit_full.csv", "coverage.csv", "ablation.csv", "cv_results.csv",
    "run_manifest.json", "run_manifest_verification.json",
    "report_validation.json", "validation_report.json",
]

SAMPLE_TRACES = [
    "sample_trace_TB02_10min.csv", "sample_trace_TB02_1hour.csv",
    "sample_trace_TB02_24hour.csv",
]


def _copy(src: Path, dst_dir: Path) -> None:
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)


def _copytree(src: Path, dst: Path) -> None:
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def main(zip_output: bool = True) -> Path:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    # 01_Source_Code
    src_root = DEST / "01_Source_Code"
    for d in ["src", "app", "evaluation", "configs", "tests", "scripts", "static"]:
        _copytree(BASE_DIR / d, src_root / d)
    for f in [
        "main.py", "generate_outputs.py", "generate_report.py", "validate_outputs.py",
        "validate_report.py", "interval_calibration.py", "convert_to_xlsx.py",
        "requirements.txt", "run_all.bat", "run_api.bat", "champion_registry.json",
        "README.md",
    ]:
        _copy(BASE_DIR / f, src_root)

    # 02_Environment
    env = DEST / "02_Environment"
    _copy(BASE_DIR / "requirements.txt", env)
    (env / "python_version.txt").write_text("Python 3.10+\nTested on 3.11 and 3.14\n", encoding="utf-8")
    (env / "setup.md").write_text(
        "# Environment setup\n\n"
        "```bash\nconda create -n wind_forecast python=3.11\n"
        "conda activate wind_forecast\npip install -r requirements.txt\n```\n",
        encoding="utf-8",
    )

    # 03_Config
    cfg = DEST / "03_Config"
    _copytree(BASE_DIR / "configs", cfg / "configs")
    _copy(BASE_DIR / "champion_registry.json", cfg)

    # 04_Data_Metadata
    meta = DEST / "04_Data_Metadata"
    for name in METADATA_FILES:
        _copy(BASE_DIR / "data" / "metadata" / name, meta)

    # 05_Models (champion artifacts + metadata jsons; skip huge dumps if absent)
    models_src = BASE_DIR / "models"
    if models_src.exists():
        _copytree(models_src, DEST / "05_Models")
    _copy(BASE_DIR / "champion_registry.json", DEST / "05_Models")

    # 06_Outputs
    out = DEST / "06_Outputs"
    for sub in ["forecasts", "figures", "xlsx"]:
        _copytree(BASE_DIR / "outputs" / sub, out / sub)
    for name in AUDIT_OUTPUTS:
        _copy(BASE_DIR / "outputs" / name, out)
    for name in SAMPLE_TRACES:
        _copy(BASE_DIR / "outputs" / "forecasts" / name, out / "audit")

    # 07_Tests
    tests = DEST / "07_Tests"
    _copytree(BASE_DIR / "tests", tests / "tests")
    for name in ["pytest_report.txt", "api_benchmark.csv", "run_log.txt"]:
        _copy(BASE_DIR / "06_test_reports" / name, tests)
        _copy(BASE_DIR / "outputs" / "forecasts" / name, tests)
        _copy(BASE_DIR / "logs" / name, tests)

    # 08_API
    api = DEST / "08_API"
    _copy(BASE_DIR / "src" / "api.py", api)
    _copy(BASE_DIR / "run_api.bat", api)
    _copy(BASE_DIR / "scripts" / "benchmark_api.py", api)
    _copy(BASE_DIR / "outputs" / "forecasts" / "api_benchmark.csv", api)
    _copy(BASE_DIR / "06_test_reports" / "api_benchmark.csv", api)
    (api / "security_notes.md").write_text(
        "# API security notes\n\n"
        "- API_KEY must be set via environment variable (fail-closed if unset).\n"
        "- Do not commit real tokens.\n"
        "- CORS is restricted; production must not use --reload.\n"
        "- Benchmarks: see api_benchmark.csv (p50/p95/p99).\n",
        encoding="utf-8",
    )

    # 09_Report
    report = DEST / "09_Report"
    _copy(BASE_DIR / "outputs" / "AMG_Wind_Power_Forecasting_Report_Revised.pdf", report)
    _copy(BASE_DIR / "outputs" / "AMG_Wind_Power_Forecasting_Report.pdf", report)
    _copy(BASE_DIR / "data" / "metadata" / "change_log.docx", report)

    # 10_Reproduction
    repro = DEST / "10_Reproduction"
    _copy(BASE_DIR / "README.md", repro)
    _copy(BASE_DIR / "outputs" / "run_manifest.json", repro)
    _copy(BASE_DIR / "outputs" / "run_manifest_verification.json", repro)
    _copy(BASE_DIR / "run_all.bat", repro)
    (repro / "clean_run.md").write_text(
        "# Clean reproduction\n\n"
        "1. Create env and `pip install -r requirements.txt`\n"
        "2. Place raw SCADA Excel files under `data/raw/`\n"
        "3. `python main.py` (full pipeline) OR `python generate_outputs.py` if models are shipped\n"
        "4. `python generate_report.py`\n"
        "5. `pytest tests/ -q`\n"
        "6. Confirm `outputs/run_manifest.json` commit/hash matches this package\n",
        encoding="utf-8",
    )

    _copy(BASE_DIR / "README.md", DEST)

    print(f"Submission packaged at: {DEST}")

    if zip_output:
        zip_path = BASE_DIR.parent / "AMG_Wind_Forecasting_Revision.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in DEST.rglob("*"):
                if path.is_file():
                    # Skip the huge forecasts.csv dump if present (>100MB)
                    if path.name == "forecasts.csv" and path.stat().st_size > 100_000_000:
                        continue
                    zf.write(path, path.relative_to(DEST.parent))
        print(f"Zip written: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
        return zip_path
    return DEST


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    main(zip_output=not args.no_zip)
