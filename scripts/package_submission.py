"""Package the submission folder AMG_Wind_Forecasting_Revision/ (doc section 6).

Run AFTER the full pipeline (python main.py --no-wf-ml), pytest and the API
benchmark, so every required evidence file exists. Pure stdlib.
"""
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEST = BASE_DIR / "AMG_Wind_Forecasting_Revision"

AUDIT_EVIDENCE = [
    "timestamp_audit.csv", "leakage_audit.csv", "leakage_audit_ridge.csv",
    "ridge_feature_columns.csv", "split_statistics.json", "inventory_summary.json",
    "raw_coverage_audit.json", "reindex_additions.json", "horizon_sample_counts.json",
    "tb12_analysis.json", "availability_report.json", "alert_screening_summary.json",
    "walk_forward_summary.json", "data_dictionary.csv", "column_mapping.json",
    "validation_report.json", "data_manifest.csv", "checksums.txt",
]

SAMPLE_TRACES = [
    "sample_trace_TB02_10min.csv", "sample_trace_TB02_1hour.csv",
    "sample_trace_TB02_24hour.csv",
]


def _copy(src: Path, dst_dir: Path):
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)


def main():
    if DEST.exists():
        shutil.rmtree(DEST)

    src_dir = BASE_DIR / "02_source_code"
    (DEST / "02_source_code").mkdir(parents=True, exist_ok=True)

    # 01_raw_data_manifest
    _copy(BASE_DIR / "data" / "metadata" / "data_manifest.csv", DEST / "01_raw_data_manifest")
    _copy(BASE_DIR / "data" / "metadata" / "checksums.txt", DEST / "01_raw_data_manifest")

    # 02_source_code
    for d in ["src", "configs", "tests", "scripts"]:
        _dir = BASE_DIR / d
        if _dir.exists():
            shutil.copytree(_dir, DEST / "02_source_code" / d)
    for f in ["main.py", "generate_outputs.py", "generate_report.py",
              "requirements.txt", "run_all.bat", "run_api.bat", "convert_to_xlsx.py"]:
        _copy(BASE_DIR / f, DEST / "02_source_code")

    # 03_audit_evidence
    ev = DEST / "03_audit_evidence"
    ev.mkdir(parents=True, exist_ok=True)
    for name in AUDIT_EVIDENCE:
        _copy(BASE_DIR / "data" / "metadata" / name, ev)
    for name in SAMPLE_TRACES:
        _copy(BASE_DIR / "outputs" / "forecasts" / name, ev)

    # 04_models
    mdir = BASE_DIR / "models"
    if mdir.exists():
        shutil.copytree(mdir, DEST / "04_models")

    # 05_outputs
    for sub in ["forecasts", "figures"]:
        _dir = BASE_DIR / "outputs" / sub
        if _dir.exists():
            shutil.copytree(_dir, DEST / "05_outputs" / sub)

    # 06_test_reports
    tr = DEST / "06_test_reports"
    tr.mkdir(parents=True, exist_ok=True)
    for name in ["pytest_report.txt", "api_benchmark.csv", "run_log.txt"]:
        _copy(BASE_DIR / name, tr)
        _copy(BASE_DIR / "outputs" / "forecasts" / name, tr)
        _copy(BASE_DIR / "logs" / name, tr)
        _copy(BASE_DIR / "06_test_reports" / name, tr)
    logs = BASE_DIR / "logs"
    if logs.exists():
        shutil.copytree(logs, tr / "logs", dirs_exist_ok=True)

    # 07_report
    r7 = DEST / "07_report"
    r7.mkdir(parents=True, exist_ok=True)
    _copy(BASE_DIR / "outputs" / "AMG_Wind_Power_Forecasting_Report_Revised.pdf", r7)
    _copy(BASE_DIR / "data" / "metadata" / "change_log.docx", r7)

    # README_REPRODUCE.md
    _copy(BASE_DIR / "README_REPRODUCE.md", DEST)

    print(f"Submission packaged at: {DEST}")


if __name__ == "__main__":
    main()
