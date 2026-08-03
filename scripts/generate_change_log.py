"""Generate the required change_log.docx (reviewer template, section 8).

Builds a valid .docx with zipfile/stdlib only (no python-docx dependency) so it
runs in a clean environment. Content rows map each review finding to root
cause, fix + files, and post-fix evidence.
"""
import zipfile
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "metadata"

ROWS = [
    ("1", "P0-01: Ridge baseline R² ≈ 1.0 (suspected data leakage)",
     "Baseline evaluation aligned target P(t+h) to the wrong row (index "
     "misalignment), letting Ridge 'predict' the value it was given.",
     "src/train_baseline.py, src/evaluate.py, src/audit.py, main.py",
     "leakage_audit.csv + leakage_audit_ridge.csv (asserts target ∉ X, "
     "timestamp_target = issue + horizon, not np.allclose(y_pred, y)); "
     "evaluation_metrics.csv shows Ridge RMSE in normal range; sample_trace_*.csv"),
    ("2", "P0-02: thời gian và số lượng dữ liệu không thống nhất",
     "Report used different reference dates and row counts; raw holes vs "
     "reindexed rows were mixed.",
     "src/audit.py, main.py",
     "timestamp_audit.csv (raw union, overall + per-year/month coverage), "
     "checksums.txt, data_manifest.csv, raw_coverage_audit.json, "
     "split_statistics.json (exact ranges + coverage_ratio per split)"),
    ("3", "P0-03: Baseline và Forecast Skill chưa tính trên cùng tập mẫu",
     "skill_score was NaN because the re-evaluation path called "
     "evaluate_all_models without baseline predictions.",
     "src/evaluate.py (append_baseline_rows), main.py",
     "outputs/forecasts/evaluation_metrics.csv now includes persistence + "
     "ridge rows and skill_vs_persistence / skill_vs_ridge computed on the "
     "same test samples (n_samples per row)"),
    ("4", "P1-01: Inventory không thống nhất model/artifact/API/test",
     "File lists were hand-maintained in the report.",
     "src/inventory.py, main.py",
     "inventory_summary.json with counts: models per type, API endpoints, "
     "test cases, raw data rows, evaluation rows"),
    ("5", "P1-02: TB12 missing rate chưa được phân tích chi tiết",
     "TB12 breakdown only existed for the test window.",
     "src/evaluate.py (analyze_tb12 split_data), main.py",
     "tb12_analysis.json -> per_split missing/stopped/frozen/mean power for "
     "train/val/test"),
    ("6", "P1-03: Availability chưa tách observed/calendar/coverage",
     "Only one availability figure was reported.",
     "src/train_failure_model.py (compute_availability), main.py",
     "availability_report.json with observed_availability_pct, "
     "calendar_availability_pct, data_coverage_pct per turbine"),
    ("7", "P1-04: Sai lệch hệ thống cấp trang trại chưa hiệu chỉnh",
     "Farm model was evaluated against farm_total_power (no NaNs) while "
     "evaluation rows used rated 2200 kW.",
     "src/evaluate.py (_rated_power_for_target, analyze_farm_bias), main.py",
     "farm_metrics.csv + farm_bias.csv + 25_farm_bias_calibration.png; "
     "evaluation farm rows now use farm rated 26400 kW; bias_kw & "
     "farm_vs_sum_turbines_kw reported"),
    ("8", "P1-05 / Latest P0-07: FAR terminology + screening semantics",
     "CSV column false_alarm_rate stored FDR=1-precision instead of "
     "FPR=FP/(FP+TN); alerts were easy to misread as confirmed fault forecasts.",
     "src/evaluate.py, generate_outputs.py, generate_report.py, "
     "tests/test_compliance.py",
     "alert_accuracy.csv / anomaly_accuracy.csv now export TP/FP/FN/TN, "
     "false_alarm_ratio (FDR), false_alarm_rate (FPR), specificity, "
     "balanced_accuracy, prevalence, method=heuristic_screening, "
     "confirmed=False, verification_status=SCREENING_ONLY; unit test "
     "recomputes FAR formulas from the confusion matrix"),
    ("9", "P2-01: API thiếu bảo mật và benchmark",
     "API used a default key file and no real latency benchmark existed; the "
     "first /predict lazy-loaded models on the request path (unbounded "
     "cold-load tail) with no timeout.",
     "src/api.py, run_api.bat, scripts/api_benchmark.py, scripts/benchmark_api.py",
     "API_KEY env-only fail-closed auth (no default, 503 when unset); "
     "run_api.bat no longer uses --reload; background pre-warm at startup "
     "(PREWARM_MODELS), per-model lock (no thundering-herd double loads), "
     "MODEL_LOAD_TIMEOUT -> 503; api_benchmark.csv with startup time, RAM, "
     "cold-vs-warm p50/p95/p99 and cold-load tail probe (worst observed "
     "request ~0.2s vs reviewer's ~96s claim)"),
    ("10", "P3-01: Báo cáo còn lỗi trình bày và truy vết",
     "ToC had an empty section 8 and hardcoded page numbers that did not match "
     "actual pages; chapters/subsections were numbered inconsistently; endpoints "
     "documented as //health //predict; 'fully compliant' claims made without "
     "citing doc code/version/criterion; conformal status contradicted itself; "
     "long table cells were clipped.",
     "generate_report.py, configs/compliance_matrix.csv, docs/feature_status.md, "
     "run_all.bat, scripts/package_submission.py",
     "ToC now generated by a two-pass render with real page numbers (no empty "
     "entries); subsection numbering fixed (9.x, 11.x); endpoints shown with a "
     "single leading slash; compliance claims cite the concrete Section 15 output "
     "schema instead of an unqualified 'fully compliant'; traceability matrix and "
     "roadmap requirement counts aligned (16 requirements, added 4.16); long table "
     "cells wrap instead of clipping; run_all.bat captures "
     "'pytest tests/ -v > pytest_report.txt' and regenerates the PDF"),
]


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _cell(text: str, width: int) -> str:
    return (f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>'
            f'<w:tcMar><w:left w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tcMar></w:tcPr>'
            f"<w:p><w:pPr><w:spacing w:before=\"20\" w:after=\"20\"/></w:pPr>"
            f"<w:r><w:t>{_escape(text)}</w:t></w:r></w:p></w:tc>")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "change_log.docx"

    width_sum = 900 + 2600 + 3200 + 3200 + 3600
    headers = _cell("STT", 900) + _cell("Vấn đề", 2600) + \
              _cell("Nguyên nhân gốc", 3200) + \
              _cell("Cách sửa và file liên quan", 3200) + \
              _cell("Bằng chứng sau sửa", 3600)
    header_row = f"<w:tr>{headers}</w:tr>"

    body_rows = ""
    for stt, issue, cause, fix, evidence in ROWS:
        body_rows += "<w:tr>" + _cell(stt, 900) + _cell(issue, 2600) + \
                     _cell(cause, 3200) + _cell(fix, 3200) + \
                     _cell(evidence, 3600) + "</w:tr>"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="28"/></w:rPr><w:t>PHIẾU BÁO CÁO THAY ĐỔI HỆ THỐNG DỰ BÁO</w:t></w:r></w:p>
<w:p><w:r><w:t>Ngày tạo: {now}  |  Người nhận xét: GS. TSKH. Ngô Đăng Lưu</w:t></w:r></w:p>
<w:p><w:r><w:t>Nguyên tắc: không sửa số liệu trực tiếp trong PDF; mọi số liệu xuất phát từ lần chạy pipeline cuối cùng.</w:t></w:r></w:p>
<w:tbl><w:tblPr><w:tblW w:w="{width_sum}" w:type="dxa"/><w:tblBorders>
<w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/>
<w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr>
{header_row}{body_rows}
</w:tbl>
<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720"/></w:sectPr>
</w:body>
</w:document>"""

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", doc_xml)

    print(f"change_log.docx written: {out_path} ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
