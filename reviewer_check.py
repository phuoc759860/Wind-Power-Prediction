#!/usr/bin/env python
"""One-command reviewer acceptance check.

Usage::

    python reviewer_check.py

Prints a PASS/FAIL matrix covering the Latest_Requirements acceptance criteria.
Exit code 0 only when every check passes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))


def check_leakage() -> None:
    path = BASE / "data" / "metadata" / "leakage_audit_full.csv"
    if not path.exists():
        check("Leakage Audit", False, "data/metadata/leakage_audit_full.csv missing")
        return
    df = pd.read_csv(path)
    flagged_col = next((c for c in df.columns if "flag" in c.lower() or c == "status"), None)
    n = len(df)
    if n < 195:
        check("Leakage Audit", False, f"expected >=195 rows, got {n}")
        return
    if flagged_col and flagged_col.lower() == "status":
        bad = df[df[flagged_col].astype(str).str.upper().isin(["FAIL", "FLAGGED", "LEAK"])]
        check("Leakage Audit", bad.empty, f"{n} rows, flagged={len(bad)}")
    else:
        check("Leakage Audit", True, f"{n} rows present")


def check_official_mask() -> None:
    try:
        from evaluation.official_mask import build_official_mask, REQUIRED_MASK_COLUMNS
        check("Official Mask", True, f"module OK; cols={len(REQUIRED_MASK_COLUMNS)}")
    except Exception as exc:
        check("Official Mask", False, str(exc))


def check_champion() -> None:
    path = BASE / "champion_registry.json"
    if not path.exists():
        check("Champion Registry", False, "champion_registry.json missing")
        return
    reg = json.loads(path.read_text(encoding="utf-8"))
    missing = []
    for level in ("turbine", "farm"):
        for horizon in ("10min", "30min", "1hour", "6hour", "24hour"):
            entry = reg.get(level, {}).get(horizon)
            if not entry:
                missing.append(f"{level}/{horizon}")
                continue
            for key in ("model_key", "model_version", "feature_version",
                        "training_cutoff", "run_id"):
                if not entry.get(key):
                    missing.append(f"{level}/{horizon}.{key}")
            mpath = BASE / entry.get("model_path", f"models/{entry.get('model_key')}_model.joblib")
            if not mpath.exists():
                missing.append(f"artifact:{mpath.name}")
    check("Champion Registry", not missing, "OK" if not missing else f"issues={missing[:5]}")


def check_prediction_interval() -> None:
    path = BASE / "outputs" / "coverage.csv"
    if not path.exists():
        check("Prediction Interval", False, "outputs/coverage.csv missing")
        return
    df = pd.read_csv(path)
    need = {"nominal", "coverage", "mean_width", "calibration_error"}
    if not need.issubset(df.columns) or df.empty:
        check("Prediction Interval", False, f"bad schema/empty: {list(df.columns)}")
        return
    # Within ±5 pp of nominal on average for the shipped bands
    err = (df["coverage"] - df["nominal"]).abs().max()
    check("Prediction Interval", err <= 0.05, f"max |coverage-nominal|={err:.4f}")


def check_api() -> None:
    try:
        import os
        os.environ.setdefault("API_KEY", "reviewer-check-key")
        from fastapi.testclient import TestClient
        import src.api as api
        from src.api import app
        api._scan_model_registry()
        api._load_champion_registry()
        client = TestClient(app, raise_server_exceptions=False)
        h = client.get("/health")
        if h.status_code != 200 or h.json().get("status") != "ok":
            check("API", False, f"/health -> {h.status_code}")
            return
        headers = {"Authorization": f"Bearer {os.environ['API_KEY']}"}
        r = client.post("/predict/champion", json={
            "level": "turbine", "horizon": "10min", "turbine_id": "TB01",
            "wind_speed": 8.5, "temperature": 22.0, "frequency": 50.0, "power": 1200,
        }, headers=headers)
        if r.status_code != 200:
            check("API", False, f"/predict/champion -> {r.status_code} {r.text[:120]}")
            return
        data = r.json()
        need = ["selected_model", "model_version", "feature_version",
                "training_cutoff", "run_id", "prediction"]
        missing = [k for k in need if k not in data or data[k] in (None, "", "unknown")]
        # Also check /predict provenance
        r2 = client.post("/predict", json={
            "turbine_id": "TB01", "wind_speed": 8.5, "temperature": 22.0,
            "frequency": 50.0, "power": 1200, "model_type": "lightgbm",
        }, headers=headers)
        if r2.status_code != 200:
            check("API", False, f"/predict -> {r2.status_code}")
            return
        row = r2.json()["predictions"][0]
        need2 = ["selected_model", "model_version", "feature_version",
                 "training_cutoff", "run_id"]
        missing2 = [k for k in need2 if k not in row or not row[k]]
        ok = not missing and not missing2
        check("API", ok, "OK" if ok else f"missing champ={missing} predict={missing2}")
    except Exception as exc:
        check("API", False, str(exc))


def check_outputs() -> None:
    required = [
        "outputs/forecasts/evaluation_metrics.csv",
        "outputs/forecasts/alert_accuracy.csv",
        "outputs/coverage.csv",
        "outputs/run_manifest.json",
        "data/metadata/leakage_audit_full.csv",
    ]
    missing = [p for p in required if not (BASE / p).exists()]
    figs = list((BASE / "outputs" / "figures").glob("*.png"))
    blank = [f.name for f in figs if f.stat().st_size < 1000]
    ok = not missing and len(figs) >= 8 and not blank
    detail = f"figs={len(figs)}"
    if missing:
        detail += f" missing={missing}"
    if blank:
        detail += f" blank={blank[:5]}"
    check("Outputs", ok, detail)


def check_report() -> None:
    pdf = BASE / "outputs" / "AMG_Wind_Power_Forecasting_Report_Revised.pdf"
    metrics = BASE / "outputs" / "forecasts" / "evaluation_metrics.csv"
    cov = BASE / "outputs" / "coverage.csv"
    if not pdf.exists():
        check("Report", False, "PDF missing")
        return
    if pdf.stat().st_size < 50_000:
        check("Report", False, f"PDF too small ({pdf.stat().st_size} bytes)")
        return
    if not metrics.exists() or not cov.exists():
        check("Report", False, "metrics/coverage missing for cross-check")
        return
    # Spot-check: nMAE formula on a sample of evaluation rows
    df = pd.read_csv(metrics)
    sample = df.sample(min(20, len(df)), random_state=0)
    bad = 0
    for _, row in sample.iterrows():
        rp = 26400.0 if str(row["target"]).startswith("farm") else 2200.0
        if abs(row["nmae_pct"] - row["mae"] / rp * 100) > 0.05:
            bad += 1
    check("Report", bad == 0 and len(df) >= 100,
          f"pdf={pdf.stat().st_size // 1024}KB metrics_rows={len(df)} nmae_mismatches={bad}")


def main() -> int:
    check_leakage()
    check_official_mask()
    check_champion()
    check_prediction_interval()
    check_api()
    check_outputs()
    check_report()

    width = max(len(n) for n, _, _ in RESULTS)
    print()
    print("AMG Wind Forecasting — Reviewer Check")
    print("=" * (width + 20))
    failed = 0
    for name, ok, detail in RESULTS:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        pad = "." * (width - len(name) + 2)
        print(f"{name} {pad} {status}  {detail}")
    print("=" * (width + 20))
    print(f"Result: {'ALL PASS' if failed == 0 else f'{failed} FAILED'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
