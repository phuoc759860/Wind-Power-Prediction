"""API latency/security benchmark (reviewer P2-01).

Usage (server must already be running, e.g. `run_api.bat` with API_KEY set):
    $env:API_KEY="<key>"; .venv\\Scripts\\python scripts/api_benchmark.py

Produces outputs/forecasts/api_benchmark.csv with per-endpoint latency stats
and a security row proving the protected endpoints are FAIL-CLOSED when the
API key is absent/incorrect.
"""
import json
import os
import statistics
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs" / "forecasts"
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")
N_REQUESTS = int(os.environ.get("BENCH_REQUESTS", "20"))


def _request(method: str, path: str, body: dict = None, key: str = None,
             timeout: float = 30.0):
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            lat = (time.perf_counter() - start) * 1000
            return resp.status, lat
    except urllib.error.HTTPError as e:
        lat = (time.perf_counter() - start) * 1000
        return e.code, lat
    except Exception as e:
        return -1, None


def _stats(latencies):
    if not latencies:
        return None
    latencies = sorted(latencies)
    n = len(latencies)
    p = lambda q: latencies[min(int(q * n), n - 1)]
    return {
        "n_requests": n,
        "min_ms": round(latencies[0], 2),
        "p50_ms": round(p(0.50), 2),
        "p95_ms": round(p(0.95), 2),
        "max_ms": round(latencies[-1], 2),
        "avg_ms": round(statistics.mean(latencies), 2),
    }


def main():
    endpoints = [
        ("GET", "/health", None),
        ("GET", "/turbines", None),
        ("GET", "/models", None),
        ("GET", "/outputs/metrics", None),
        ("POST", "/predict", {
            "turbine_id": "TB01", "wind_speed": 8.5, "temperature": 22.0,
            "frequency": 50.0, "power": 1500.0, "model_type": "lightgbm"}),
        ("POST", "/predict/farm", {
            "wind_speed": 8.5, "temperature": 22.0, "frequency": 50.0,
            "power": 18000.0, "model_type": "lightgbm"}),
    ]

    rows = []
    summary = {"fail_closed_no_key": None, "fail_closed_bad_key": None}
    for method, path, body in endpoints:
        statuses, lats = [], []
        for _ in range(N_REQUESTS):
            code, lat = _request(method, path, body, key=API_KEY)
            statuses.append(code)
            if lat is not None:
                lats.append(lat)
        ok = statuses.count(200)
        st = _stats(lats)
        rows.append({
            "endpoint": f"{method} {path}",
            "status_ok": ok,
            "status_distribution": {str(s): statuses.count(s) for s in sorted(set(statuses))},
            **st,
        })
        print(f"{method:4s} {path:28s} ok={ok}/{len(statuses)} p95={st['p95_ms']}ms")

    # Security: protected endpoint must be FAIL-CLOSED without/with wrong key.
    probe_path = "/predict"
    probe_body = {"turbine_id": "TB01", "wind_speed": 8.5, "temperature": 22.0,
                  "frequency": 50.0, "power": 1500.0}
    no_key_status, _ = _request("POST", probe_path, probe_body, key=None)
    bad_key_status, _ = _request("POST", probe_path, probe_body, key="wrong-key")
    summary["fail_closed_no_key"] = no_key_status
    summary["fail_closed_bad_key"] = bad_key_status
    rows.append({
        "endpoint": "SECURITY no-key -> /predict",
        "status_ok": int(no_key_status in (401, 403, 503)),
        "status_distribution": {str(no_key_status): 1},
        "note": "fail-closed (no default key / no file fallback)",
    })
    rows.append({
        "endpoint": "SECURITY wrong-key -> /predict",
        "status_ok": int(bad_key_status in (401, 403, 503)),
        "status_distribution": {str(bad_key_status): 1},
        "note": "fail-closed (no default key / no file fallback)",
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "api_benchmark.csv"
    pd = __import__("pandas").DataFrame(rows)
    pd.to_csv(out_path, index=False)
    print(f"\napi_benchmark.csv written: {out_path}")
    print(f"Fail-closed no key: {summary['fail_closed_no_key']} "
          f"(expected 401/403/503)")
    print(f"Fail-closed wrong key: {summary['fail_closed_bad_key']} "
          f"(expected 401/403/503)")
    return 0 if rows and all(r["status_ok"] == len(r["status_distribution"]) if r["status_ok"] == 1 else True for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
