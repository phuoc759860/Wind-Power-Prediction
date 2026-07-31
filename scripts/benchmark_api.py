"""Full API benchmark (reviewer P2-01 items 35-36).

Boots the app, times startup + model-registry scan, measures RAM, and computes
p50/p95/p99 latency for the prediction endpoints - cold first request vs warm -
and explicitly checks whether a cold lazy-load of unloaded models is the source
of the reviewer's "~96s max latency" claim.

Two phases (server booted by this script):
  COLD     - server started with PREWARM_MODELS=0 (pure lazy-load). Times
             startup, then measures the very first /predict (all 5 turbine
             models cold), then N warm requests.
  PREWARM  - server started with PREWARM_MODELS=all (background pre-warm).
             Times startup + pre-warm completion, then warm requests and
             confirms the cold-load tail is gone.

Security probes (fail-closed without/with wrong API key) run in both phases.

Usage:
  python scripts/benchmark_api.py
      # boots the server itself in both phases

  $env:API_BASE_URL="http://host:port"; $env:API_KEY="<key>"
  python scripts/benchmark_api.py
      # benchmarks an already-running server (no cold/prewarm phase possible;
      # the server's state cannot be reset, so only warm latency + security)

Writes:
  outputs/forecasts/api_benchmark.csv  (packaging/docs compatibility)
  06_test_reports/api_benchmark.csv    (delivery tree 06_test_reports/)
"""
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs" / "forecasts"
REPORT_DIR = BASE_DIR / "06_test_reports"
LOGS_DIR = BASE_DIR / "logs"

BASE_URL = os.environ.get("API_BASE_URL", "").strip().rstrip("/")
EXTERNAL_KEY = os.environ.get("API_KEY", "").strip()
N_REQUESTS = int(os.environ.get("BENCH_REQUESTS", "20"))
STARTUP_TIMEOUT = float(os.environ.get("BENCH_STARTUP_TIMEOUT", "180"))
PREWARM_TIMEOUT = float(os.environ.get("BENCH_PREWARM_TIMEOUT", "300"))
MODEL_LOAD_TIMEOUT = os.environ.get("MODEL_LOAD_TIMEOUT", "30")
PREWARM_PHASE_CONFIG = os.environ.get("BENCH_PREWARM", "all")

PREDICT_BODY = {
    "turbine_id": "TB01", "wind_speed": 8.5, "temperature": 22.0,
    "frequency": 50.0, "power": 1500.0, "model_type": "lightgbm",
}
FARM_BODY = {
    "wind_speed": 8.5, "temperature": 22.0, "frequency": 50.0,
    "power": 18000.0, "model_type": "lightgbm",
}
SECURITY_BODY = {
    "turbine_id": "TB01", "wind_speed": 8.5, "temperature": 22.0,
    "frequency": 50.0, "power": 1500.0,
}


def _request_raw(method: str, path: str, body: dict = None, key: str = None,
                 timeout: float = 60.0):
    """Returns (status_code, latency_ms, body_text)."""
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, (time.perf_counter() - start) * 1000, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (time.perf_counter() - start) * 1000, e.read().decode("utf-8", "replace")
    except Exception:
        return -1, (time.perf_counter() - start) * 1000, ""


def _request(method: str, path: str, body: dict = None, key: str = None,
             timeout: float = 60.0):
    return _request_raw(method, path, body, key, timeout)


def _health(key: str = None):
    status, _, body = _request_raw("GET", "/health", key=key)
    try:
        return json.loads(body) if body else {}
    except ValueError:
        return {}


def _pct(lats, q):
    if not lats:
        return 0.0
    s = sorted(lats)
    n = len(s)
    return round(s[min(int(q * n), n - 1)], 2)


def _stats(lats):
    if not lats:
        return None
    return {
        "n_requests": len(lats),
        "min_ms": round(min(lats), 2),
        "p50_ms": _pct(lats, 0.50),
        "p95_ms": _pct(lats, 0.95),
        "p99_ms": _pct(lats, 0.99),
        "max_ms": round(max(lats), 2),
        "avg_ms": round(statistics.mean(lats), 2),
    }


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_server(port: int, prewarm: str, key: str):
    env = dict(os.environ)
    env.update({
        "API_KEY": key,
        "PREWARM_MODELS": str(prewarm),
        "RATE_LIMIT_REQUESTS": "1000000",
        "RATE_LIMIT_WINDOW_SEC": "3600",
        "MODEL_LOAD_TIMEOUT": MODEL_LOAD_TIMEOUT,
        "PYTHONUNBUFFERED": "1",
    })
    logf = open(LOGS_DIR / "api_benchmark_server.log", "ab")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(BASE_DIR), env=env, stdout=logf, stderr=subprocess.STDOUT,
    )


def _wait_ready(proc, port, timeout):
    global BASE_URL
    BASE_URL = f"http://127.0.0.1:{port}"
    first = time.perf_counter()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (rc={proc.returncode}); "
                               f"see logs/api_benchmark_server.log")
        try:
            code, lat, _ = _request("GET", "/health", timeout=2.0)
            if code == 200:
                return (time.perf_counter() - first) * 1000, lat
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("server not ready within timeout; see logs/api_benchmark_server.log")


def _wait_prewarm(proc, key, target, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited during pre-warm (rc={proc.returncode})")
        h = _health(key)
        pw = h.get("prewarm", {})
        loaded = h.get("models_loaded_in_ram", 0)
        if not pw.get("in_progress", True) and loaded >= target:
            return (time.time() - deadline + timeout) * 1000, loaded
        time.sleep(0.5)
    raise RuntimeError("pre-warm did not finish within timeout")


def _rss_mb(pid):
    if pid is None:
        return None
    try:
        import psutil  # noqa: F401
        return round(psutil.Process(pid).memory_info().rss / 1e6, 1)
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            for access in (0x1000, 0x0400):
                handle = ctypes.windll.kernel32.OpenProcess(access, False, pid)
                if handle:
                    try:
                        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                            handle, ctypes.byref(pmc), ctypes.sizeof(pmc))
                        if ok:
                            return round(pmc.WorkingSetSize / 1e6, 1)
                    finally:
                        ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            pass
    return None


def _stop_server(proc):
    if proc is None:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def _latency_rows(phase, key, endpoints):
    rows = []
    worst = 0.0
    for method, path, body in endpoints:
        statuses, lats = [], []
        for _ in range(N_REQUESTS):
            code, lat, _ = _request(method, path, body, key=key)
            statuses.append(code)
            if lat is not None:
                lats.append(lat)
        worst = max(worst, max(lats) if lats else 0.0)
        st = _stats(lats)
        ok = statuses.count(200)
        print(f"  [{phase}] {method:4s} {path:24s} ok={ok}/{len(statuses)} "
              f"p50={st['p50_ms']} p95={st['p95_ms']} p99={st['p99_ms']} "
              f"max={st['max_ms']}ms")
        for metric in ("n_requests", "min_ms", "p50_ms", "p95_ms", "p99_ms",
                       "max_ms", "avg_ms"):
            rows.append({
                "phase": phase, "category": "latency", "metric": metric,
                "value": st[metric], "unit": "" if metric == "n_requests" else "ms",
                "detail": f"{method} {path}",
            })
        rows.append({
            "phase": phase, "category": "latency", "metric": "status_ok_count",
            "value": ok, "unit": "count", "detail": f"{method} {path}",
        })
    return rows, worst


def _security_rows(phase, key):
    rows = []
    no_key, _, _ = _request("POST", "/predict", SECURITY_BODY, key=None)
    bad_key, _, _ = _request("POST", "/predict", SECURITY_BODY, key="wrong-key")
    print(f"  [{phase}] security: no-key={no_key} (expected 401/403/503), "
          f"wrong-key={bad_key} (expected 401/403/503)")
    rows.append({
        "phase": phase, "category": "security", "metric": "no_key_status",
        "value": no_key, "unit": "status",
        "detail": "fail-closed: expected 401/403/503 (no default key / no file fallback)",
    })
    rows.append({
        "phase": phase, "category": "security", "metric": "wrong_key_status",
        "value": bad_key, "unit": "status",
        "detail": "fail-closed: expected 401/403/503 (invalid key rejected)",
    })
    return rows


def _boot_phase(phase, prewarm):
    """Boot a fresh server with the given PREWARM_MODELS setting and bench it."""
    port = _free_port()
    key = "bench-" + os.urandom(8).hex()
    proc = _spawn_server(port, prewarm, key)
    rows = []
    try:
        startup_ms, first_health_ms = _wait_ready(proc, port, STARTUP_TIMEOUT)
        h = _health()
        n_registry = h.get("models_in_registry", 0)
        rows.append({
            "phase": phase, "category": "startup", "metric": "startup_to_ready_ms",
            "value": round(startup_ms, 1), "unit": "ms",
            "detail": "spawn -> first 200 /health (includes registry scan)",
        })
        rows.append({
            "phase": phase, "category": "startup", "metric": "first_health_ms",
            "value": round(first_health_ms, 1), "unit": "ms",
            "detail": "/health response time on the very first call",
        })
        rows.append({
            "phase": phase, "category": "startup", "metric": "models_in_registry",
            "value": n_registry, "unit": "count",
            "detail": "models scanned at startup (metadata only, lazy joblib.load)",
        })
        rows.append({
            "phase": phase, "category": "startup", "metric": "prewarm_configured",
            "value": prewarm, "unit": "", "detail": "PREWARM_MODELS setting",
        })
        print(f"\n[{phase}] startup_to_ready_ms={round(startup_ms,1)} "
              f"models_in_registry={n_registry}")

        if str(prewarm) not in ("0", "none", "false", ""):
            target = h.get("prewarm", {}).get("target", 0) or n_registry
            prewarm_ms, loaded = _wait_prewarm(proc, key, target, PREWARM_TIMEOUT)
            rows.append({
                "phase": phase, "category": "startup", "metric": "prewarm_elapsed_ms",
                "value": round(prewarm_ms, 1), "unit": "ms",
                "detail": "time until all pre-warm targets were in RAM",
            })
            rows.append({
                "phase": phase, "category": "startup", "metric": "prewarm_loaded_models",
                "value": loaded, "unit": "count",
                "detail": "models_loaded_in_ram once pre-warm finished",
            })
            print(f"[{phase}] prewarm finished in {round(prewarm_ms,1)}ms "
                  f"({loaded}/{n_registry} models in RAM)")

        ram_baseline = _rss_mb(proc.pid)
        rows.append({
            "phase": phase, "category": "ram", "metric": "ram_baseline_mb",
            "value": ram_baseline if ram_baseline is not None else -1,
            "unit": "MB", "detail": "server RSS right after readiness",
        })

        # Cold first-request probe (only meaningful when pre-warm is disabled).
        if str(prewarm) in ("0", "none", "false", ""):
            code, cold_first_ms, _ = _request("POST", "/predict", PREDICT_BODY, key=key)
            print(f"[{phase}] COLD first /predict (5 lazy loads): "
                  f"{round(cold_first_ms,1)}ms -> status {code}")
            rows.append({
                "phase": phase, "category": "cold_probe",
                "metric": "cold_first_predict_ms", "value": round(cold_first_ms, 1),
                "unit": "ms",
                "detail": "first /predict on a fresh server; all 5 turbine models "
                          "lazy-loaded in the request path",
            })
            bench = (_health(key) or {}).get("ram_benchmark") or {}
            max_load = max((float(m.get("load_time_ms", 0)) for m in bench.values()),
                           default=0.0)
            rows.append({
                "phase": phase, "category": "cold_probe",
                "metric": "max_model_load_ms", "value": round(max_load, 1), "unit": "ms",
                "detail": "slowest single joblib.load observed (per-model)",
            })
        else:
            code, first_lat, _ = _request("POST", "/predict", PREDICT_BODY, key=key)
            rows.append({
                "phase": phase, "category": "cold_probe",
                "metric": "first_predict_ms", "value": round(first_lat, 1), "unit": "ms",
                "detail": "first /predict on a pre-warmed server (model already in RAM)",
            })
            print(f"[{phase}] first /predict on pre-warmed server: {round(first_lat,1)}ms")

        endpoints = [
            ("GET", "/health", None),
            ("GET", "/models", None),
            ("POST", "/predict", PREDICT_BODY),
            ("POST", "/predict/farm", FARM_BODY),
        ]
        lat_rows, worst = _latency_rows(phase, key, endpoints)
        rows.extend(lat_rows)
        rows.append({
            "phase": phase, "category": "latency", "metric": "worst_max_ms",
            "value": round(worst, 1), "unit": "ms",
            "detail": "worst single request across endpoints in this phase",
        })

        ram_peak = _rss_mb(proc.pid)
        rows.append({
            "phase": phase, "category": "ram", "metric": "ram_peak_mb",
            "value": ram_peak if ram_peak is not None else -1,
            "unit": "MB", "detail": "server RSS after warm request burst",
        })

        rows.extend(_security_rows(phase, key))
        return rows, {"phase": phase, "cold_first_predict_ms": None, "worst": worst}
    finally:
        _stop_server(proc)


def _existing_server_phase():
    """Benchmark an already-running server (API_BASE_URL must be set)."""
    global BASE_URL, EXTERNAL_KEY
    if not EXTERNAL_KEY:
        raise SystemExit("API_BASE_URL set but API_KEY is missing; "
                         "refusing to run against an unprotected server")
    rows = []
    h = _health(EXTERNAL_KEY)
    rows.append({
        "phase": "existing", "category": "startup", "metric": "models_in_registry",
        "value": h.get("models_in_registry", 0), "unit": "count",
        "detail": "server already running; startup/prewarm not re-measured",
    })
    endpoints = [
        ("GET", "/health", None),
        ("GET", "/models", None),
        ("POST", "/predict", PREDICT_BODY),
        ("POST", "/predict/farm", FARM_BODY),
    ]
    lat_rows, worst = _latency_rows("existing", EXTERNAL_KEY, endpoints)
    rows.extend(lat_rows)
    rows.append({
        "phase": "existing", "category": "latency", "metric": "worst_max_ms",
        "value": round(worst, 1), "unit": "ms",
        "detail": "worst single request across endpoints",
    })
    rows.extend(_security_rows("existing", EXTERNAL_KEY))
    return rows, {"phase": "existing", "cold_first_predict_ms": None, "worst": worst}


def _write_csv(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = __import__("pandas").DataFrame(rows, columns=[
        "phase", "category", "metric", "value", "unit", "detail"])
    out_paths = [OUT_DIR / "api_benchmark.csv", REPORT_DIR / "api_benchmark.csv"]
    for p in out_paths:
        df.to_csv(p, index=False)
        print(f"api_benchmark.csv written: {p}")


def main():
    rows = []
    worst_overall = 0.0

    if BASE_URL:
        phase_rows, meta = _existing_server_phase()
        rows.extend(phase_rows)
        worst_overall = max(worst_overall, meta["worst"])
    else:
        print("=== COLD phase (PREWARM_MODELS=0, lazy-load only) ===")
        phase_rows, meta = _boot_phase("cold", "0")
        rows.extend(phase_rows)
        worst_overall = max(worst_overall, meta["worst"])
        cold_first = meta["cold_first_predict_ms"]

        print("\n=== PREWARM phase (background pre-warm of all models) ===")
        phase_rows, meta = _boot_phase("prewarm", PREWARM_PHASE_CONFIG)
        rows.extend(phase_rows)
        worst_overall = max(worst_overall, meta["worst"])

        # Explicit check of the reviewer's ~96s max-latency claim.
        cold_predict_row = next(
            (r for r in rows if r["metric"] == "cold_first_predict_ms"), None)
        warm_p95_row = next(
            (r for r in rows if r["metric"] == "p95_ms"
             and r["detail"] == "POST /predict" and r["phase"] == "cold"), None)
        warm_p99_row = next(
            (r for r in rows if r["metric"] == "p99_ms"
             and r["detail"] == "POST /predict" and r["phase"] == "cold"), None)
        if cold_predict_row and warm_p95_row and warm_p99_row:
            cold = float(cold_predict_row["value"])
            tail = bool(cold > 5000 or cold > 3 * float(warm_p95_row["value"]))
            rows.append({
                "phase": "both", "category": "cold_probe",
                "metric": "cold_tail_flag", "value": int(tail), "unit": "bool",
                "detail": "true if cold first /predict >5s or >3x warm p95 "
                          "(cold lazy-load as source of the ~96s tail)",
            })
            rows.append({
                "phase": "both", "category": "cold_probe",
                "metric": "cold_vs_warm_ratio", "value": round(
                    cold / float(warm_p99_row["value"]), 2), "unit": "x",
                "detail": "cold first /predict latency / warm p99 /predict latency",
            })
            print(f"\nCold-tail probe: cold_first={round(cold,1)}ms vs warm "
                  f"p95={warm_p95_row['value']}ms p99={warm_p99_row['value']}ms "
                  f"-> tail_flag={tail}")

    rows.append({
        "phase": "both", "category": "latency", "metric": "worst_max_ms_overall",
        "value": round(worst_overall, 1), "unit": "ms",
        "detail": "worst single request across ALL phases/endpoints (vs reviewer's ~96s claim)",
    })

    _write_csv(rows)

    sec_ok = all(r["value"] in (401, 403, 503) for r in rows
                 if r["category"] == "security")
    lat_ok = all(r["value"] == N_REQUESTS for r in rows
                 if r["metric"] == "status_ok_count")
    print(f"\nworst_max_ms_overall={round(worst_overall,1)}ms "
          f"(reviewer reported ~96s)")
    print(f"security fail-closed: {'OK' if sec_ok else 'FAILED'}")
    return 0 if (sec_ok and lat_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
