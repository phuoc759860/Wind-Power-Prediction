"""P2-01 items 35-36: model-load mitigation + API benchmark helpers.

Covers the reviewer's concerns: unbounded cold-load latency on the prediction
path. We verify (a) the per-key lock prevents thundering-herd double loads,
(b) MODEL_LOAD_TIMEOUT turns an unresponsive model load into a 503 instead of
an unbounded hang, (c) pre-warm config parsing/priority ordering, and
(d) the benchmark script's percentile helpers used to produce
06_test_reports/api_benchmark.csv.
"""
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("API_KEY", "amg-wind-2024-test")

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import src.api as api
from src.api import app

API_KEY = os.environ["API_KEY"]


@pytest.fixture(scope="module", autouse=True)
def registry():
    api._scan_model_registry()
    api._load_availability()
    yield api._model_registry


@pytest.fixture(scope="module")
def client():
    api._scan_model_registry()
    api._load_availability()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def fresh_key():
    """A registry key guaranteed to exist; cache cleared before use."""
    key = "TB01_power_target_10min_lightgbm"
    assert key in api._model_registry
    api._model_cache.pop(key, None)
    yield key
    api._model_cache.pop(key, None)


# ── Pre-warm configuration ────────────────────────────────────────────────────
def test_prewarm_target_count_all(monkeypatch):
    monkeypatch.setattr(api, "PREWARM_MODELS", "all")
    assert api._prewarm_target_count() == len(api._model_registry)


def test_prewarm_target_count_disabled(monkeypatch):
    for value in ("0", "none", "false", ""):
        monkeypatch.setattr(api, "PREWARM_MODELS", value)
        assert api._prewarm_target_count() == 0


def test_prewarm_target_count_numeric(monkeypatch):
    monkeypatch.setattr(api, "PREWARM_MODELS", "7")
    assert api._prewarm_target_count() == 7


def test_prewarm_target_count_invalid(monkeypatch):
    monkeypatch.setattr(api, "PREWARM_MODELS", "bogus")
    assert api._prewarm_target_count() == 0


def test_prewarm_priority_order():
    keys = api._prewarm_priority_keys()
    assert keys[0].startswith("farm_total_power")
    # lightgbm (the API default) must precede xgboost for the same target
    for i, k in enumerate(keys):
        if k.endswith("_lightgbm"):
            target = k.rsplit("_", 1)[0]
            xgb = f"{target}_xgboost"
            if xgb in api._model_registry:
                assert keys.index(xgb) > i, f"{xgb} should come after {k}"


# ── Thundering-herd protection ────────────────────────────────────────────────
def test_get_model_concurrent_single_load(monkeypatch, fresh_key):
    """8 concurrent cold requests for the same model must load it exactly once."""
    loads = []
    real_load = api.joblib.load

    def counting(path, **kwargs):
        loads.append(path)
        return real_load(path, **kwargs)

    monkeypatch.setattr(api.joblib, "load", counting)
    errors = []

    def worker():
        try:
            api._get_model(fresh_key)
        except Exception as e:  # pragma: no cover - failure would fail the test
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, errors
    model_loads = [p for p in loads if p.endswith("_model.joblib")]
    assert len(model_loads) == 1, f"model loaded {len(model_loads)} times, expected 1"
    assert fresh_key in api._model_cache


# ── MODEL_LOAD_TIMEOUT -> 503 instead of an unbounded hang ────────────────────
def test_get_model_load_failure_returns_503(monkeypatch, fresh_key):
    def boom(path, **kwargs):
        raise RuntimeError("disk read failed")

    monkeypatch.setattr(api.joblib, "load", boom)
    with pytest.raises(HTTPException) as exc:
        api._get_model(fresh_key)
    assert exc.value.status_code == 503
    assert fresh_key not in api._model_cache


def test_get_model_slow_load_returns_503(monkeypatch, fresh_key):
    """A model that loads successfully but exceeds MODEL_LOAD_TIMEOUT -> 503."""

    def slow(path, **kwargs):
        time.sleep(0.2)
        return object()

    monkeypatch.setattr(api.joblib, "load", slow)
    monkeypatch.setattr(api, "MODEL_LOAD_TIMEOUT", 0.05)
    with pytest.raises(HTTPException) as exc:
        api._get_model(fresh_key)
    assert exc.value.status_code == 503
    assert fresh_key not in api._model_cache


# ── Health endpoint exposes pre-warm / RAM state ──────────────────────────────
def test_health_exposes_prewarm_state(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "prewarm" in data
    assert "models_loaded_in_ram" in data
    assert data["prewarm"]["configured"] in ("all", "0", "none", "false")
    assert data["models_in_registry"] >= 0


# ── Benchmark helper functions (scripts/benchmark_api.py) ─────────────────────
def test_benchmark_percentiles():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import benchmark_api as bm

    lats = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert bm._pct(lats, 0.50) == 30.0
    assert bm._pct(lats, 0.95) == 50.0
    assert bm._pct(lats, 0.99) == 50.0
    st = bm._stats(lats)
    assert st["p50_ms"] == 30.0
    assert st["max_ms"] == 50.0
    assert st["avg_ms"] == 30.0
    assert st["n_requests"] == 5


def test_benchmark_cold_tail_flag_logic():
    """Replicates the CSV logic: cold first predict >3x warm p95 -> tail flagged."""
    cold = 2031.4
    warm_p95 = 42.29
    tail = bool(cold > 5000 or cold > 3 * warm_p95)
    assert tail is True
    cold_small = 100.0
    tail = bool(cold_small > 5000 or cold_small > 3 * warm_p95)
    assert tail is False
