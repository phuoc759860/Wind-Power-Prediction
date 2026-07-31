"""P1-01: inventory must count API endpoints from app.routes at runtime.

The reviewer asked to read "trực tiếp app.routes" instead of grepping
'@app.get(...)' decorators out of the source text. These tests pin that
behaviour: the count reflects routes registered on the live app object,
including ones added dynamically.
"""
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("API_KEY", "amg-wind-2024-test")

from fastapi.routing import APIRoute  # noqa: E402

from src import inventory  # noqa: E402
from src.api import app  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
AUTO_DOC_PATHS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def _expected_from_app_routes():
    counts = Counter()
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path not in AUTO_DOC_PATHS:
            for m in (route.methods or set()):
                counts[m.upper()] += 1
    return counts


def test_count_api_endpoints_matches_app_routes():
    got = inventory._count_api_endpoints(BASE_DIR)
    expected = _expected_from_app_routes()
    assert got["total"] == sum(expected.values())
    for method in ("GET", "POST", "PUT", "DELETE"):
        assert got[method] == expected.get(method, 0), (method, got, expected)
    assert got["total"] >= 20  # sanity: the app really has many endpoints


def test_count_api_endpoints_tracks_runtime_route_registration():
    """A route added to the live app is counted -> reads app.routes, not source."""
    before = inventory._count_api_endpoints(BASE_DIR)

    def _tmp_endpoint():
        return {}

    app.add_api_route("/__bench_tmp__", _tmp_endpoint, methods=["GET"])
    try:
        after = inventory._count_api_endpoints(BASE_DIR)
    finally:
        app.routes[:] = [r for r in app.routes
                         if getattr(r, "path", None) != "/__bench_tmp__"]

    assert after["GET"] == before["GET"] + 1
    assert after["total"] == before["total"] + 1


def test_count_api_endpoints_returns_stable_schema():
    got = inventory._count_api_endpoints(BASE_DIR)
    assert set(got.keys()) == {"GET", "POST", "PUT", "DELETE", "total"}
    assert all(isinstance(v, int) for v in got.values())
