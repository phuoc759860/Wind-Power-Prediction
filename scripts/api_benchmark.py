"""API latency/security benchmark (reviewer P2-01 items 35-36).

Compatibility wrapper: the comprehensive benchmark now lives in
scripts/benchmark_api.py (startup timing, RAM, cold-vs-warm p50/p95/p99,
cold-load tail probe and fail-closed security rows).

Usage:
    $env:API_KEY="<key>"; .venv\\Scripts\\python scripts/api_benchmark.py
    # or: python scripts/api_benchmark.py  (boots the server itself)

Writes 06_test_reports/api_benchmark.csv and outputs/forecasts/api_benchmark.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_api  # noqa: E402

if __name__ == "__main__":
    sys.exit(benchmark_api.main())
