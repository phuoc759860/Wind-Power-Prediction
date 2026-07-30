"""Run pytest and auto-fill compliance_matrix.csv with Pass/Fail + last_run_date.

Usage:
    python scripts/run_compliance.py [--junitxml results.xml] [--commit]

Reads compliance_matrix.csv, runs the referenced tests (from the 'tests' column),
skips rows where tests is empty or marked 'document-only',
and writes back test_result and last_run_date columns.
"""

import csv
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MATRIX_PATH = BASE / "configs" / "compliance_matrix.csv"


def run_tests(test_specs: str) -> bool:
    """Run pytest on the given test specs. Return True if all pass."""
    if not test_specs or test_specs.strip() == "":
        return True
    specs = [s.strip() for s in test_specs.replace(",", " ").split() if s.strip()]
    if not specs:
        return True

    xml = tempfile.mktemp(suffix=".xml")
    cmd = [sys.executable, "-m", "pytest", *specs, "--junitxml", xml, "-q", "--tb=no"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE))
    return result.returncode == 0


def main():
    if not MATRIX_PATH.exists():
        print(f"Compliance matrix not found: {MATRIX_PATH}")
        sys.exit(1)

    with open(MATRIX_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    # Add new columns if missing
    if "test_result" not in fieldnames:
        fieldnames.append("test_result")
    if "last_run_date" not in fieldnames:
        fieldnames.append("last_run_date")

    today = datetime.now().strftime("%Y-%m-%d")

    for row in rows:
        status = row.get("status", "").strip().lower()
        if status == "document-only":
            row["test_result"] = "N/A"
            row["last_run_date"] = today
            continue

        test_col = row.get("tests", "").strip()
        if not test_col:
            row["test_result"] = "N/A (no tests)"
            row["last_run_date"] = today
            continue

        print(f"  [{row['requirement_id']}] {test_col} ... ", end="", flush=True)
        passed = run_tests(test_col)
        row["test_result"] = "PASS" if passed else "FAIL"
        row["last_run_date"] = today
        print(row["test_result"])

    with open(MATRIX_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nUpdated {MATRIX_PATH} with test results ({today})")


if __name__ == "__main__":
    main()
