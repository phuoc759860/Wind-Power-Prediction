from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

AVAILABLE_HORIZONS = ["10min", "30min", "1hour", "6hour", "24hour"]
DEFAULT_NOMINAL_LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)

# Files written as part of provenance that must NOT perturb the output hash,
# otherwise the manifest's own output_hash becomes a self-referential
# fixed-point and can never be verified against the directory it describes.
MANIFEST_EXCLUDE_FILES = {"run_manifest.json", "run_manifest_verification.json"}


def _safe_predictions(
    predictions: Mapping[str, Mapping[str, object]] | Sequence[Mapping[str, object]],
    df: pd.DataFrame | None = None,
) -> list[dict]:
    if isinstance(predictions, Mapping):
        items = list(predictions.values())
    else:
        items = list(predictions)

    cleaned = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        preds = item.get("predictions")
        if preds is None:
            continue
        actual = item.get("actual")
        target = item.get("target", "unknown")
        if actual is None:
            if df is not None and target in df.columns:
                n = len(preds)
                actual = df[target].values[:n]
            else:
                continue
        cleaned.append({
            "target": target,
            "model": item.get("model_name", item.get("model", "unknown")),
            "predictions": np.asarray(preds, dtype=float),
            "actual": np.asarray(actual, dtype=float),
        })
    return cleaned


def build_coverage_summary(
    df: pd.DataFrame,
    predictions: Mapping[str, Mapping[str, object]] | Sequence[Mapping[str, object]],
    nominal_levels: Iterable[float] | None = None,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a single calibration summary CSV with reviewer-required columns.

    The output schema is intentionally restricted to:
    nominal, coverage, mean_width, calibration_error

    Calibration is estimated split-conformally: prediction residuals are pooled
    across targets, split into disjoint calibration/evaluation halves, and the
    nominal quantile is fit on the calibration half while coverage is measured
    on the evaluation half. This avoids the circular zero calibration error that
    results from fitting and evaluating on the same rows.
    """
    if df is None or df.empty:
        raise ValueError("build_coverage_summary requires a non-empty dataframe")

    nominal_levels = list(DEFAULT_NOMINAL_LEVELS if nominal_levels is None else nominal_levels)
    if mask is not None:
        mask = pd.Series(mask, index=df.index).astype(bool)
        work = df.loc[mask].copy()
    else:
        work = df.copy()

    if work.empty:
        raise ValueError("No rows remain after applying the official evaluation mask")

    records = []
    pred_items = _safe_predictions(predictions, df=work)
    if not pred_items:
        raise ValueError("No usable prediction targets were supplied for interval calibration")

    pooled_actual = []
    pooled_preds = []
    for item in pred_items:
        actual = item["actual"]
        preds = item["predictions"]
        n = min(len(actual), len(preds))
        if n <= 0:
            continue
        actual = actual[:n]
        preds = preds[:n]
        valid = ~(np.isnan(actual) | np.isnan(preds))
        if valid.sum() < 10:
            continue
        pooled_actual.append(actual[valid])
        pooled_preds.append(preds[valid])

    if not pooled_actual:
        raise ValueError("No calibration rows could be produced (no valid targets)")

    pooled_actual = np.concatenate(pooled_actual)
    pooled_preds = np.concatenate(pooled_preds)
    idx = np.arange(len(pooled_actual))
    cal_idx = idx[::2]
    val_idx = idx[1::2]
    cal_errors = np.abs(pooled_actual[cal_idx] - pooled_preds[cal_idx])
    val_actual = pooled_actual[val_idx]
    val_preds = pooled_preds[val_idx]
    for nominal in nominal_levels:
        if nominal <= 0 or nominal >= 1:
            continue
        q = float(np.quantile(cal_errors, nominal))
        lower = val_preds - q
        upper = val_preds + q
        inside = (val_actual >= lower) & (val_actual <= upper)
        coverage = float(np.mean(inside))
        mean_width = float(np.mean(upper - lower))
        calibration_error = float(abs(coverage - nominal))
        records.append({
            "nominal": float(nominal),
            "coverage": round(coverage, 6),
            "mean_width": round(mean_width, 6),
            "calibration_error": round(calibration_error, 6),
        })

    if not records:
        raise ValueError("No calibration rows could be produced")

    out = pd.DataFrame(records)
    out = out.sort_values(["nominal"]).reset_index(drop=True)
    return out.drop_duplicates(subset=["nominal"], keep="last").reset_index(drop=True)


def save_coverage_summary(
    df: pd.DataFrame,
    predictions: Mapping[str, Mapping[str, object]] | Sequence[Mapping[str, object]],
    output_path: str | Path = "outputs/coverage.csv",
    nominal_levels: Iterable[float] | None = None,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Persist a reviewer-facing coverage summary to CSV and return it."""
    summary = build_coverage_summary(df=df, predictions=predictions, nominal_levels=nominal_levels, mask=mask)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    return summary


def write_coverage_csv_from_prediction_bundle(
    test_df: pd.DataFrame,
    predictions: Mapping[str, Mapping[str, object]] | Sequence[Mapping[str, object]],
    output_path: str | Path = "outputs/coverage.csv",
    nominal_levels: Iterable[float] | None = None,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper for the final one-command pipeline.

    The module deliberately writes a single CSV that the report path only reads.
    """
    return save_coverage_summary(
        df=test_df,
        predictions=predictions,
        output_path=output_path,
        nominal_levels=nominal_levels,
        mask=mask,
    )


def _default_coverage_path(base_dir: str | Path = ".") -> Path:
    return Path(base_dir) / "outputs" / "coverage.csv"


def build_run_manifest(base_dir: str | Path = ".",
                       config: Mapping | None = None,
                       run_id: str | None = None,
                       timestamp: str | None = None) -> dict:
    """Generate a deterministic provenance manifest for a single run.

    run_id/timestamp are preserved across the run (written at start, re-saved
    with final hashes at the end) so the finished manifest still identifies the
    original invocation. output_hash excludes the manifest files themselves so
    the hash can be recomputed and compared by verify_run_manifest.
    """
    base_dir = Path(base_dir)
    import hashlib
    import platform
    import subprocess

    def _hash_path(path: Path, exclude: set | None = None) -> str:
        exclude = exclude or set()
        hasher = hashlib.sha256()
        for file_path in sorted(path.rglob("*")):
            if not file_path.is_file():
                continue
            rel = str(file_path.relative_to(path)).replace("\\", "/")
            if rel in exclude:
                continue
            hasher.update(rel.encode("utf-8"))
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    hasher.update(chunk)
        return hasher.hexdigest()

    def _git_commit() -> str:
        try:
            result = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base_dir, text=True).strip()
            return result[:12]
        except Exception:
            return "unknown"

    def _python_version() -> str:
        return platform.python_version()

    try:
        packages = {}
        from importlib.metadata import distributions
        for dist in distributions():
            try:
                packages[dist.metadata["Name"]] = dist.version
            except Exception:
                continue
    except Exception:
        packages = {}

    manifest = {
        "run_id": run_id or f"run-{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
        "git_commit": _git_commit(),
        "config_hash": hashlib.sha256(json.dumps(config or {}, sort_keys=True).encode("utf-8")).hexdigest(),
        "seed": int((config or {}).get("seed", 0)),
        "python": _python_version(),
        "packages": packages,
        "data_hash": _hash_path(base_dir / "data"),
        "output_hash": _hash_path(base_dir / "outputs", exclude=MANIFEST_EXCLUDE_FILES),
        "timestamp": timestamp or pd.Timestamp.utcnow().isoformat(),
    }
    return manifest


def save_run_manifest(base_dir: str | Path = ".", config: Mapping | None = None,
                      output_path: str | Path = "outputs/run_manifest.json",
                      run_id: str | None = None,
                      timestamp: str | None = None) -> dict:
    manifest = build_run_manifest(base_dir=base_dir, config=config,
                                  run_id=run_id, timestamp=timestamp)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def verify_run_manifest(manifest_path: str | Path = "outputs/run_manifest.json",
                        base_dir: str | Path = ".",
                        config: Mapping | None = None,
                        output_path: str | Path | None = None,
                        strict: bool = True) -> dict:
    """Recompute every reproducible manifest field and compare it against the
    current files/environment. Writes a verification report (reviewer evidence)
    and, in strict mode, raises RuntimeError on the first failing field."""
    manifest_path = Path(manifest_path)
    base_dir = Path(base_dir)
    if not manifest_path.exists():
        raise RuntimeError(f"run_manifest.json not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = build_run_manifest(
        base_dir=base_dir,
        config=config,
        run_id=manifest.get("run_id"),
        timestamp=manifest.get("timestamp"),
    )

    def _check(field: str, expected, actual) -> dict:
        return {"field": field, "expected": expected, "actual": actual,
                "match": bool(expected == actual)}

    run_id = manifest.get("run_id")
    checks = [
        {"field": "run_id",
         "expected": "well-formed run-* id",
         "actual": run_id,
         "match": bool(isinstance(run_id, str) and run_id.startswith("run-") and len(run_id) > 4)},
        _check("git_commit", current["git_commit"], manifest.get("git_commit")),
        _check("config_hash", current["config_hash"], manifest.get("config_hash")),
        _check("seed", current["seed"], manifest.get("seed")),
        _check("python", current["python"], manifest.get("python")),
        _check("packages", current["packages"], manifest.get("packages")),
        _check("data_hash", current["data_hash"], manifest.get("data_hash")),
        _check("output_hash", current["output_hash"], manifest.get("output_hash")),
    ]

    failed = [c for c in checks if not c["match"]]
    report = {
        "verified_at": pd.Timestamp.utcnow().isoformat(),
        "manifest_path": str(manifest_path),
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
    }

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    if strict and failed:
        names = ", ".join(c["field"] for c in failed)
        raise RuntimeError(f"Run manifest verification FAILED: {names}")

    return report
