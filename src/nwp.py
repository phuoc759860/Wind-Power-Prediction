"""NWP ingestion interface + SCADA-only vs SCADA+NWP ablation (P1-05).

The shipped 6h/24h models are SCADA-only time-series baselines (no weather
forecasts). Real NWP grids are not in the repository, so this module provides:

  * load_nwp()          — ingestion interface: reads an NWP forecast CSV
                          (timestamp, lead_minutes, wind_speed, wind_direction,
                          temperature) if one is supplied.
  * build_stub_nwp()    — deterministic STUB NWP (perfect-forecast upper bound,
                          observed value at t+lead plus small seeded noise)
                          persisted to disk so the ingestion path is exercised
                          end-to-end even without a real NWP provider feed.
  * add_nwp_features()  — merges NWP forecasts (issued at t, valid at t+lead)
                          into the feature frame for one horizon.
  * run_nwp_ablation()  — trains a small Ridge per (target, horizon) on the
                          official split with SCADA-only vs SCADA+NWP features
                          and reports test R2/RMSE/MAE.

The ablation is deliberately honest: without real NWP data the NWP columns are
labelled 'stub' and represent an upper-bound, NOT an operational improvement.
"""
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NWP_COLUMNS = ["wind_speed", "wind_direction", "temperature"]


def load_nwp(nwp_path: str, timestamp_col: str = "timestamp") -> Optional[pd.DataFrame]:
    """Ingest an NWP forecast CSV.

    Expected schema: timestamp (issue time), lead_minutes, plus one column per
    NWP variable in NWP_COLUMNS. Returns None if the file does not exist.
    """
    path = Path(nwp_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    missing = [c for c in ["lead_minutes", *NWP_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"NWP file {path} is missing required columns: {missing}")
    df["lead_minutes"] = pd.to_numeric(df["lead_minutes"])
    logger.info(f"NWP ingested: {path} ({len(df)} forecast rows, "
                f"leads {sorted(df['lead_minutes'].unique().tolist())})")
    return df


def build_stub_nwp(feature_data: pd.DataFrame, config: dict,
                   out_path: str, horizons: Optional[List[dict]] = None) -> pd.DataFrame:
    """Deterministic stub NWP persisted as a real CSV (perfect-forecast upper bound).

    For every issue timestamp t and every horizon h, the NWP forecast valid at
    t+h is the observed farm wind/temperature at t+h plus small seeded noise.
    This is an ORACLE/UPPER-BOUND forecast; it is used only so the ingestion +
    ablation machinery can be demonstrated without a real NWP feed.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if horizons is None:
        horizons = config.get("forecasting", {}).get("horizons", [])

    turbine_ids = config.get("turbines", {}).get("ids", [])
    ws_cols = [f"{t}_wind_speed" for t in turbine_ids if f"{t}_wind_speed" in feature_data.columns]
    tp_cols = [f"{t}_temperature" for t in turbine_ids if f"{t}_temperature" in feature_data.columns]

    df = feature_data[["timestamp"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if ws_cols:
        df["farm_wind"] = feature_data[ws_cols].mean(axis=1, skipna=True)
    else:
        df["farm_wind"] = np.nan
    if tp_cols:
        df["farm_temp"] = feature_data[tp_cols].mean(axis=1, skipna=True)
    else:
        df["farm_temp"] = np.nan
    df["diurnal"] = 90 + 45 * np.sin(2 * np.pi * np.arange(len(df)) / 144.0)

    rng = np.random.default_rng(42)
    noise_w = 1.0 + rng.normal(0.0, 0.02, size=len(df))
    noise_t = rng.normal(0.0, 0.5, size=len(df))

    records = []
    for h in horizons:
        steps = int(h.get("steps", 0))
        lead_min = steps * 10
        if lead_min <= 0:
            continue
        wind = df["farm_wind"].shift(-steps) * noise_w
        temp = df["farm_temp"].shift(-steps) + noise_t
        for _, row in df.iterrows():
            records.append({
                "timestamp": row["timestamp"],
                "lead_minutes": lead_min,
                "wind_speed": round(float(wind.loc[row.name]) if pd.notna(wind.loc[row.name]) else np.nan, 3),
                "wind_direction": round(float(row["diurnal"]), 3),
                "temperature": round(float(temp.loc[row.name]) if pd.notna(temp.loc[row.name]) else np.nan, 3),
            })
    nwp = pd.DataFrame(records)
    nwp.to_csv(out_path, index=False)
    logger.info(f"Stub NWP written: {out_path} ({len(nwp)} rows, leads "
                f"{sorted(nwp['lead_minutes'].unique().tolist())})")
    return nwp


def add_nwp_features(df: pd.DataFrame, nwp: pd.DataFrame,
                     lead_minutes: int) -> pd.DataFrame:
    """Merge NWP forecasts valid at t+lead (issued at t) as features at time t.

    Returns a copy of df with added columns nwp_wind_speed, nwp_wind_direction,
    nwp_temperature aligned on the issue timestamp.
    """
    out = df.copy()
    sub = nwp[nwp["lead_minutes"] == lead_minutes][["timestamp", *NWP_COLUMNS]].copy()
    if sub.empty:
        for c in NWP_COLUMNS:
            out[f"nwp_{c}"] = np.nan
        return out
    sub = sub.rename(columns={c: f"nwp_{c}" for c in NWP_COLUMNS})
    out["_ts"] = pd.to_datetime(out["timestamp"])
    sub["_ts"] = pd.to_datetime(sub["timestamp"])
    sub = sub.drop(columns=["timestamp"]).drop_duplicates("_ts")
    out = out.merge(sub, on="_ts", how="left")
    out = out.drop(columns=["_ts"])
    return out


def run_nwp_ablation(train_df: pd.DataFrame, val_df: pd.DataFrame,
                     test_df: pd.DataFrame, config: dict, nwp: Optional[pd.DataFrame],
                     nwp_source: str = "stub_synthetic") -> pd.DataFrame:
    """SCADA-only vs SCADA+NWP Ridge ablation on the official split (P1-05).

    Trains a small Ridge (with scaling) per (target, horizon) in `cases` on the
    train split and reports test R2/RMSE/MAE for both feature configurations.
    Returns one row per (target, horizon, feature_set).
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from src.train_power_model import prepare_features

    cases = [
        ("TB02_power", "6hour"),
        ("TB02_power", "24hour"),
        ("TB01_power", "24hour"),
        ("farm_total_power", "6hour"),
        ("farm_total_power", "24hour"),
    ]
    steps_map = {h["name"]: int(h["steps"]) * 10 for h in config.get("forecasting", {}).get("horizons", [])}
    rated = config.get("turbines", {}).get("rated_power_kw", 2200)

    rows = []
    for base_target, horizon in cases:
        target = f"{base_target}_target_{horizon}"
        if target not in train_df.columns:
            logger.warning(f"  NWP ablation: target {target} missing, skipping")
            continue
        lead_min = steps_map.get(horizon, 0)

        for label, with_nwp in [("scada_only", False), ("scada_plus_nwp", True)]:
            train_in = train_df
            test_in = test_df
            if with_nwp and nwp is not None:
                train_in = add_nwp_features(train_df, nwp, lead_min)
                test_in = add_nwp_features(test_df, nwp, lead_min)

            X_tr, y_tr, fcols = prepare_features(train_in, target)
            X_te, y_te, _ = prepare_features(test_in, target, fcols)
            if len(X_tr) < 100 or len(X_te) < 10:
                continue

            scaler = StandardScaler().fit(X_tr.values)
            model = Ridge(alpha=1.0).fit(scaler.transform(X_tr.values), y_tr.values)
            pred = model.predict(scaler.transform(X_te.values))
            yv = y_te.values
            ss_res = float(np.nansum((yv - pred) ** 2))
            ss_tot = float(np.nansum((yv - np.nanmean(yv)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rmse = float(np.sqrt(np.nanmean((yv - pred) ** 2)))
            mae = float(np.nanmean(np.abs(yv - pred)))
            rows.append({
                "target": base_target,
                "horizon": horizon,
                "feature_set": label,
                "n_features": len(fcols),
                "r2": round(r2, 4),
                "rmse_kw": round(rmse, 2),
                "nmae_pct": round(mae / rated * 100, 3),
                "n_samples": int(len(yv)),
                "nwp_source": nwp_source,
            })
            logger.info(f"  NWP ablation {base_target} {horizon} {label:15s}: "
                        f"R2={r2:.4f} RMSE={rmse:.1f} kW ({len(fcols)} features)")

    return pd.DataFrame(rows)
