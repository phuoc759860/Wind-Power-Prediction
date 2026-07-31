import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

HORIZON_COLORS = {
    "10min": "#2ecc71",
    "30min": "#3498db",
    "1hour": "#9b59b6",
    "6hour": "#e67e22",
    "24hour": "#e74c3c",
}

HORIZON_LABELS = {
    "10min": "10 min",
    "30min": "30 min",
    "1hour": "1 h",
    "6hour": "6 h",
    "24hour": "24 h",
}

TURBINE_IDS = [f"TB{i:02d}" for i in range(1, 13)]


def _model_display_name(name: str) -> str:
    mapping = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "persistence": "Persistence"}
    return mapping.get(name.lower(), name)


def _horizon_label(h: str) -> str:
    return HORIZON_LABELS.get(h, h)


def _rated_power_for_target(target: str, default: float = 2200,
                            farm_rated: float = 26400) -> float:
    """Farm-total targets are normalized against farm rated power (12 x 2200 kW)."""
    return farm_rated if "farm_total_power" in target else default


def compute_metrics(actual: np.ndarray, predicted: np.ndarray, rated_power: float = 2200,
                    exclude_mask: np.ndarray = None) -> Dict:
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    if exclude_mask is not None:
        valid = valid & ~np.asarray(exclude_mask, dtype=bool)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {"mae": np.nan, "rmse": np.nan, "nrmse_pct": np.nan, "r2": np.nan, "max_error": np.nan}

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    nmae = (mae / rated_power * 100) if rated_power > 0 else np.nan
    nrmse = (rmse / rated_power * 100) if rated_power > 0 else np.nan
    r2 = r2_score(actual, predicted)
    max_err = np.max(np.abs(actual - predicted))
    bias = np.mean(predicted - actual)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "nmae_pct": round(nmae, 4),
        "nrmse_pct": round(nrmse, 4),
        "bias": round(bias, 4),
        "r2": round(r2, 4),
        "max_error": round(max_err, 4),
        "n_samples": int(len(actual)),
    }


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return np.nan
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_skill_score(rmse_model: float, rmse_baseline: float) -> float:
    if rmse_baseline == 0 or np.isnan(rmse_baseline) or np.isnan(rmse_model):
        return np.nan
    return round(1 - rmse_model / rmse_baseline, 4)


def evaluate_all_models(test_data: pd.DataFrame, predictions: Dict, baseline_metrics: Dict,
                        config: dict, baseline_predictions: Dict = None,
                        ridge_predictions: Dict = None,
                        rated_power: float = 2200) -> pd.DataFrame:
    results = []

    for model_key, pred_info in predictions.items():
        model_name = pred_info.get("model_name", "unknown")
        target = pred_info.get("target", model_key)
        pred_values = pred_info.get("predictions")

        if pred_values is None:
            continue

        actual = test_data[target].values[:len(pred_values)] if target in test_data.columns else None
        if actual is None:
            continue

        valid = ~(np.isnan(actual) | np.isnan(pred_values))

        rated_power = _rated_power_for_target(target, rated_power)

        # Segment decomposition (reviewer P0-03): near-rated plateau and near-zero
        cap_mask = (actual >= rated_power * 0.95) & valid
        zero_mask = (actual <= rated_power * 0.01) & valid
        seg_mask = cap_mask | zero_mask

        metrics = compute_metrics(actual, pred_values, rated_power)
        metrics_excl = compute_metrics(actual, pred_values, rated_power, exclude_mask=seg_mask)

        horizon = "unknown"
        for h in config.get("forecasting", {}).get("horizons", []):
            if f"_target_{h['name']}" in target:
                horizon = h["name"]
                break

        base_target = target.replace(f"_target_{horizon}", "") if horizon != "unknown" else target
        baseline_key = f"{base_target}_{horizon}" if horizon != "unknown" else base_target
        if baseline_key in baseline_metrics:
            rmse_base = baseline_metrics[baseline_key].get("rmse", np.nan)
            skill = compute_skill_score(metrics["rmse"], rmse_base)
        else:
            skill = np.nan

        # Skill vs persistence on the SAME test samples (P0-03 fix).
        skill_vs_persistence = np.nan
        skill_vs_ridge = np.nan
        if baseline_predictions is not None and target in baseline_predictions:
            base_pred = np.asarray(baseline_predictions[target][:len(pred_values)], dtype=float)
            both = valid & ~np.isnan(base_pred)
            if both.sum() > 0:
                skill_vs_persistence = compute_skill_score(
                    _rmse(actual[both], pred_values[both]),
                    _rmse(actual[both], base_pred[both]))
        if ridge_predictions is not None and target in ridge_predictions:
            ridge_pred = np.asarray(ridge_predictions[target][:len(pred_values)], dtype=float)
            both = valid & ~np.isnan(ridge_pred)
            if both.sum() > 0:
                skill_vs_ridge = compute_skill_score(
                    _rmse(actual[both], pred_values[both]),
                    _rmse(actual[both], ridge_pred[both]))

        hor_label = horizon.replace("hour", "h").replace("min", "min")
        results.append({
            "target": target,
            "model": model_name,
            "horizon": horizon,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "nmae_pct": metrics["nmae_pct"],
            "nrmse_pct": metrics["nrmse_pct"],
            "bias": metrics["bias"],
            "r2": metrics["r2"],
            "max_error": metrics["max_error"],
            "skill_score": skill,
            "skill_vs_persistence": skill_vs_persistence,
            "skill_vs_ridge": skill_vs_ridge,
            "rmse_excl_capacity_zero": metrics_excl["rmse"],
            "n_samples": metrics["n_samples"],
            "n_at_capacity": int(cap_mask.sum()),
            "n_zero_power": int(zero_mask.sum()),
        })

    return pd.DataFrame(results)


def append_baseline_rows(results_df: pd.DataFrame, test_data: pd.DataFrame,
                         baseline_predictions: Dict, ridge_predictions: Dict,
                         config: dict, rated_power: float = 2200) -> pd.DataFrame:
    """Append persistence + ridge rows evaluated on the SAME test samples (P0-03).

    Each baseline row carries explicit RMSE/R2/skill columns and its own
    n_samples so the report can compare every model against the baselines on
    identical (turbine, horizon) samples without index-misalignment leakage.
    """
    if baseline_predictions is None:
        baseline_predictions = {}
    if ridge_predictions is None:
        ridge_predictions = {}

    horizons = config.get("forecasting", {}).get("horizons", [])
    targets = sorted(set(list(baseline_predictions.keys()) + list(ridge_predictions.keys())))
    rows = []
    for target in targets:
        horizon = "unknown"
        for h in horizons:
            if f"_target_{h['name']}" in target:
                horizon = h["name"]
                break
        if target not in test_data.columns:
            continue
        actual = test_data[target].values
        base = target.replace(f"_target_{horizon}", "") if horizon != "unknown" else target

        for mdl_name, pred_key in [("persistence", "persistence"), ("ridge", "ridge")]:
            pred_map = baseline_predictions if mdl_name == "persistence" else ridge_predictions
            preds = np.asarray(pred_map.get(target, []), dtype=float)
            if preds is None or len(preds) == 0:
                continue
            preds = preds[:len(actual)]
            valid = ~(np.isnan(actual) | np.isnan(preds))
            if valid.sum() == 0:
                continue
            rated_power = _rated_power_for_target(target, rated_power)
            m = compute_metrics(actual, preds, rated_power)
            cap_mask = (actual >= rated_power * 0.95) & valid
            zero_mask = (actual <= rated_power * 0.01) & valid
            rows.append({
                "target": target,
                "model": mdl_name,
                "horizon": horizon,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "nmae_pct": m["nmae_pct"],
                "nrmse_pct": m["nrmse_pct"],
                "bias": m["bias"],
                "r2": m["r2"],
                "max_error": m["max_error"],
                "skill_score": np.nan,
                "skill_vs_persistence": np.nan,
                "skill_vs_ridge": np.nan,
                "rmse_excl_capacity_zero": np.nan,
                "n_samples": m["n_samples"],
                "n_at_capacity": int(cap_mask.sum()),
                "n_zero_power": int(zero_mask.sum()),
            })
    if rows:
        results_df = pd.concat([results_df, pd.DataFrame(rows)], ignore_index=True)
    return results_df


def _parse_target(target: str):
    for h in ["10min", "30min", "1hour", "6hour", "24hour"]:
        if target.endswith(f"_power_target_{h}"):
            return target.replace(f"_power_target_{h}", ""), h
    return target, None


def plot_actual_vs_predicted(actual: np.ndarray, predicted: np.ndarray,
                             title: str, save_path: str = None):
    fig, ax = plt.subplots(figsize=(14, 6))
    n = min(len(actual), len(predicted))
    ax.plot(actual[:n], label="Actual", alpha=0.7, linewidth=0.8)
    ax.plot(predicted[:n], label="Predicted", alpha=0.7, linewidth=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Power (kW)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_performance_heatmap(results_df: pd.DataFrame, save_path: str = None):
    if results_df.empty:
        return

    df = results_df.copy()
    df[["turbine", "horizon"]] = df["target"].apply(lambda x: pd.Series(_parse_target(x)))
    df = df[df["turbine"].isin(TURBINE_IDS)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for i, model in enumerate(["xgboost", "lightgbm"]):
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="turbine", columns="horizon", values="r2")
        pivot = pivot.reindex(columns=[h for h in HORIZON_COLORS if h in pivot.columns])
        pivot = pivot.reindex(index=TURBINE_IDS)
        pivot.columns = [_horizon_label(c) for c in pivot.columns]

        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
                    ax=axes[i], linewidths=0.5, cbar_kws={"label": "R\u00B2"})
        display = _model_display_name(model)
        axes[i].set_title(f"{display} — R\u00B2 by Turbine & Horizon", fontsize=12)
        axes[i].set_xlabel("Forecast Horizon")
        axes[i].set_ylabel("Turbine")

    plt.suptitle("Model Performance Heatmap", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_horizon_decay(results_df: pd.DataFrame, save_path: str = None):
    if results_df.empty:
        return

    df = results_df.copy()
    df[["turbine", "horizon"]] = df["target"].apply(lambda x: pd.Series(_parse_target(x)))
    df = df[df["turbine"].isin(TURBINE_IDS)]

    horizon_order = [h for h in HORIZON_COLORS if h in df["horizon"].unique()]
    horizon_labels_display = [_horizon_label(h) for h in horizon_order]
    x_pos = range(len(horizon_order))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, model in enumerate(["xgboost", "lightgbm"]):
        sub = df[df["model"] == model]
        if sub.empty:
            continue

        for tid in TURBINE_IDS:
            tsub = sub[sub["turbine"] == tid]
            if tsub.empty:
                continue
            vals = tsub.set_index("horizon").reindex(horizon_order)["r2"]
            axes[i].plot(x_pos, vals.values, marker="o", alpha=0.5, linewidth=1.2,
                         label=tid, markersize=4)

        avg = sub.groupby("horizon")["r2"].mean().reindex(horizon_order)
        axes[i].plot(x_pos, avg.values, marker="D", linewidth=3, color="black",
                     label="Mean turbine R\u00B2", markersize=7, zorder=10)

        display = _model_display_name(model)
        axes[i].set_title(f"{display} — R\u00B2 vs Forecast Horizon", fontsize=12)
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(horizon_labels_display)
        axes[i].set_xlabel("Forecast Horizon")
        axes[i].set_ylabel("R\u00B2")
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=7, ncol=3, loc="lower left")

    plt.suptitle("Forecast Skill Decay Across Horizons", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_best_model_scatter(results_df: pd.DataFrame, test_data: pd.DataFrame,
                            predictions: Dict, save_path: str = None):
    if results_df.empty:
        return

    # Baselines have no prediction arrays in `predictions`; keep them for the
    # metrics tables but out of the "best model" plots.
    results_df = results_df[~results_df["model"].isin(["persistence", "ridge"])]
    if results_df.empty:
        return

    horizon_order = ["10min", "30min", "1hour", "6hour", "24hour"]
    horizons_present = [h for h in horizon_order if h in results_df["horizon"].values]
    if not horizons_present:
        return

    fig, axes = plt.subplots(1, len(horizons_present), figsize=(5 * len(horizons_present), 5))
    if len(horizons_present) == 1:
        axes = [axes]

    for i, h in enumerate(horizons_present):
        hsub = results_df[results_df["horizon"] == h]
        best_row = hsub.loc[hsub["r2"].idxmax()]
        target = best_row["target"]
        model_name = best_row["model"]
        model_key = f"{target}_{model_name}"

        if model_key not in predictions:
            continue

        pred_vals = predictions[model_key]["predictions"]
        actual_vals = test_data[target].values[:len(pred_vals)]
        valid = ~(np.isnan(actual_vals) | np.isnan(pred_vals))
        actual_vals = actual_vals[valid]
        pred_vals = pred_vals[valid]

        axes[i].scatter(actual_vals, pred_vals, alpha=0.15, s=1, color=HORIZON_COLORS.get(h, "steelblue"))
        lims = [0, 2200]
        axes[i].plot(lims, lims, "r--", linewidth=1.5, alpha=0.8, label="Perfect")
        axes[i].set_xlim(lims)
        axes[i].set_ylim(lims)
        axes[i].set_aspect("equal")
        axes[i].set_title(f"{_horizon_label(h)}\nR\u00B2={best_row['r2']:.3f}  MAE={best_row['mae']:.0f}kW\n({_model_display_name(model_name)})",
                          fontsize=10)
        axes[i].set_xlabel("Actual Power (kW)")
        if i == 0:
            axes[i].set_ylabel("Predicted Power (kW)")
        axes[i].grid(True, alpha=0.2)

    plt.suptitle("Best Model — Actual vs Predicted by Horizon", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_histogram(results_df: pd.DataFrame, test_data: pd.DataFrame,
                         predictions: Dict, save_path: str = None):
    if results_df.empty:
        return

    results_df = results_df[~results_df["model"].isin(["persistence", "ridge"])]
    if results_df.empty:
        return

    horizon_order = ["10min", "30min", "1hour", "6hour", "24hour"]
    horizons_present = [h for h in horizon_order if h in results_df["horizon"].values]
    if not horizons_present:
        return

    fig, axes = plt.subplots(1, len(horizons_present), figsize=(5 * len(horizons_present), 4))
    if len(horizons_present) == 1:
        axes = [axes]

    for i, h in enumerate(horizons_present):
        hsub = results_df[results_df["horizon"] == h]
        best_row = hsub.loc[hsub["r2"].idxmax()]
        target = best_row["target"]
        model_name = best_row["model"]
        model_key = f"{target}_{model_name}"

        if model_key not in predictions:
            continue

        pred_vals = predictions[model_key]["predictions"]
        actual_vals = test_data[target].values[:len(pred_vals)]
        valid = ~(np.isnan(actual_vals) | np.isnan(pred_vals))
        errors = (actual_vals[valid] - pred_vals[valid])

        axes[i].hist(errors, bins=80, color=HORIZON_COLORS.get(h, "steelblue"),
                     alpha=0.8, edgecolor="none", density=True)
        axes[i].axvline(x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
        axes[i].set_title(f"{_horizon_label(h)} (\u03C3={np.std(errors):.0f}kW)", fontsize=10)
        axes[i].set_xlabel("Error (kW)")
        if i == 0:
            axes[i].set_ylabel("Density")
        axes[i].grid(True, alpha=0.2)

    plt.suptitle("Prediction Error Distribution by Horizon", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_farm_timeseries(test_data: pd.DataFrame, predictions: Dict, save_path: str = None):
    farm_key = None
    for mk in predictions:
        if "farm_total_power" in mk and "10min" in mk and predictions[mk].get("model_name") == "lightgbm":
            farm_key = mk
            break
    if farm_key is None:
        for mk in predictions:
            if "farm_total_power" in mk and "10min" in mk:
                farm_key = mk
                break
    if farm_key is None:
        return

    target = predictions[farm_key]["target"]
    pred_vals = predictions[farm_key]["predictions"]
    actual_vals = test_data[target].values[:len(pred_vals)]

    n_display = 2000
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(actual_vals[:n_display], label="Actual", linewidth=1, alpha=0.8)
    ax.plot(pred_vals[:n_display], label="Predicted (10min)", linewidth=1, alpha=0.8)
    ax.set_title("Farm Total Power — 10min Ahead Forecast (First 2000 Steps)", fontsize=13)
    ax.set_xlabel("Time Step (10-min intervals)")
    ax.set_ylabel("Power (kW)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_radar_summary(results_df: pd.DataFrame, save_path: str = None):
    if results_df.empty:
        return

    df = results_df.copy()
    df[["turbine", "horizon"]] = df["target"].apply(lambda x: pd.Series(_parse_target(x)))
    df = df[df["turbine"].isin(TURBINE_IDS)]

    horizon_order = [h for h in ["10min", "30min", "1hour", "6hour", "24hour"] if h in df["horizon"].unique()]
    if len(horizon_order) < 3:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for model in ["xgboost", "lightgbm"]:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        avg_r2 = sub.groupby("horizon")["r2"].mean().reindex(horizon_order)
        avg_mae = sub.groupby("horizon")["mae"].mean().reindex(horizon_order) / 2200

        x = range(len(horizon_order))
        color = "steelblue" if model == "xgboost" else "darkorange"
        display = _model_display_name(model)
        ax.plot(x, avg_r2.values, marker="o", linewidth=2.5, color=color,
                label=f"{display} R\u00B2", markersize=8)
        ax.plot(x, 1 - avg_mae.values, marker="s", linewidth=2.5, color=color,
                linestyle="--", label=f"{display} (1-NRMSE)", markersize=8, alpha=0.7)

    ax.set_xticks(range(len(horizon_order)))
    ax.set_xticklabels([_horizon_label(h) for h in horizon_order], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Forecast Horizon", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Quality Across Horizons (Farm-Level Average)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_farm_level_metrics(test_data: pd.DataFrame, predictions: Dict,
                                config: dict, rated_power: float = 26400) -> pd.DataFrame:
    farm_results = []
    farm_actual = test_data["farm_total_power"].values if "farm_total_power" in test_data.columns else None
    if farm_actual is None:
        return pd.DataFrame()

    for model_key, pred_info in predictions.items():
        model_name = pred_info.get("model_name", "unknown")
        target = pred_info.get("target", model_key)
        if "farm_total_power" not in target:
            continue
        pred_values = pred_info.get("predictions")
        if pred_values is None:
            continue

        horizon = "unknown"
        for h in config.get("forecasting", {}).get("horizons", []):
            if f"_target_{h['name']}" in target:
                horizon = h["name"]
                break

        n = min(len(farm_actual), len(pred_values))
        metrics = compute_metrics(farm_actual[:n], pred_values[:n], rated_power)
        farm_results.append({
            "target": target,
            "model": model_name,
            "horizon": horizon,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "nmae_pct": metrics["nmae_pct"],
            "nrmse_pct": metrics["nrmse_pct"],
            "bias": metrics["bias"],
            "r2": metrics["r2"],
            "max_error": metrics["max_error"],
            "n_samples": metrics["n_samples"],
            "level": "farm_total",
        })

    return pd.DataFrame(farm_results)


def analyze_farm_bias(test_data: pd.DataFrame, predictions: Dict,
                      config: dict, rated_power: float = 26400) -> pd.DataFrame:
    """Farm-level bias analysis (reviewer P1-04).

    Compares the direct farm_total_power model forecast with the SUM of the 12
    individual turbine forecasts, and reports bias in kW and % of rated farm
    power, plus whether forecasts at capacity/zero dominate the error.
    """
    horizons = config.get("forecasting", {}).get("horizons", [])
    records = []

    for h in horizons:
        h_name = h["name"]
        farm_target = f"farm_total_power_target_{h_name}"

        farm_keys = [k for k in predictions if predictions[k].get("target") == farm_target]
        if not farm_keys or farm_target not in test_data.columns:
            continue

        farm_key = farm_keys[0]
        farm_pred = np.asarray(predictions[farm_key]["predictions"], dtype=float)
        actual = test_data[farm_target].values[:len(farm_pred)]
        valid = ~(np.isnan(actual) | np.isnan(farm_pred))
        farm_pred = farm_pred[valid]
        actual = actual[valid]

        # Sum of the 12 individual turbine forecasts (same horizon, any model).
        sum_pred = np.zeros(len(actual))
        have_all = True
        for tb in TURBINE_IDS:
            t_target = f"{tb}_power_target_{h_name}"
            t_keys = [k for k in predictions if predictions[k].get("target") == t_target]
            if not t_keys:
                have_all = False
                break
            p = np.asarray(predictions[t_keys[0]]["predictions"], dtype=float)
            sum_pred += p[:len(sum_pred)]

        if not have_all:
            sum_pred = np.full(len(actual), np.nan)

        bias_kw = float(np.mean(farm_pred - actual)) if len(actual) else np.nan
        bias_pct_rated = bias_kw / rated_power * 100 if rated_power else np.nan
        mae_kw = float(np.mean(np.abs(farm_pred - actual))) if len(actual) else np.nan

        diff_farm_vs_sum = float(np.mean(farm_pred - sum_pred)) if have_all and not np.isnan(sum_pred).all() else np.nan

        records.append({
            "horizon": h_name,
            "n_samples": int(len(actual)),
            "actual_mean_kw": round(float(np.mean(actual)), 2),
            "farm_model_mean_kw": round(float(np.mean(farm_pred)), 2),
            "bias_kw": round(bias_kw, 2) if bias_kw == bias_kw else None,
            "bias_pct_rated": round(bias_pct_rated, 3) if bias_pct_rated == bias_pct_rated else None,
            "mae_kw": round(mae_kw, 2) if mae_kw == mae_kw else None,
            "farm_vs_sum_turbines_kw": round(diff_farm_vs_sum, 2) if diff_farm_vs_sum == diff_farm_vs_sum else None,
            "n_at_capacity": int((actual >= rated_power * 0.95).sum()),
            "n_zero_power": int((actual <= rated_power * 0.01).sum()),
            "note": ("Direct farm model vs sum of 12 turbine models. "
                     "Bias > 0 means the farm model over-forecasts on average."),
        })

    return pd.DataFrame(records)


def plot_farm_bias_calibration(test_data: pd.DataFrame, predictions: Dict,
                               config: dict, save_path: str = None):
    """Scatter of actual farm power vs farm-model forecast + calibration line."""
    df = analyze_farm_bias(test_data, predictions, config)
    if df.empty:
        return

    farm_target = "farm_total_power_target_10min"
    farm_key = None
    for k in predictions:
        if predictions[k].get("target") == farm_target:
            farm_key = k
            break
    if farm_key is None:
        return

    farm_pred = np.asarray(predictions[farm_key]["predictions"], dtype=float)
    actual = test_data[farm_target].values[:len(farm_pred)]
    valid = ~(np.isnan(actual) | np.isnan(farm_pred))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual[valid], farm_pred[valid], alpha=0.15, s=2, color="#e67e22")
    lims = [0, 26400]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_title("Farm Total Power — Direct Model Calibration (10min)")
    ax.set_xlabel("Actual Farm Power (kW)")
    ax.set_ylabel("Farm Model Prediction (kW)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_tb12(test_data: pd.DataFrame, results_df: pd.DataFrame,
                 split_data: Dict = None) -> Dict:
    tb12_analysis = {}
    tb12_power = test_data.get("TB12_power") if "TB12_power" in test_data.columns else None
    if tb12_power is None:
        return tb12_analysis

    # P1-02: missing/stopped/frozen breakdown per split (train/val/test).
    if split_data:
        tb12_analysis["per_split"] = {}
        for split_name, df in split_data.items():
            col = f"TB12_power"
            if col not in df.columns:
                continue
            vals = df[col].values
            frozen = _detect_frozen_data(vals)
            tb12_analysis["per_split"][split_name] = {
                "n_rows": int(len(df)),
                "missing_rate_pct": round(np.isnan(vals).mean() * 100, 2) if len(vals) else None,
                "stopped_rate_pct": round(((vals == 0) | (vals < 5)).sum() / len(vals) * 100, 2) if len(vals) else None,
                "frozen_blocks": frozen["count"],
                "frozen_ratio_pct": round(frozen["ratio"] * 100, 2),
                "mean_power_kw": round(float(np.nanmean(vals)), 1) if len(vals) else None,
            }

    tb12_power_values = tb12_power.values
    tb12_analysis["missing_rate"] = round(np.isnan(tb12_power_values).mean() * 100, 2)
    tb12_analysis["stopped_rate"] = round(((tb12_power_values == 0) | (tb12_power_values < 5)).sum() / len(tb12_power_values) * 100, 2)
    tb12_analysis["mean_power_when_operating"] = round(np.nanmean(tb12_power_values[tb12_power_values > 5]), 2)

    frozen = _detect_frozen_data(tb12_power_values)
    tb12_analysis["frozen_data_blocks"] = frozen["count"]
    tb12_analysis["frozen_data_ratio"] = round(frozen["ratio"] * 100, 2)

    for ref_tb in [t for t in TURBINE_IDS if t != "TB12"]:
        ref_col = f"{ref_tb}_power"
        if ref_col not in test_data.columns:
            continue
        ref_vals = test_data[ref_col].values
        valid = ~(np.isnan(tb12_power_values) | np.isnan(ref_vals))
        if valid.sum() > 10:
            corr = np.corrcoef(tb12_power_values[valid], ref_vals[valid])[0, 1]
            tb12_analysis[f"power_corr_with_{ref_tb}"] = round(float(corr), 4)
            tb12_analysis[f"mean_power_{ref_tb}"] = round(float(np.nanmean(ref_vals)), 1)

    tb12_analysis["mean_power_TB12"] = round(float(np.nanmean(tb12_power_values)), 1)
    tb12_analysis["std_power_TB12"] = round(float(np.nanstd(tb12_power_values[tb12_power_values > 5])), 1)

    ws_col = "TB12_wind_speed"
    if ws_col in test_data.columns:
        ws = test_data[ws_col].values
        valid_ws = ~(np.isnan(ws) | np.isnan(tb12_power_values))
        if valid_ws.sum() > 10:
            tb12_analysis["wind_speed_corr_with_power"] = round(
                float(np.corrcoef(tb12_power_values[valid_ws], ws[valid_ws])[0, 1]), 4)
            tb12_analysis["mean_wind_speed"] = round(float(np.nanmean(ws)), 2)

        if "TB09_wind_speed" in test_data.columns:
            tb12_analysis["ws_corr_TB12_vs_TB09"] = round(
                float(np.corrcoef(ws[valid_ws], test_data["TB09_wind_speed"].values[valid_ws])[0, 1]), 4)
            tb12_analysis["mean_wind_speed_TB09"] = round(float(np.nanmean(test_data["TB09_wind_speed"].values)), 2)
        if "TB11_wind_speed" in test_data.columns:
            tb12_analysis["mean_wind_speed_TB11"] = round(float(np.nanmean(test_data["TB11_wind_speed"].values)), 2)
            tb12_analysis["ws_corr_TB12_vs_TB11"] = round(
                float(np.corrcoef(ws[valid_ws], test_data["TB11_wind_speed"].values[valid_ws])[0, 1]), 4)

    tb12_results = results_df[results_df["target"].str.startswith("TB12")] if not results_df.empty else pd.DataFrame()
    if not tb12_results.empty:
        tb12_analysis["mean_r2"] = round(tb12_results["r2"].mean(), 4)
        for _, row in tb12_results.iterrows():
            tb12_analysis[f"r2_{row['horizon']}_{row['model']}"] = row["r2"]

    findings = []
    if tb12_analysis.get("missing_rate", 0) > 20:
        findings.append("HIGH_MISSING: missing rate >20% - possible communication loss")
    if tb12_analysis.get("frozen_data_ratio", 0) > 10:
        findings.append("FROZEN_SIGNAL: frozen data ratio >10% - possible sensor drift")
    if tb12_analysis.get("wind_speed_corr_with_power", 1) < 0.5:
        findings.append("LOW_WS_POWER_CORR: wind-power correlation <0.5 - power curve deviation")
    neighbor_corr = [v for k, v in tb12_analysis.items() if k.startswith("power_corr_with_")]
    if neighbor_corr and np.mean(neighbor_corr) < 0.4:
        findings.append("LOW_NEIGHBOR_CORR: avg power correlation with neighbors <0.4 - anomalous behavior")
    if tb12_analysis.get("stopped_rate", 0) > 30:
        findings.append("HIGH_STOPPED: stopped rate >30% - possible maintenance/fault issue")
    if tb12_analysis.get("mean_power_TB12", 1000) < 200:
        findings.append("LOW_MEAN_POWER: mean power <200kW - significantly underperforming")
    tb12_analysis["findings"] = findings
    if not findings:
        tb12_analysis["findings"] = ["NO_ANOMALY_DETECTED: TB12 behavior within expected range"]

    return tb12_analysis


def _detect_frozen_data(values: np.ndarray, threshold: int = 5) -> Dict:
    frozen_blocks = 0
    frozen_samples = 0
    i = 0
    n = len(values)
    while i < n - threshold:
        if np.isnan(values[i]):
            i += 1
            continue
        block = [values[i]]
        j = i + 1
        while j < n and abs(values[j] - values[i]) < 1.0:
            block.append(values[j])
            j += 1
        if len(block) >= threshold:
            frozen_blocks += 1
            frozen_samples += len(block)
        i = j
    return {"count": frozen_blocks, "ratio": frozen_samples / n if n > 0 else 0}


def plot_tb12_distribution(test_data: pd.DataFrame, save_path: str = None):
    if "TB12_power" not in test_data.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for tb, ax, color in [("TB12", axes[0], "#e74c3c"), ("TB09", axes[0], "#3498db"),
                           ("TB12", axes[1], "#e74c3c"), ("TB04", axes[1], "#2ecc71")]:
        col = f"{tb}_power"
        if col not in test_data.columns:
            continue
        vals = test_data[col].dropna().values
        ax_idx = 0 if tb in ["TB12", "TB09"] else 1
        axes[ax_idx].hist(vals, bins=80, alpha=0.5, color=color, label=tb, density=True)
    axes[0].set_title("Power Distribution: TB12 vs TB09")
    axes[0].set_xlabel("Power (kW)"); axes[0].set_ylabel("Density"); axes[0].legend()
    axes[1].set_title("Power Distribution: TB12 vs TB04")
    axes[1].set_xlabel("Power (kW)"); axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.2)
    plt.suptitle("TB12 Power Distribution Comparison", fontsize=13)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_by_wind_speed(test_data: pd.DataFrame, predictions: Dict,
                               results_df: pd.DataFrame, save_path: str = None):
    if results_df.empty:
        return

    horizon_order = ["10min", "30min", "1hour", "6hour", "24hour"]
    horizons_present = [h for h in horizon_order if h in results_df["horizon"].values]
    if not horizons_present:
        return

    fig, axes = plt.subplots(1, len(horizons_present), figsize=(5 * len(horizons_present), 4))
    if len(horizons_present) == 1:
        axes = [axes]

    for i, h in enumerate(horizons_present):
        hsub = results_df[results_df["horizon"] == h]
        if hsub.empty:
            continue
        best_row = hsub.loc[hsub["r2"].idxmax()]
        target = best_row["target"]
        turbine_id = target.replace("_power_target_10min", "").replace("_power_target_30min", "").replace("_power_target_1hour", "").replace("_power_target_6hour", "").replace("_power_target_24hour", "").replace("_power", "")
        model_name = best_row["model"]
        model_key = f"{target}_{model_name}"
        if model_key not in predictions:
            continue

        pred_vals = predictions[model_key]["predictions"]
        actual_vals = test_data[target].values[:len(pred_vals)]

        ws_col = f"{turbine_id}_wind_speed" if turbine_id in TURBINE_IDS else None
        if ws_col is None or ws_col not in test_data.columns:
            ws_col = "TB01_wind_speed"
            if ws_col not in test_data.columns:
                continue
        ws_vals = test_data[ws_col].values[:len(pred_vals)]

        valid = ~(np.isnan(actual_vals) | np.isnan(pred_vals) | np.isnan(ws_vals))
        actual = actual_vals[valid]
        pred = pred_vals[valid]
        ws = ws_vals[valid]

        bins = [0, 3, 6, 9, 12, 15, 20, 30, 60]
        labels = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-20", "20-25", "25+"]
        bin_idx = np.digitize(ws, bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(labels) - 1)

        mae_per_bin = []
        bin_labels = []
        for b in range(len(labels)):
            mask = bin_idx == b
            if mask.sum() > 5:
                bin_labels.append(labels[b])
                mae_per_bin.append(np.mean(np.abs(actual[mask] - pred[mask])))

        if bin_labels:
            axes[i].bar(bin_labels, mae_per_bin, color=HORIZON_COLORS.get(h, "steelblue"), alpha=0.7)
            axes[i].set_title(_horizon_label(h))
            axes[i].tick_params(axis="x", rotation=45)

    plt.suptitle("Prediction Error by Wind Speed", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_by_power_region(test_data: pd.DataFrame, predictions: Dict,
                                 results_df: pd.DataFrame, save_path: str = None):
    if results_df.empty or "farm_total_power" not in test_data.columns:
        return

    horizon_order = ["10min", "30min", "1hour", "6hour", "24hour"]
    horizons_present = [h for h in horizon_order if h in results_df["horizon"].values]
    if not horizons_present:
        return

    regions = [(0, 500, "Low (0-500kW)"), (500, 1500, "Medium (500-1500kW)"),
               (1500, 2200, "High (1500-2200kW)")]

    fig, axes = plt.subplots(1, len(horizons_present), figsize=(5 * len(horizons_present), 4))
    if len(horizons_present) == 1:
        axes = [axes]

    for i, h in enumerate(horizons_present):
        hsub = results_df[results_df["horizon"] == h]
        if hsub.empty:
            continue
        best_row = hsub.loc[hsub["r2"].idxmax()]
        target = best_row["target"]
        model_name = best_row["model"]
        model_key = f"{target}_{model_name}"
        if model_key not in predictions:
            continue

        pred_vals = predictions[model_key]["predictions"]
        actual_vals = test_data[target].values[:len(pred_vals)]
        valid = ~(np.isnan(actual_vals) | np.isnan(pred_vals))
        actual = actual_vals[valid]
        pred = pred_vals[valid]
        errors = np.abs(actual - pred)

        labels = []
        mae_list = []
        for lo, hi, label in regions:
            mask = (actual >= lo) & (actual < hi)
            if mask.sum() > 0:
                labels.append(label)
                mae_list.append(np.mean(errors[mask]))

        if labels:
            axes[i].bar(labels, mae_list, color=HORIZON_COLORS.get(h, "steelblue"), alpha=0.7)
            axes[i].set_title(_horizon_label(h))
            axes[i].set_ylabel("MAE (kW)")
            axes[i].tick_params(axis="x", rotation=15)

    plt.suptitle("Prediction Error by Power Region", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_by_season(test_data: pd.DataFrame, predictions: Dict,
                          results_df: pd.DataFrame, save_path: str = None):
    if results_df.empty or "season" not in test_data.columns:
        return
    if "farm_total_power" not in test_data.columns:
        return

    season_map = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}
    horizon_order = ["10min", "30min", "1hour", "6hour", "24hour"]
    horizons_present = [h for h in horizon_order if h in results_df["horizon"].values]
    if not horizons_present:
        return

    fig, axes = plt.subplots(1, len(horizons_present), figsize=(5 * len(horizons_present), 4))
    if len(horizons_present) == 1:
        axes = [axes]

    for i, h in enumerate(horizons_present):
        hsub = results_df[results_df["horizon"] == h]
        if hsub.empty:
            continue
        best_row = hsub.loc[hsub["r2"].idxmax()]
        target = best_row["target"]
        model_name = best_row["model"]
        model_key = f"{target}_{model_name}"
        if model_key not in predictions:
            continue

        pred_vals = predictions[model_key]["predictions"]
        actual_vals = test_data[target].values[:len(pred_vals)]
        season_vals = test_data["season"].values[:len(pred_vals)]

        valid = ~(np.isnan(actual_vals) | np.isnan(pred_vals) | pd.isna(season_vals))
        actual = actual_vals[valid]
        pred = pred_vals[valid]
        seasons = season_vals[valid]
        errors = np.abs(actual - pred)

        labels = []
        mae_list = []
        for s in sorted(set(seasons)):
            mask = seasons == s
            if mask.sum() > 0:
                labels.append(season_map.get(s, f"Season {s}"))
                mae_list.append(np.mean(errors[mask]))

        if labels:
            axes[i].bar(labels, mae_list, color=HORIZON_COLORS.get(h, "steelblue"), alpha=0.7)
            axes[i].set_title(_horizon_label(h))
            axes[i].tick_params(axis="x", rotation=15)

    plt.suptitle("Prediction Error by Season", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_residual_analysis(test_data: pd.DataFrame, predictions: Dict,
                            results_df: pd.DataFrame, save_path: str = None):
    if results_df.empty:
        return

    horizon_order = ["10min", "30min", "1hour", "6hour", "24hour"]
    horizons_present = [h for h in horizon_order if h in results_df["horizon"].values]
    if not horizons_present:
        return

    fig, axes = plt.subplots(2, len(horizons_present), figsize=(5 * len(horizons_present), 8))

    for i, h in enumerate(horizons_present):
        hsub = results_df[results_df["horizon"] == h]
        if hsub.empty:
            continue
        best_row = hsub.loc[hsub["r2"].idxmax()]
        target = best_row["target"]
        model_name = best_row["model"]
        model_key = f"{target}_{model_name}"
        if model_key not in predictions:
            continue

        pred_vals = predictions[model_key]["predictions"]
        actual_vals = test_data[target].values[:len(pred_vals)]
        valid = ~(np.isnan(actual_vals) | np.isnan(pred_vals))
        actual = actual_vals[valid]
        pred = pred_vals[valid]
        residuals = actual - pred

        axes[0, i].scatter(pred, residuals, alpha=0.1, s=2, color=HORIZON_COLORS.get(h, "steelblue"))
        axes[0, i].axhline(y=0, color="red", linestyle="--", linewidth=1)
        axes[0, i].set_xlabel("Predicted (kW)")
        axes[0, i].set_title(_horizon_label(h))
        if i == 0:
            axes[0, i].set_ylabel("Residual (kW)")

        axes[1, i].hist(residuals, bins=60, color=HORIZON_COLORS.get(h, "steelblue"),
                        alpha=0.7, edgecolor="none", density=True)
        axes[1, i].axvline(x=0, color="red", linestyle="--", linewidth=1)
        axes[1, i].set_xlabel("Residual (kW)")
        if i == 0:
            axes[1, i].set_ylabel("Density")

    plt.suptitle("Residual Analysis by Horizon", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_baseline_comparison(results_df: pd.DataFrame, baseline_results: Dict,
                              save_path: str = None):
    if results_df.empty or not baseline_results:
        return

    ml_summary = results_df.groupby(["horizon", "model"])["rmse"].mean().reset_index()
    horizon_order = ["10min", "30min", "1hour", "6hour", "24hour"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(horizon_order))
    width = 0.18

    for i, mdl in enumerate(["persistence", "ridge", "xgboost", "lightgbm"]):
        if mdl in ("persistence", "ridge"):
            vals = []
            for h in horizon_order:
                suffix = "" if mdl == "persistence" else "_ridge"
                key = [k for k in baseline_results if h in k and k.endswith(suffix)]
                if key:
                    vals.append(np.mean([baseline_results[k]["rmse"] for k in key]))
                else:
                    vals.append(0)
        else:
            sub = ml_summary[ml_summary["model"] == mdl]
            vals = [sub[sub["horizon"] == h]["rmse"].mean() if h in sub["horizon"].values else 0 for h in horizon_order]

        display = _model_display_name(mdl)
        ax.bar([p + i * width for p in x], vals, width, label=display,
               alpha=0.8, color=list(HORIZON_COLORS.values())[i] if i < len(HORIZON_COLORS) else None)

    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels([_horizon_label(h) for h in horizon_order])
    ax.set_ylabel("RMSE (kW)")
    ax.set_title("Model Comparison: Baseline vs ML Models")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_by_day_night(test_data: pd.DataFrame, predictions: Dict,
                              results_df: pd.DataFrame, save_path: str = None):
    if results_df.empty or "hour_of_day" not in test_data.columns:
        return

    horizon_order = ["10min", "30min", "1hour", "6hour", "24hour"]
    horizons_present = [h for h in horizon_order if h in results_df["horizon"].values]
    if not horizons_present:
        return

    fig, axes = plt.subplots(1, len(horizons_present), figsize=(5 * len(horizons_present), 4))
    if len(horizons_present) == 1:
        axes = [axes]

    for i, h in enumerate(horizons_present):
        hsub = results_df[results_df["horizon"] == h]
        if hsub.empty:
            continue
        best_row = hsub.loc[hsub["r2"].idxmax()]
        target = best_row["target"]
        model_name = best_row["model"]
        model_key = f"{target}_{model_name}"
        if model_key not in predictions:
            continue

        pred_vals = predictions[model_key]["predictions"]
        actual_vals = test_data[target].values[:len(pred_vals)]
        hour_vals = test_data["hour_of_day"].values[:len(pred_vals)]

        valid = ~(np.isnan(actual_vals) | np.isnan(pred_vals) | pd.isna(hour_vals))
        actual = actual_vals[valid]
        pred = pred_vals[valid]
        hours = hour_vals[valid]
        errors = np.abs(actual - pred)

        day_mask = (hours >= 6) & (hours < 18)
        night_mask = ~day_mask

        labels = ["Day (6-18h)", "Night (18-6h)"]
        day_mae = np.mean(errors[day_mask]) if day_mask.sum() > 0 else 0
        night_mae = np.mean(errors[night_mask]) if night_mask.sum() > 0 else 0
        axes[i].bar(labels, [day_mae, night_mae], color=[HORIZON_COLORS.get(h, "steelblue"), "#555555"], alpha=0.7)
        axes[i].set_title(_horizon_label(h))
        axes[i].tick_params(axis="x", rotation=15)

    plt.suptitle("Prediction Error: Day vs Night", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None):
    if results_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if "model" in results_df.columns and "rmse" in results_df.columns:
        model_rmse = results_df.groupby("model")["rmse"].mean().sort_values()
        model_rmse.plot(kind="barh", ax=axes[0], color="steelblue")
        axes[0].set_title("Average RMSE by Model")
        axes[0].set_xlabel("RMSE (kW)")

    if "model" in results_df.columns and "r2" in results_df.columns:
        model_r2 = results_df.groupby("model")["r2"].mean().sort_values(ascending=False)
        model_r2.plot(kind="barh", ax=axes[1], color="darkgreen")
        axes[1].set_title("Average R2 by Model")
        axes[1].set_xlabel("R2")

    if "model" in results_df.columns and "mae" in results_df.columns:
        model_mae = results_df.groupby("model")["mae"].mean().sort_values()
        model_mae.plot(kind="barh", ax=axes[2], color="darkorange")
        axes[2].set_title("Average MAE by Model")
        axes[2].set_xlabel("MAE (kW)")

    plt.suptitle("Model Performance Comparison", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_horizon_comparison(results_df: pd.DataFrame, save_path: str = None):
    if results_df.empty or "horizon" not in results_df.columns:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, metric in enumerate(["mae", "rmse", "r2"]):
        if metric in results_df.columns:
            pivot = results_df.pivot_table(index="model", columns="horizon", values=metric)
            pivot.plot(kind="bar", ax=axes[i], rot=45)
            axes[i].set_title(metric.upper())
            axes[i].set_ylabel(metric)
            axes[i].legend(title="Horizon")

    plt.suptitle("Model Comparison Across Horizons", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_alert_accuracy(test_data: pd.DataFrame, predictions: Dict,
                             ramp_threshold: float = 0.5) -> Dict:
    results = {}
    TURBINE_IDS = [f"TB{i:02d}" for i in range(1, 13)]

    # Evaluate per turbine × horizon — use actual power for ground truth
    for model_key, pred_info in predictions.items():
        model_name = pred_info.get("model_name", "unknown")
        target = pred_info.get("target", model_key)
        pred_vals = pred_info.get("predictions")
        if pred_vals is None or len(pred_vals) < 10:
            continue

        # Identify turbine and horizon from target
        turbine_id = None
        horizon = None
        for tb in TURBINE_IDS:
            if target.startswith(tb) and "_power_target_" in target:
                turbine_id = tb
                horizon = target.split("_target_")[1]
                break
        if turbine_id is None:
            continue

        # Ground truth: use actual power values (or if available, the target column)
        actual_col = f"{turbine_id}_power"
        if actual_col not in test_data.columns:
            continue
        actual_vals = test_data[actual_col].values[:len(pred_vals)]

        # Ramp events on actual
        rated = test_data.get(f"{turbine_id}_rated_power", pd.Series([2200])).iloc[0]
        ramp_actual_pct = np.abs(np.diff(actual_vals, prepend=actual_vals[0])) / rated * 100
        actual_events = ramp_actual_pct > ramp_threshold

        # Ramp events on predicted
        n = min(len(pred_vals), len(actual_events))
        pred_ramp_pct = np.abs(np.diff(pred_vals[:n], prepend=pred_vals[0])) / rated * 100
        pred_events = pred_ramp_pct > ramp_threshold
        actual_events = actual_events[:n]

        tp = np.sum(pred_events & actual_events)
        fp = np.sum(pred_events & ~actual_events)
        fn = np.sum(~pred_events & actual_events)
        tn = np.sum(~pred_events & ~actual_events)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        far = fp / (fp + tp) if (fp + tp) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        balanced_acc = (recall + specificity) / 2

        key = f"{model_name}_{turbine_id}_{horizon}"
        results[key] = {
            "turbine_id": turbine_id,
            "horizon": horizon,
            "model": model_name,
            "definition": ("Ramp event = |power_change| > {:.0f}% of rated per 10min. "
                           "Ground truth from actual {} power. "
                           "TP=alarm+event, FP=alarm+no_event, FN=no_alarm+event, TN=no_alarm+no_event").format(ramp_threshold * 100, turbine_id),
            "threshold_pct": ramp_threshold * 100,
            "n_actual_events": int(actual_events.sum()),
            "n_predicted_events": int(pred_events.sum()),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "false_alarm_rate": round(far, 4),
            "specificity": round(specificity, 4), "fpr": round(fpr, 4),
            "balanced_accuracy": round(balanced_acc, 4),
        }

    return results


def evaluate_anomaly_detection(test_data: pd.DataFrame) -> Dict:
    results = {}
    TURBINE_IDS = [f"TB{i:02d}" for i in range(1, 13)]
    RATED_POWER = 2200

    for tb in TURBINE_IDS:
        pwr_col = f"{tb}_power"
        ws_col = f"{tb}_wind_speed"
        if pwr_col not in test_data.columns or ws_col not in test_data.columns:
            continue

        power = test_data[pwr_col].values
        ws = test_data[ws_col].values
        n = len(power)

        # Ground-truth anomalies (domain-rule based)
        gt_anomaly = np.zeros(n, dtype=bool)
        gt_power = np.zeros(n, dtype=bool)
        gt_ws = np.zeros(n, dtype=bool)
        for i in range(n):
            if np.isnan(power[i]) or np.isnan(ws[i]):
                continue
            if power[i] < 0:
                gt_anomaly[i] = True
                gt_power[i] = True
            elif power[i] > RATED_POWER:
                gt_anomaly[i] = True
                gt_power[i] = True
            elif ws[i] < 3 and power[i] > RATED_POWER * 0.8:
                gt_anomaly[i] = True
                gt_ws[i] = True
            elif ws[i] > 15 and power[i] < RATED_POWER * 0.1:
                gt_anomaly[i] = True
                gt_ws[i] = True

        # Detected anomalies (z-score method, same logic as generate_anomaly_alert)
        valid = ~(np.isnan(power) | np.isnan(ws))
        if valid.sum() < 100:
            continue
        p_valid = power[valid]
        mean_p, std_p = np.mean(p_valid), np.std(p_valid)
        detected = np.zeros(n, dtype=bool)
        for i in range(n):
            if np.isnan(power[i]) or np.isnan(ws[i]):
                continue
            z_score = abs(power[i] - mean_p) / (std_p + 1e-6)
            if z_score > 3.0:
                detected[i] = True

        tp = int(np.sum(detected & gt_anomaly))
        fp = int(np.sum(detected & ~gt_anomaly))
        fn = int(np.sum(~detected & gt_anomaly))
        tn = int(np.sum(~detected & ~gt_anomaly))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        far = fp / (fp + tp) if (fp + tp) > 0 else 0

        # Per-type breakdown
        def _type_stats(mask_type):
            t = int(np.sum(detected & mask_type))
            f = int(np.sum(detected & gt_anomaly & ~mask_type))
            if (t + f) == 0:
                return 0, 0
            return t, t / (t + f) if (t + f) > 0 else 0

        tp_power, prec_power = _type_stats(gt_power)
        tp_ws_curve, prec_ws_curve = _type_stats(gt_ws)

        results[tb] = {
            "method": "z-score > 3 on power (global mean/std)",
            "gt_definition": ("Anomaly = power < 0 OR power > rated OR "
                              "(wind < 3m/s & power > 1760kW) OR (wind > 15m/s & power < 220kW)"),
            "n_gt_anomalies": int(gt_anomaly.sum()),
            "n_detected": int(detected.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_alarm_rate": round(far, 4),
            "gt_power_anomalies": int(gt_power.sum()),
            "detected_power_anomalies": tp_power,
            "power_anomaly_precision": round(prec_power, 4) if prec_power > 0 else 0,
            "gt_wind_curve_anomalies": int(gt_ws.sum()),
            "detected_wind_curve_anomalies": tp_ws_curve,
            "wind_curve_anomaly_precision": round(prec_ws_curve, 4) if prec_ws_curve > 0 else 0,
        }

    return results


def generate_evaluation_report(results_df: pd.DataFrame, save_dir: str,
                                test_data: pd.DataFrame = None,
                                predictions: dict = None,
                                baseline_results: dict = None):
    os.makedirs(save_dir, exist_ok=True)

    if not results_df.empty:
        results_df.to_csv(os.path.join(save_dir, "evaluation_results.csv"), index=False)

        # Baselines have no prediction arrays under {target}_{model} keys, so they
        # must not compete for the "best row" used by the per-horizon plots.
        plot_df = results_df[~results_df["model"].isin(["persistence", "ridge"])]

        plot_model_comparison(plot_df, os.path.join(save_dir, "model_comparison.png"))
        plot_horizon_comparison(plot_df, os.path.join(save_dir, "horizon_comparison.png"))

        if test_data is not None and predictions is not None:
            plot_error_by_power_region(test_data, predictions, plot_df,
                                        os.path.join(save_dir, "08_error_by_power_region.png"))
            plot_error_by_season(test_data, predictions, plot_df,
                                  os.path.join(save_dir, "09_error_by_season.png"))
            plot_error_by_day_night(test_data, predictions, plot_df,
                                     os.path.join(save_dir, "10_error_by_day_night.png"))
            plot_error_by_wind_speed(test_data, predictions, plot_df,
                                      os.path.join(save_dir, "13_error_by_wind_speed.png"))
            plot_residual_analysis(test_data, predictions, plot_df,
                                    os.path.join(save_dir, "11_residual_analysis.png"))
            plot_tb12_distribution(test_data,
                                    os.path.join(save_dir, "14_tb12_distribution.png"))

        if baseline_results is not None:
            plot_baseline_comparison(plot_df, baseline_results,
                                      os.path.join(save_dir, "12_baseline_comparison.png"))

        agg_cols = ["mae", "rmse", "nmae_pct", "nrmse_pct", "bias", "r2"]
        available = [c for c in agg_cols if c in results_df.columns]
        summary = results_df.groupby("model")[available].mean().round(4)
        summary.to_csv(os.path.join(save_dir, "model_summary.csv"))

        logger.info("Evaluation report generated:")
        logger.info(f"\n{summary.to_string()}")

    return results_df


def evaluate_coverage_calibration(test_data: pd.DataFrame, predictions: Dict,
                                  config: dict = None, confidence_levels: List[float] = None,
                                  output_dir: str = None) -> pd.DataFrame:
    if confidence_levels is None:
        confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
    records = []

    for model_key, pred_info in predictions.items():
        model_name = pred_info.get("model_name", "unknown")
        target = pred_info.get("target", model_key)
        pred_values = pred_info.get("predictions")
        if pred_values is None:
            continue

        actual = test_data[target].values[:len(pred_values)] if target in test_data.columns else None
        if actual is None:
            continue

        horizon = "unknown"
        if config:
            for h in config.get("forecasting", {}).get("horizons", []):
                if f"_target_{h['name']}" in target:
                    horizon = h["name"]
                    break

        errors = np.abs(actual - pred_values)
        n = len(errors)
        if n == 0:
            continue

        for conf in confidence_levels:
            alpha = 1 - conf
            q = np.nanquantile(errors, conf)
            lower = pred_values - q
            upper = pred_values + q
            inside = (actual >= lower) & (actual <= upper)
            emp_coverage = np.mean(inside)
            interval_width = np.mean(upper - lower)
            calibration_error = abs(emp_coverage - conf)

            records.append({
                "target": target,
                "model": model_name,
                "horizon": horizon,
                "nominal_confidence": conf,
                "empirical_coverage": round(emp_coverage, 4),
                "mean_interval_width": round(interval_width, 2),
                "calibration_error": round(calibration_error, 4),
                "n_samples": n,
            })

    df = pd.DataFrame(records)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "coverage_calibration.csv"), index=False)
    return df


def plot_conformal_forecast_timeseries(power_df: pd.DataFrame, save_path: str = None):
    df = power_df.copy()
    if df.empty:
        return
    df["ts"] = pd.to_datetime(df["timestamp_target"])
    sample_turbine = df["turbine_id"].iloc[0] if "turbine_id" in df.columns else "TB01"
    sub = df[df["turbine_id"] == sample_turbine].sort_values("ts")
    if sub.empty:
        return
    sub = sub.head(500)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(sub["ts"], sub["y_low"], sub["y_high"], alpha=0.25, color="#3498db", label="80% conformal interval")
    ax.plot(sub["ts"], sub["y_pred"], color="#2c3e50", linewidth=1.2, label="Prediction")
    ax.set_title(f"Conformal Power Forecast — {sample_turbine}", fontsize=13)
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Time")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_coverage_calibration_curve(coverage_df: pd.DataFrame, save_path: str = None):
    df = coverage_df.copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    for mdl in df["model"].unique():
        sub = df[df["model"] == mdl]
        agg = sub.groupby("nominal_confidence")["empirical_coverage"].mean().reset_index()
        ax.plot(agg["nominal_confidence"], agg["empirical_coverage"], marker="o", linewidth=2,
                label=_model_display_name(mdl), markersize=6)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Nominal Confidence", fontsize=11)
    ax.set_ylabel("Empirical Coverage", fontsize=11)
    ax.set_title("Conformal Prediction — Calibration Curve", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0.4, 1.01)
    ax.set_ylim(0.2, 1.01)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_alert_accuracy_dashboard(acc_df: pd.DataFrame, save_path: str = None):
    df = acc_df.copy()
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_list = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 Score"), ("false_alarm_rate", "False Alarm Rate")]
    for ax, (col, label) in zip(axes.flat, metrics_list):
        pivot = df.pivot_table(index="turbine_id", columns="horizon", values=col, aggfunc="mean")
        pivot = pivot.reindex(index=sorted(pivot.index))
        pivot.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="white", linewidth=0.5)
        ax.set_title(label, fontsize=12)
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.legend(title="Horizon", fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_ylim(0, 1.05)
    plt.suptitle("Alert Accuracy by Turbine & Horizon", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_failure_risk_heatmap(failure_df: pd.DataFrame, save_path: str = None):
    df = failure_df.copy()
    if df.empty:
        return
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["ts"].dt.date
    pivot = df.pivot_table(index="turbine_id", columns="date", values="stop_risk_score", aggfunc="mean")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3, cbar_kws={"label": "Stop Risk Score"},
                vmin=0, vmax=1.0)
    ax.set_title("Failure Risk (Stop Risk Score) by Turbine — Daily Average", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Turbine")
    fig.autofmt_xdate()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_data_quality_bars(dq_df: pd.DataFrame, save_path: str = None):
    df = dq_df.copy()
    if df.empty or "missing_rate_pct" not in df.columns:
        return
    df["missing_pct"] = pd.to_numeric(df["missing_rate_pct"].str.rstrip("%"), errors="coerce").fillna(0)
    df = df.sort_values("missing_pct", ascending=False).head(30)

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.3)))
    colors = ["#e74c3c" if v > 5 else "#f39c12" if v > 1 else "#2ecc71" for v in df["missing_pct"]]
    ax.barh(range(len(df)), df["missing_pct"].values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["column"].values, fontsize=8)
    ax.set_xlabel("Missing Rate (%)")
    ax.set_title("Data Quality — Missing Rate by Column", fontsize=13)
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ramp_alert_timeline(ramp_df: pd.DataFrame, save_path: str = None):
    df = ramp_df.copy()
    if df.empty:
        return
    df["ts"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("ts").head(500)
    ramp_types = df["ramp_type"].unique()
    palette = {t: c for t, c in zip(ramp_types, sns.color_palette("Set2", len(ramp_types)))}

    fig, ax = plt.subplots(figsize=(16, 5))
    for rtype in ramp_types:
        sub = df[df["ramp_type"] == rtype]
        ax.scatter(sub["ts"], sub["expected_change"], label=rtype, alpha=0.7, s=sub["probability"] * 200,
                   color=[palette[rtype]], edgecolors="black", linewidth=0.3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Expected Ramp Change (kW)")
    ax.set_title("Ramp Alerts — Size & Probability", fontsize=13)
    ax.legend(fontsize=9, title="Ramp Type")
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_farm_forecast_summary(farm_df: pd.DataFrame, save_path: str = None):
    df = farm_df.copy()
    if df.empty:
        return
    df["ts"] = pd.to_datetime(df["timestamp_target"])
    df = df.sort_values("ts").head(500)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(df["ts"], df["farm_power_low"], df["farm_power_high"], alpha=0.25, color="#e67e22", label="80% conformal interval")
    ax.plot(df["ts"], df["farm_power_pred"], color="#d35400", linewidth=1.5, label="Farm power forecast")
    ax.plot(df["ts"], df["farm_energy_pred"], color="#2ecc71", linewidth=1, alpha=0.7, label="Farm energy forecast")
    ax.set_title("Farm-Level Power & Energy Forecast", fontsize=13)
    ax.set_ylabel("Power (kW) / Energy (kWh)")
    ax.set_xlabel("Time")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_metrics_by_turbine(metrics_df: pd.DataFrame, save_path: str = None):
    df = metrics_df.copy()
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric, label in zip(axes, ["RMSE", "nRMSE", "R2"], ["RMSE (kW)", "nRMSE (%)", "R\u00B2"]):
        pivot = df.pivot_table(index="turbine_id", columns="model", values=metric, aggfunc="mean")
        pivot = pivot.reindex(index=sorted(pivot.index))
        if pivot.empty:
            continue
        pivot.plot(kind="bar", ax=ax, edgecolor="white", linewidth=0.5)
        ax.set_title(label, fontsize=12)
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")
    plt.suptitle("Model Performance by Turbine", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_farm_metrics_overview(farm_metrics_df: pd.DataFrame, save_path: str = None):
    df = farm_metrics_df.copy()
    if df.empty:
        return

    df["label"] = df["target"] + " (" + df["model"] + ")"
    df = df.sort_values("rmse", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, label) in zip(axes, [("rmse", "RMSE (kW)"), ("mae", "MAE (kW)"), ("r2", "R\u00B2")]):
        colors = ["#3498db" if "lightgbm" in t else "#e74c3c" for t in df["label"]]
        bars = ax.barh(range(len(df)), df[metric].values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["label"].values, fontsize=7)
        ax.set_xlabel(label)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.2, axis="x")
    fig.legend([plt.Rectangle((0, 0), 1, 1, fc="#3498db"), plt.Rectangle((0, 0), 1, 1, fc="#e74c3c")],
               ["LightGBM", "XGBoost"], loc="lower right", fontsize=9)
    plt.suptitle("Farm-Level Metrics by Horizon & Model", fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_forecast_quality_distribution(power_df: pd.DataFrame, save_path: str = None):
    df = power_df.copy()
    if df.empty or "forecast_quality" not in df.columns:
        return
    counts = df["forecast_quality"].value_counts()
    colors = {"ok": "#2ecc71", "low_confidence": "#f39c12", "stale": "#e74c3c"}
    bar_colors = [colors.get(k, "#95a5a6") for k in counts.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=bar_colors, startangle=90, wedgeprops={"edgecolor": "white"})
    ax1.set_title("Forecast Quality Distribution", fontsize=12)
    ax2.bar(range(len(counts)), counts.values, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(range(len(counts)))
    ax2.set_xticklabels(counts.index)
    ax2.set_ylabel("Forecast Count")
    ax2.set_title("Forecast Quality Counts", fontsize=12)
    ax2.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
