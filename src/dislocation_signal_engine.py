from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config_institutional import (
    REGRESSION_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    MODEL_READY_MAX_BUSINESS_GAP,
    ensure_directories,
)

GROUP_COL = "cusip_id"
DATE_COL = "date"
TARGET_COL = "final_vwap_return"
ISSUER_COL = "trace_company_symbol"
MATURITY_BUCKET_COL = "peer_maturity_bucket"
SPLIT_COL = "sample_split"

PEER_VARIANT = "raw"
M4_MODEL = "M4_rates_equity_vix_peer_raw"
M5_MODEL = "M5_rates_equity_vix_peer_raw_microstructure_clean"
M3_MODEL = "M3_rates_equity_vix"
M1_MODEL = "M1_rates"

Z_CANDIDATE_THRESHOLD = 2.0
Z_SEVERE_THRESHOLD = 3.0
MIN_CUSIP_HISTORY_OBS = 20
CUSIP_ROLLING_WINDOW_OBS = 60
MIN_GROUP_HISTORY_OBS = 100
MIN_GLOBAL_HISTORY_OBS = 500
FUTURE_HORIZONS = [1, 3, 5]
SKIP_OBS_FOR_FUTURE_RETURNS = 1
MIN_ROBUST_GROUP_HISTORY_OBS = 100
MAD_TO_SIGMA = 1.4826
ROBUST_SIGMA_FLOOR_MULTIPLE = 0.05
M5_ABS_RESIDUAL_REDUCTION_THRESHOLD = 0.50
MAX_CONVERGENCE_HORIZON_OBS = 10
CONVERGENCE_Z_THRESHOLD = 1.0
HALF_RESIDUAL_FRACTION = 0.50
TRANSACTION_COST_SCENARIOS_BPS = [0, 5, 10]
PEER_HEDGE_FACTOR_COL = "peer_raw_bank_sector_maturity"
FIXED_HORIZON_MAIN_HORIZON_OBS = 5
FIXED_HORIZON_MAIN_COST_BPS = 10
FIXED_HORIZON_MAIN_PNL_COL = (
    f"unhedged_signal_pnl_{FIXED_HORIZON_MAIN_HORIZON_OBS}obs"
    f"_net_{FIXED_HORIZON_MAIN_COST_BPS}bp"
)
EVENT_STRATEGY_SPLITS = ["validation", "test"]
EVENT_STRATEGY_ENTRY_THRESHOLDS = [2.0, 3.0]
EVENT_STRATEGY_EXIT_Z = 1.0
EVENT_STRATEGY_MAX_HOLDING_OBS = 30
EVENT_STRATEGY_COSTS_BPS = [0, 5, 10]
EVENT_STRATEGY_HIGH_CONFIDENCE_CLASS = "high_confidence_dislocation_candidate"
INTERVAL_CALIBRATION_SPLIT = "validation"
INTERVAL_EVALUATION_SPLITS = ["validation", "test"]
PREDICTION_INTERVAL_LEVELS = [0.90, 0.95]
PNL_BOOTSTRAP_N = 2000
PNL_BOOTSTRAP_SEED = 42
PNL_BOOTSTRAP_BLOCK_COL = "_event_month"
LOW_TRADE_COUNT_THRESHOLD = 1
HIGH_GAP_WARNING_THRESHOLD = 3
QUALITY_TRAIN_QUANTILE = 0.95


PREDICTIONS_PATH = REGRESSION_DIR / "peer_baseline_gap5_model_predictions.parquet"
PANEL_WITH_PEERS_PATH = REGRESSION_DIR / "regression_panel_gap5_with_peer_factors.parquet"
OUTPUT_SIGNALS_PARQUET = REGRESSION_DIR / "dislocation_signals_gap5_m4_m5.parquet"
OUTPUT_SIGNALS_CSV = TABLES_DIR / "dislocation_signals_gap5_m4_m5.csv"
OUTPUT_TOP_EVENTS = TABLES_DIR / "top_dislocation_events_gap5_m4.csv"
OUTPUT_TOP_EVENTS_HIGH_CONF = TABLES_DIR / "top_dislocation_events_gap5_m4_high_confidence.csv"
OUTPUT_TOP_EVENTS_ONE_PER_CUSIP = TABLES_DIR / "top_dislocation_events_gap5_m4_one_per_cusip.csv"
OUTPUT_TOP_EVENTS_HIGH_CONF_CLEAN = TABLES_DIR / "top_dislocation_events_gap5_m4_high_confidence_clean.csv"
OUTPUT_EVENT_SUMMARY_ISSUER = TABLES_DIR / "dislocation_summary_by_issuer.csv"
OUTPUT_EVENT_SUMMARY_BUCKET = TABLES_DIR / "dislocation_summary_by_maturity_bucket.csv"
OUTPUT_EVENT_SUMMARY_YEAR = TABLES_DIR / "dislocation_summary_by_year.csv"
OUTPUT_EVENT_SUMMARY_MONTH = TABLES_DIR / "dislocation_summary_by_month.csv"
OUTPUT_EVENT_RATE_MONTH = TABLES_DIR / "dislocation_event_rate_by_month.csv"
OUTPUT_Z_SUMMARY = TABLES_DIR / "dislocation_zscore_summary.csv"
OUTPUT_MEAN_REVERSION = TABLES_DIR / "dislocation_mean_reversion_by_z_bucket.csv"
OUTPUT_MEAN_REVERSION_SKIP = TABLES_DIR / "dislocation_mean_reversion_skip1_by_z_bucket.csv"
OUTPUT_SIGNAL_CLASS_SUMMARY = TABLES_DIR / "dislocation_signal_quality_class_summary.csv"
OUTPUT_TOP_EVENTS_HIGH_CONF_ONE_PER_CUSIP = (
    TABLES_DIR / "top_dislocation_events_gap5_m4_high_confidence_one_per_cusip.csv"
)
OUTPUT_TOP_EVENTS_HIGH_CONF_CLEAN_ONE_PER_CUSIP = (
    TABLES_DIR / "top_dislocation_events_gap5_m4_high_confidence_clean_one_per_cusip.csv"
)
OUTPUT_MEAN_REVERSION_BY_SIDE_QUALITY = (
    TABLES_DIR / "dislocation_mean_reversion_by_side_and_quality.csv"
)
OUTPUT_CONVERGENCE_EVENTS = TABLES_DIR / "dislocation_time_to_convergence_events.csv"
OUTPUT_CONVERGENCE_SUMMARY = TABLES_DIR / "dislocation_convergence_summary.csv"
OUTPUT_STRATEGY_PNL_EVENTS = TABLES_DIR / "dislocation_strategy_pnl_events.csv"
OUTPUT_STRATEGY_PNL_SUMMARY = TABLES_DIR / "dislocation_strategy_pnl_summary.csv"
OUTPUT_EVENT_DRIVEN_STRATEGY_TRADES = TABLES_DIR / "dislocation_event_driven_strategy_trades.csv"
OUTPUT_EVENT_DRIVEN_STRATEGY_SUMMARY = TABLES_DIR / "dislocation_event_driven_strategy_summary.csv"
OUTPUT_EVENT_DRIVEN_STRATEGY_CUMULATIVE = TABLES_DIR / "dislocation_event_driven_strategy_cumulative.csv"
OUTPUT_FAIR_VALUE_INTERVALS = TABLES_DIR / "fair_value_prediction_intervals_m4.csv"
OUTPUT_FAIR_VALUE_INTERVAL_COVERAGE = TABLES_DIR / "fair_value_prediction_interval_coverage_m4.csv"
OUTPUT_STRATEGY_PNL_UNCERTAINTY = TABLES_DIR / "dislocation_strategy_pnl_uncertainty.csv"
OUTPUT_MANIFEST = TABLES_DIR / "dislocation_signal_manifest.json"

FIG_Z_DIST = FIGURES_DIR / "dislocation_m4_zscore_distribution.png"
FIG_EVENT_MONTH = FIGURES_DIR / "dislocation_event_count_by_month.png"
FIG_EVENT_RATE_MONTH = FIGURES_DIR / "dislocation_event_rate_by_month.png"
FIG_TOP_EVENTS = FIGURES_DIR / "dislocation_top_events_over_time.png"
FIG_MEAN_REVERSION = FIGURES_DIR / "dislocation_mean_reversion_5obs_by_bucket.png"
FIG_MEAN_REVERSION_SKIP = FIGURES_DIR / "dislocation_mean_reversion_skip1_5obs_by_bucket.png"
FIG_M4_M5_REDUCTION = FIGURES_DIR / "dislocation_m4_vs_m5_abs_residual.png"
FIG_ISSUER_EVENTS = FIGURES_DIR / "dislocation_candidate_rate_by_issuer.png"
FIG_CONVERGENCE_SUMMARY = FIGURES_DIR / "dislocation_convergence_within_5obs_by_side_quality.png"
FIG_STRATEGY_PNL_5OBS = FIGURES_DIR / "dislocation_strategy_pnl_5obs_by_side_quality.png"
FIG_EVENT_DRIVEN_STRATEGY_CUMULATIVE = FIGURES_DIR / "dislocation_event_driven_strategy_cumulative_test_bond_net10bp.png"


def _assert_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def _safe_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _read_required_parquet(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            "Run peer_factor_models.py first."
        )
    return pd.read_parquet(path)


def _available(cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns]


def load_prediction_block(predictions_path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    preds = _read_required_parquet(predictions_path, "peer model predictions")
    preds = _safe_to_datetime(preds, DATE_COL)

    required = [
        "model",
        "peer_variant",
        SPLIT_COL,
        GROUP_COL,
        DATE_COL,
        TARGET_COL,
        "fitted_return",
        "residual_return",
    ]
    _assert_columns(preds, required, "peer model predictions")

    if PEER_VARIANT not in set(preds["peer_variant"].dropna().unique()):
        raise ValueError(
            f"Peer variant '{PEER_VARIANT}' not found in predictions. "
            f"Available variants: {sorted(preds['peer_variant'].dropna().unique())}"
        )

    available_models = set(preds["model"].dropna().unique())
    for required_model in [M4_MODEL, M5_MODEL]:
        if required_model not in available_models:
            raise ValueError(
                f"Required model '{required_model}' not found in predictions. "
                f"Available models: {sorted(available_models)}"
            )

    return preds


def build_m4_m5_signal_base(preds: pd.DataFrame) -> pd.DataFrame:
    key_cols = [GROUP_COL, DATE_COL]
    if "_sample_index" in preds.columns:
        key_cols = ["_sample_index", GROUP_COL, DATE_COL]

    m4 = preds.loc[
        preds["peer_variant"].eq(PEER_VARIANT)
        & preds["model"].eq(M4_MODEL)
    ].copy()

    m5 = preds.loc[
        preds["peer_variant"].eq(PEER_VARIANT)
        & preds["model"].eq(M5_MODEL)
    ].copy()

    if m4.empty:
        raise ValueError(f"No rows found for {M4_MODEL} / {PEER_VARIANT}.")

    if m5.empty:
        raise ValueError(f"No rows found for {M5_MODEL} / {PEER_VARIANT}.")

    rename_m4 = {
        "fitted_return": "m4_fitted_return",
        "residual_return": "m4_residual_return",
    }
    m4 = m4.rename(columns=rename_m4)

    keep_m5 = key_cols + ["fitted_return", "residual_return"]
    m5 = m5[keep_m5].rename(
        columns={
            "fitted_return": "m5_fitted_return",
            "residual_return": "m5_residual_return",
        }
    )

    signals = m4.merge(m5, on=key_cols, how="left", validate="one_to_one")

    signals["m4_abs_residual"] = signals["m4_residual_return"].abs()
    signals["m5_abs_residual"] = signals["m5_residual_return"].abs()
    signals["m4_minus_m5_abs_residual"] = (
        signals["m4_abs_residual"] - signals["m5_abs_residual"]
    )
    signals["m5_abs_residual_reduction_share"] = np.where(
        signals["m4_abs_residual"].gt(0),
        1.0 - signals["m5_abs_residual"] / signals["m4_abs_residual"],
        np.nan,
    )

    signals = signals.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)
    return signals


def _cusip_rolling_zscore(
    df: pd.DataFrame,
    value_col: str,
    out_prefix: str,
    window: int = CUSIP_ROLLING_WINDOW_OBS,
    min_obs: int = MIN_CUSIP_HISTORY_OBS,
) -> pd.DataFrame:
    out = df.copy().sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    def past_rolling_mean(x: pd.Series) -> pd.Series:
        return x.shift(1).rolling(window=window, min_periods=min_obs).mean()

    def past_rolling_std(x: pd.Series) -> pd.Series:
        return x.shift(1).rolling(window=window, min_periods=min_obs).std(ddof=1)

    out[f"{out_prefix}_cusip_rolling_mean"] = (
        out.groupby(GROUP_COL, group_keys=False)[value_col].apply(past_rolling_mean)
    )
    out[f"{out_prefix}_cusip_rolling_std"] = (
        out.groupby(GROUP_COL, group_keys=False)[value_col].apply(past_rolling_std)
    )

    std_col = f"{out_prefix}_cusip_rolling_std"
    mean_col = f"{out_prefix}_cusip_rolling_mean"
    z_col = f"{out_prefix}_z_cusip_rolling"

    out[z_col] = np.where(
        out[std_col].gt(0),
        (out[value_col] - out[mean_col]) / out[std_col],
        np.nan,
    )

    return out


def _past_expanding_stats_by_group_date(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    out_prefix: str,
    min_obs: int,
) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")

    if group_cols:
        for col in group_cols:
            if col not in out.columns:
                out[col] = "missing"
            out[col] = out[col].astype("object").where(out[col].notna(), "missing")
        agg_cols = group_cols + [DATE_COL]
    else:
        out["_global_group"] = "all"
        group_cols = ["_global_group"]
        agg_cols = group_cols + [DATE_COL]

    temp = out[agg_cols + [value_col]].dropna(subset=[DATE_COL, value_col]).copy()
    temp["_sum"] = temp[value_col]
    temp["_sumsq"] = temp[value_col] ** 2
    temp["_count"] = 1

    daily = (
        temp.groupby(agg_cols, dropna=False)
        .agg(_daily_count=("_count", "sum"), _daily_sum=("_sum", "sum"), _daily_sumsq=("_sumsq", "sum"))
        .reset_index()
        .sort_values(group_cols + [DATE_COL])
    )

    for stat in ["_daily_count", "_daily_sum", "_daily_sumsq"]:
        daily[f"_past{stat[6:]}"] = (
            daily.groupby(group_cols, dropna=False)[stat]
            .cumsum()
            .groupby([daily[c] for c in group_cols], dropna=False)
            .shift(1)
        )

    count = daily["_past_count"]
    mean = daily["_past_sum"] / count
    var = (daily["_past_sumsq"] - (daily["_past_sum"] ** 2) / count) / (count - 1)
    var = var.where(var > 0)

    daily[f"{out_prefix}_past_count"] = count
    daily[f"{out_prefix}_past_mean"] = mean
    daily[f"{out_prefix}_past_std"] = np.sqrt(var)

    stats_cols = agg_cols + [
        f"{out_prefix}_past_count",
        f"{out_prefix}_past_mean",
        f"{out_prefix}_past_std",
    ]

    out = out.merge(daily[stats_cols], on=agg_cols, how="left")
    out[f"{out_prefix}_z"] = np.where(
        out[f"{out_prefix}_past_count"].ge(min_obs)
        & out[f"{out_prefix}_past_std"].gt(0),
        (out[value_col] - out[f"{out_prefix}_past_mean"]) / out[f"{out_prefix}_past_std"],
        np.nan,
    )

    if "_global_group" in out.columns:
        out = out.drop(columns=["_global_group"])

    return out


def add_past_only_zscores(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()

    out = _cusip_rolling_zscore(
        out,
        value_col="m4_residual_return",
        out_prefix="m4",
        window=CUSIP_ROLLING_WINDOW_OBS,
        min_obs=MIN_CUSIP_HISTORY_OBS,
    )
    out = _cusip_rolling_zscore(
        out,
        value_col="m5_residual_return",
        out_prefix="m5",
        window=CUSIP_ROLLING_WINDOW_OBS,
        min_obs=MIN_CUSIP_HISTORY_OBS,
    )

    group_cols = [c for c in [ISSUER_COL, MATURITY_BUCKET_COL] if c in out.columns]
    out = _past_expanding_stats_by_group_date(
        out,
        value_col="m4_residual_return",
        group_cols=group_cols,
        out_prefix="m4_issuer_maturity",
        min_obs=MIN_GROUP_HISTORY_OBS,
    )
    out = _past_expanding_stats_by_group_date(
        out,
        value_col="m5_residual_return",
        group_cols=group_cols,
        out_prefix="m5_issuer_maturity",
        min_obs=MIN_GROUP_HISTORY_OBS,
    )

    out = _past_expanding_stats_by_group_date(
        out,
        value_col="m4_residual_return",
        group_cols=[],
        out_prefix="m4_global",
        min_obs=MIN_GLOBAL_HISTORY_OBS,
    )
    out = _past_expanding_stats_by_group_date(
        out,
        value_col="m5_residual_return",
        group_cols=[],
        out_prefix="m5_global",
        min_obs=MIN_GLOBAL_HISTORY_OBS,
    )

    out["m4_z_main"] = out["m4_issuer_maturity_z"]
    out["m4_z_main_source"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[out["m4_z_main"].notna(), "m4_z_main_source"] = "issuer_maturity_past"

    fallback_cusip = out["m4_z_main"].isna() & out["m4_z_cusip_rolling"].notna()
    out.loc[fallback_cusip, "m4_z_main"] = out.loc[fallback_cusip, "m4_z_cusip_rolling"]
    out.loc[fallback_cusip, "m4_z_main_source"] = "cusip_rolling"

    fallback_global = out["m4_z_main"].isna() & out["m4_global_z"].notna()
    out.loc[fallback_global, "m4_z_main"] = out.loc[fallback_global, "m4_global_z"]
    out.loc[fallback_global, "m4_z_main_source"] = "global_past"

    out["m5_z_main"] = out["m5_issuer_maturity_z"]
    fallback_cusip_m5 = out["m5_z_main"].isna() & out["m5_z_cusip_rolling"].notna()
    out.loc[fallback_cusip_m5, "m5_z_main"] = out.loc[fallback_cusip_m5, "m5_z_cusip_rolling"]
    fallback_global_m5 = out["m5_z_main"].isna() & out["m5_global_z"].notna()
    out.loc[fallback_global_m5, "m5_z_main"] = out.loc[fallback_global_m5, "m5_global_z"]

    out["m4_abs_z_main"] = out["m4_z_main"].abs()
    out["m5_abs_z_main"] = out["m5_z_main"].abs()

    out["m4_sigma_main"] = np.nan
    out.loc[out["m4_z_main_source"].eq("issuer_maturity_past"), "m4_sigma_main"] = out.loc[
        out["m4_z_main_source"].eq("issuer_maturity_past"),
        "m4_issuer_maturity_past_std",
    ]

    out.loc[out["m4_z_main_source"].eq("cusip_rolling"), "m4_sigma_main"] = out.loc[
        out["m4_z_main_source"].eq("cusip_rolling"),
        "m4_cusip_rolling_std",
    ]

    out.loc[out["m4_z_main_source"].eq("global_past"), "m4_sigma_main"] = out.loc[
        out["m4_z_main_source"].eq("global_past"),
        "m4_global_past_std",
    ]

    out["m4_abs_scaled_residual_main"] = np.where(
        out["m4_sigma_main"].gt(0),
        out["m4_residual_return"].abs() / out["m4_sigma_main"],
        np.nan,
    )

    return out

def _empirical_validation_quantile(scores: pd.Series, level: float) -> float:
    x = pd.to_numeric(scores, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan

    x = np.sort(x)
    rank = int(np.ceil((len(x) + 1) * level)) - 1
    rank = min(max(rank, 0), len(x) - 1)
    return float(x[rank])


def add_fair_value_prediction_intervals(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()

    required = [
        SPLIT_COL,
        TARGET_COL,
        "m4_fitted_return",
        "m4_residual_return",
        "m4_sigma_main",
        "m4_abs_scaled_residual_main",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns for fair-value intervals: {missing}")

    calibration = out.loc[
        out[SPLIT_COL].eq(INTERVAL_CALIBRATION_SPLIT)
        & out["m4_abs_scaled_residual_main"].notna()
        & out["m4_sigma_main"].gt(0)
    ].copy()

    for level in PREDICTION_INTERVAL_LEVELS:
        pct = int(round(level * 100))
        q = _empirical_validation_quantile(
            calibration["m4_abs_scaled_residual_main"],
            level=level,
        )

        out[f"m4_interval_q_{pct}"] = q
        out[f"m4_fv_lower_{pct}"] = out["m4_fitted_return"] - q * out["m4_sigma_main"]
        out[f"m4_fv_upper_{pct}"] = out["m4_fitted_return"] + q * out["m4_sigma_main"]

        inside_col = f"m4_inside_fv_interval_{pct}"

        inside = (
                out[TARGET_COL].ge(out[f"m4_fv_lower_{pct}"])
                & out[TARGET_COL].le(out[f"m4_fv_upper_{pct}"])
        ).astype(float)

        invalid_interval = (
                out[f"m4_fv_lower_{pct}"].isna()
                | out[f"m4_fv_upper_{pct}"].isna()
        )

        inside.loc[invalid_interval] = np.nan
        out[inside_col] = inside

        out[f"m4_fv_interval_width_{pct}"] = (
            out[f"m4_fv_upper_{pct}"] - out[f"m4_fv_lower_{pct}"]
        )

    return out


def build_fair_value_interval_coverage(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for split in INTERVAL_EVALUATION_SPLITS:
        g = signals.loc[signals[SPLIT_COL].eq(split)].copy()
        if g.empty:
            continue

        for level in PREDICTION_INTERVAL_LEVELS:
            pct = int(round(level * 100))
            inside_col = f"m4_inside_fv_interval_{pct}"
            width_col = f"m4_fv_interval_width_{pct}"

            valid = g.loc[g[inside_col].notna()].copy()
            rows.append(
                {
                    "sample_split": split,
                    "interval_level": level,
                    "interval_pct": pct,
                    "n_obs": int(len(valid)),
                    "empirical_coverage": float(valid[inside_col].mean()) if len(valid) else np.nan,
                    "target_coverage": level,
                    "mean_interval_width": float(valid[width_col].mean()) if len(valid) else np.nan,
                    "median_interval_width": float(valid[width_col].median()) if len(valid) else np.nan,
                    "mean_abs_residual": float(valid["m4_abs_residual"].mean()) if "m4_abs_residual" in valid.columns and len(valid) else np.nan,
                }
            )

    return pd.DataFrame(rows)

def _robust_past_group_zscore(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    out_prefix: str,
    min_obs: int = MIN_ROBUST_GROUP_HISTORY_OBS,
) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")

    cols = [c for c in group_cols if c in out.columns]
    if not cols:
        out[f"{out_prefix}_robust_past_count"] = np.nan
        out[f"{out_prefix}_robust_past_median"] = np.nan
        out[f"{out_prefix}_robust_past_mad"] = np.nan
        out[f"{out_prefix}_robust_past_sigma"] = np.nan
        out[f"{out_prefix}_robust_z"] = np.nan
        return out

    for col in cols:
        out[col] = out[col].astype("object").where(out[col].notna(), "missing")

    train_mask = out[SPLIT_COL].eq("train") if SPLIT_COL in out.columns else pd.Series(True, index=out.index)
    train_std = pd.to_numeric(out.loc[train_mask, value_col], errors="coerce").std(ddof=1)
    sigma_floor = max(float(train_std) * ROBUST_SIGMA_FLOOR_MULTIPLE, 1e-8) if pd.notna(train_std) else 1e-8

    stat_cols = [
        f"{out_prefix}_robust_past_count",
        f"{out_prefix}_robust_past_median",
        f"{out_prefix}_robust_past_mad",
        f"{out_prefix}_robust_past_sigma",
    ]
    for col in stat_cols:
        out[col] = np.nan

    out = out.sort_values(cols + [DATE_COL, GROUP_COL]).copy()

    for _, g in out.groupby(cols, dropna=False, sort=False):
        past_values: list[float] = []
        for date_value, gd in g.groupby(DATE_COL, sort=True):
            idx = gd.index
            if len(past_values) >= min_obs:
                hist = np.asarray(past_values, dtype=float)
                hist = hist[np.isfinite(hist)]
                if len(hist) >= min_obs:
                    med = float(np.nanmedian(hist))
                    mad = float(np.nanmedian(np.abs(hist - med)))
                    sigma = max(MAD_TO_SIGMA * mad, sigma_floor)
                    out.loc[idx, f"{out_prefix}_robust_past_count"] = len(hist)
                    out.loc[idx, f"{out_prefix}_robust_past_median"] = med
                    out.loc[idx, f"{out_prefix}_robust_past_mad"] = mad
                    out.loc[idx, f"{out_prefix}_robust_past_sigma"] = sigma

            current = pd.to_numeric(gd[value_col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(current):
                past_values.extend(current[np.isfinite(current)].tolist())

    sigma_col = f"{out_prefix}_robust_past_sigma"
    med_col = f"{out_prefix}_robust_past_median"
    out[f"{out_prefix}_robust_z"] = np.where(
        out[sigma_col].gt(0),
        (out[value_col] - out[med_col]) / out[sigma_col],
        np.nan,
    )

    return out.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)


def add_robust_zscore_diagnostics(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    group_cols = [c for c in [ISSUER_COL, MATURITY_BUCKET_COL] if c in out.columns]
    out = _robust_past_group_zscore(
        out,
        value_col="m4_residual_return",
        group_cols=group_cols,
        out_prefix="m4_issuer_maturity",
        min_obs=MIN_ROBUST_GROUP_HISTORY_OBS,
    )
    out["m4_abs_robust_z"] = out["m4_issuer_maturity_robust_z"].abs()
    out["m4_extreme_standard_but_not_robust"] = (
        out["m4_abs_z_main"].ge(Z_SEVERE_THRESHOLD)
        & out["m4_abs_robust_z"].notna()
        & out["m4_abs_robust_z"].lt(Z_CANDIDATE_THRESHOLD)
    )
    return out

def add_price_quality_flags(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()

    train_mask = out[SPLIT_COL].eq("train") if SPLIT_COL in out.columns else pd.Series(False, index=out.index)

    for col in ["price_dispersion_rel_filled", "price_range_rel_filled"]:
        flag_col = f"high_{col}"
        if col in out.columns:
            base = pd.to_numeric(out.loc[train_mask, col], errors="coerce")
            threshold = base.quantile(QUALITY_TRAIN_QUANTILE) if base.notna().any() else np.nan
            out[f"{col}_train_p95"] = threshold
            out[flag_col] = pd.to_numeric(out[col], errors="coerce").gt(threshold) if pd.notna(threshold) else False
        else:
            out[flag_col] = False

    if "single_trade_day" in out.columns:
        out["low_trade_count_flag"] = pd.to_numeric(out["single_trade_day"], errors="coerce").eq(1.0)
    elif "log_n_trades" in out.columns:
        out["low_trade_count_flag"] = np.expm1(pd.to_numeric(out["log_n_trades"], errors="coerce")).le(LOW_TRADE_COUNT_THRESHOLD)
    else:
        out["low_trade_count_flag"] = False

    if "business_gap_days" in out.columns:
        out["large_gap_warning_flag"] = pd.to_numeric(out["business_gap_days"], errors="coerce").gt(HIGH_GAP_WARNING_THRESHOLD)
    else:
        out["large_gap_warning_flag"] = False

    out["price_quality_warning_flag"] = (
        out["low_trade_count_flag"].fillna(False)
        | out["large_gap_warning_flag"].fillna(False)
        | out.get("high_price_dispersion_rel_filled", False)
        | out.get("high_price_range_rel_filled", False)
    )

    return out


def classify_dislocation_events(signals: pd.DataFrame) -> pd.DataFrame:
    out = add_price_quality_flags(signals)

    out["m4_candidate_flag"] = out["m4_abs_z_main"].ge(Z_CANDIDATE_THRESHOLD)
    out["m4_severe_flag"] = out["m4_abs_z_main"].ge(Z_SEVERE_THRESHOLD)

    out["dislocation_side"] = "ordinary"
    out.loc[out["m4_z_main"].le(-Z_CANDIDATE_THRESHOLD), "dislocation_side"] = (
        "negative_residual_underperformed_potentially_cheap"
    )
    out.loc[out["m4_z_main"].ge(Z_CANDIDATE_THRESHOLD), "dislocation_side"] = (
        "positive_residual_outperformed_potentially_rich"
    )

    out["severity"] = "ordinary"
    out.loc[out["m4_candidate_flag"], "severity"] = "candidate_abs_z_ge_2"
    out.loc[out["m4_severe_flag"], "severity"] = "severe_abs_z_ge_3"

    out["m5_abs_z_available"] = out["m5_abs_z_main"].notna()
    out["m5_absorbs_m4_residual_flag"] = (
        out["m5_abs_z_available"]
        & out["m4_candidate_flag"]
        & (
            out["m5_abs_z_main"].lt(Z_CANDIDATE_THRESHOLD)
            | out["m5_abs_residual_reduction_share"].ge(M5_ABS_RESIDUAL_REDUCTION_THRESHOLD)
        )
    )

    out["signal_quality_class"] = "ordinary_residual"

    low_quality_candidate = out["m4_candidate_flag"] & out["price_quality_warning_flag"].fillna(False)
    out.loc[low_quality_candidate, "signal_quality_class"] = "low_price_quality_warning"

    micro_sensitive = out["m4_candidate_flag"] & out["m5_absorbs_m4_residual_flag"]
    out.loc[micro_sensitive, "signal_quality_class"] = "microstructure_sensitive"

    high_conf = (
        out["m4_candidate_flag"]
        & out["m5_abs_z_main"].ge(Z_CANDIDATE_THRESHOLD)
        & ~out["price_quality_warning_flag"].fillna(False)
    )
    out.loc[high_conf, "signal_quality_class"] = "high_confidence_dislocation_candidate"

    out.loc[micro_sensitive, "signal_quality_class"] = "microstructure_sensitive"

    return out


def _load_future_return_source(signals: pd.DataFrame) -> pd.DataFrame:

    preferred_cols = [GROUP_COL, DATE_COL, TARGET_COL, PEER_HEDGE_FACTOR_COL]

    if PANEL_WITH_PEERS_PATH.exists():
        try:
            panel = pd.read_parquet(PANEL_WITH_PEERS_PATH)
            panel = _safe_to_datetime(panel, DATE_COL)
            if TARGET_COL in panel.columns:
                cols = [c for c in preferred_cols if c in panel.columns]
                return panel[cols].drop_duplicates([GROUP_COL, DATE_COL]).copy()
        except Exception as exc:
            print(f"Warning: could not load full panel for future returns ({exc}). Falling back to signal sample.")

    cols = [c for c in preferred_cols if c in signals.columns]
    return signals[cols].drop_duplicates([GROUP_COL, DATE_COL]).copy()


def _sum_forward_by_group(df: pd.DataFrame, value_col: str, start_k: int, end_k: int) -> tuple[pd.Series, pd.Series]:
    pieces = []
    for k in range(start_k, end_k + 1):
        pieces.append(df.groupby(GROUP_COL)[value_col].shift(-k))

    if not pieces:
        empty = pd.Series(np.nan, index=df.index)
        return empty, pd.Series(0, index=df.index)

    total = np.sum(pieces, axis=0)
    valid_count = sum(x.notna().astype(int) for x in pieces)
    total = pd.Series(total, index=df.index, dtype="float64")
    total.loc[valid_count < len(pieces)] = np.nan
    return total, valid_count


def add_future_returns(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    future_source = _load_future_return_source(out)
    future_source = future_source.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    has_peer_hedge = PEER_HEDGE_FACTOR_COL in future_source.columns

    for h in FUTURE_HORIZONS:
        future_source[f"future_return_{h}obs"], _ = _sum_forward_by_group(
            future_source,
            TARGET_COL,
            start_k=1,
            end_k=h,
        )

        start_k = SKIP_OBS_FOR_FUTURE_RETURNS + 1
        end_k = SKIP_OBS_FOR_FUTURE_RETURNS + h
        future_source[f"future_return_skip1_{h}obs"], _ = _sum_forward_by_group(
            future_source,
            TARGET_COL,
            start_k=start_k,
            end_k=end_k,
        )

        if has_peer_hedge:
            future_source[f"future_peer_return_{h}obs"], _ = _sum_forward_by_group(
                future_source,
                PEER_HEDGE_FACTOR_COL,
                start_k=1,
                end_k=h,
            )
            future_source[f"future_peer_return_skip1_{h}obs"], _ = _sum_forward_by_group(
                future_source,
                PEER_HEDGE_FACTOR_COL,
                start_k=start_k,
                end_k=end_k,
            )

    future_cols = [GROUP_COL, DATE_COL]
    future_cols += [f"future_return_{h}obs" for h in FUTURE_HORIZONS]
    future_cols += [f"future_return_skip1_{h}obs" for h in FUTURE_HORIZONS]
    if has_peer_hedge:
        future_cols += [f"future_peer_return_{h}obs" for h in FUTURE_HORIZONS]
        future_cols += [f"future_peer_return_skip1_{h}obs" for h in FUTURE_HORIZONS]

    out = out.merge(future_source[future_cols], on=[GROUP_COL, DATE_COL], how="left")

    out["signal_direction"] = -np.sign(out["m4_z_main"])
    out.loc[~out["m4_candidate_flag"].fillna(False), "signal_direction"] = 0.0

    for h in FUTURE_HORIZONS:
        col = f"future_return_{h}obs"
        out[f"future_reversal_return_{h}obs"] = out["signal_direction"] * out[col]
        out[f"future_reversal_hit_{h}obs"] = out[f"future_reversal_return_{h}obs"].gt(0)
        out[f"unhedged_signal_pnl_{h}obs"] = out["signal_direction"] * out[col]

        skip_col = f"future_return_skip1_{h}obs"
        out[f"future_reversal_return_skip1_{h}obs"] = out["signal_direction"] * out[skip_col]
        out[f"future_reversal_hit_skip1_{h}obs"] = out[f"future_reversal_return_skip1_{h}obs"].gt(0)
        out[f"unhedged_signal_pnl_skip1_{h}obs"] = out["signal_direction"] * out[skip_col]

        peer_col = f"future_peer_return_{h}obs"
        if peer_col in out.columns:
            out[f"future_excess_return_vs_peer_{h}obs"] = out[col] - out[peer_col]
            out[f"peer_hedged_signal_pnl_{h}obs"] = (
                out["signal_direction"] * out[f"future_excess_return_vs_peer_{h}obs"]
            )

        peer_skip_col = f"future_peer_return_skip1_{h}obs"
        if peer_skip_col in out.columns:
            out[f"future_excess_return_vs_peer_skip1_{h}obs"] = out[skip_col] - out[peer_skip_col]
            out[f"peer_hedged_signal_pnl_skip1_{h}obs"] = (
                out["signal_direction"] * out[f"future_excess_return_vs_peer_skip1_{h}obs"]
            )

        for cost_bps in TRANSACTION_COST_SCENARIOS_BPS:
            cost_decimal = cost_bps / 10000.0
            out[f"unhedged_signal_pnl_{h}obs_net_{cost_bps}bp"] = (
                out[f"unhedged_signal_pnl_{h}obs"] - cost_decimal
            )
            out[f"unhedged_signal_pnl_skip1_{h}obs_net_{cost_bps}bp"] = (
                out[f"unhedged_signal_pnl_skip1_{h}obs"] - cost_decimal
            )

            if f"peer_hedged_signal_pnl_{h}obs" in out.columns:
                out[f"peer_hedged_signal_pnl_{h}obs_net_{cost_bps}bp"] = (
                        out[f"peer_hedged_signal_pnl_{h}obs"] - cost_decimal
                )
            if f"peer_hedged_signal_pnl_skip1_{h}obs" in out.columns:
                out[f"peer_hedged_signal_pnl_skip1_{h}obs_net_{cost_bps}bp"] = (
                        out[f"peer_hedged_signal_pnl_skip1_{h}obs"] - cost_decimal
                )

    return out


def add_convergence_and_strategy_metrics(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy().sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    for k in range(1, MAX_CONVERGENCE_HORIZON_OBS + 1):
        out[f"future_m4_z_{k}obs"] = out.groupby(GROUP_COL)["m4_z_main"].shift(-k)
        out[f"future_m4_residual_{k}obs"] = out.groupby(GROUP_COL)["m4_residual_return"].shift(-k)

    event_mask = out[SPLIT_COL].eq("test") & out["m4_candidate_flag"].fillna(False)

    out["time_to_abs_z_below_1_obs"] = np.nan
    out["time_to_half_residual_obs"] = np.nan
    out["time_to_residual_sign_flip_obs"] = np.nan

    event_idx = out.index[event_mask]
    if len(event_idx) > 0:
        current_abs_resid = out.loc[event_idx, "m4_abs_residual"].astype(float)
        current_sign = np.sign(out.loc[event_idx, "m4_residual_return"].astype(float))

        for k in range(1, MAX_CONVERGENCE_HORIZON_OBS + 1):
            future_z = out.loc[event_idx, f"future_m4_z_{k}obs"].astype(float)
            future_resid = out.loc[event_idx, f"future_m4_residual_{k}obs"].astype(float)
            future_sign = np.sign(future_resid)

            z_hit = future_z.abs().lt(CONVERGENCE_Z_THRESHOLD)
            half_hit = future_resid.abs().le(HALF_RESIDUAL_FRACTION * current_abs_resid)
            sign_hit = future_resid.notna() & current_sign.ne(0) & future_sign.ne(0) & future_sign.ne(current_sign)

            missing_z = out.loc[event_idx, "time_to_abs_z_below_1_obs"].isna()
            out.loc[event_idx[missing_z & z_hit], "time_to_abs_z_below_1_obs"] = k

            missing_half = out.loc[event_idx, "time_to_half_residual_obs"].isna()
            out.loc[event_idx[missing_half & half_hit], "time_to_half_residual_obs"] = k

            missing_sign = out.loc[event_idx, "time_to_residual_sign_flip_obs"].isna()
            out.loc[event_idx[missing_sign & sign_hit], "time_to_residual_sign_flip_obs"] = k

    for h in [1, 3, 5, 10]:
        out[f"converged_abs_z_below_1_within_{h}obs"] = out["time_to_abs_z_below_1_obs"].le(h)
        out[f"converged_half_residual_within_{h}obs"] = out["time_to_half_residual_obs"].le(h)
        out[f"converged_sign_flip_within_{h}obs"] = out["time_to_residual_sign_flip_obs"].le(h)

    out["residual_side_simple"] = np.where(
        out["m4_z_main"].lt(0),
        "negative_residual_potentially_cheap",
        np.where(out["m4_z_main"].gt(0), "positive_residual_potentially_rich", "zero_residual"),
    )

    return out


def add_z_buckets(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()

    bins = [-np.inf, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, np.inf]
    labels = [
        "z_le_-3_severe_negative",
        "-3_lt_z_le_-2_candidate_negative",
        "-2_lt_z_le_-1_mild_negative",
        "-1_lt_z_lt_1_normal",
        "1_le_z_lt_2_mild_positive",
        "2_le_z_lt_3_candidate_positive",
        "z_ge_3_severe_positive",
    ]
    out["m4_z_bucket"] = pd.cut(out["m4_z_main"], bins=bins, labels=labels, right=True)
    return out


def build_mean_reversion_summary(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test = signals.loc[signals[SPLIT_COL].eq("test")].copy()

    for bucket, g in test.groupby("m4_z_bucket", dropna=False):
        row = {
            "sample_split": "test",
            "m4_z_bucket": str(bucket),
            "n_obs": int(len(g)),
            "mean_m4_z": float(g["m4_z_main"].mean()) if len(g) else np.nan,
            "mean_m4_residual": float(g["m4_residual_return"].mean()) if len(g) else np.nan,
            "mean_abs_m4_residual": float(g["m4_abs_residual"].mean()) if len(g) else np.nan,
            "candidate_share": float(g["m4_candidate_flag"].mean()) if len(g) else np.nan,
        }

        for h in FUTURE_HORIZONS:
            future_col = f"future_return_{h}obs"
            reversal_col = f"future_reversal_return_{h}obs"
            hit_col = f"future_reversal_hit_{h}obs"

            row[f"n_future_{h}obs"] = int(g[future_col].notna().sum()) if future_col in g.columns else 0
            row[f"mean_future_return_{h}obs"] = float(g[future_col].mean()) if future_col in g.columns else np.nan
            row[f"mean_reversal_return_{h}obs"] = float(g[reversal_col].mean()) if reversal_col in g.columns else np.nan
            row[f"median_reversal_return_{h}obs"] = float(g[reversal_col].median()) if reversal_col in g.columns else np.nan
            row[f"reversal_hit_rate_{h}obs"] = float(g.loc[g[future_col].notna(), hit_col].mean()) if future_col in g.columns and g[future_col].notna().any() else np.nan

            skip_future_col = f"future_return_skip1_{h}obs"
            skip_reversal_col = f"future_reversal_return_skip1_{h}obs"
            skip_hit_col = f"future_reversal_hit_skip1_{h}obs"
            row[f"n_future_skip1_{h}obs"] = int(g[skip_future_col].notna().sum()) if skip_future_col in g.columns else 0
            row[f"mean_future_return_skip1_{h}obs"] = float(g[skip_future_col].mean()) if skip_future_col in g.columns else np.nan
            row[f"mean_reversal_return_skip1_{h}obs"] = float(g[skip_reversal_col].mean()) if skip_reversal_col in g.columns else np.nan
            row[f"median_reversal_return_skip1_{h}obs"] = float(g[skip_reversal_col].median()) if skip_reversal_col in g.columns else np.nan
            row[f"reversal_hit_rate_skip1_{h}obs"] = float(g.loc[g[skip_future_col].notna(), skip_hit_col].mean()) if skip_future_col in g.columns and g[skip_future_col].notna().any() else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def build_mean_reversion_by_side_and_quality(signals: pd.DataFrame) -> pd.DataFrame:
    test = signals.loc[
        signals[SPLIT_COL].eq("test") & signals["m4_candidate_flag"]
    ].copy()
    if test.empty:
        return pd.DataFrame()

    if "residual_side_simple" not in test.columns:
        test["residual_side_simple"] = np.where(
            test["m4_z_main"].lt(0),
            "negative_residual_potentially_cheap",
            "positive_residual_potentially_rich",
        )

    group_cols = ["residual_side_simple", "signal_quality_class", "severity"]
    rows = []

    for keys, g in test.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {
            "residual_side_simple": keys[0],
            "signal_quality_class": keys[1],
            "severity": keys[2],
            "n_events": int(len(g)),
            "n_cusips": int(g[GROUP_COL].nunique()),
            "mean_m4_z": float(g["m4_z_main"].mean()),
            "median_m4_z": float(g["m4_z_main"].median()),
            "mean_abs_m4_z": float(g["m4_abs_z_main"].mean()),
            "mean_m4_residual": float(g["m4_residual_return"].mean()),
            "mean_abs_m4_residual": float(g["m4_abs_residual"].mean()),
            "mean_m5_abs_residual_reduction_share": float(g["m5_abs_residual_reduction_share"].mean())
            if "m5_abs_residual_reduction_share" in g.columns else np.nan,
        }

        for h in FUTURE_HORIZONS:
            for prefix in ["", "skip1_"]:
                if prefix:
                    future_col = f"future_return_skip1_{h}obs"
                    reversal_col = f"future_reversal_return_skip1_{h}obs"
                    hit_col = f"future_reversal_hit_skip1_{h}obs"
                    out_prefix = f"skip1_{h}obs"
                else:
                    future_col = f"future_return_{h}obs"
                    reversal_col = f"future_reversal_return_{h}obs"
                    hit_col = f"future_reversal_hit_{h}obs"
                    out_prefix = f"{h}obs"

                if future_col in g.columns:
                    valid = g[future_col].notna()
                    row[f"n_future_{out_prefix}"] = int(valid.sum())
                    row[f"mean_future_return_{out_prefix}"] = float(g.loc[valid, future_col].mean())
                    row[f"mean_reversal_return_{out_prefix}"] = float(g.loc[valid, reversal_col].mean())
                    row[f"median_reversal_return_{out_prefix}"] = float(g.loc[valid, reversal_col].median())
                    row[f"reversal_hit_rate_{out_prefix}"] = float(g.loc[valid, hit_col].mean())
                else:
                    row[f"n_future_{out_prefix}"] = 0
                    row[f"mean_future_return_{out_prefix}"] = np.nan
                    row[f"mean_reversal_return_{out_prefix}"] = np.nan
                    row[f"median_reversal_return_{out_prefix}"] = np.nan
                    row[f"reversal_hit_rate_{out_prefix}"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(group_cols).reset_index(drop=True) if not out.empty else out


def build_convergence_event_table(signals: pd.DataFrame) -> pd.DataFrame:
    test = signals.loc[
        signals[SPLIT_COL].eq("test") & signals["m4_candidate_flag"]
    ].copy()
    if test.empty:
        return pd.DataFrame()

    preferred_cols = [
        DATE_COL,
        GROUP_COL,
        ISSUER_COL,
        MATURITY_BUCKET_COL,
        "residual_side_simple",
        "dislocation_side",
        "severity",
        "signal_quality_class",
        "m4_z_main",
        "m4_residual_return",
        "m4_abs_residual",
        "m5_abs_residual_reduction_share",
        "time_to_abs_z_below_1_obs",
        "time_to_half_residual_obs",
        "time_to_residual_sign_flip_obs",
    ]
    for h in [1, 3, 5, 10]:
        preferred_cols += [
            f"converged_abs_z_below_1_within_{h}obs",
            f"converged_half_residual_within_{h}obs",
            f"converged_sign_flip_within_{h}obs",
        ]
    cols = [c for c in preferred_cols if c in test.columns]
    return test[cols].sort_values("m4_abs_residual", ascending=False)


def build_convergence_summary(signals: pd.DataFrame) -> pd.DataFrame:
    test = signals.loc[
        signals[SPLIT_COL].eq("test") & signals["m4_candidate_flag"]
    ].copy()
    if test.empty:
        return pd.DataFrame()

    group_cols = ["residual_side_simple", "signal_quality_class", "severity"]
    rows = []
    for keys, g in test.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {
            "residual_side_simple": keys[0],
            "signal_quality_class": keys[1],
            "severity": keys[2],
            "n_events": int(len(g)),
            "n_cusips": int(g[GROUP_COL].nunique()),
            "median_time_to_abs_z_below_1_obs": float(g["time_to_abs_z_below_1_obs"].median()),
            "median_time_to_half_residual_obs": float(g["time_to_half_residual_obs"].median()),
            "median_time_to_residual_sign_flip_obs": float(g["time_to_residual_sign_flip_obs"].median()),
        }
        for h in [1, 3, 5, 10]:
            for metric in ["abs_z_below_1", "half_residual", "sign_flip"]:
                col = f"converged_{metric}_within_{h}obs"
                row[f"share_{col}"] = float(g[col].mean()) if col in g.columns else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_strategy_pnl_event_table(signals: pd.DataFrame) -> pd.DataFrame:
    test = signals.loc[
        signals[SPLIT_COL].eq("test") & signals["m4_candidate_flag"]
    ].copy()
    if test.empty:
        return pd.DataFrame()

    preferred_cols = [
        DATE_COL,
        GROUP_COL,
        ISSUER_COL,
        MATURITY_BUCKET_COL,
        "residual_side_simple",
        "dislocation_side",
        "severity",
        "signal_quality_class",
        "m4_z_main",
        "signal_direction",
    ]
    for h in FUTURE_HORIZONS:
        preferred_cols += [
            f"future_return_{h}obs",
            f"future_return_skip1_{h}obs",
            f"unhedged_signal_pnl_{h}obs",
            f"unhedged_signal_pnl_skip1_{h}obs",
            f"peer_hedged_signal_pnl_{h}obs",
            f"peer_hedged_signal_pnl_skip1_{h}obs",
        ]
        for cost_bps in TRANSACTION_COST_SCENARIOS_BPS:
            preferred_cols += [
                f"unhedged_signal_pnl_{h}obs_net_{cost_bps}bp",
                f"unhedged_signal_pnl_skip1_{h}obs_net_{cost_bps}bp",
                f"peer_hedged_signal_pnl_{h}obs_net_{cost_bps}bp",
                f"peer_hedged_signal_pnl_skip1_{h}obs_net_{cost_bps}bp",
            ]
    cols = [c for c in preferred_cols if c in test.columns]
    return test[cols].sort_values("m4_z_main")


def build_strategy_pnl_summary(signals: pd.DataFrame) -> pd.DataFrame:
    test = signals.loc[
        signals[SPLIT_COL].eq("test") & signals["m4_candidate_flag"]
    ].copy()
    if test.empty:
        return pd.DataFrame()

    group_cols = ["residual_side_simple", "signal_quality_class", "severity"]
    pnl_cols = []
    for h in FUTURE_HORIZONS:
        pnl_cols += [
            f"unhedged_signal_pnl_{h}obs",
            f"unhedged_signal_pnl_skip1_{h}obs",
            f"peer_hedged_signal_pnl_{h}obs",
            f"peer_hedged_signal_pnl_skip1_{h}obs",
        ]
        for cost_bps in TRANSACTION_COST_SCENARIOS_BPS:
            pnl_cols += [
                f"unhedged_signal_pnl_{h}obs_net_{cost_bps}bp",
                f"unhedged_signal_pnl_skip1_{h}obs_net_{cost_bps}bp",
                f"peer_hedged_signal_pnl_{h}obs_net_{cost_bps}bp",
                f"peer_hedged_signal_pnl_skip1_{h}obs_net_{cost_bps}bp",
            ]
    pnl_cols = [c for c in pnl_cols if c in test.columns]

    rows = []
    for keys, g in test.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {
            "residual_side_simple": keys[0],
            "signal_quality_class": keys[1],
            "severity": keys[2],
            "n_events": int(len(g)),
            "n_cusips": int(g[GROUP_COL].nunique()),
        }
        for col in pnl_cols:
            valid = g[col].dropna()
            row[f"n_{col}"] = int(len(valid))
            row[f"mean_{col}"] = float(valid.mean()) if len(valid) else np.nan
            row[f"median_{col}"] = float(valid.median()) if len(valid) else np.nan
            row[f"hit_rate_{col}"] = float(valid.gt(0).mean()) if len(valid) else np.nan
            row[f"t_stat_{col}"] = float(valid.mean() / (valid.std(ddof=1) / np.sqrt(len(valid)))) if len(valid) > 1 and valid.std(ddof=1) > 0 else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

def _bootstrap_mean_ci_by_month(
    g: pd.DataFrame,
    value_col: str,
    block_col: str = PNL_BOOTSTRAP_BLOCK_COL,
    n_boot: int = PNL_BOOTSTRAP_N,
    seed: int = PNL_BOOTSTRAP_SEED,
) -> dict:
    valid = g[[value_col, block_col]].dropna().copy()
    valid[value_col] = pd.to_numeric(valid[value_col], errors="coerce")
    valid = valid.dropna(subset=[value_col])

    if valid.empty:
        return {
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
            "bootstrap_method": "not_available",
        }

    rng = np.random.default_rng(seed)
    blocks = valid[block_col].dropna().unique()

    if len(blocks) >= 2:
        block_values = {
            block: valid.loc[valid[block_col].eq(block), value_col].to_numpy(dtype=float)
            for block in blocks
        }

        boot_means = []
        for _ in range(n_boot):
            sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
            sampled_values = np.concatenate([block_values[b] for b in sampled_blocks])
            boot_means.append(float(np.mean(sampled_values)))

        method = "monthly_block_bootstrap"
    else:
        values = valid[value_col].to_numpy(dtype=float)
        boot_means = []
        for _ in range(n_boot):
            sampled_values = rng.choice(values, size=len(values), replace=True)
            boot_means.append(float(np.mean(sampled_values)))

        method = "event_bootstrap_fallback"

    boot_means = np.asarray(boot_means, dtype=float)

    return {
        "ci95_lower": float(np.quantile(boot_means, 0.025)),
        "ci95_upper": float(np.quantile(boot_means, 0.975)),
        "bootstrap_method": method,
    }


def build_strategy_pnl_uncertainty(signals: pd.DataFrame) -> pd.DataFrame:
    test = signals.loc[
        signals[SPLIT_COL].eq("test") & signals["m4_candidate_flag"]
    ].copy()

    if test.empty:
        return pd.DataFrame()

    test[PNL_BOOTSTRAP_BLOCK_COL] = test[DATE_COL].dt.to_period("M").astype(str)

    group_cols = ["residual_side_simple", "signal_quality_class", "severity"]

    pnl_cols = []
    for h in FUTURE_HORIZONS:
        pnl_cols += [
            f"unhedged_signal_pnl_{h}obs",
            f"unhedged_signal_pnl_skip1_{h}obs",
            f"peer_hedged_signal_pnl_{h}obs",
            f"peer_hedged_signal_pnl_skip1_{h}obs",
        ]
        for cost_bps in TRANSACTION_COST_SCENARIOS_BPS:
            pnl_cols += [
                f"unhedged_signal_pnl_{h}obs_net_{cost_bps}bp",
                f"unhedged_signal_pnl_skip1_{h}obs_net_{cost_bps}bp",
                f"peer_hedged_signal_pnl_{h}obs_net_{cost_bps}bp",
                f"peer_hedged_signal_pnl_skip1_{h}obs_net_{cost_bps}bp",
            ]

    pnl_cols = [c for c in pnl_cols if c in test.columns]

    rows = []
    for keys, g in test.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        for col in pnl_cols:
            valid = pd.to_numeric(g[col], errors="coerce").dropna()

            if valid.empty:
                continue

            p05 = float(valid.quantile(0.05))
            left_tail = valid.loc[valid.le(p05)]

            ci = _bootstrap_mean_ci_by_month(
                g=g,
                value_col=col,
                block_col=PNL_BOOTSTRAP_BLOCK_COL,
                n_boot=PNL_BOOTSTRAP_N,
                seed=PNL_BOOTSTRAP_SEED,
            )

            rows.append(
                {
                    "residual_side_simple": keys[0],
                    "signal_quality_class": keys[1],
                    "severity": keys[2],
                    "pnl_metric": col,
                    "n_events": int(len(valid)),
                    "n_cusips": int(g[GROUP_COL].nunique()),
                    "n_months": int(g[PNL_BOOTSTRAP_BLOCK_COL].nunique()),
                    "mean_pnl": float(valid.mean()),
                    "median_pnl": float(valid.median()),
                    "hit_rate": float(valid.gt(0).mean()),
                    "loss_probability": float(valid.lt(0).mean()),
                    "p05_pnl": p05,
                    "p25_pnl": float(valid.quantile(0.25)),
                    "p75_pnl": float(valid.quantile(0.75)),
                    "p95_pnl": float(valid.quantile(0.95)),
                    "expected_shortfall_5pct": float(left_tail.mean()) if len(left_tail) else np.nan,
                    "ci95_lower_mean_pnl": ci["ci95_lower"],
                    "ci95_upper_mean_pnl": ci["ci95_upper"],
                    "bootstrap_method": ci["bootstrap_method"],
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["is_headline_fixed_horizon_metric"] = out["pnl_metric"].eq(
        FIXED_HORIZON_MAIN_PNL_COL
    )

    return out.sort_values(
        ["residual_side_simple", "signal_quality_class", "severity", "pnl_metric"]
    ).reset_index(drop=True)

def _event_strategy_entry_condition(z_value: float, side: str, threshold: float) -> bool:
    if pd.isna(z_value):
        return False
    if side == "cheap":
        return float(z_value) <= -float(threshold)
    if side == "rich":
        return float(z_value) >= float(threshold)
    raise ValueError(f"Unknown event-strategy side: {side}")


def _event_strategy_exit_condition(z_value: float, side: str) -> bool:
    if pd.isna(z_value):
        return False
    if side == "cheap":
        return float(z_value) >= -EVENT_STRATEGY_EXIT_Z
    if side == "rich":
        return float(z_value) <= EVENT_STRATEGY_EXIT_Z
    raise ValueError(f"Unknown event-strategy side: {side}")


def build_event_driven_strategy_trades(signals: pd.DataFrame) -> pd.DataFrame:
    required = [
        SPLIT_COL,
        GROUP_COL,
        DATE_COL,
        TARGET_COL,
        "m4_z_main",
        "m4_residual_return",
        "signal_quality_class",
    ]
    missing = [c for c in required if c not in signals.columns]
    if missing:
        raise ValueError(f"Missing columns for event-driven strategy: {missing}")

    df = signals.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values([SPLIT_COL, GROUP_COL, DATE_COL]).reset_index(drop=True)

    rows = []
    strategy_sides = ["cheap", "rich"]

    for split in EVENT_STRATEGY_SPLITS:
        split_df = df.loc[df[SPLIT_COL].eq(split)].copy()
        if split_df.empty:
            continue

        for threshold in EVENT_STRATEGY_ENTRY_THRESHOLDS:
            for side in strategy_sides:
                direction = 1.0 if side == "cheap" else -1.0

                for cusip, g in split_df.groupby(GROUP_COL, sort=False):
                    g = g.sort_values(DATE_COL).copy()
                    idx_list = g.index.to_list()
                    pos = 0

                    while pos < len(idx_list) - 1:
                        entry_idx = idx_list[pos]
                        entry_row = df.loc[entry_idx]
                        entry_z = entry_row["m4_z_main"]

                        is_high_conf = (
                            entry_row["signal_quality_class"]
                            == EVENT_STRATEGY_HIGH_CONFIDENCE_CLASS
                        )
                        is_entry = is_high_conf and _event_strategy_entry_condition(
                            entry_z,
                            side=side,
                            threshold=threshold,
                        )

                        if not is_entry:
                            pos += 1
                            continue

                        max_exit_pos = min(
                            pos + EVENT_STRATEGY_MAX_HOLDING_OBS,
                            len(idx_list) - 1,
                        )

                        exit_pos = None
                        exit_reason = None

                        for candidate_pos in range(pos + 1, max_exit_pos + 1):
                            candidate_idx = idx_list[candidate_pos]
                            candidate_z = df.at[candidate_idx, "m4_z_main"]
                            if _event_strategy_exit_condition(candidate_z, side=side):
                                exit_pos = candidate_pos
                                exit_reason = "residual_normalisation"
                                break

                        if exit_pos is None:
                            exit_pos = max_exit_pos
                            if exit_pos == pos:
                                pos += 1
                                continue
                            exit_reason = (
                                "time_stop"
                                if exit_pos == pos + EVENT_STRATEGY_MAX_HOLDING_OBS
                                else "end_of_split"
                            )

                        exit_idx = idx_list[exit_pos]
                        path_idx = idx_list[pos + 1: exit_pos + 1]

                        path_returns = pd.to_numeric(
                            df.loc[path_idx, TARGET_COL],
                            errors="coerce",
                        )
                        path_residuals = pd.to_numeric(
                            df.loc[path_idx, "m4_residual_return"],
                            errors="coerce",
                        )

                        gross_bond_payoff = (
                            direction * float(path_returns.sum())
                            if path_returns.notna().all() and len(path_returns) > 0
                            else np.nan
                        )
                        gross_residual_payoff = (
                            direction * float(path_residuals.sum())
                            if path_residuals.notna().all() and len(path_residuals) > 0
                            else np.nan
                        )

                        exit_z = df.at[exit_idx, "m4_z_main"]
                        overshoot_exit = bool(
                            (side == "cheap" and pd.notna(exit_z) and float(exit_z) > 0)
                            or (side == "rich" and pd.notna(exit_z) and float(exit_z) < 0)
                        )

                        row = {
                            "split": split,
                            GROUP_COL: cusip,
                            ISSUER_COL: entry_row.get(ISSUER_COL, np.nan),
                            MATURITY_BUCKET_COL: entry_row.get(MATURITY_BUCKET_COL, np.nan),
                            "event_date": entry_row[DATE_COL],
                            "entry_date": entry_row[DATE_COL],
                            "exit_date": df.at[exit_idx, DATE_COL],
                            "side": side,
                            "direction": direction,
                            "entry_threshold_abs_z": float(threshold),
                            "entry_z": float(entry_z) if pd.notna(entry_z) else np.nan,
                            "exit_z": float(exit_z) if pd.notna(exit_z) else np.nan,
                            "exit_reason": exit_reason,
                            "holding_obs": int(exit_pos - pos),
                            "overshoot_exit_flag": overshoot_exit,
                            "gross_bond_payoff": gross_bond_payoff,
                            "gross_residual_payoff": gross_residual_payoff,
                            "entry_m4_residual": entry_row.get("m4_residual_return", np.nan),
                            "exit_m4_residual": df.at[exit_idx, "m4_residual_return"],
                            "signal_quality_class": entry_row.get("signal_quality_class", np.nan),
                            "event_severity": entry_row.get("severity", np.nan),
                        }

                        for cost_bps in EVENT_STRATEGY_COSTS_BPS:
                            cost_decimal = float(cost_bps) / 10000.0
                            row[f"net_{cost_bps}bp_bond_payoff"] = (
                                gross_bond_payoff - cost_decimal
                                if pd.notna(gross_bond_payoff)
                                else np.nan
                            )
                            row[f"net_{cost_bps}bp_residual_payoff"] = (
                                gross_residual_payoff - cost_decimal
                                if pd.notna(gross_residual_payoff)
                                else np.nan
                            )

                        rows.append(row)

                        pos = exit_pos + 1

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce")
    out["exit_date"] = pd.to_datetime(out["exit_date"], errors="coerce")
    return out.sort_values(
        ["split", "side", "entry_threshold_abs_z", "event_date", GROUP_COL]
    ).reset_index(drop=True)


def _event_driven_payoff_long(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    rows = []
    base_cols = [
        "split",
        GROUP_COL,
        ISSUER_COL,
        MATURITY_BUCKET_COL,
        "event_date",
        "entry_date",
        "exit_date",
        "side",
        "entry_threshold_abs_z",
        "entry_z",
        "exit_z",
        "exit_reason",
        "holding_obs",
        "overshoot_exit_flag",
    ]
    base_cols = [c for c in base_cols if c in trades.columns]

    payoff_specs = [
        ("bond_return", "gross_bond_payoff", "net_{cost}bp_bond_payoff"),
        ("model_adjusted_residual", "gross_residual_payoff", "net_{cost}bp_residual_payoff"),
    ]

    for _, trade in trades.iterrows():
        base = {c: trade[c] for c in base_cols}
        for payoff_type, gross_col, net_template in payoff_specs:
            for cost_bps in EVENT_STRATEGY_COSTS_BPS:
                payoff_col = gross_col if cost_bps == 0 else net_template.format(cost=cost_bps)
                if payoff_col not in trades.columns:
                    continue
                row = base.copy()
                row["payoff_type"] = payoff_type
                row["cost_bp"] = int(cost_bps)
                row["payoff"] = trade[payoff_col]
                rows.append(row)

    return pd.DataFrame(rows)


def build_event_driven_strategy_summary(trades: pd.DataFrame) -> pd.DataFrame:
    long = _event_driven_payoff_long(trades)
    if long.empty:
        return pd.DataFrame()

    long["payoff"] = pd.to_numeric(long["payoff"], errors="coerce")
    valid = long.dropna(subset=["payoff"]).copy()
    if valid.empty:
        return pd.DataFrame()

    group_cols = ["split", "side", "entry_threshold_abs_z", "payoff_type", "cost_bp"]
    rows = []

    for keys, g in valid.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        payoff = g["payoff"].astype(float)
        holding = pd.to_numeric(g["holding_obs"], errors="coerce")

        row = {
            "split": keys[0],
            "side": keys[1],
            "entry_threshold_abs_z": float(keys[2]),
            "payoff_type": keys[3],
            "cost_bp": int(keys[4]),
            "n_trades": int(len(g)),
            "n_cusips": int(g[GROUP_COL].nunique()) if GROUP_COL in g.columns else np.nan,
            "mean_payoff": float(payoff.mean()),
            "median_payoff": float(payoff.median()),
            "total_unit_notional_payoff": float(payoff.sum()),
            "hit_rate": float(payoff.gt(0).mean()),
            "loss_probability": float(payoff.lt(0).mean()),
            "p05_payoff": float(payoff.quantile(0.05)),
            "p25_payoff": float(payoff.quantile(0.25)),
            "p75_payoff": float(payoff.quantile(0.75)),
            "p95_payoff": float(payoff.quantile(0.95)),
            "mean_holding_obs": float(holding.mean()),
            "median_holding_obs": float(holding.median()),
            "min_holding_obs": float(holding.min()),
            "max_holding_obs": float(holding.max()),
            "p75_holding_obs": float(holding.quantile(0.75)),
            "p90_holding_obs": float(holding.quantile(0.90)),
            "share_exit_residual_normalisation": float(g["exit_reason"].eq("residual_normalisation").mean()),
            "share_exit_time_stop": float(g["exit_reason"].eq("time_stop").mean()),
            "share_exit_end_of_split": float(g["exit_reason"].eq("end_of_split").mean()),
            "share_overshoot_exit": float(g["overshoot_exit_flag"].fillna(False).astype(bool).mean()),
            "mean_entry_z": float(pd.to_numeric(g["entry_z"], errors="coerce").mean()),
            "mean_exit_z": float(pd.to_numeric(g["exit_z"], errors="coerce").mean()),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_event_driven_strategy_cumulative(trades: pd.DataFrame) -> pd.DataFrame:
    long = _event_driven_payoff_long(trades)
    if long.empty:
        return pd.DataFrame()

    long["payoff"] = pd.to_numeric(long["payoff"], errors="coerce")
    long = long.dropna(subset=["payoff", "exit_date"]).copy()
    if long.empty:
        return pd.DataFrame()

    group_cols = ["split", "side", "entry_threshold_abs_z", "payoff_type", "cost_bp"]
    parts = []
    for keys, g in long.groupby(group_cols, dropna=False):
        g = g.sort_values(["exit_date", "event_date", GROUP_COL]).copy()
        g["trade_number"] = np.arange(1, len(g) + 1)
        g["cum_payoff"] = g["payoff"].cumsum()
        parts.append(
            g[
                group_cols
                + [
                    "trade_number",
                    "event_date",
                    "entry_date",
                    "exit_date",
                    GROUP_COL,
                    "payoff",
                    "cum_payoff",
                    "holding_obs",
                    "exit_reason",
                ]
            ]
        )

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_zscore_summary(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, g in signals.groupby(SPLIT_COL, dropna=False):
        for z_col in ["m4_z_main", "m4_z_cusip_rolling", "m4_issuer_maturity_z", "m4_issuer_maturity_robust_z", "m4_global_z", "m5_z_main"]:
            if z_col not in g.columns:
                continue
            x = pd.to_numeric(g[z_col], errors="coerce")
            rows.append(
                {
                    "sample_split": split,
                    "z_score": z_col,
                    "n_obs": int(len(x)),
                    "n_nonmissing": int(x.notna().sum()),
                    "nonmissing_share": float(x.notna().mean()),
                    "mean": float(x.mean()),
                    "std": float(x.std()),
                    "p01": float(x.quantile(0.01)),
                    "p05": float(x.quantile(0.05)),
                    "p50": float(x.quantile(0.50)),
                    "p95": float(x.quantile(0.95)),
                    "p99": float(x.quantile(0.99)),
                    "share_abs_ge_2": float(x.abs().ge(2).mean()),
                    "share_abs_ge_3": float(x.abs().ge(3).mean()),
                }
            )
    return pd.DataFrame(rows)


def _event_summary(signals: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in group_cols if c in signals.columns]
    if not cols:
        return pd.DataFrame()

    test = signals.loc[signals[SPLIT_COL].eq("test")].copy()
    if test.empty:
        return pd.DataFrame()

    summary = (
        test.groupby(cols, dropna=False)
        .agg(
            rows=(GROUP_COL, "size"),
            cusips=(GROUP_COL, "nunique"),
            candidate_events=("m4_candidate_flag", "sum"),
            severe_events=("m4_severe_flag", "sum"),
            mean_abs_z=("m4_abs_z_main", "mean"),
            p95_abs_z=("m4_abs_z_main", lambda x: x.quantile(0.95)),
            mean_abs_m4_residual=("m4_abs_residual", "mean"),
            high_confidence_events=("signal_quality_class", lambda x: (x == "high_confidence_dislocation_candidate").sum()),
            microstructure_sensitive_events=("signal_quality_class", lambda x: (x == "microstructure_sensitive").sum()),
            low_price_quality_warning_events=("signal_quality_class", lambda x: (x == "low_price_quality_warning").sum()),
        )
        .reset_index()
    )
    summary["candidate_event_rate"] = summary["candidate_events"] / summary["rows"]
    summary["severe_event_rate"] = summary["severe_events"] / summary["rows"]

    return summary.sort_values("candidate_event_rate", ascending=False)


def build_top_events(signals: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    test = signals.loc[signals[SPLIT_COL].eq("test")].copy()
    test = test.loc[test["m4_candidate_flag"]].copy()

    if test.empty:
        return pd.DataFrame()

    preferred_cols = [
        DATE_COL,
        GROUP_COL,
        ISSUER_COL,
        MATURITY_BUCKET_COL,
        "dislocation_side",
        "severity",
        "signal_quality_class",
        TARGET_COL,
        "m4_fitted_return",
        "m4_residual_return",
        "m4_z_main",
        "m4_z_main_source",
        "m4_issuer_maturity_robust_z",
        "m4_extreme_standard_but_not_robust",
        "m5_residual_return",
        "m5_z_main",
        "m5_abs_residual_reduction_share",
        "business_gap_days",
        "log_n_trades",
        "log_total_volume",
        "price_dispersion_rel_filled",
        "price_range_rel_filled",
        "single_trade_day",
        "future_return_1obs",
        "future_return_3obs",
        "future_return_5obs",
        "future_reversal_return_1obs",
        "future_reversal_return_3obs",
        "future_reversal_return_5obs",
        "future_return_skip1_1obs",
        "future_return_skip1_3obs",
        "future_return_skip1_5obs",
        "future_reversal_return_skip1_1obs",
        "future_reversal_return_skip1_3obs",
        "future_reversal_return_skip1_5obs",
        "time_to_abs_z_below_1_obs",
        "time_to_half_residual_obs",
        "time_to_residual_sign_flip_obs",
        "unhedged_signal_pnl_5obs",
        "unhedged_signal_pnl_skip1_5obs",
        "peer_hedged_signal_pnl_5obs",
        "peer_hedged_signal_pnl_skip1_5obs",
        "unhedged_signal_pnl_5obs_net_10bp",
        "peer_hedged_signal_pnl_5obs_net_10bp",
        "unhedged_signal_pnl_skip1_5obs_net_10bp",
        "peer_hedged_signal_pnl_skip1_5obs_net_10bp",
    ]
    cols = [c for c in preferred_cols if c in test.columns]

    return test.sort_values("m4_abs_z_main", ascending=False).head(n)[cols]


def build_filtered_top_event_tables(signals: pd.DataFrame, n: int = 100) -> dict[str, pd.DataFrame]:
    test = signals.loc[
        signals[SPLIT_COL].eq("test") & signals["m4_candidate_flag"]
    ].copy()

    empty = {
        "high_confidence": pd.DataFrame(),
        "one_per_cusip": pd.DataFrame(),
        "high_confidence_clean": pd.DataFrame(),
        "high_confidence_one_per_cusip": pd.DataFrame(),
        "high_confidence_clean_one_per_cusip": pd.DataFrame(),
    }
    if test.empty:
        return empty

    cols = [c for c in build_top_events(signals, n=1).columns if c in test.columns]

    high_conf = test.loc[
        test["signal_quality_class"].eq("high_confidence_dislocation_candidate")
    ].copy()

    high_conf_clean = high_conf.loc[
        ~high_conf["low_trade_count_flag"].fillna(False)
        & ~high_conf["large_gap_warning_flag"].fillna(False)
    ].copy()

    if "single_trade_day" in high_conf_clean.columns:
        high_conf_clean = high_conf_clean.loc[
            ~pd.to_numeric(high_conf_clean["single_trade_day"], errors="coerce").eq(1.0)
        ].copy()

    all_one_per_cusip = (
        test.sort_values("m4_abs_z_main", ascending=False)
        .drop_duplicates(GROUP_COL, keep="first")
        .head(n)
    )

    high_conf_one_per_cusip = (
        high_conf.sort_values("m4_abs_z_main", ascending=False)
        .drop_duplicates(GROUP_COL, keep="first")
        .head(n)
    )

    high_conf_clean_one_per_cusip = (
        high_conf_clean.sort_values("m4_abs_z_main", ascending=False)
        .drop_duplicates(GROUP_COL, keep="first")
        .head(n)
    )

    return {
        "high_confidence": high_conf.sort_values("m4_abs_z_main", ascending=False).head(n)[cols],
        "one_per_cusip": all_one_per_cusip[cols],
        "high_confidence_clean": high_conf_clean.sort_values("m4_abs_z_main", ascending=False).head(n)[cols],
        "high_confidence_one_per_cusip": high_conf_one_per_cusip[cols],
        "high_confidence_clean_one_per_cusip": high_conf_clean_one_per_cusip[cols],
    }


def build_event_rate_by_month(signals: pd.DataFrame) -> pd.DataFrame:
    test = signals.loc[signals[SPLIT_COL].eq("test")].copy()
    if test.empty:
        return pd.DataFrame()
    test["month"] = test[DATE_COL].dt.to_period("M").astype(str)
    monthly = (
        test.groupby("month", dropna=False)
        .agg(
            rows=(GROUP_COL, "size"),
            cusips=(GROUP_COL, "nunique"),
            candidate_events=("m4_candidate_flag", "sum"),
            severe_events=("m4_severe_flag", "sum"),
            high_confidence_events=("signal_quality_class", lambda x: (x == "high_confidence_dislocation_candidate").sum()),
            microstructure_sensitive_events=("signal_quality_class", lambda x: (x == "microstructure_sensitive").sum()),
            low_price_quality_warning_events=("signal_quality_class", lambda x: (x == "low_price_quality_warning").sum()),
        )
        .reset_index()
        .sort_values("month")
    )
    monthly["candidate_event_rate"] = monthly["candidate_events"] / monthly["rows"]
    monthly["severe_event_rate"] = monthly["severe_events"] / monthly["rows"]
    monthly["high_confidence_event_rate"] = monthly["high_confidence_events"] / monthly["rows"]
    return monthly


def write_outputs(signals: pd.DataFrame, mean_reversion: pd.DataFrame) -> None:
    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    signals.to_parquet(OUTPUT_SIGNALS_PARQUET, index=False)
    interval_cols = [
        DATE_COL,
        GROUP_COL,
        ISSUER_COL,
        MATURITY_BUCKET_COL,
        SPLIT_COL,
        TARGET_COL,
        "m4_fitted_return",
        "m4_residual_return",
        "m4_sigma_main",
        "m4_abs_scaled_residual_main",
    ]

    for level in PREDICTION_INTERVAL_LEVELS:
        pct = int(round(level * 100))
        interval_cols += [
            f"m4_interval_q_{pct}",
            f"m4_fv_lower_{pct}",
            f"m4_fv_upper_{pct}",
            f"m4_inside_fv_interval_{pct}",
            f"m4_fv_interval_width_{pct}",
        ]

    interval_cols = [c for c in interval_cols if c in signals.columns]
    signals.loc[signals[SPLIT_COL].isin(INTERVAL_EVALUATION_SPLITS), interval_cols].to_csv(
        OUTPUT_FAIR_VALUE_INTERVALS,
        index=False,
    )

    build_fair_value_interval_coverage(signals).to_csv(
        OUTPUT_FAIR_VALUE_INTERVAL_COVERAGE,
        index=False,
    )

    signals_for_csv = signals.loc[
        signals[SPLIT_COL].eq("test") & signals["m4_candidate_flag"]
    ].copy()
    signals_for_csv.to_csv(OUTPUT_SIGNALS_CSV, index=False)

    build_top_events(signals, n=150).to_csv(OUTPUT_TOP_EVENTS, index=False)
    filtered_top_tables = build_filtered_top_event_tables(signals, n=150)
    filtered_top_tables["high_confidence"].to_csv(OUTPUT_TOP_EVENTS_HIGH_CONF, index=False)
    filtered_top_tables["one_per_cusip"].to_csv(OUTPUT_TOP_EVENTS_ONE_PER_CUSIP, index=False)
    filtered_top_tables["high_confidence_clean"].to_csv(OUTPUT_TOP_EVENTS_HIGH_CONF_CLEAN, index=False)
    filtered_top_tables["high_confidence_one_per_cusip"].to_csv(
        OUTPUT_TOP_EVENTS_HIGH_CONF_ONE_PER_CUSIP,
        index=False,
    )
    filtered_top_tables["high_confidence_clean_one_per_cusip"].to_csv(
        OUTPUT_TOP_EVENTS_HIGH_CONF_CLEAN_ONE_PER_CUSIP,
        index=False,
    )

    _event_summary(signals, [ISSUER_COL]).to_csv(OUTPUT_EVENT_SUMMARY_ISSUER, index=False)
    _event_summary(signals, [MATURITY_BUCKET_COL]).to_csv(OUTPUT_EVENT_SUMMARY_BUCKET, index=False)

    tmp = signals.copy()
    tmp["year"] = tmp[DATE_COL].dt.year
    tmp["month"] = tmp[DATE_COL].dt.to_period("M").astype(str)
    _event_summary(tmp, ["year"]).to_csv(OUTPUT_EVENT_SUMMARY_YEAR, index=False)
    _event_summary(tmp, ["month"]).to_csv(OUTPUT_EVENT_SUMMARY_MONTH, index=False)

    build_zscore_summary(signals).to_csv(OUTPUT_Z_SUMMARY, index=False)
    mean_reversion.to_csv(OUTPUT_MEAN_REVERSION, index=False)
    mean_reversion.to_csv(OUTPUT_MEAN_REVERSION_SKIP, index=False)
    build_mean_reversion_by_side_and_quality(signals).to_csv(
        OUTPUT_MEAN_REVERSION_BY_SIDE_QUALITY,
        index=False,
    )
    build_event_rate_by_month(signals).to_csv(OUTPUT_EVENT_RATE_MONTH, index=False)
    build_convergence_event_table(signals).to_csv(OUTPUT_CONVERGENCE_EVENTS, index=False)
    build_convergence_summary(signals).to_csv(OUTPUT_CONVERGENCE_SUMMARY, index=False)
    build_strategy_pnl_event_table(signals).to_csv(OUTPUT_STRATEGY_PNL_EVENTS, index=False)
    build_strategy_pnl_summary(signals).to_csv(OUTPUT_STRATEGY_PNL_SUMMARY, index=False)
    build_strategy_pnl_uncertainty(signals).to_csv(
        OUTPUT_STRATEGY_PNL_UNCERTAINTY,
        index=False,
    )
    event_driven_trades = build_event_driven_strategy_trades(signals)
    event_driven_trades.to_csv(OUTPUT_EVENT_DRIVEN_STRATEGY_TRADES, index=False)
    build_event_driven_strategy_summary(event_driven_trades).to_csv(
        OUTPUT_EVENT_DRIVEN_STRATEGY_SUMMARY,
        index=False,
    )
    build_event_driven_strategy_cumulative(event_driven_trades).to_csv(
        OUTPUT_EVENT_DRIVEN_STRATEGY_CUMULATIVE,
        index=False,
    )

    class_summary = (
        signals.loc[signals[SPLIT_COL].eq("test")]
        .groupby("signal_quality_class", dropna=False)
        .agg(rows=(GROUP_COL, "size"), candidate_events=("m4_candidate_flag", "sum"), severe_events=("m4_severe_flag", "sum"))
        .reset_index()
    )
    class_summary["row_share"] = class_summary["rows"] / class_summary["rows"].sum()
    class_summary.to_csv(OUTPUT_SIGNAL_CLASS_SUMMARY, index=False)

def plot_z_distribution(signals: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    test = signals.loc[signals[SPLIT_COL].eq("test")].copy()
    x = test["m4_z_main"].replace([np.inf, -np.inf], np.nan).dropna()

    plt.figure(figsize=(7.0, 4.5))
    plt.hist(x.clip(-6, 6), bins=80)
    plt.axvline(-Z_CANDIDATE_THRESHOLD, linestyle="--", linewidth=1)
    plt.axvline(Z_CANDIDATE_THRESHOLD, linestyle="--", linewidth=1)
    plt.axvline(-Z_SEVERE_THRESHOLD, linestyle=":", linewidth=1)
    plt.axvline(Z_SEVERE_THRESHOLD, linestyle=":", linewidth=1)
    plt.xlabel("Past-only M4 residual z-score")
    plt.ylabel("Number of test observations")
    plt.title("M4 dislocation-score distribution, test sample")
    plt.tight_layout()
    plt.savefig(FIG_Z_DIST, dpi=300, bbox_inches="tight")
    plt.close()


def plot_event_count_by_month(signals: pd.DataFrame) -> None:
    test = signals.loc[signals[SPLIT_COL].eq("test")].copy()
    test["month"] = test[DATE_COL].dt.to_period("M").dt.to_timestamp()
    monthly = (
        test.groupby("month")
        .agg(candidate_events=("m4_candidate_flag", "sum"), severe_events=("m4_severe_flag", "sum"))
        .reset_index()
        .sort_values("month")
    )

    plt.figure(figsize=(8.0, 4.5))
    plt.plot(monthly["month"], monthly["candidate_events"], marker="o", label="|z| >= 2")
    plt.plot(monthly["month"], monthly["severe_events"], marker="o", label="|z| >= 3")
    plt.xlabel("Month")
    plt.ylabel("Number of test-set events")
    plt.title("Dislocation candidate count by month")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_EVENT_MONTH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_event_rate_by_month(signals: pd.DataFrame) -> None:
    monthly = build_event_rate_by_month(signals)
    if monthly.empty:
        return
    monthly["month_ts"] = pd.to_datetime(monthly["month"])

    plt.figure(figsize=(8.0, 4.5))
    plt.plot(monthly["month_ts"], monthly["candidate_event_rate"], marker="o", label="|z| >= 2")
    plt.plot(monthly["month_ts"], monthly["severe_event_rate"], marker="o", label="|z| >= 3")
    if "high_confidence_event_rate" in monthly.columns:
        plt.plot(monthly["month_ts"], monthly["high_confidence_event_rate"], marker="o", label="high-confidence")
    plt.xlabel("Month")
    plt.ylabel("Event rate")
    plt.title("Dislocation candidate rate by month")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_EVENT_RATE_MONTH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_events_over_time(signals: pd.DataFrame) -> None:
    test = signals.loc[signals[SPLIT_COL].eq("test") & signals["m4_candidate_flag"]].copy()
    if test.empty:
        return

    plt.figure(figsize=(8.0, 4.5))
    plt.scatter(test[DATE_COL], test["m4_z_main"], s=10, alpha=0.5)
    plt.axhline(Z_CANDIDATE_THRESHOLD, linestyle="--", linewidth=1)
    plt.axhline(-Z_CANDIDATE_THRESHOLD, linestyle="--", linewidth=1)
    plt.axhline(Z_SEVERE_THRESHOLD, linestyle=":", linewidth=1)
    plt.axhline(-Z_SEVERE_THRESHOLD, linestyle=":", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Past-only M4 residual z-score")
    plt.title("M4 dislocation candidates over time")
    plt.tight_layout()
    plt.savefig(FIG_TOP_EVENTS, dpi=300, bbox_inches="tight")
    plt.close()


def plot_mean_reversion(mean_reversion: pd.DataFrame, horizon: int = 5, skip1: bool = False) -> None:
    if mean_reversion.empty:
        return

    if skip1:
        col = f"mean_reversal_return_skip1_{horizon}obs"
        n_col = f"n_future_skip1_{horizon}obs"
        output_path = FIG_MEAN_REVERSION_SKIP
        ylabel = f"Mean reversal return, skip 1 then next {horizon} obs"
        title = "Skip-window future mean reversion by M4 residual z-score bucket"
    else:
        col = f"mean_reversal_return_{horizon}obs"
        n_col = f"n_future_{horizon}obs"
        output_path = FIG_MEAN_REVERSION
        ylabel = f"Mean reversal return over next {horizon} observations"
        title = "Future mean reversion by M4 residual z-score bucket"

    if col not in mean_reversion.columns:
        return

    df = mean_reversion.copy()
    df = df.loc[df[n_col].gt(0)].copy()

    plt.figure(figsize=(8.5, 4.8))
    plt.bar(df["m4_z_bucket"].astype(str), df[col])
    plt.axhline(0.0, linewidth=1)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_m4_m5_residual_reduction(signals: pd.DataFrame) -> None:
    test = signals.loc[signals[SPLIT_COL].eq("test")].copy()
    test = test.dropna(subset=["m4_abs_residual", "m5_abs_residual"])
    if test.empty:
        return

    if len(test) > 50000:
        test = test.sample(n=50000, random_state=42)

    plt.figure(figsize=(6.0, 5.5))
    plt.scatter(test["m4_abs_residual"], test["m5_abs_residual"], s=5, alpha=0.25)
    max_val = np.nanpercentile(test[["m4_abs_residual", "m5_abs_residual"]].to_numpy(), 99.5)
    plt.plot([0, max_val], [0, max_val], linewidth=1)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xlabel("Absolute M4 residual")
    plt.ylabel("Absolute M5 residual")
    plt.title("Residual reduction after price-quality controls")
    plt.tight_layout()
    plt.savefig(FIG_M4_M5_REDUCTION, dpi=300, bbox_inches="tight")
    plt.close()


def plot_candidate_rate_by_issuer(signals: pd.DataFrame) -> None:
    if ISSUER_COL not in signals.columns:
        return
    summary = _event_summary(signals, [ISSUER_COL])
    if summary.empty:
        return
    summary = summary.sort_values("candidate_event_rate")

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(summary[ISSUER_COL].astype(str), summary["candidate_event_rate"])
    plt.ylabel("Candidate-event rate, |z| >= 2")
    plt.xlabel("Issuer")
    plt.title("M4 dislocation candidate rate by issuer")
    plt.tight_layout()
    plt.savefig(FIG_ISSUER_EVENTS, dpi=300, bbox_inches="tight")
    plt.close()




def plot_convergence_summary(signals: pd.DataFrame) -> None:
    summary = build_convergence_summary(signals)
    if summary.empty:
        return
    col = "share_converged_abs_z_below_1_within_5obs"
    if col not in summary.columns:
        return
    df = summary.loc[summary["severity"].ne("ordinary")].copy()
    if df.empty:
        return
    df["label"] = (
        df["residual_side_simple"].astype(str).str.replace("_residual_potentially_", " ", regex=False)
        + " | "
        + df["signal_quality_class"].astype(str).str.replace("_", " ")
        + " | "
        + df["severity"].astype(str).str.replace("_", " ")
    )
    df = df.sort_values(col)
    plt.figure(figsize=(9.0, 5.0))
    plt.barh(df["label"], df[col])
    plt.xlabel("Share converged: |z| < 1 within 5 observations")
    plt.title("Dislocation convergence by signal side and quality")
    plt.tight_layout()
    plt.savefig(FIG_CONVERGENCE_SUMMARY, dpi=300, bbox_inches="tight")
    plt.close()


def plot_strategy_pnl_summary(signals: pd.DataFrame) -> None:
    summary = build_strategy_pnl_summary(signals)
    if summary.empty:
        return
    col = f"mean_{FIXED_HORIZON_MAIN_PNL_COL}"
    if col not in summary.columns:
        return
    df = summary.loc[summary["severity"].ne("ordinary")].copy()
    if df.empty:
        return
    df["label"] = (
        df["residual_side_simple"].astype(str).str.replace("_residual_potentially_", " ", regex=False)
        + " | "
        + df["signal_quality_class"].astype(str).str.replace("_", " ")
        + " | "
        + df["severity"].astype(str).str.replace("_", " ")
    )
    df = df.sort_values(col)
    plt.figure(figsize=(9.0, 5.0))
    plt.barh(df["label"], df[col])
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Mean unhedged signal payoff, first 5 observations, net 10 bp")
    plt.title("Stylised dislocation-signal P&L by side and quality")
    plt.tight_layout()
    plt.savefig(FIG_STRATEGY_PNL_5OBS, dpi=300, bbox_inches="tight")
    plt.close()


def plot_event_driven_strategy_cumulative(signals: pd.DataFrame) -> None:
    trades = build_event_driven_strategy_trades(signals)
    cumulative = build_event_driven_strategy_cumulative(trades)
    if cumulative.empty:
        return

    df = cumulative.loc[
        cumulative["split"].eq("test")
        & cumulative["payoff_type"].eq("bond_return")
        & cumulative["cost_bp"].eq(10)
    ].copy()
    if df.empty:
        return

    plt.figure(figsize=(9.0, 5.0))
    for keys, g in df.groupby(["side", "entry_threshold_abs_z"], dropna=False):
        side, threshold = keys
        label = f"{side}, |z|>={threshold:g}, net 10bp"
        g = g.sort_values("exit_date")
        plt.plot(g["exit_date"], g["cum_payoff"], label=label)

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Exit date")
    plt.ylabel("Cumulative unit-notional payoff")
    plt.title("Event-driven convergence payoff, test sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_EVENT_DRIVEN_STRATEGY_CUMULATIVE, dpi=300, bbox_inches="tight")
    plt.close()

def write_figures(signals: pd.DataFrame, mean_reversion: pd.DataFrame) -> None:
    plot_z_distribution(signals)
    plot_event_count_by_month(signals)
    plot_event_rate_by_month(signals)
    plot_top_events_over_time(signals)
    plot_mean_reversion(mean_reversion, horizon=5, skip1=False)
    plot_mean_reversion(mean_reversion, horizon=5, skip1=True)
    plot_m4_m5_residual_reduction(signals)
    plot_candidate_rate_by_issuer(signals)
    plot_convergence_summary(signals)
    plot_strategy_pnl_summary(signals)
    plot_event_driven_strategy_cumulative(signals)



def write_manifest(signals: pd.DataFrame, mean_reversion: pd.DataFrame) -> None:
    manifest = {
        "script": "dislocation_signal_engine_v3.py",
        "input_predictions_path": str(PREDICTIONS_PATH),
        "input_panel_with_peers_path": str(PANEL_WITH_PEERS_PATH),
        "peer_variant": PEER_VARIANT,
        "m4_model": M4_MODEL,
        "m5_model": M5_MODEL,
        "baseline_gap_threshold_bd": MODEL_READY_MAX_BUSINESS_GAP,
        "z_candidate_threshold": Z_CANDIDATE_THRESHOLD,
        "z_severe_threshold": Z_SEVERE_THRESHOLD,
        "future_horizons_obs": FUTURE_HORIZONS,
        "skip_obs_for_future_returns": SKIP_OBS_FOR_FUTURE_RETURNS,
        "fixed_horizon_payoff": {
            "main_horizon_obs": FIXED_HORIZON_MAIN_HORIZON_OBS,
            "main_cost_bps": FIXED_HORIZON_MAIN_COST_BPS,
            "main_pnl_metric": FIXED_HORIZON_MAIN_PNL_COL,
            "skip_first_observation_for_main_payoff": False,
            "skip1_metrics_retained_as_robustness": True,
            "cost_convention": (
                "cost_bps is treated as a total round-trip return haircut "
                "and is subtracted once for all payoff variants"
            ),
        },
        "event_driven_strategy": {
            "splits": EVENT_STRATEGY_SPLITS,
            "entry_thresholds_abs_z": EVENT_STRATEGY_ENTRY_THRESHOLDS,
            "exit_z_threshold": EVENT_STRATEGY_EXIT_Z,
            "max_holding_obs": EVENT_STRATEGY_MAX_HOLDING_OBS,
            "costs_bps_roundtrip": EVENT_STRATEGY_COSTS_BPS,
            "entry_mark": "same observation TRACE VWAP",
            "exit_rule": "cheap exits when z >= -1; rich exits when z <= 1; otherwise time stop",
            "payoff_types": ["bond_return", "model_adjusted_residual"],
            "overlap_rule": "no overlapping trades for same CUSIP, side, threshold and split",
        },
        "robust_zscore_method": "issuer-maturity past median/MAD diagnostic",
        "rows_total": int(len(signals)),
        "rows_train": int(signals[SPLIT_COL].eq("train").sum()) if SPLIT_COL in signals.columns else None,
        "rows_validation": int(signals[SPLIT_COL].eq("validation").sum()) if SPLIT_COL in signals.columns else None,
        "rows_test": int(signals[SPLIT_COL].eq("test").sum()) if SPLIT_COL in signals.columns else None,
        "test_candidate_events": int(signals.loc[signals[SPLIT_COL].eq("test"), "m4_candidate_flag"].sum()) if SPLIT_COL in signals.columns else None,
        "test_severe_events": int(signals.loc[signals[SPLIT_COL].eq("test"), "m4_severe_flag"].sum()) if SPLIT_COL in signals.columns else None,
        "outputs": {
            "signals_parquet": str(OUTPUT_SIGNALS_PARQUET),
            "signals_candidate_csv": str(OUTPUT_SIGNALS_CSV),
            "top_events": str(OUTPUT_TOP_EVENTS),
            "top_events_high_confidence": str(OUTPUT_TOP_EVENTS_HIGH_CONF),
            "top_events_one_per_cusip": str(OUTPUT_TOP_EVENTS_ONE_PER_CUSIP),
            "top_events_high_confidence_clean": str(OUTPUT_TOP_EVENTS_HIGH_CONF_CLEAN),
            "top_events_high_confidence_one_per_cusip": str(OUTPUT_TOP_EVENTS_HIGH_CONF_ONE_PER_CUSIP),
            "top_events_high_confidence_clean_one_per_cusip": str(OUTPUT_TOP_EVENTS_HIGH_CONF_CLEAN_ONE_PER_CUSIP),
            "mean_reversion": str(OUTPUT_MEAN_REVERSION),
            "mean_reversion_skip1": str(OUTPUT_MEAN_REVERSION_SKIP),
            "mean_reversion_by_side_quality": str(OUTPUT_MEAN_REVERSION_BY_SIDE_QUALITY),
            "event_rate_month": str(OUTPUT_EVENT_RATE_MONTH),
            "z_summary": str(OUTPUT_Z_SUMMARY),
            "convergence_events": str(OUTPUT_CONVERGENCE_EVENTS),
            "convergence_summary": str(OUTPUT_CONVERGENCE_SUMMARY),
            "strategy_pnl_events": str(OUTPUT_STRATEGY_PNL_EVENTS),
            "strategy_pnl_summary": str(OUTPUT_STRATEGY_PNL_SUMMARY),
            "fair_value_intervals": str(OUTPUT_FAIR_VALUE_INTERVALS),
            "fair_value_interval_coverage": str(OUTPUT_FAIR_VALUE_INTERVAL_COVERAGE),
            "strategy_pnl_uncertainty": str(OUTPUT_STRATEGY_PNL_UNCERTAINTY),
            "event_driven_strategy_trades": str(OUTPUT_EVENT_DRIVEN_STRATEGY_TRADES),
            "event_driven_strategy_summary": str(OUTPUT_EVENT_DRIVEN_STRATEGY_SUMMARY),
            "event_driven_strategy_cumulative": str(OUTPUT_EVENT_DRIVEN_STRATEGY_CUMULATIVE),
            "figures": [
                str(FIG_Z_DIST),
                str(FIG_EVENT_MONTH),
                str(FIG_EVENT_RATE_MONTH),
                str(FIG_TOP_EVENTS),
                str(FIG_MEAN_REVERSION),
                str(FIG_MEAN_REVERSION_SKIP),
                str(FIG_M4_M5_REDUCTION),
                str(FIG_ISSUER_EVENTS),
                str(FIG_CONVERGENCE_SUMMARY),
                str(FIG_STRATEGY_PNL_5OBS),
                str(FIG_EVENT_DRIVEN_STRATEGY_CUMULATIVE),
            ],
        },
    }

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)



def main() -> None:
    ensure_directories()

    print("Loading peer model predictions...")
    preds = load_prediction_block(PREDICTIONS_PATH)
    print("Prediction rows:", f"{len(preds):,}")
    print("Available models:", sorted(preds["model"].dropna().unique()))

    print("\nBuilding M4/M5 residual panel...")
    signals = build_m4_m5_signal_base(preds)
    print("M4 signal rows:", f"{len(signals):,}")
    print("Train rows:", f"{signals[SPLIT_COL].eq('train').sum():,}")
    print("Validation rows:", f"{signals[SPLIT_COL].eq('validation').sum():,}")
    print("Test rows:", f"{signals[SPLIT_COL].eq('test').sum():,}")

    print("\nAdding past-only residual z-scores...")
    signals = add_past_only_zscores(signals)

    print("\nAdding robust past-only z-score diagnostics...")
    signals = add_robust_zscore_diagnostics(signals)

    print("\nAdding validation-calibrated fair-value prediction intervals...")
    signals = add_fair_value_prediction_intervals(signals)

    print("\nClassifying dislocation events...")
    signals = classify_dislocation_events(signals)
    signals = add_z_buckets(signals)

    print("\nAdding future returns for mean-reversion diagnostics...")
    signals = add_future_returns(signals)

    print("\nAdding time-to-convergence and stylised signal-P&L diagnostics...")
    signals = add_convergence_and_strategy_metrics(signals)

    print("\nBuilding mean-reversion summaries...")
    mean_reversion = build_mean_reversion_summary(signals)

    print("\nWriting outputs...")
    write_outputs(signals, mean_reversion)
    write_figures(signals, mean_reversion)
    write_manifest(signals, mean_reversion)

    test = signals.loc[signals[SPLIT_COL].eq("test")].copy()
    n_test = len(test)
    n_candidates = int(test["m4_candidate_flag"].sum())
    n_severe = int(test["m4_severe_flag"].sum())

    print("\nDISLOCATION SIGNAL SUMMARY")
    print("Test rows:", f"{n_test:,}")
    print("Candidate events |z| >= 2:", f"{n_candidates:,}", f"({n_candidates / n_test:.2%})" if n_test else "")
    print("Severe events |z| >= 3:", f"{n_severe:,}", f"({n_severe / n_test:.2%})" if n_test else "")
    print("\nSignal class summary:")
    print(
        test.groupby("signal_quality_class")
        .agg(rows=(GROUP_COL, "size"), candidates=("m4_candidate_flag", "sum"), severe=("m4_severe_flag", "sum"))
        .reset_index()
        .to_string(index=False)
    )

    print("\nSaved key files:")
    for path in [
        OUTPUT_SIGNALS_PARQUET,
        OUTPUT_FAIR_VALUE_INTERVALS,
        OUTPUT_FAIR_VALUE_INTERVAL_COVERAGE,
        OUTPUT_STRATEGY_PNL_UNCERTAINTY,
        OUTPUT_TOP_EVENTS,
        OUTPUT_TOP_EVENTS_HIGH_CONF,
        OUTPUT_TOP_EVENTS_ONE_PER_CUSIP,
        OUTPUT_TOP_EVENTS_HIGH_CONF_ONE_PER_CUSIP,
        OUTPUT_TOP_EVENTS_HIGH_CONF_CLEAN_ONE_PER_CUSIP,
        OUTPUT_MEAN_REVERSION,
        OUTPUT_MEAN_REVERSION_SKIP,
        OUTPUT_MEAN_REVERSION_BY_SIDE_QUALITY,
        OUTPUT_CONVERGENCE_SUMMARY,
        OUTPUT_STRATEGY_PNL_SUMMARY,
        OUTPUT_EVENT_DRIVEN_STRATEGY_TRADES,
        OUTPUT_EVENT_DRIVEN_STRATEGY_SUMMARY,
        OUTPUT_EVENT_DRIVEN_STRATEGY_CUMULATIVE,
        OUTPUT_EVENT_RATE_MONTH,
        OUTPUT_EVENT_SUMMARY_ISSUER,
        OUTPUT_Z_SUMMARY,
        FIG_Z_DIST,
        FIG_MEAN_REVERSION,
        FIG_MEAN_REVERSION_SKIP,
        FIG_CONVERGENCE_SUMMARY,
        FIG_STRATEGY_PNL_5OBS,
        FIG_EVENT_DRIVEN_STRATEGY_CUMULATIVE,
        OUTPUT_MANIFEST,
    ]:
        print(" -", path)


if __name__ == "__main__":
    main()
