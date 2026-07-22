from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from config_institutional import (
    REGRESSION_PANEL_PATH,
    REGRESSION_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    GAP_THRESHOLDS,
    MODEL_READY_MAX_BUSINESS_GAP,
    ensure_directories,
)


GROUP_COL = "cusip_id"
DATE_COL = "date"
PREV_DATE_COL = "prev_date"
ISSUER_COL = "trace_company_symbol"
TARGET_CANDIDATES = ["final_vwap_return"]
TRAIN_END_DATE = pd.Timestamp("2023-01-01")
VALIDATION_END_DATE = pd.Timestamp("2024-01-01")
SPLIT_COL = "sample_split"

RATES_FEATURES = [
    "d_dgs2_interval",
    "d_dgs5_interval",
    "d_dgs10_interval",
    "d_dgs30_interval",
]

EQUITY_FEATURES = [
    "issuer_equity_log_return_interval",
]

MARKET_EQUITY_FEATURES = [
    "sp500_log_return_interval",
]

ISSUER_EXCESS_EQUITY_FEATURES = [
    "issuer_equity_excess_sp500_log_return_interval",
]

VIX_FEATURES = [
    "d_vix_interval",
]

MACRO_BENCHMARK_FEATURES = [
    "d_dgs2_interval",
    "d_dgs5_interval",
    "d_dgs10_interval",
    "d_dgs30_interval",
    "d_baa_10y_spread_interval",
    "d_vix_interval",
]

PEER_BASE_NAMES = [
    "same_issuer_maturity",
    "other_bank_maturity",
    "bank_sector_maturity",
]

PEER_VARIANTS = ["raw", "resid_rates", "resid_standard"]
MODEL_PEER_VARIANTS = ["raw"]
DIAGNOSTIC_PEER_VARIANTS = PEER_VARIANTS

MAX_DESIGN_DIAGNOSTIC_ROWS = 200_000
RANDOM_SEED = 42

MAX_PEER_PRICE_STALENESS_BD = 5
MIN_PEERS = 3

MATURITY_BINS = [1, 3, 5, 7, 10, 15, 30, np.inf]
MATURITY_LABELS = ["1-3y", "3-5y", "5-7y", "7-10y", "10-15y", "15-30y", "30y+"]


def choose_target(panel: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in panel.columns:
            return col
    raise ValueError(f"No target column found. Tried: {TARGET_CANDIDATES}")


def choose_log_price_column(panel: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    panel = panel.copy()

    for col in ["final_log_vwap_price", "log_vwap_price"]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
            return panel, col

    if "vwap_price" not in panel.columns:
        raise ValueError(
            "Cannot build peer factors: missing final_log_vwap_price, "
            "log_vwap_price and vwap_price."
        )

    panel["_log_vwap_price_for_peer"] = np.log(
        pd.to_numeric(panel["vwap_price"], errors="coerce").where(
            pd.to_numeric(panel["vwap_price"], errors="coerce") > 0
        )
    )

    return panel, "_log_vwap_price_for_peer"


def prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()

    panel[DATE_COL] = pd.to_datetime(panel[DATE_COL], errors="coerce")
    panel[PREV_DATE_COL] = pd.to_datetime(panel[PREV_DATE_COL], errors="coerce")
    panel[GROUP_COL] = panel[GROUP_COL].astype(str).str.strip()

    if ISSUER_COL in panel.columns:
        panel[ISSUER_COL] = panel[ISSUER_COL].astype(str).str.upper().str.strip()

    if "total_volume" in panel.columns:
        panel["log_total_volume"] = np.log1p(pd.to_numeric(panel["total_volume"], errors="coerce"))

    if "n_trades" in panel.columns:
        panel["log_n_trades"] = np.log1p(pd.to_numeric(panel["n_trades"], errors="coerce"))
        panel["single_trade_day"] = pd.to_numeric(panel["n_trades"], errors="coerce").le(1).astype(float)

    if "price_dispersion_rel" in panel.columns:
        panel["price_dispersion_rel_filled"] = pd.to_numeric(
            panel["price_dispersion_rel"], errors="coerce"
        ).fillna(0.0)

    if "price_range_rel" in panel.columns:
        panel["price_range_rel_filled"] = pd.to_numeric(
            panel["price_range_rel"], errors="coerce"
        ).fillna(0.0)

    if "final_amihud_daily" in panel.columns:
        x = pd.to_numeric(panel["final_amihud_daily"], errors="coerce")
        panel["log_final_amihud_daily"] = np.log1p(x.where(x >= 0))

    return panel


def add_static_peer_metadata(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = panel.copy()

    if "years_to_maturity" not in panel.columns:
        raise ValueError("years_to_maturity is required to form maturity peer buckets.")

    required = [GROUP_COL, ISSUER_COL, "years_to_maturity"]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing required columns for peer metadata: {missing}")

    meta = (
        panel.dropna(subset=[GROUP_COL, ISSUER_COL, "years_to_maturity"])
        .groupby(GROUP_COL)
        .agg(
            peer_issuer=(ISSUER_COL, lambda x: x.dropna().mode().iloc[0] if not x.dropna().mode().empty else x.dropna().iloc[0]),
            median_years_to_maturity=("years_to_maturity", "median"),
            n_obs_for_peer_meta=(DATE_COL, "size"),
        )
        .reset_index()
    )

    meta["peer_maturity_bucket"] = pd.cut(
        meta["median_years_to_maturity"],
        bins=MATURITY_BINS,
        labels=MATURITY_LABELS,
        right=False,
    ).astype(str)

    meta.loc[meta["peer_maturity_bucket"].eq("nan"), "peer_maturity_bucket"] = np.nan

    panel = panel.merge(
        meta[[GROUP_COL, "peer_issuer", "peer_maturity_bucket"]],
        on=GROUP_COL,
        how="left",
    )

    return panel, meta


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_true - y_pred
    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "sse": sse,
        "n_obs": int(len(y_true)),
    }

def assign_sample_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[SPLIT_COL] = np.select(
        [
            out[DATE_COL] < TRAIN_END_DATE,
            (out[DATE_COL] >= TRAIN_END_DATE) & (out[DATE_COL] < VALIDATION_END_DATE),
            out[DATE_COL] >= VALIDATION_END_DATE,
        ],
        ["train", "validation", "test"],
        default="excluded",
    )
    return out


def oos_r2_from_sse(model_sse: float, benchmark_sse: float) -> float:
    if benchmark_sse > 0:
        return float(1.0 - model_sse / benchmark_sse)
    return np.nan

def fit_fe_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    group_col: str = GROUP_COL,
) -> dict:
    train = train.copy()
    test = test.copy()

    global_y_mean = float(train[target].mean())
    y_means = train.groupby(group_col)[target].mean()

    if len(features) == 0:
        train_pred = train[group_col].map(y_means).fillna(global_y_mean).to_numpy()
        test_pred = test[group_col].map(y_means).fillna(global_y_mean).to_numpy()

        return {
            "features": features,
            "coefficients": pd.Series(dtype=float),
            "train_pred": train_pred,
            "test_pred": test_pred,
            "y_means": y_means,
            "x_means": None,
            "global_y_mean": global_y_mean,
            "global_x_mean": None,
        }

    X_train = train[features].astype(float)
    y_train = train[target].astype(float)

    x_means = train.groupby(group_col)[features].mean()
    global_x_mean = train[features].mean()

    X_bar = pd.DataFrame(index=train.index)
    for feature in features:
        X_bar[feature] = train[group_col].map(x_means[feature])

    y_bar = train[group_col].map(y_means)

    X_centered = X_train - X_bar
    y_centered = y_train - y_bar

    model = LinearRegression(fit_intercept=False)
    model.fit(X_centered.to_numpy(), y_centered.to_numpy())

    coefficients = pd.Series(model.coef_, index=features, name="coefficient")

    train_pred = predict_fe_model(
        df=train,
        features=features,
        coefficients=coefficients,
        y_means=y_means,
        x_means=x_means,
        global_y_mean=global_y_mean,
        global_x_mean=global_x_mean,
        group_col=group_col,
    )

    test_pred = predict_fe_model(
        df=test,
        features=features,
        coefficients=coefficients,
        y_means=y_means,
        x_means=x_means,
        global_y_mean=global_y_mean,
        global_x_mean=global_x_mean,
        group_col=group_col,
    )

    return {
        "features": features,
        "coefficients": coefficients,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "y_means": y_means,
        "x_means": x_means,
        "global_y_mean": global_y_mean,
        "global_x_mean": global_x_mean,
    }


def predict_fe_model(
    df: pd.DataFrame,
    features: list[str],
    coefficients: pd.Series,
    y_means: pd.Series,
    x_means: pd.DataFrame | None,
    global_y_mean: float,
    global_x_mean: pd.Series | None,
    group_col: str = GROUP_COL,
) -> np.ndarray:
    if len(features) == 0:
        return df[group_col].map(y_means).fillna(global_y_mean).to_numpy()

    if x_means is None or global_x_mean is None:
        raise ValueError("x_means and global_x_mean are required when features are non-empty.")

    X = df[features].astype(float)
    y_bar = df[group_col].map(y_means).fillna(global_y_mean)

    X_bar = pd.DataFrame(index=df.index)
    for feature in features:
        X_bar[feature] = df[group_col].map(x_means[feature]).fillna(global_x_mean[feature])

    beta = coefficients.loc[features].to_numpy()
    alpha = y_bar.to_numpy() - X_bar.to_numpy() @ beta

    return alpha + X.to_numpy() @ beta


def _build_price_and_age_matrices(
    panel: pd.DataFrame,
    log_price_col: str,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp], list[str], dict[pd.Timestamp, int], dict[str, int]]:
    cols = [GROUP_COL, DATE_COL, log_price_col]
    px = panel[cols].dropna(subset=[GROUP_COL, DATE_COL, log_price_col]).copy()
    px = px.sort_values([DATE_COL, GROUP_COL])
    px = px.drop_duplicates([DATE_COL, GROUP_COL], keep="last")

    price_wide = px.pivot(index=DATE_COL, columns=GROUP_COL, values=log_price_col).sort_index()
    all_dates = list(price_wide.index)
    all_cusips = list(price_wide.columns)

    raw_price = price_wide.to_numpy(dtype=float)
    observed = np.isfinite(raw_price)

    date_index = np.arange(len(all_dates), dtype=float)[:, None]
    obs_date_index = np.where(observed, date_index, np.nan)

    log_price_ffill = pd.DataFrame(raw_price, index=all_dates, columns=all_cusips).ffill().to_numpy(dtype=float)
    obs_index_ffill = pd.DataFrame(obs_date_index, index=all_dates, columns=all_cusips).ffill().to_numpy(dtype=float)

    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(all_dates)}
    cusip_to_col = {c: j for j, c in enumerate(all_cusips)}

    return log_price_ffill, obs_index_ffill, all_dates, all_cusips, date_to_idx, cusip_to_col


def _asof_date_index(date_to_idx: dict[pd.Timestamp, int], dates: list[pd.Timestamp], value: pd.Timestamp) -> int | None:
    value = pd.Timestamp(value)

    if value in date_to_idx:
        return date_to_idx[value]

    pos = np.searchsorted(np.array(dates, dtype="datetime64[ns]"), np.datetime64(value), side="right") - 1
    if pos < 0:
        return None
    return int(pos)


def build_peer_candidate_groups(
    meta: pd.DataFrame,
    cusip_to_col: dict[str, int],
) -> tuple[dict, dict, dict, dict[str, int]]:
    meta = meta.copy()
    meta = meta[meta[GROUP_COL].isin(cusip_to_col)].dropna(
        subset=["peer_issuer", "peer_maturity_bucket"]
    )

    meta["col"] = meta[GROUP_COL].map(cusip_to_col)

    same_issuer_bucket: dict[tuple[str, str], list[int]] = {}
    other_bank_bucket: dict[tuple[str, str], list[int]] = {}
    sector_bucket: dict[str, list[int]] = {}

    for bucket, g_bucket in meta.groupby("peer_maturity_bucket"):
        sector_bucket[str(bucket)] = g_bucket["col"].astype(int).tolist()

        issuers = sorted(g_bucket["peer_issuer"].dropna().unique())
        for issuer in issuers:
            key = (str(issuer), str(bucket))
            same = g_bucket.loc[g_bucket["peer_issuer"].eq(issuer), "col"].astype(int).tolist()
            other = g_bucket.loc[~g_bucket["peer_issuer"].eq(issuer), "col"].astype(int).tolist()
            same_issuer_bucket[key] = same
            other_bank_bucket[key] = other

    return same_issuer_bucket, other_bank_bucket, sector_bucket, cusip_to_col


def _mean_peer_return_for_row(
    end_idx: int,
    start_idx: int,
    target_col: int,
    candidate_cols: list[int],
    log_price_ffill: np.ndarray,
    obs_index_ffill: np.ndarray,
    max_staleness_bd: int,
    min_peers: int,
) -> tuple[float, int]:
    if not candidate_cols:
        return np.nan, 0

    cols = np.asarray(candidate_cols, dtype=int)
    cols = cols[cols != int(target_col)]

    if cols.size == 0:
        return np.nan, 0

    end_px = log_price_ffill[end_idx, cols]
    start_px = log_price_ffill[start_idx, cols]

    end_obs_idx = obs_index_ffill[end_idx, cols]
    start_obs_idx = obs_index_ffill[start_idx, cols]

    end_age = end_idx - end_obs_idx
    start_age = start_idx - start_obs_idx

    ret = end_px - start_px

    valid = (
        np.isfinite(ret)
        & np.isfinite(end_age)
        & np.isfinite(start_age)
        & (end_age <= max_staleness_bd)
        & (start_age <= max_staleness_bd)
    )

    n_valid = int(valid.sum())
    if n_valid < min_peers:
        return np.nan, n_valid

    return float(np.nanmean(ret[valid])), n_valid


def build_raw_peer_factors(
    panel: pd.DataFrame,
    log_price_col: str,
    max_staleness_bd: int = MAX_PEER_PRICE_STALENESS_BD,
    min_peers: int = MIN_PEERS,
) -> pd.DataFrame:
    panel = panel.copy()
    panel = panel.sort_values([DATE_COL, GROUP_COL]).reset_index(drop=False).rename(columns={"index": "_original_index"})

    matrices = _build_price_and_age_matrices(panel, log_price_col=log_price_col)
    log_price_ffill, obs_index_ffill, all_dates, all_cusips, date_to_idx, cusip_to_col = matrices

    meta = (
        panel[[GROUP_COL, "peer_issuer", "peer_maturity_bucket"]]
        .drop_duplicates(GROUP_COL)
        .copy()
    )

    same_issuer_bucket, other_bank_bucket, sector_bucket, _ = build_peer_candidate_groups(meta, cusip_to_col)

    out_cols = []
    for name in PEER_BASE_NAMES:
        out_cols.append(f"peer_raw_{name}")
        out_cols.append(f"peer_n_{name}")

    peer_values = {col: np.full(len(panel), np.nan) for col in out_cols}

    for pos, row in enumerate(panel.itertuples(index=False)):
        if pos % 25000 == 0:
            print(f"Building raw peer factors: row {pos:,} / {len(panel):,}")

        cusip = getattr(row, GROUP_COL)
        date_value = getattr(row, DATE_COL)
        prev_value = getattr(row, PREV_DATE_COL)
        issuer = getattr(row, "peer_issuer")
        bucket = getattr(row, "peer_maturity_bucket")

        if pd.isna(date_value) or pd.isna(prev_value) or pd.isna(issuer) or pd.isna(bucket):
            continue

        if cusip not in cusip_to_col:
            continue

        end_idx = _asof_date_index(date_to_idx, all_dates, pd.Timestamp(date_value))
        start_idx = _asof_date_index(date_to_idx, all_dates, pd.Timestamp(prev_value))
        if end_idx is None or start_idx is None or start_idx >= end_idx:
            continue

        target_col = cusip_to_col[cusip]
        key = (str(issuer), str(bucket))

        candidate_map = {
            "same_issuer_maturity": same_issuer_bucket.get(key, []),
            "other_bank_maturity": other_bank_bucket.get(key, []),
            "bank_sector_maturity": sector_bucket.get(str(bucket), []),
        }

        for name, candidates in candidate_map.items():
            value, n_valid = _mean_peer_return_for_row(
                end_idx=end_idx,
                start_idx=start_idx,
                target_col=target_col,
                candidate_cols=candidates,
                log_price_ffill=log_price_ffill,
                obs_index_ffill=obs_index_ffill,
                max_staleness_bd=max_staleness_bd,
                min_peers=min_peers,
            )

            peer_values[f"peer_raw_{name}"][pos] = value
            peer_values[f"peer_n_{name}"][pos] = n_valid

    peer_df = panel[["_original_index"]].copy()
    for col, values in peer_values.items():
        peer_df[col] = values

    panel = panel.merge(peer_df, on="_original_index", how="left")
    panel = panel.sort_values("_original_index").drop(columns=["_original_index"]).reset_index(drop=True)

    return panel


def add_residualised_peer_factors(
    panel: pd.DataFrame,
    base_features: list[str],
    variant_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = panel.copy()
    features = [c for c in base_features if c in panel.columns]

    if not features:
        raise ValueError(f"No available base features for {variant_name} residualisation.")

    coef_rows = []

    for name in PEER_BASE_NAMES:
        raw_col = f"peer_raw_{name}"
        resid_col = f"peer_{variant_name}_{name}"

        if raw_col not in panel.columns:
            continue

        required = [raw_col] + features + [DATE_COL]
        sample = panel.dropna(subset=required).copy()
        train = sample.loc[sample[DATE_COL] < TRAIN_END_DATE].copy()

        panel[resid_col] = np.nan

        if train.empty:
            coef_rows.append(
                {
                    "peer_base": name,
                    "peer_variant": variant_name,
                    "raw_feature": raw_col,
                    "term": "status",
                    "coefficient": np.nan,
                    "n_train": 0,
                    "train_r2": np.nan,
                    "note": "empty training sample",
                }
            )
            continue

        model = LinearRegression(fit_intercept=True)
        model.fit(train[features].astype(float).to_numpy(), train[raw_col].astype(float).to_numpy())

        complete = panel[features].notna().all(axis=1) & panel[raw_col].notna()
        pred = np.full(len(panel), np.nan)
        pred[complete.to_numpy()] = model.predict(panel.loc[complete, features].astype(float).to_numpy())
        panel[resid_col] = panel[raw_col].to_numpy(dtype=float) - pred

        train_pred = model.predict(train[features].astype(float).to_numpy())
        y_train = train[raw_col].astype(float).to_numpy()
        sse = float(np.sum((y_train - train_pred) ** 2))
        sst = float(np.sum((y_train - y_train.mean()) ** 2))
        train_r2 = 1.0 - sse / sst if sst > 0 else np.nan

        coef_rows.append(
            {
                "peer_base": name,
                "peer_variant": variant_name,
                "raw_feature": raw_col,
                "term": "intercept",
                "coefficient": float(model.intercept_),
                "n_train": int(len(train)),
                "train_r2": float(train_r2),
                "note": "",
            }
        )

        for feature, coefficient in zip(features, model.coef_):
            coef_rows.append(
                {
                    "peer_base": name,
                    "peer_variant": variant_name,
                    "raw_feature": raw_col,
                    "term": feature,
                    "coefficient": float(coefficient),
                    "n_train": int(len(train)),
                    "train_r2": float(train_r2),
                    "note": "",
                }
            )

    return panel, pd.DataFrame(coef_rows)


def available(cols: Iterable[str], panel: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in panel.columns]


def get_microstructure_features(panel: pd.DataFrame) -> list[str]:
    candidates = [
        "log_n_trades",
        "log_total_volume",
        "price_dispersion_rel_filled",
        "price_range_rel_filled",
        "business_gap_days",
        "single_trade_day",
        "institutional_trade_share",
        "potential_agency_duplicate_share",
        "ats_trade_share",
        "correction_report_share",
        "buy_sell_imbalance",
    ]
    return available(candidates, panel)


def get_model_specs(panel: pd.DataFrame, peer_variant: str) -> dict[str, list[str]]:
    if peer_variant not in set(PEER_VARIANTS):
        raise ValueError(f"peer_variant must be one of {PEER_VARIANTS}.")

    rates = available(RATES_FEATURES, panel)
    equity = available(EQUITY_FEATURES, panel)
    vix = available(VIX_FEATURES, panel)
    micro = get_microstructure_features(panel)

    standard = rates + equity + vix

    peer_same = available([f"peer_{peer_variant}_same_issuer_maturity"], panel)
    peer_other = available([f"peer_{peer_variant}_other_bank_maturity"], panel)
    peer_sector = available([f"peer_{peer_variant}_bank_sector_maturity"], panel)
    peer_all = peer_same + peer_other + peer_sector

    specs = {
        "M0_bond_fe_only": [],
        "M1_rates": rates,
        "M2_rates_equity": rates + equity,
        "M3_rates_equity_vix": standard,

        f"M4a_rates_equity_vix_peer_same_issuer_{peer_variant}": standard + peer_same,
        f"M4b_rates_equity_vix_peer_other_bank_{peer_variant}": standard + peer_other,
        f"M4c_rates_equity_vix_peer_sector_{peer_variant}": standard + peer_sector,
        f"M4d_rates_equity_vix_peer_same_other_{peer_variant}": standard + peer_same + peer_other,

        f"M4_rates_equity_vix_peer_{peer_variant}": standard + peer_all,

        f"M5_rates_equity_vix_peer_{peer_variant}_microstructure_clean": standard + peer_all + micro,
    }

    specs["B1_old_macro_rates_credit_vix"] = available(MACRO_BENCHMARK_FEATURES, panel)
    specs["B2_old_macro_rates_credit_vix_equity"] = available(
        MACRO_BENCHMARK_FEATURES + EQUITY_FEATURES,
        panel,
    )

    return specs

def get_equity_proxy_horse_race_specs(
    panel: pd.DataFrame,
    peer_variant: str = "raw",
) -> dict[str, list[str]]:
    if peer_variant not in set(PEER_VARIANTS):
        raise ValueError(f"peer_variant must be one of {PEER_VARIANTS}.")

    peer_features = [f"peer_{peer_variant}_{name}" for name in PEER_BASE_NAMES]
    peer_features = available(peer_features, panel)

    rates = available(RATES_FEATURES, panel)
    issuer_equity = available(EQUITY_FEATURES, panel)
    market_equity = available(MARKET_EQUITY_FEATURES, panel)
    issuer_excess = available(ISSUER_EXCESS_EQUITY_FEATURES, panel)
    vix = available(VIX_FEATURES, panel)

    if not market_equity:
        raise ValueError(
            "Missing sp500_log_return_interval. "
            "Run fred_data.py and build_regression_panel_institutional.py after adding SP500."
        )

    if not issuer_excess:
        raise ValueError(
            "Missing issuer_equity_excess_sp500_log_return_interval. "
            "Check add_interval_features in build_regression_panel_institutional.py."
        )

    specs = {
        "EQ0_bond_fe_only": [],
        "EQ1_rates": rates,

        "EQ2a_rates_issuer_equity": rates + issuer_equity,
        "EQ2b_rates_sp500": rates + market_equity,
        "EQ2c_rates_issuer_equity_sp500": rates + issuer_equity + market_equity,
        "EQ2d_rates_sp500_issuer_excess": rates + market_equity + issuer_excess,

        "EQ3a_rates_issuer_equity_vix": rates + issuer_equity + vix,
        "EQ3b_rates_sp500_vix": rates + market_equity + vix,
        "EQ3c_rates_issuer_equity_sp500_vix": rates + issuer_equity + market_equity + vix,
        "EQ3d_rates_sp500_issuer_excess_vix": rates + market_equity + issuer_excess + vix,

        f"EQ4a_rates_issuer_equity_vix_peer_{peer_variant}": (
            rates + issuer_equity + vix + peer_features
        ),
        f"EQ4b_rates_sp500_vix_peer_{peer_variant}": (
            rates + market_equity + vix + peer_features
        ),
        f"EQ4c_rates_issuer_equity_sp500_vix_peer_{peer_variant}": (
            rates + issuer_equity + market_equity + vix + peer_features
        ),
        f"EQ4d_rates_sp500_issuer_excess_vix_peer_{peer_variant}": (
            rates + market_equity + issuer_excess + vix + peer_features
        ),
    }

    return specs

def restrict_to_equity_horse_race_common_sample(
    panel: pd.DataFrame,
    peer_variant: str = "raw",
) -> pd.DataFrame:

    peer_features = [f"peer_{peer_variant}_{name}" for name in PEER_BASE_NAMES]

    required = (
        RATES_FEATURES
        + VIX_FEATURES
        + EQUITY_FEATURES
        + MARKET_EQUITY_FEATURES
        + ISSUER_EXCESS_EQUITY_FEATURES
        + peer_features
        + [TARGET_CANDIDATES[0], DATE_COL, GROUP_COL]
    )

    required = [c for c in required if c in panel.columns]

    out = panel.dropna(subset=required).copy()

    print("\nEquity proxy horse-race common sample")
    print(f"Rows before restriction: {len(panel):,}")
    print(f"Rows after restriction:  {len(out):,}")
    print(f"CUSIPs after restriction: {out[GROUP_COL].nunique():,}")
    print(f"First date: {out[DATE_COL].min()}")
    print(f"Last date:  {out[DATE_COL].max()}")

    return out


def run_model_specs(
    panel: pd.DataFrame,
    target: str,
    model_specs: dict[str, list[str]],
    gap_threshold: int | None,
    peer_variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = panel.copy()

    if gap_threshold is not None:
        gap_flag = f"valid_return_gap_{gap_threshold}bd"
        if gap_flag not in df.columns:
            raise ValueError(f"Missing required gap flag: {gap_flag}")
        df = df.loc[df[gap_flag]].copy()

    df = df.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    print("\nMODEL SAMPLE")
    print("Gap threshold:", gap_threshold)
    print("Peer variant:", peer_variant)
    print("Target:", target)
    print("Rows before model-specific dropna:", len(df))
    print("CUSIPs before model-specific dropna:", df[GROUP_COL].nunique())
    print("Date range:", df[DATE_COL].min(), "to", df[DATE_COL].max())

    results = []
    coef_rows = []
    pred_parts = []

    for model_name, raw_features in model_specs.items():
        features = [f for f in raw_features if f in df.columns]
        required_cols = [GROUP_COL, DATE_COL, target] + features

        sample = df.dropna(subset=required_cols).copy()
        sample = (
            sample.sort_values([GROUP_COL, DATE_COL])
            .reset_index(drop=False)
            .rename(columns={"index": "_sample_index"})
        )
        sample = assign_sample_split(sample)
        sample = sample.loc[sample[SPLIT_COL].isin(["train", "validation", "test"])].copy()

        train = sample.loc[sample[SPLIT_COL].eq("train")].copy()
        validation = sample.loc[sample[SPLIT_COL].eq("validation")].copy()
        test = sample.loc[sample[SPLIT_COL].eq("test")].copy()
        eval_df = pd.concat([validation, test], ignore_index=False).copy()

        print(
            f"{model_name}: rows={len(sample):,}, "
            f"train={len(train):,}, validation={len(validation):,}, test={len(test):,}, "
            f"cusips={sample[GROUP_COL].nunique():,}, features={len(features)}"
        )

        if train.empty or validation.empty or test.empty:
            print(f"Skipping {model_name}: empty train, validation or test sample.")
            continue

        fitted = fit_fe_model(
            train=train,
            test=eval_df,
            features=features,
            target=target,
        )

        fitted_m0_same = fit_fe_model(
            train=train,
            test=eval_df,
            features=[],
            target=target,
        )

        train = train.copy()
        eval_df = eval_df.copy()

        train["_fitted_return"] = fitted["train_pred"]
        train["_residual_return"] = train[target] - train["_fitted_return"]

        eval_df["_fitted_return"] = fitted["test_pred"]
        eval_df["_residual_return"] = eval_df[target] - eval_df["_fitted_return"]
        eval_df["_m0_fitted_return"] = fitted_m0_same["test_pred"]

        y_train = train[target].to_numpy()
        train_metrics = compute_metrics(y_train, train["_fitted_return"].to_numpy())
        train_m0_metrics = compute_metrics(
            y_train,
            fitted_m0_same["train_pred"],
        )

        train_r2_vs_bond_fe = oos_r2_from_sse(
            train_metrics["sse"],
            train_m0_metrics["sse"],
        )

        split_metrics = {}
        m0_metrics = {}
        oos_r2 = {}

        for split in ["validation", "test"]:
            g = eval_df.loc[eval_df[SPLIT_COL].eq(split)].copy()
            split_metrics[split] = compute_metrics(
                g[target].to_numpy(),
                g["_fitted_return"].to_numpy(),
            )
            m0_metrics[split] = compute_metrics(
                g[target].to_numpy(),
                g["_m0_fitted_return"].to_numpy(),
            )
            oos_r2[split] = oos_r2_from_sse(
                split_metrics[split]["sse"],
                m0_metrics[split]["sse"],
            )

        result = {
            "gap_threshold_bd": gap_threshold,
            "peer_variant": peer_variant,
            "model": model_name,
            "model_order": model_order(model_name),
            "n_features": len(features),
            "features": ", ".join(features),
            "sample_rows": len(sample),
            "sample_cusips": sample[GROUP_COL].nunique(),

            "train_rows": train_metrics["n_obs"],
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "train_sse": train_metrics["sse"],
            "m0_same_sample_train_sse": train_m0_metrics["sse"],
            "train_r2_vs_bond_fe": train_r2_vs_bond_fe,

            "validation_rows": split_metrics["validation"]["n_obs"],
            "validation_rmse": split_metrics["validation"]["rmse"],
            "validation_mae": split_metrics["validation"]["mae"],
            "validation_r2": split_metrics["validation"]["r2"],
            "validation_sse": split_metrics["validation"]["sse"],
            "m0_same_sample_validation_sse": m0_metrics["validation"]["sse"],
            "validation_oos_r2_vs_bond_fe": oos_r2["validation"],

            "test_rows": split_metrics["test"]["n_obs"],
            "test_rmse": split_metrics["test"]["rmse"],
            "test_mae": split_metrics["test"]["mae"],
            "test_r2": split_metrics["test"]["r2"],
            "test_sse": split_metrics["test"]["sse"],
            "m0_same_sample_test_sse": m0_metrics["test"]["sse"],
            "test_oos_r2_vs_bond_fe": oos_r2["test"],

            "oos_r2_vs_bond_fe": oos_r2["test"],
        }
        results.append(result)

        for feature, coefficient in fitted["coefficients"].items():
            coef_rows.append(
                {
                    "gap_threshold_bd": gap_threshold,
                    "peer_variant": peer_variant,
                    "model": model_name,
                    "feature": feature,
                    "coefficient": coefficient,
                }
            )

        prediction_cols = ["_sample_index", GROUP_COL, DATE_COL, target]
        optional_prediction_cols = [
            ISSUER_COL,
            "peer_maturity_bucket",
            "business_gap_days",
            "log_n_trades",
            "log_total_volume",
            "price_dispersion_rel_filled",
            "price_range_rel_filled",
            "single_trade_day",
            "institutional_trade_share",
            "potential_agency_duplicate_share",
            "ats_trade_share",
            "correction_report_share",
            "buy_sell_imbalance",
        ]
        for col in optional_prediction_cols:
            if col in sample.columns and col not in prediction_cols:
                prediction_cols.append(col)

        pred_train = train[prediction_cols].copy()
        pred_train[SPLIT_COL] = "train"
        pred_train["model"] = model_name
        pred_train["peer_variant"] = peer_variant
        pred_train["fitted_return"] = train["_fitted_return"].to_numpy()
        pred_train["residual_return"] = train["_residual_return"].to_numpy()

        pred_eval = eval_df[prediction_cols].copy()
        pred_eval[SPLIT_COL] = eval_df[SPLIT_COL].to_numpy()
        pred_eval["model"] = model_name
        pred_eval["peer_variant"] = peer_variant
        pred_eval["fitted_return"] = eval_df["_fitted_return"].to_numpy()
        pred_eval["residual_return"] = eval_df["_residual_return"].to_numpy()

        pred_parts.append(pd.concat([pred_train, pred_eval], ignore_index=True))

    results_df = pd.DataFrame(results)
    coef_df = pd.DataFrame(coef_rows)
    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()

    return results_df, coef_df, pred_df


def run_gap_sensitivity(
    panel: pd.DataFrame,
    target: str,
    peer_variants: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_parts = []
    coef_parts = []

    for peer_variant in peer_variants:
        specs = get_model_specs(panel, peer_variant=peer_variant)
        for gap in GAP_THRESHOLDS:
            results, coefs, _ = run_model_specs(
                panel=panel,
                target=target,
                model_specs=specs,
                gap_threshold=gap,
                peer_variant=peer_variant,
            )
            result_parts.append(results)
            if not coefs.empty:
                coef_parts.append(coefs)

    all_results = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()
    all_coefs = pd.concat(coef_parts, ignore_index=True) if coef_parts else pd.DataFrame()

    return all_results, all_coefs


def build_peer_factor_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for name in PEER_BASE_NAMES:
        n_col = f"peer_n_{name}"

        for variant in PEER_VARIANTS:
            col = f"peer_{variant}_{name}"
            if col not in panel.columns:
                continue

            valid = panel[col].notna()
            row = {
                "peer_base": name,
                "peer_variant": variant,
                "feature": col,
                "n_obs": int(len(panel)),
                "n_nonmissing": int(valid.sum()),
                "share_nonmissing": float(valid.mean()),
                "mean": float(panel[col].mean(skipna=True)),
                "std": float(panel[col].std(skipna=True)),
                "p01": float(panel[col].quantile(0.01)),
                "p50": float(panel[col].quantile(0.50)),
                "p99": float(panel[col].quantile(0.99)),
            }

            if n_col in panel.columns:
                row["mean_n_peers"] = float(panel.loc[valid, n_col].mean())
                row["median_n_peers"] = float(panel.loc[valid, n_col].median())
                row["p10_n_peers"] = float(panel.loc[valid, n_col].quantile(0.10))
            else:
                row["mean_n_peers"] = np.nan
                row["median_n_peers"] = np.nan
                row["p10_n_peers"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def build_feature_correlation(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    cols = []
    cols += available(RATES_FEATURES, panel)
    for variant in PEER_VARIANTS:
        cols += [f"peer_{variant}_{name}" for name in PEER_BASE_NAMES if f"peer_{variant}_{name}" in panel.columns]
    cols += available(EQUITY_FEATURES + VIX_FEATURES + ["d_baa_10y_spread_interval"], panel)
    cols += get_microstructure_features(panel)
    cols += available(["log_final_amihud_daily"], panel)
    cols = [target] + list(dict.fromkeys(cols))

    return panel[cols].corr()



def peer_ablation_model_names(peer_variant: str) -> list[str]:
    return [
        f"M4a_rates_equity_vix_peer_same_issuer_{peer_variant}",
        f"M4b_rates_equity_vix_peer_other_bank_{peer_variant}",
        f"M4c_rates_equity_vix_peer_sector_{peer_variant}",
        f"M4d_rates_equity_vix_peer_same_other_{peer_variant}",
        f"M4_rates_equity_vix_peer_{peer_variant}",
    ]


def core_model_names(
    peer_variant: str,
    include_m5: bool = True,
    include_peer_ablations: bool = False,
) -> list[str]:
    names = [
        "M0_bond_fe_only",
        "M1_rates",
        "M2_rates_equity",
        "M3_rates_equity_vix",
    ]

    if include_peer_ablations:
        names += peer_ablation_model_names(peer_variant)
    else:
        names.append(f"M4_rates_equity_vix_peer_{peer_variant}")

    if include_m5:
        names.append(f"M5_rates_equity_vix_peer_{peer_variant}_microstructure_clean")

    return names


def model_order(model_name: str) -> float:
    if model_name == "M0_bond_fe_only":
        return 0.0
    if model_name == "M1_rates":
        return 1.0
    if model_name == "M2_rates_equity":
        return 2.0
    if model_name == "M3_rates_equity_vix":
        return 3.0
    if model_name.startswith("M4a_"):
        return 4.1
    if model_name.startswith("M4b_"):
        return 4.2
    if model_name.startswith("M4c_"):
        return 4.3
    if model_name.startswith("M4d_"):
        return 4.4
    if model_name.startswith("M4_"):
        return 4.9
    if model_name.startswith("M5_"):
        return 5.0
    if model_name.startswith("B1_"):
        return 6.0
    if model_name.startswith("B2_"):
        return 7.0
    return 99.0


def short_model_label(model_name: str) -> str:
    if model_name == "M0_bond_fe_only":
        return "M0 FE"
    if model_name == "M1_rates":
        return "M1 rates"
    if model_name == "M2_rates_equity":
        return "M2 + equity"
    if model_name == "M3_rates_equity_vix":
        return "M3 + VIX"
    if model_name.startswith("M4a_"):
        return "M4a same"
    if model_name.startswith("M4b_"):
        return "M4b other"
    if model_name.startswith("M4c_"):
        return "M4c sector"
    if model_name.startswith("M4d_"):
        return "M4d same+other"
    if model_name.startswith("M4_"):
        return "M4 all peers"
    if model_name.startswith("M5_"):
        return "M5 + micro"
    if model_name.startswith("B1_"):
        return "B1 macro"
    if model_name.startswith("B2_"):
        return "B2 macro+eq"
    return model_name


def run_balanced_peer_sample_comparison(
    panel: pd.DataFrame,
    target: str,
    gap_threshold: int = MODEL_READY_MAX_BUSINESS_GAP,
    peer_variants: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    peer_variants = peer_variants or MODEL_PEER_VARIANTS

    result_parts = []
    coef_parts = []

    for peer_variant in peer_variants:
        specs = get_model_specs(panel, peer_variant=peer_variant)

        m4_name = f"M4_rates_equity_vix_peer_{peer_variant}"
        m4_features = specs[m4_name]

        df = panel.copy()
        gap_flag = f"valid_return_gap_{gap_threshold}bd"
        if gap_flag not in df.columns:
            raise ValueError(f"Missing required gap flag: {gap_flag}")
        df = df.loc[df[gap_flag]].copy()

        balanced_required = [GROUP_COL, DATE_COL, target] + m4_features
        balanced = df.dropna(subset=balanced_required).copy()
        balanced = balanced.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

        print("\nBALANCED PEER SAMPLE")
        print("Gap threshold:", gap_threshold)
        print("Peer variant:", peer_variant)
        print("Reference model:", m4_name)
        print(f"Rows={len(balanced):,}, cusips={balanced[GROUP_COL].nunique():,}")

        names = core_model_names(
            peer_variant,
            include_m5=False,
            include_peer_ablations=True,
        )
        balanced_specs = {name: specs[name] for name in names if name in specs}

        results, coefs, _ = run_model_specs(
            panel=balanced,
            target=target,
            model_specs=balanced_specs,
            gap_threshold=None,
            peer_variant=peer_variant,
        )

        if not results.empty:
            results["sample_type"] = "balanced_m4_complete_cases"
            results["balanced_reference_model"] = m4_name
            result_parts.append(results)

        if not coefs.empty:
            coefs["sample_type"] = "balanced_m4_complete_cases"
            coefs["balanced_reference_model"] = m4_name
            coef_parts.append(coefs)

    results_df = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()
    coefs_df = pd.concat(coef_parts, ignore_index=True) if coef_parts else pd.DataFrame()

    return results_df, coefs_df


def build_incremental_contribution_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    out_parts = []
    group_cols = [c for c in ["gap_threshold_bd", "peer_variant", "sample_type"] if c in results.columns]

    temp = results.copy()
    if not group_cols:
        temp["_all"] = "all"
        group_cols = ["_all"]

    for _, g in temp.groupby(group_cols, dropna=False):
        g = g.copy()

        peer_variant = str(g["peer_variant"].dropna().iloc[0]) if "peer_variant" in g.columns and g["peer_variant"].notna().any() else "raw"

        nested_models = core_model_names(
            peer_variant,
            include_m5=True,
            include_peer_ablations=False,
        )

        g = g.loc[g["model"].isin(nested_models)].copy()

        if g.empty:
            continue

        g["model_order"] = g["model"].map(model_order)
        g = g.sort_values("model_order").reset_index(drop=True)

        g["previous_model"] = g["model"].shift(1)
        g["delta_oos_r2_vs_previous"] = g["oos_r2_vs_bond_fe"].diff()
        g["delta_test_rmse_vs_previous"] = g["test_rmse"].diff()
        g["delta_test_mae_vs_previous"] = g["test_mae"].diff()

        if "_all" in g.columns:
            g = g.drop(columns=["_all"])

        out_parts.append(g)

    return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()

def build_peer_validation_selection(results: pd.DataFrame) -> pd.DataFrame:

    if results.empty:
        return pd.DataFrame()

    peer_rows = results.loc[
        results["model"].str.startswith("M4", na=False)
        & results["validation_rmse"].notna()
    ].copy()

    if peer_rows.empty:
        return pd.DataFrame()

    peer_rows = peer_rows.sort_values(
        ["peer_variant", "validation_rmse", "validation_mae"],
        ascending=[True, True, True],
    )

    peer_rows["validation_rank_rmse"] = (
        peer_rows.groupby("peer_variant")["validation_rmse"]
        .rank(method="first", ascending=True)
        .astype(int)
    )

    cols = [
        "peer_variant",
        "model",
        "model_order",
        "n_features",
        "features",
        "sample_rows",
        "train_rows",
        "validation_rows",
        "test_rows",
        "validation_rmse",
        "validation_mae",
        "validation_oos_r2_vs_bond_fe",
        "test_rmse",
        "test_mae",
        "test_oos_r2_vs_bond_fe",
        "validation_rank_rmse",
    ]

    return peer_rows[[c for c in cols if c in peer_rows.columns]].reset_index(drop=True)


def build_model_estimation_summary(results: pd.DataFrame, target: str) -> pd.DataFrame:

    if results.empty:
        return pd.DataFrame()

    out = results.copy()

    out["estimator"] = "CUSIP fixed-effect within OLS"
    out["fixed_effects"] = "CUSIP"
    out["coefficient_scope"] = "common slopes across CUSIPs; CUSIP-specific intercepts"
    out["regularization"] = "none"
    out["target"] = target

    out["fit_sample"] = f"date < {TRAIN_END_DATE.date()}"
    out["validation_sample"] = (
        f"{TRAIN_END_DATE.date()} <= date < {VALIDATION_END_DATE.date()}"
    )
    out["test_sample"] = f"date >= {VALIDATION_END_DATE.date()}"

    out["prediction_type"] = (
        "out-of-sample contemporaneous fair-value application"
    )

    cols = [
        "model",
        "peer_variant",
        "estimator",
        "fixed_effects",
        "regularization",
        "target",
        "fit_sample",
        "validation_sample",
        "test_sample",
        "coefficient_scope",
        "prediction_type",
        "n_features",
        "features",
        "train_rows",
        "validation_rows",
        "test_rows",
        "validation_rmse",
        "validation_mae",
        "validation_oos_r2_vs_bond_fe",
        "test_rmse",
        "test_mae",
        "test_oos_r2_vs_bond_fe",
    ]

    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)

def _design_sample(
    panel: pd.DataFrame,
    target: str,
    features: list[str],
    gap_threshold: int,
) -> pd.DataFrame:
    df = panel.copy()
    gap_flag = f"valid_return_gap_{gap_threshold}bd"
    if gap_flag not in df.columns:
        raise ValueError(f"Missing required gap flag: {gap_flag}")
    df = df.loc[df[gap_flag]].copy()
    required = [GROUP_COL, DATE_COL, target] + features
    return df.dropna(subset=required).copy()


def build_design_diagnostics(
    panel: pd.DataFrame,
    target: str,
    gap_threshold: int = MODEL_READY_MAX_BUSINESS_GAP,
    peer_variants: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    peer_variants = peer_variants or PEER_VARIANTS
    design_rows = []
    vif_rows = []

    rng = np.random.default_rng(RANDOM_SEED)

    for peer_variant in peer_variants:
        specs = get_model_specs(panel, peer_variant=peer_variant)
        for model_name in core_model_names(peer_variant, include_m5=True):
            features = specs[model_name]
            sample = _design_sample(panel, target, features, gap_threshold)

            row = {
                "gap_threshold_bd": gap_threshold,
                "peer_variant": peer_variant,
                "model": model_name,
                "model_order": model_order(model_name),
                "n_features": len(features),
                "sample_rows": len(sample),
                "sample_cusips": sample[GROUP_COL].nunique() if GROUP_COL in sample.columns else np.nan,
                "condition_number": np.nan,
                "max_abs_feature_corr": np.nan,
                "mean_abs_feature_corr": np.nan,
                "note": "",
            }

            if len(features) == 0:
                row["note"] = "no regressors"
                design_rows.append(row)
                continue

            if sample.empty:
                row["note"] = "empty sample"
                design_rows.append(row)
                continue

            if len(sample) > MAX_DESIGN_DIAGNOSTIC_ROWS:
                idx = rng.choice(sample.index.to_numpy(), size=MAX_DESIGN_DIAGNOSTIC_ROWS, replace=False)
                diag_sample = sample.loc[idx].copy()
                row["note"] = f"diagnostics computed on {MAX_DESIGN_DIAGNOSTIC_ROWS:,} row subsample"
            else:
                diag_sample = sample

            X = diag_sample[features].astype(float)
            std = X.std(axis=0, ddof=0)
            nonconstant = std.gt(0) & std.notna()
            X = X.loc[:, nonconstant]

            if X.shape[1] == 0:
                row["note"] = "all features constant after filtering"
                design_rows.append(row)
                continue

            Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
            Xz_values = Xz.to_numpy(dtype=float)

            if Xz_values.shape[0] > Xz_values.shape[1]:
                try:
                    row["condition_number"] = float(np.linalg.cond(Xz_values))
                except np.linalg.LinAlgError:
                    row["condition_number"] = np.nan

            if X.shape[1] > 1:
                corr = X.corr().abs()
                mask = ~np.eye(corr.shape[0], dtype=bool)
                offdiag = corr.where(mask).stack()
                if not offdiag.empty:
                    row["max_abs_feature_corr"] = float(offdiag.max())
                    row["mean_abs_feature_corr"] = float(offdiag.mean())


                for feature in X.columns:
                    other = [c for c in X.columns if c != feature]
                    if not other:
                        continue
                    y = X[feature].to_numpy(dtype=float)
                    X_other = X[other].to_numpy(dtype=float)
                    model = LinearRegression(fit_intercept=True)
                    model.fit(X_other, y)
                    pred = model.predict(X_other)
                    sse = float(np.sum((y - pred) ** 2))
                    sst = float(np.sum((y - y.mean()) ** 2))
                    r2 = 1.0 - sse / sst if sst > 0 else np.nan
                    vif = 1.0 / (1.0 - r2) if np.isfinite(r2) and r2 < 1.0 else np.inf
                    vif_rows.append(
                        {
                            "gap_threshold_bd": gap_threshold,
                            "peer_variant": peer_variant,
                            "model": model_name,
                            "feature": feature,
                            "vif": float(vif),
                            "auxiliary_r2": float(r2) if np.isfinite(r2) else np.nan,
                            "diagnostic_rows": int(len(X)),
                        }
                    )

            design_rows.append(row)

    return pd.DataFrame(design_rows), pd.DataFrame(vif_rows)


def build_residual_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:

    if predictions.empty:
        return pd.DataFrame()

    df = predictions.loc[predictions[SPLIT_COL].isin(["validation", "test"])].copy()
    if df.empty:
        return pd.DataFrame()

    df["abs_residual_return"] = df["residual_return"].abs()
    rows = []

    def summarise(temp: pd.DataFrame, split: str, breakdown: str, bucket: str) -> dict:
        resid = temp["residual_return"].astype(float)
        abs_resid = resid.abs()
        return {
            "sample_split": split,
            "breakdown": breakdown,
            "bucket": bucket,
            "peer_variant": temp["peer_variant"].iloc[0] if "peer_variant" in temp.columns and len(temp) else np.nan,
            "model": temp["model"].iloc[0] if "model" in temp.columns and len(temp) else np.nan,
            "n_obs": int(len(temp)),
            "mean_residual": float(resid.mean()),
            "std_residual": float(resid.std()),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "mae": float(abs_resid.mean()),
            "p95_abs_residual": float(abs_resid.quantile(0.95)),
            "p99_abs_residual": float(abs_resid.quantile(0.99)),
        }

    for (split, peer_variant, model_name), g in df.groupby([SPLIT_COL, "peer_variant", "model"]):
        rows.append(summarise(g, split, "all", "all"))

        if ISSUER_COL in g.columns:
            for issuer, gi in g.groupby(ISSUER_COL, dropna=False):
                rows.append(summarise(gi, split, "issuer", str(issuer)))

        if "peer_maturity_bucket" in g.columns:
            for bucket, gb in g.groupby("peer_maturity_bucket", dropna=False):
                rows.append(summarise(gb, split, "maturity_bucket", str(bucket)))

    return pd.DataFrame(rows)


def build_test_residual_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    return build_residual_diagnostics(predictions)


def plot_balanced_gap5_oos_r2(balanced_results: pd.DataFrame, peer_variant: str = "raw") -> None:
    if balanced_results.empty:
        return
    df = balanced_results.loc[balanced_results["peer_variant"].eq(peer_variant)].copy()
    if df.empty:
        return
    df = df.sort_values("model_order")
    plt.figure(figsize=(7, 4.5))
    plt.bar(np.arange(len(df)), df["oos_r2_vs_bond_fe"])
    plt.xticks(np.arange(len(df)), [short_model_label(x) for x in df["model"]], rotation=30, ha="right")
    plt.ylabel("OOS R2 vs bond FE")
    plt.title("Balanced peer-sample model performance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"peer_balanced_gap5_oos_r2_{peer_variant}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_balanced_incremental_oos_r2(incremental_results: pd.DataFrame, peer_variant: str = "raw") -> None:
    if incremental_results.empty:
        return
    df = incremental_results.loc[incremental_results["peer_variant"].eq(peer_variant)].copy()
    df = df.loc[df["model_order"].gt(0)].sort_values("model_order")
    if df.empty:
        return
    plt.figure(figsize=(7, 4.5))
    plt.bar(np.arange(len(df)), df["delta_oos_r2_vs_previous"])
    plt.xticks(np.arange(len(df)), [short_model_label(x) for x in df["model"]], rotation=30, ha="right")
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("Incremental OOS R2 vs previous model")
    plt.title("Incremental contribution on balanced peer sample")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"peer_balanced_gap5_incremental_oos_r2_{peer_variant}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_test_actual_vs_fitted(predictions: pd.DataFrame, peer_variant: str = "raw") -> None:
    if predictions.empty:
        return
    model_name = f"M4_rates_equity_vix_peer_{peer_variant}"
    df = predictions.loc[
        predictions["sample_split"].eq("test")
        & predictions["peer_variant"].eq(peer_variant)
        & predictions["model"].eq(model_name)
    ].copy()
    if df.empty:
        return
    if len(df) > 50_000:
        df = df.sample(n=50_000, random_state=RANDOM_SEED)
    x = df["fitted_return"].astype(float)
    y = df[TARGET_CANDIDATES[0]].astype(float) if TARGET_CANDIDATES[0] in df.columns else df.iloc[:, 0]
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(x, y, s=4, alpha=0.10)
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("Fitted return")
    plt.ylabel("Realised return")
    plt.title("M4 test-set actual vs fitted returns")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"peer_m4_test_actual_vs_fitted_{peer_variant}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_test_absolute_residual_ecdf(predictions: pd.DataFrame, peer_variant: str = "raw") -> None:
    if predictions.empty:
        return
    models = [
        "M1_rates",
        "M3_rates_equity_vix",
        f"M4_rates_equity_vix_peer_{peer_variant}",
        f"M5_rates_equity_vix_peer_{peer_variant}_microstructure_clean",
    ]
    plt.figure(figsize=(7, 4.5))
    plotted = False
    for model_name in models:
        df = predictions.loc[
            predictions["sample_split"].eq("test")
            & predictions["peer_variant"].eq(peer_variant)
            & predictions["model"].eq(model_name)
        ].copy()
        if df.empty:
            continue
        x = np.sort(df["residual_return"].abs().dropna().to_numpy())
        if len(x) == 0:
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=short_model_label(model_name))
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("Absolute residual return")
    plt.ylabel("Empirical CDF")
    plt.title("Test-set absolute residual distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"peer_test_abs_residual_ecdf_{peer_variant}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_m4_rmse_by_issuer(predictions: pd.DataFrame, peer_variant: str = "raw") -> None:
    if predictions.empty or ISSUER_COL not in predictions.columns:
        return
    model_name = f"M4_rates_equity_vix_peer_{peer_variant}"
    df = predictions.loc[
        predictions["sample_split"].eq("test")
        & predictions["peer_variant"].eq(peer_variant)
        & predictions["model"].eq(model_name)
    ].copy()
    if df.empty:
        return
    rmse = (
        df.groupby(ISSUER_COL)["residual_return"]
        .apply(lambda x: np.sqrt(np.mean(np.square(x.astype(float)))))
        .sort_values()
    )
    plt.figure(figsize=(6.5, 4))
    plt.bar(np.arange(len(rmse)), rmse.to_numpy())
    plt.xticks(np.arange(len(rmse)), rmse.index.astype(str), rotation=0)
    plt.ylabel("Test RMSE")
    plt.title("M4 test RMSE by issuer")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"peer_m4_test_rmse_by_issuer_{peer_variant}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_results(results: pd.DataFrame, output_prefix: str) -> None:
    if results.empty:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    baseline_gap = MODEL_READY_MAX_BUSINESS_GAP
    df = results.loc[results["gap_threshold_bd"].eq(baseline_gap)].copy()

    if not df.empty:
        df = df.sort_values(["peer_variant", "model"])
        labels = df["peer_variant"].astype(str) + " | " + df["model"].astype(str)

        plt.figure(figsize=(11, 5))
        plt.bar(np.arange(len(df)), df["oos_r2_vs_bond_fe"])
        plt.xticks(np.arange(len(df)), labels, rotation=80, ha="right")
        plt.ylabel("OOS R2 vs bond FE")
        plt.title(f"Peer-factor model performance, gap <= {baseline_gap}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{output_prefix}_gap5_oos_r2.png", dpi=300, bbox_inches="tight")
        plt.close()

    df_no_m0 = results.loc[~results["model"].eq("M0_bond_fe_only")].copy()
    if not df_no_m0.empty:
        plt.figure(figsize=(9, 5))
        for (peer_variant, model_name), g in df_no_m0.groupby(["peer_variant", "model"]):
            if model_name.startswith("B"):
                continue
            g = g.sort_values("gap_threshold_bd")
            plt.plot(g["gap_threshold_bd"], g["oos_r2_vs_bond_fe"], marker="o", label=f"{peer_variant}: {model_name}")

        plt.axvline(MODEL_READY_MAX_BUSINESS_GAP, linestyle="--", linewidth=1)
        plt.xlabel("Maximum business-day gap")
        plt.ylabel("OOS R2 vs bond FE")
        plt.title("Peer-factor model performance by gap threshold")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{output_prefix}_gap_sensitivity_oos_r2.png", dpi=300, bbox_inches="tight")
        plt.close()


def write_manifest(output_paths: list[Path]) -> None:
    manifest = {
        "script": "peer_factor_models.py",
        "train_end_date": str(TRAIN_END_DATE.date()),
        "validation_end_date": str(VALIDATION_END_DATE.date()),
        "sample_split_definition": {
            "train": f"date < {TRAIN_END_DATE.date()}",
            "validation": f"{TRAIN_END_DATE.date()} <= date < {VALIDATION_END_DATE.date()}",
            "test": f"date >= {VALIDATION_END_DATE.date()}",
        },
        "model_peer_variants": MODEL_PEER_VARIANTS,
        "diagnostic_peer_variants": DIAGNOSTIC_PEER_VARIANTS,
        "model_ready_max_business_gap": MODEL_READY_MAX_BUSINESS_GAP,
        "gap_thresholds": GAP_THRESHOLDS,
        "max_peer_price_staleness_bd": MAX_PEER_PRICE_STALENESS_BD,
        "min_peers": MIN_PEERS,
        "peer_base_names": PEER_BASE_NAMES,
        "peer_variants": PEER_VARIANTS,
        "core_model_ordering": [
            "M0 = bond fixed effects only",
            "M1 = rates",
            "M2 = rates + issuer equity",
            "M3 = rates + issuer equity + VIX",
            "M4 = rates + issuer equity + VIX + peer factors",
            "M5 = M4 + clean microstructure / price-quality controls",
        ],
        "additional_figures": [
            "peer_balanced_gap5_oos_r2_raw.png",
            "peer_balanced_gap5_incremental_oos_r2_raw.png",
            "peer_m4_test_actual_vs_fitted_raw.png",
            "peer_test_abs_residual_ecdf_raw.png",
            "peer_m4_test_rmse_by_issuer_raw.png",
        ],
        "outputs": [str(p) for p in output_paths],
    }

    path = TABLES_DIR / "peer_model_manifest.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)



def main() -> None:
    ensure_directories()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading regression panel:", REGRESSION_PANEL_PATH)
    panel = pd.read_parquet(REGRESSION_PANEL_PATH)
    panel = prepare_panel(panel)
    panel, log_price_col = choose_log_price_column(panel)
    panel, peer_meta = add_static_peer_metadata(panel)

    target = choose_target(panel)

    print("Target:", target)
    print("Log price column for peer construction:", log_price_col)
    print("Rows:", len(panel))
    print("CUSIPs:", panel[GROUP_COL].nunique())
    print("Date range:", panel[DATE_COL].min(), "to", panel[DATE_COL].max())

    peer_meta_path = TABLES_DIR / "peer_static_metadata.csv"
    peer_meta.to_csv(peer_meta_path, index=False)

    print("\nBuilding raw peer factors...")
    panel = build_raw_peer_factors(
        panel=panel,
        log_price_col=log_price_col,
        max_staleness_bd=MAX_PEER_PRICE_STALENESS_BD,
        min_peers=MIN_PEERS,
    )

    print("\nAdding residualised peer factors...")
    residualisation_parts = []

    panel, resid_rates_coefs = add_residualised_peer_factors(
        panel=panel,
        base_features=available(RATES_FEATURES, panel),
        variant_name="resid_rates",
    )
    residualisation_parts.append(resid_rates_coefs)

    panel, resid_standard_coefs = add_residualised_peer_factors(
        panel=panel,
        base_features=available(RATES_FEATURES + EQUITY_FEATURES + VIX_FEATURES, panel),
        variant_name="resid_standard",
    )
    residualisation_parts.append(resid_standard_coefs)

    residualisation_coefs = pd.concat(residualisation_parts, ignore_index=True)
    residualisation_coefs_path = TABLES_DIR / "peer_residualisation_coefficients.csv"
    residualisation_coefs.to_csv(residualisation_coefs_path, index=False)

    peer_panel_path = REGRESSION_DIR / "regression_panel_with_peer_factors.parquet"
    panel.to_parquet(peer_panel_path, index=False)

    baseline_flag = f"valid_return_gap_{MODEL_READY_MAX_BUSINESS_GAP}bd"
    if baseline_flag not in panel.columns:
        raise ValueError(f"Missing baseline gap flag: {baseline_flag}")

    peer_panel_gap5_path = REGRESSION_DIR / "regression_panel_gap5_with_peer_factors.parquet"
    panel.loc[panel[baseline_flag]].copy().to_parquet(peer_panel_gap5_path, index=False)

    print("\nWriting peer factor diagnostics...")
    peer_coverage = build_peer_factor_coverage(panel)
    peer_coverage_path = TABLES_DIR / "peer_factor_coverage.csv"
    peer_coverage.to_csv(peer_coverage_path, index=False)

    feature_corr = build_feature_correlation(panel.loc[panel[baseline_flag]].copy(), target=target)
    feature_corr_path = TABLES_DIR / "peer_model_feature_correlation_gap5.csv"
    feature_corr.to_csv(feature_corr_path)

    print("\nRunning baseline gap5 peer models...")
    baseline_result_parts = []
    baseline_coef_parts = []
    baseline_pred_parts = []

    for peer_variant in MODEL_PEER_VARIANTS:
        specs = get_model_specs(panel, peer_variant=peer_variant)
        results, coefs, preds = run_model_specs(
            panel=panel,
            target=target,
            model_specs=specs,
            gap_threshold=MODEL_READY_MAX_BUSINESS_GAP,
            peer_variant=peer_variant,
        )
        baseline_result_parts.append(results)
        baseline_coef_parts.append(coefs)
        baseline_pred_parts.append(preds)

    baseline_results = pd.concat(baseline_result_parts, ignore_index=True)
    baseline_coefs = pd.concat(baseline_coef_parts, ignore_index=True)
    baseline_preds = pd.concat(baseline_pred_parts, ignore_index=True)

    baseline_results_path = TABLES_DIR / "peer_baseline_gap5_model_results.csv"
    baseline_coefs_path = TABLES_DIR / "peer_baseline_gap5_model_coefficients.csv"
    baseline_preds_path = REGRESSION_DIR / "peer_baseline_gap5_model_predictions.parquet"

    baseline_results.to_csv(baseline_results_path, index=False)
    baseline_coefs.to_csv(baseline_coefs_path, index=False)
    baseline_preds.to_parquet(baseline_preds_path, index=False)

    validation_selection = build_peer_validation_selection(baseline_results)
    validation_selection_path = TABLES_DIR / "peer_validation_model_selection_gap5.csv"
    validation_selection.to_csv(validation_selection_path, index=False)

    model_estimation_summary = build_model_estimation_summary(baseline_results, target=target)
    model_estimation_summary_path = TABLES_DIR / "peer_model_estimation_summary_gap5.csv"
    model_estimation_summary.to_csv(model_estimation_summary_path, index=False)

    print("\nRunning equity proxy horse-race on raw peer variant...")

    equity_hr_panel = restrict_to_equity_horse_race_common_sample(
        panel=panel,
        peer_variant="raw",
    )

    equity_horse_race_specs = get_equity_proxy_horse_race_specs(
        panel=equity_hr_panel,
        peer_variant="raw",
    )

    equity_hr_results, equity_hr_coefs, equity_hr_preds = run_model_specs(
        panel=equity_hr_panel,
        target=target,
        model_specs=equity_horse_race_specs,
        gap_threshold=MODEL_READY_MAX_BUSINESS_GAP,
        peer_variant="raw",
    )

    equity_hr_results_path = TABLES_DIR / "equity_proxy_horse_race_gap5_results.csv"
    equity_hr_coefs_path = TABLES_DIR / "equity_proxy_horse_race_gap5_coefficients.csv"
    equity_hr_preds_path = REGRESSION_DIR / "equity_proxy_horse_race_gap5_predictions.parquet"

    equity_hr_results.to_csv(equity_hr_results_path, index=False)
    equity_hr_coefs.to_csv(equity_hr_coefs_path, index=False)
    equity_hr_preds.to_parquet(equity_hr_preds_path, index=False)

    print(f"Saved equity proxy horse-race results to: {equity_hr_results_path}")
    print(f"Saved equity proxy horse-race coefficients to: {equity_hr_coefs_path}")
    print(f"Saved equity proxy horse-race predictions to: {equity_hr_preds_path}")

    print("\nRunning balanced peer-sample comparison...")
    balanced_results, balanced_coefs = run_balanced_peer_sample_comparison(
        panel=panel,
        target=target,
        gap_threshold=MODEL_READY_MAX_BUSINESS_GAP,
        peer_variants=MODEL_PEER_VARIANTS,
    )

    balanced_results_path = TABLES_DIR / "peer_balanced_gap5_model_results.csv"
    balanced_coefs_path = TABLES_DIR / "peer_balanced_gap5_model_coefficients.csv"
    balanced_results.to_csv(balanced_results_path, index=False)
    balanced_coefs.to_csv(balanced_coefs_path, index=False)
    balanced_validation_selection = build_peer_validation_selection(balanced_results)
    balanced_validation_selection_path = TABLES_DIR / "peer_balanced_validation_model_selection_gap5.csv"
    balanced_validation_selection.to_csv(balanced_validation_selection_path, index=False)

    baseline_incremental = build_incremental_contribution_summary(baseline_results)
    balanced_incremental = build_incremental_contribution_summary(balanced_results)
    baseline_incremental_path = TABLES_DIR / "peer_baseline_gap5_incremental_contribution.csv"
    balanced_incremental_path = TABLES_DIR / "peer_balanced_gap5_incremental_contribution.csv"
    baseline_incremental.to_csv(baseline_incremental_path, index=False)
    balanced_incremental.to_csv(balanced_incremental_path, index=False)

    print("\nRunning feature-design diagnostics...")
    design_diagnostics, vif_diagnostics = build_design_diagnostics(
        panel=panel,
        target=target,
        gap_threshold=MODEL_READY_MAX_BUSINESS_GAP,
        peer_variants=DIAGNOSTIC_PEER_VARIANTS,
    )
    design_diagnostics_path = TABLES_DIR / "peer_design_diagnostics_gap5.csv"
    vif_diagnostics_path = TABLES_DIR / "peer_vif_diagnostics_gap5.csv"
    design_diagnostics.to_csv(design_diagnostics_path, index=False)
    vif_diagnostics.to_csv(vif_diagnostics_path, index=False)

    residual_diagnostics = build_residual_diagnostics(baseline_preds)
    residual_diagnostics_path = TABLES_DIR / "peer_validation_test_residual_diagnostics_gap5.csv"
    residual_diagnostics.to_csv(residual_diagnostics_path, index=False)

    plot_balanced_gap5_oos_r2(balanced_results, peer_variant="raw")
    plot_balanced_incremental_oos_r2(balanced_incremental, peer_variant="raw")
    plot_test_actual_vs_fitted(baseline_preds, peer_variant="raw")
    plot_test_absolute_residual_ecdf(baseline_preds, peer_variant="raw")
    plot_m4_rmse_by_issuer(baseline_preds, peer_variant="raw")

    print("\nRunning gap sensitivity peer models...")
    gap_results, gap_coefs = run_gap_sensitivity(
        panel=panel,
        target=target,
        peer_variants=MODEL_PEER_VARIANTS,
    )

    gap_results_path = TABLES_DIR / "peer_gap_sensitivity_model_results.csv"
    gap_coefs_path = TABLES_DIR / "peer_gap_sensitivity_model_coefficients.csv"
    gap_results.to_csv(gap_results_path, index=False)
    gap_coefs.to_csv(gap_coefs_path, index=False)

    plot_model_results(gap_results, output_prefix="peer_models")

    output_paths = [
        peer_meta_path,
        residualisation_coefs_path,
        peer_panel_path,
        peer_panel_gap5_path,
        peer_coverage_path,
        feature_corr_path,
        baseline_results_path,
        baseline_coefs_path,
        balanced_validation_selection_path,
        baseline_preds_path,
        validation_selection_path,
        model_estimation_summary_path,
        balanced_results_path,
        balanced_coefs_path,
        baseline_incremental_path,
        balanced_incremental_path,
        design_diagnostics_path,
        vif_diagnostics_path,
        residual_diagnostics_path,
        gap_results_path,
        gap_coefs_path,
    ]
    write_manifest(output_paths)

    print("\nPEER FACTOR COVERAGE")
    print(peer_coverage)

    print("\nBASELINE GAP5 MODEL RESULTS")
    cols = [
        "peer_variant",
        "model",
        "sample_rows",
        "train_rows",
        "validation_rows",
        "test_rows",
        "validation_rmse",
        "validation_mae",
        "validation_oos_r2_vs_bond_fe",
        "test_rmse",
        "test_mae",
        "test_oos_r2_vs_bond_fe",
    ]
    print(baseline_results[[c for c in cols if c in baseline_results.columns]])

    print("\nBALANCED GAP5 MODEL RESULTS")
    print(balanced_results[[c for c in cols if c in balanced_results.columns]])

    print("\nBALANCED GAP5 INCREMENTAL CONTRIBUTION")
    inc_cols = [
        "peer_variant",
        "model",
        "previous_model",
        "oos_r2_vs_bond_fe",
        "delta_oos_r2_vs_previous",
        "test_rmse",
        "delta_test_rmse_vs_previous",
    ]
    print(balanced_incremental[[c for c in inc_cols if c in balanced_incremental.columns]])

    print("\nSaved outputs:")
    for path in output_paths:
        print("-", path)


if __name__ == "__main__":
    main()
