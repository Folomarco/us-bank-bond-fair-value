from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config_institutional import (
    REGRESSION_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    ensure_directories,
)


GROUP_COL = "cusip_id"
DATE_COL = "date"
SPLIT_COL = "sample_split"
TARGET_COL = "final_vwap_return"
ISSUER_COL = "trace_company_symbol"
MATURITY_BUCKET_COL = "peer_maturity_bucket"

TRAIN_END_DATE = pd.Timestamp("2023-01-01")
VALIDATION_END_DATE = pd.Timestamp("2024-01-01")

PANEL_PATH = REGRESSION_DIR / "regression_panel_gap5_with_peer_factors.parquet"

OUTPUT_CANDIDATES = TABLES_DIR / "dynamic_state_space_extended_candidate_results.csv"
OUTPUT_SELECTED_RESULTS = TABLES_DIR / "dynamic_state_space_extended_selected_model_results.csv"
OUTPUT_RESIDUAL_DIAGNOSTICS = TABLES_DIR / "dynamic_state_space_extended_residual_diagnostics.csv"
OUTPUT_COEFFICIENTS = TABLES_DIR / "dynamic_state_space_extended_coefficients_by_date.csv"
OUTPUT_PREDICTIONS = REGRESSION_DIR / "dynamic_state_space_extended_gap5_model_predictions.parquet"
OUTPUT_MANIFEST = TABLES_DIR / "dynamic_state_space_extended_manifest.json"

FIG_VALIDATION_RMSE = FIGURES_DIR / "dynamic_state_space_extended_validation_rmse.png"
FIG_TEST_OOS_R2 = FIGURES_DIR / "dynamic_state_space_extended_test_oos_r2_vs_expanding_m0.png"
FIG_COEFFICIENTS_GLOBAL = FIGURES_DIR / "dynamic_state_space_extended_global_coefficients_over_time.png"
FIG_ABS_RESIDUAL_ECDF = FIGURES_DIR / "dynamic_state_space_extended_abs_residual_ecdf.png"

RATES_FEATURES = [
    "d_dgs2_interval",
    "d_dgs5_interval",
    "d_dgs10_interval",
    "d_dgs30_interval",
]

EQUITY_FEATURES = [
    "issuer_equity_log_return_interval",
]

VIX_FEATURES = [
    "d_vix_interval",
]

PEER_FEATURES_RAW = [
    "peer_raw_same_issuer_maturity",
    "peer_raw_other_bank_maturity",
    "peer_raw_bank_sector_maturity",
]

M4_FEATURES = RATES_FEATURES + EQUITY_FEATURES + VIX_FEATURES + PEER_FEATURES_RAW

DISCOUNT_FACTORS = [1.0, 0.999, 0.995, 0.990, 0.985, 0.980, 0.970, 0.950]
OBS_VARIANCE_MULTIPLIERS = [0.5, 1.0, 2.0, 4.0]

RIDGE_EPS = 1e-10
COV_FLOOR = 1e-12
INITIAL_COV_MULTIPLIER = 10.0
MIN_OBS_PER_GLOBAL_UPDATE_DATE = 25
MIN_OBS_PER_GROUP_UPDATE_DATE = 8
MIN_GROUP_TRAIN_OBS = 2_000
MIN_STATIC_WINDOW_OBS = 20_000
MIN_STATIC_WINDOW_CUSIPS = 200

SAVE_TRAIN_PREDICTIONS = False



def json_safe(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def assign_sample_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")

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


def assert_required_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "n_obs": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "sse": sse,
        "mean_residual": float(np.mean(residual)),
        "std_residual": float(np.std(residual, ddof=1)),
        "mean_abs_residual": float(np.mean(np.abs(residual))),
        "p95_abs_residual": float(np.quantile(np.abs(residual), 0.95)),
        "p99_abs_residual": float(np.quantile(np.abs(residual), 0.99)),
        "p01_residual": float(np.quantile(residual, 0.01)),
        "p05_residual": float(np.quantile(residual, 0.05)),
        "p50_residual": float(np.quantile(residual, 0.50)),
        "p95_residual": float(np.quantile(residual, 0.95)),
        "p99_residual": float(np.quantile(residual, 0.99)),
    }


def oos_r2_from_sse(model_sse: float, benchmark_sse: float) -> float:
    if benchmark_sse > 0:
        return float(1.0 - model_sse / benchmark_sse)
    return np.nan


def residual_autocorr_by_cusip(df: pd.DataFrame, residual_col: str) -> float:
    values = []

    for _, g in df.sort_values([GROUP_COL, DATE_COL]).groupby(GROUP_COL):
        r = pd.to_numeric(g[residual_col], errors="coerce").dropna()

        if len(r) < 3:
            continue

        if float(r.std(ddof=1)) == 0.0:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ac = r.autocorr(lag=1)

        if pd.notna(ac) and np.isfinite(ac):
            values.append(float(ac))

    if not values:
        return np.nan

    return float(np.mean(values))


def get_month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out[GROUP_COL] = out[GROUP_COL].astype(str).str.strip()

    if ISSUER_COL in out.columns:
        out[ISSUER_COL] = out[ISSUER_COL].astype(str).str.upper().str.strip()
    else:
        out[ISSUER_COL] = "UNKNOWN"

    if MATURITY_BUCKET_COL not in out.columns:
        if "years_to_maturity" in out.columns:
            bins = [1, 3, 5, 7, 10, 15, 30, np.inf]
            labels = ["1-3y", "3-5y", "5-7y", "7-10y", "10-15y", "15-30y", "30y+"]
            out[MATURITY_BUCKET_COL] = pd.cut(
                pd.to_numeric(out["years_to_maturity"], errors="coerce"),
                bins=bins,
                labels=labels,
                right=False,
            ).astype(str)
        else:
            out[MATURITY_BUCKET_COL] = "unknown_maturity"

    out[MATURITY_BUCKET_COL] = out[MATURITY_BUCKET_COL].astype(str).replace({"nan": "unknown_maturity"})

    for col in [TARGET_COL] + M4_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return assign_sample_split(out)


def load_model_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing peer-factor regression panel: {PANEL_PATH}\n"
            "Run peer_factor_models.py first."
        )

    panel = pd.read_parquet(PANEL_PATH)
    panel = prepare_panel(panel)

    required = [GROUP_COL, DATE_COL, SPLIT_COL, TARGET_COL, ISSUER_COL, MATURITY_BUCKET_COL] + M4_FEATURES
    assert_required_columns(panel, required, "extended dynamic state-space panel")

    complete = panel.dropna(subset=[TARGET_COL] + M4_FEATURES).copy()
    complete = complete.sort_values([DATE_COL, GROUP_COL]).reset_index(drop=True)
    complete["_sample_index"] = np.arange(len(complete))
    complete["_month_start"] = get_month_start(complete[DATE_COL])

    return complete



def fit_fe_context(
    train: pd.DataFrame,
    features: list[str],
    target: str = TARGET_COL,
    group_col: str = GROUP_COL,
) -> dict[str, Any]:
    train = train.copy()

    y_means = train.groupby(group_col)[target].mean()

    if features:
        x_means = train.groupby(group_col)[features].mean()
        global_x_mean = train[features].mean()
    else:
        x_means = pd.DataFrame(index=y_means.index)
        global_x_mean = pd.Series(dtype=float)

    return {
        "features": features,
        "target": target,
        "group_col": group_col,
        "y_means": y_means,
        "x_means": x_means,
        "global_y_mean": float(train[target].mean()),
        "global_x_mean": global_x_mean,
        "n_obs": int(len(train)),
        "n_cusips": int(train[group_col].nunique()),
        "window_start": str(train[DATE_COL].min().date()) if len(train) else None,
        "window_end_exclusive": None,
    }


def y_base(df: pd.DataFrame, context: dict[str, Any]) -> np.ndarray:
    return (
        df[context["group_col"]]
        .map(context["y_means"])
        .fillna(context["global_y_mean"])
        .to_numpy(dtype=float)
    )


def center_y(df: pd.DataFrame, context: dict[str, Any]) -> np.ndarray:
    y = pd.to_numeric(df[context["target"]], errors="coerce").to_numpy(dtype=float)
    return y - y_base(df, context)


def center_x(df: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    features = context["features"]

    if not features:
        return pd.DataFrame(index=df.index)

    group_col = context["group_col"]
    X = df[features].astype(float)
    X_bar = pd.DataFrame(index=df.index)

    for feature in features:
        X_bar[feature] = (
            df[group_col]
            .map(context["x_means"][feature])
            .fillna(context["global_x_mean"][feature])
        )

    return X - X_bar


def build_expanding_contexts(sample: pd.DataFrame, eval_months: list[pd.Timestamp]) -> dict[pd.Timestamp, dict[str, Any]]:
    contexts: dict[pd.Timestamp, dict[str, Any]] = {}

    for month_start in eval_months:
        hist = sample.loc[sample[DATE_COL] < month_start].copy()

        if len(hist) < MIN_STATIC_WINDOW_OBS or hist[GROUP_COL].nunique() < MIN_STATIC_WINDOW_CUSIPS:
            raise ValueError(
                f"Insufficient expanding history before {month_start.date()}: "
                f"n_obs={len(hist)}, n_cusips={hist[GROUP_COL].nunique()}"
            )

        ctx_m0 = fit_fe_context(hist, features=[])
        ctx_m4 = fit_fe_context(hist, features=M4_FEATURES)
        ctx_m0["window_end_exclusive"] = str(month_start.date())
        ctx_m4["window_end_exclusive"] = str(month_start.date())

        contexts[month_start] = {
            "m0": ctx_m0,
            "m4": ctx_m4,
            "window": hist,
            "window_start": str(hist[DATE_COL].min().date()),
            "window_end_exclusive": str(month_start.date()),
        }

    return contexts



def fit_fe_ols(train: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
    features = context["features"]

    if not features:
        return {"context": context, "coef": pd.Series(dtype=float), "features": []}

    Xc = center_x(train, context)[features].to_numpy(dtype=float)
    yc = center_y(train, context)

    beta, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
    coef = pd.Series(beta, index=features, name="coefficient")

    return {"context": context, "coef": coef, "features": features}


def predict_fe_ols(df: pd.DataFrame, fitted: dict[str, Any]) -> np.ndarray:
    context = fitted["context"]
    features = fitted["features"]
    base = y_base(df, context)

    if not features:
        return base

    Xc = center_x(df, context)[features].to_numpy(dtype=float)
    beta = fitted["coef"].loc[features].to_numpy(dtype=float)
    return base + Xc @ beta


def make_prediction_frame(
    df: pd.DataFrame,
    model_name: str,
    model_family: str,
    fitted_value: np.ndarray,
    own_m0_fitted_value: np.ndarray,
    own_m0_model: str,
    evaluation_protocol: str,
    refit_month: str | None = None,
    window_start: str | None = None,
    window_end_exclusive: str | None = None,
) -> pd.DataFrame:
    keep_cols = ["_sample_index", GROUP_COL, DATE_COL, SPLIT_COL, ISSUER_COL, MATURITY_BUCKET_COL]
    keep_cols = [c for c in keep_cols if c in df.columns]

    out = df[keep_cols].copy()
    out["model"] = model_name
    out["model_family"] = model_family
    out["target_value"] = df[TARGET_COL].to_numpy(dtype=float)
    out["fitted_value"] = fitted_value
    out["residual"] = out["target_value"] - out["fitted_value"]
    out["own_m0_fitted_value"] = own_m0_fitted_value
    out["own_m0_model"] = own_m0_model
    out["evaluation_protocol"] = evaluation_protocol
    out["refit_month"] = refit_month
    out["window_start"] = window_start
    out["window_end_exclusive"] = window_end_exclusive
    return out


def build_expanding_static_predictions(
    sample: pd.DataFrame,
    contexts: dict[pd.Timestamp, dict[str, Any]],
    eval_months: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_parts = []
    coef_rows = []

    for month_start in eval_months:
        month_rows = sample.loc[sample["_month_start"].eq(month_start)].copy()
        month_rows = month_rows.loc[month_rows[SPLIT_COL].isin(["validation", "test"])]
        if month_rows.empty:
            continue

        ctx_pack = contexts[month_start]
        ctx_m0 = ctx_pack["m0"]
        ctx_m4 = ctx_pack["m4"]
        hist = ctx_pack["window"]

        m0_pred = y_base(month_rows, ctx_m0)
        prediction_parts.append(
            make_prediction_frame(
                df=month_rows,
                model_name="Expanding_M0_FE",
                model_family="expanding_fixed_effects",
                fitted_value=m0_pred,
                own_m0_fitted_value=m0_pred,
                own_m0_model="Expanding_M0_FE",
                evaluation_protocol="monthly_expanding",
                refit_month=str(month_start.date()),
                window_start=ctx_pack["window_start"],
                window_end_exclusive=ctx_pack["window_end_exclusive"],
            )
        )

        fitted_m4 = fit_fe_ols(hist, ctx_m4)
        m4_pred = predict_fe_ols(month_rows, fitted_m4)
        prediction_parts.append(
            make_prediction_frame(
                df=month_rows,
                model_name="Expanding_OLS_M4",
                model_family="expanding_fixed_effects",
                fitted_value=m4_pred,
                own_m0_fitted_value=m0_pred,
                own_m0_model="Expanding_M0_FE",
                evaluation_protocol="monthly_expanding",
                refit_month=str(month_start.date()),
                window_start=ctx_pack["window_start"],
                window_end_exclusive=ctx_pack["window_end_exclusive"],
            )
        )

        for feature, coef in fitted_m4["coef"].items():
            coef_rows.append(
                {
                    "model": "Expanding_OLS_M4",
                    "model_family": "expanding_fixed_effects",
                    "date": month_start,
                    "state_group": "all",
                    "feature": feature,
                    "coefficient": float(coef),
                    "window_start": ctx_pack["window_start"],
                    "window_end_exclusive": ctx_pack["window_end_exclusive"],
                    "n_obs": int(len(hist)),
                    "n_cusips": int(hist[GROUP_COL].nunique()),
                }
            )

    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    coefficients = pd.DataFrame(coef_rows)
    return predictions, coefficients



def fit_static_ols_standardised(train: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
    features = context["features"]
    Xc_df = center_x(train, context)[features]
    yc = center_y(train, context)

    scale = Xc_df.std(axis=0, ddof=1).replace(0.0, np.nan)
    if scale.isna().any():
        bad = scale[scale.isna()].index.tolist()
        raise ValueError(f"Cannot standardise features with zero/NaN scale: {bad}")

    Xs = (Xc_df / scale).to_numpy(dtype=float)
    beta_std, _, _, _ = np.linalg.lstsq(Xs, yc, rcond=None)

    fitted_centered = Xs @ beta_std
    residual = yc - fitted_centered
    resid_var = float(np.var(residual, ddof=len(features)))

    xpx = Xs.T @ Xs
    beta_cov_std = resid_var * np.linalg.pinv(xpx + RIDGE_EPS * np.eye(len(features)))

    return {
        "beta_std": beta_std,
        "beta_original": pd.Series(beta_std / scale.to_numpy(dtype=float), index=features),
        "beta_cov_std": beta_cov_std,
        "feature_scale": scale,
        "residual_variance": resid_var,
        "train_residual": residual,
    }


def initialise_group_states(
    sample: pd.DataFrame,
    group_cols: list[str],
    train_context: dict[str, Any],
    global_init: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[tuple, str]]:
    train = sample.loc[sample[SPLIT_COL].eq("train")].copy()
    features = train_context["features"]
    feature_scale = global_init["feature_scale"]

    states: dict[str, dict[str, Any]] = {}
    group_map: dict[tuple, str] = {}

    global_state_key = "GLOBAL_FALLBACK"
    states[global_state_key] = {
        "beta": global_init["beta_std"].copy(),
        "cov": global_init["beta_cov_std"].copy() * INITIAL_COV_MULTIPLIER,
        "n_train": int(len(train)),
        "source": "global_fallback",
    }

    if not group_cols:
        states = {
            "GLOBAL": {
                "beta": global_init["beta_std"].copy(),
                "cov": global_init["beta_cov_std"].copy() * INITIAL_COV_MULTIPLIER,
                "n_train": int(len(train)),
                "source": "global",
            }
        }
        group_map[("GLOBAL",)] = "GLOBAL"
        return states, group_map

    for key, g in train.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        state_key = "|".join(str(x) for x in key_tuple)

        if len(g) < MIN_GROUP_TRAIN_OBS:
            group_map[key_tuple] = global_state_key
            continue

        state_source = "group_specific"

        try:
            ctx = fit_fe_context(g, features=features)
            Xc_df = center_x(g, ctx)[features]
            yc = center_y(g, ctx)
            Xs = (Xc_df / feature_scale).to_numpy(dtype=float)
            beta_std, _, _, _ = np.linalg.lstsq(Xs, yc, rcond=None)
            residual = yc - Xs @ beta_std
            resid_var = float(np.var(residual, ddof=len(features)))
            xpx = Xs.T @ Xs
            cov = resid_var * np.linalg.pinv(
                xpx + RIDGE_EPS * np.eye(len(features))
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            warnings.warn(
                f"Falling back to the global initial state for {state_key}: {exc}",
                RuntimeWarning,
            )
            beta_std = global_init["beta_std"].copy()
            cov = global_init["beta_cov_std"].copy()
            state_source = "global_fallback_after_initialisation_error"

        states[state_key] = {
            "beta": np.asarray(beta_std, dtype=float).copy(),
            "cov": np.asarray(cov, dtype=float).copy() * INITIAL_COV_MULTIPLIER,
            "n_train": int(len(g)),
            "source": state_source,
        }

        group_map[key_tuple] = state_key

    return states, group_map


def state_key_for_row(row: pd.Series, group_cols: list[str], group_map: dict[tuple, str]) -> str:
    if not group_cols:
        return "GLOBAL"
    key_tuple = tuple(row[c] for c in group_cols)
    return group_map.get(key_tuple, "GLOBAL_FALLBACK")


def kalman_batch_update(
    beta_prior: np.ndarray,
    cov_prior: np.ndarray,
    H: np.ndarray,
    y: np.ndarray,
    obs_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    p = len(beta_prior)
    obs_var = max(float(obs_var), COV_FLOOR)

    prior_precision = np.linalg.pinv(cov_prior + COV_FLOOR * np.eye(p))
    data_precision = (H.T @ H) / obs_var

    post_precision = prior_precision + data_precision + RIDGE_EPS * np.eye(p)
    post_cov = np.linalg.pinv(post_precision)

    rhs = prior_precision @ beta_prior + (H.T @ y) / obs_var
    beta_post = post_cov @ rhs

    post_cov = 0.5 * (post_cov + post_cov.T)
    return beta_post, post_cov




def kalman_predictive_loglik_low_rank(
    beta_prior: np.ndarray,
    cov_prior: np.ndarray,
    H: np.ndarray,
    y: np.ndarray,
    obs_var: float,
) -> float:
    if H.size == 0 or y.size == 0:
        return np.nan

    obs_var = max(float(obs_var), COV_FLOOR)
    p = len(beta_prior)
    n = int(len(y))

    P = 0.5 * (np.asarray(cov_prior, dtype=float) + np.asarray(cov_prior, dtype=float).T)
    P = P + COV_FLOOR * np.eye(p)
    e = np.asarray(y, dtype=float) - H @ np.asarray(beta_prior, dtype=float)

    P_inv = np.linalg.pinv(P)
    A = P_inv + (H.T @ H) / obs_var
    A = 0.5 * (A + A.T) + RIDGE_EPS * np.eye(p)

    sign_p, logdet_p = np.linalg.slogdet(P)
    sign_a, logdet_a = np.linalg.slogdet(A)
    if sign_p <= 0 or sign_a <= 0:
        return np.nan

    hte = H.T @ e
    try:
        solved = np.linalg.solve(A, hte)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(A) @ hte

    quad = float((e @ e) / obs_var - (hte @ solved) / (obs_var ** 2))
    logdet_s = float(n * np.log(obs_var) + logdet_p + logdet_a)
    return float(-0.5 * (n * np.log(2.0 * np.pi) + logdet_s + quad))


def predict_with_context(
    df: pd.DataFrame,
    context: dict[str, Any],
    beta_std: np.ndarray,
    feature_scale: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = y_base(df, context)
    Xc = center_x(df, context)[context["features"]]
    Xs = (Xc / feature_scale).to_numpy(dtype=float)
    pred = base + Xs @ beta_std
    y_centered = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=float) - base
    return pred, Xs, y_centered


def run_kalman_expanding_fe(
    sample: pd.DataFrame,
    contexts: dict[pd.Timestamp, dict[str, Any]],
    eval_months: list[pd.Timestamp],
    group_cols: list[str],
    model_name: str,
    discount_factor: float,
    obs_variance_multiplier: float,
    global_init: dict[str, Any],
    train_context: dict[str, Any],
    evaluation_splits: list[str],
    save_coefficients: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_scale = global_init["feature_scale"]
    base_obs_var = global_init["residual_variance"]
    obs_var = max(base_obs_var * obs_variance_multiplier, COV_FLOOR)

    states, group_map = initialise_group_states(
        sample=sample,
        group_cols=group_cols,
        train_context=train_context,
        global_init=global_init,
    )

    prediction_parts = []
    coef_rows = []

    eval_sample = sample.loc[sample[SPLIT_COL].isin(evaluation_splits)].copy()
    eval_sample = eval_sample.loc[eval_sample["_month_start"].isin(eval_months)].copy()
    eval_sample = eval_sample.sort_values([DATE_COL, GROUP_COL]).copy()

    for date_value, day in eval_sample.groupby(DATE_COL, sort=True):
        month_start = pd.Timestamp(day["_month_start"].iloc[0])
        ctx_pack = contexts[month_start]
        ctx_m0 = ctx_pack["m0"]
        ctx_m4 = ctx_pack["m4"]

        day = day.copy()
        day["_state_key"] = day.apply(lambda row: state_key_for_row(row, group_cols, group_map), axis=1)

        for state_key, gd in day.groupby("_state_key", sort=False):
            if state_key not in states:
                state_key = "GLOBAL_FALLBACK"

            state = states[state_key]
            beta_prior = state["beta"].copy()
            cov_prior = state["cov"].copy()

            if discount_factor < 1.0:
                cov_prior = cov_prior / max(discount_factor, 1e-6)

            pred, H, y_centered = predict_with_context(
                df=gd,
                context=ctx_m4,
                beta_std=beta_prior,
                feature_scale=feature_scale,
            )
            m0_pred = y_base(gd, ctx_m0)

            min_update = MIN_OBS_PER_GLOBAL_UPDATE_DATE if not group_cols else MIN_OBS_PER_GROUP_UPDATE_DATE
            valid = np.isfinite(y_centered) & np.isfinite(H).all(axis=1)
            n_update = int(valid.sum())
            predictive_loglik = np.nan
            predictive_loglik_per_obs = np.nan
            predictive_rmse = np.nan

            if n_update >= min_update:
                H_valid = H[valid]
                y_valid = y_centered[valid]
                predictive_loglik = kalman_predictive_loglik_low_rank(
                    beta_prior=beta_prior,
                    cov_prior=cov_prior,
                    H=H_valid,
                    y=y_valid,
                    obs_var=obs_var,
                )
                predictive_loglik_per_obs = float(predictive_loglik / n_update) if np.isfinite(predictive_loglik) else np.nan
                innovation = y_valid - H_valid @ beta_prior
                predictive_rmse = float(np.sqrt(np.mean(innovation ** 2)))

            pred_frame = make_prediction_frame(
                df=gd,
                model_name=model_name,
                model_family="state_space_kalman_expanding_fe",
                fitted_value=pred,
                own_m0_fitted_value=m0_pred,
                own_m0_model="Expanding_M0_FE",
                evaluation_protocol="prequential_monthly_expanding_fe",
                refit_month=str(month_start.date()),
                window_start=ctx_pack["window_start"],
                window_end_exclusive=ctx_pack["window_end_exclusive"],
            )
            pred_frame["state_key"] = state_key
            pred_frame["n_obs_state_date"] = int(len(gd))
            pred_frame["n_obs_used_for_update"] = n_update
            pred_frame["predictive_loglik_state_date"] = predictive_loglik
            pred_frame["predictive_loglik_per_obs_state_date"] = predictive_loglik_per_obs
            pred_frame["predictive_rmse_state_date"] = predictive_rmse
            prediction_parts.append(pred_frame)

            if n_update >= min_update:
                beta_post, cov_post = kalman_batch_update(
                    beta_prior=beta_prior,
                    cov_prior=cov_prior,
                    H=H_valid,
                    y=y_valid,
                    obs_var=obs_var,
                )
                state["beta"] = beta_post
                state["cov"] = cov_post
            else:
                state["beta"] = beta_prior
                state["cov"] = cov_prior

            states[state_key] = state

            if save_coefficients:
                beta_original = state["beta"] / feature_scale.to_numpy(dtype=float)
                for feature, coef in zip(M4_FEATURES, beta_original):
                    coef_rows.append(
                        {
                            "model": model_name,
                            "model_family": "state_space_kalman_expanding_fe",
                            "date": pd.Timestamp(date_value),
                            "state_group": state_key,
                            "state_source": state.get("source", "unknown"),
                            "feature": feature,
                            "coefficient": float(coef),
                            "discount_factor": float(discount_factor),
                            "obs_variance_multiplier": float(obs_variance_multiplier),
                            "obs_variance": float(obs_var),
                            "predictive_loglik_state_date": predictive_loglik,
                            "predictive_loglik_per_obs_state_date": predictive_loglik_per_obs,
                            "predictive_rmse_state_date": predictive_rmse,
                            "n_obs_used_for_update": n_update,
                            "window_start": ctx_pack["window_start"],
                            "window_end_exclusive": ctx_pack["window_end_exclusive"],
                            "n_obs_state_train": int(state.get("n_train", 0)),
                        }
                    )

    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    coefficients = pd.DataFrame(coef_rows)
    return predictions, coefficients



def summarise_predictions(predictions: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    rows = []

    for model in model_order:
        m = predictions.loc[predictions["model"].eq(model)].copy()
        if m.empty:
            continue

        for split in ["validation", "test"]:
            g = m.loc[m[SPLIT_COL].eq(split)].copy()
            if g.empty:
                continue

            metrics = compute_metrics(
                g["target_value"].to_numpy(dtype=float),
                g["fitted_value"].to_numpy(dtype=float),
            )

            bench_resid = g["target_value"].to_numpy(dtype=float) - g["own_m0_fitted_value"].to_numpy(dtype=float)
            bench_sse = float(np.sum(bench_resid ** 2))

            rows.append(
                {
                    "model": model,
                    "model_family": g["model_family"].iloc[0],
                    "sample_split": split,
                    **metrics,
                    "benchmark_model": g["own_m0_model"].iloc[0],
                    "benchmark_sse": bench_sse,
                    "oos_r2_vs_expanding_m0": oos_r2_from_sse(metrics["sse"], bench_sse),
                    "evaluation_protocol": g["evaluation_protocol"].iloc[0],
                }
            )

    return pd.DataFrame(rows)


def build_residual_diagnostics(predictions: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    rows = []

    for model in model_order:
        m = predictions.loc[predictions["model"].eq(model)].copy()
        if m.empty:
            continue

        for split in ["validation", "test"]:
            g = m.loc[m[SPLIT_COL].eq(split)].copy()
            if g.empty:
                continue

            r = g["residual"].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model,
                    "sample_split": split,
                    "n_obs": int(len(g)),
                    "mean_residual": float(np.mean(r)),
                    "std_residual": float(np.std(r, ddof=1)),
                    "mean_abs_residual": float(np.mean(np.abs(r))),
                    "p95_abs_residual": float(np.quantile(np.abs(r), 0.95)),
                    "p99_abs_residual": float(np.quantile(np.abs(r), 0.99)),
                    "residual_lag1_autocorr_by_cusip": residual_autocorr_by_cusip(g, "residual"),
                }
            )

    return pd.DataFrame(rows)


def evaluate_candidate(
    predictions: pd.DataFrame,
    model_name: str,
    discount_factor: float,
    obs_variance_multiplier: float,
) -> dict[str, Any]:
    g = predictions.loc[predictions[SPLIT_COL].eq("validation")].copy()
    if g.empty:
        raise ValueError(f"No validation predictions for {model_name}")

    metrics = compute_metrics(
        g["target_value"].to_numpy(dtype=float),
        g["fitted_value"].to_numpy(dtype=float),
    )
    bench_resid = g["target_value"].to_numpy(dtype=float) - g["own_m0_fitted_value"].to_numpy(dtype=float)
    bench_sse = float(np.sum(bench_resid ** 2))

    validation_predictive_loglik = np.nan
    validation_predictive_n_obs = 0
    validation_avg_loglik_per_obs = np.nan
    ll_required = {"state_key", "predictive_loglik_state_date", "n_obs_used_for_update"}
    if ll_required.issubset(g.columns):
        ll_blocks = (
            g.dropna(subset=["predictive_loglik_state_date"])
            .drop_duplicates(["model", DATE_COL, "state_key"])
        )
        if not ll_blocks.empty:
            validation_predictive_loglik = float(ll_blocks["predictive_loglik_state_date"].sum())
            validation_predictive_n_obs = int(ll_blocks["n_obs_used_for_update"].sum())
            if validation_predictive_n_obs > 0:
                validation_avg_loglik_per_obs = float(validation_predictive_loglik / validation_predictive_n_obs)

    return {
        "model": model_name,
        "discount_factor": float(discount_factor),
        "obs_variance_multiplier": float(obs_variance_multiplier),
        "validation_n_obs": metrics["n_obs"],
        "validation_rmse": metrics["rmse"],
        "validation_mae": metrics["mae"],
        "validation_sse": metrics["sse"],
        "validation_oos_r2_vs_expanding_m0": oos_r2_from_sse(metrics["sse"], bench_sse),
        "validation_predictive_loglik": validation_predictive_loglik,
        "validation_predictive_n_obs": validation_predictive_n_obs,
        "validation_avg_loglik_per_obs": validation_avg_loglik_per_obs,
        "selection_criterion": "max_validation_predictive_loglik",
    }


def plot_validation_candidates(candidate_results: pd.DataFrame) -> None:
    if candidate_results.empty:
        return

    plt.figure(figsize=(8, 4.8))
    for model, g in candidate_results.groupby("model"):
        summary = (
            g.groupby("discount_factor")["validation_rmse"]
            .min()
            .reset_index()
            .sort_values("discount_factor")
        )
        plt.plot(summary["discount_factor"], summary["validation_rmse"], marker="o", label=model)

    plt.xlabel("Discount factor")
    plt.ylabel("Best validation RMSE")
    plt.title("Extended DLM validation RMSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_VALIDATION_RMSE, dpi=300, bbox_inches="tight")
    plt.close()


def plot_test_oos_r2(results: pd.DataFrame) -> None:
    test = results.loc[results["sample_split"].eq("test")].copy()
    if test.empty:
        return

    test = test.sort_values("oos_r2_vs_expanding_m0")
    plt.figure(figsize=(8, 4.8))
    plt.barh(test["model"], test["oos_r2_vs_expanding_m0"])
    plt.xlabel("Test OOS R2 vs Expanding M0")
    plt.title("Extended dynamic state-space model comparison")
    plt.tight_layout()
    plt.savefig(FIG_TEST_OOS_R2, dpi=300, bbox_inches="tight")
    plt.close()


def plot_global_coefficients(coefficients: pd.DataFrame) -> None:
    if coefficients.empty:
        return

    g = coefficients.loc[
        coefficients["model"].eq("DLM_Kalman_M4_expandingFE_global_beta")
        & coefficients["state_group"].eq("GLOBAL")
    ].copy()
    if g.empty:
        return

    plt.figure(figsize=(11, 5.5))
    for feature, gf in g.groupby("feature"):
        plt.plot(gf["date"], gf["coefficient"], label=feature)

    plt.xlabel("Date")
    plt.ylabel("Filtered coefficient, original feature units")
    plt.title("Kalman filtered M4 coefficients with expanding fixed effects")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_COEFFICIENTS_GLOBAL, dpi=300, bbox_inches="tight")
    plt.close()


def plot_abs_residual_ecdf(predictions: pd.DataFrame, model_order: list[str]) -> None:
    plt.figure(figsize=(8, 4.8))

    any_line = False
    for model in model_order:
        g = predictions.loc[
            predictions["model"].eq(model)
            & predictions[SPLIT_COL].eq("test")
        ].copy()
        if g.empty:
            continue

        x = np.sort(np.abs(g["residual"].to_numpy(dtype=float)))
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=model)
        any_line = True

    if not any_line:
        plt.close()
        return

    plt.xlabel("Absolute test residual")
    plt.ylabel("Empirical CDF")
    plt.title("Test absolute residual distribution")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_ABS_RESIDUAL_ECDF, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_directories()

    panel = load_model_panel()
    print("Extended dynamic state-space panel")
    print("Rows:", len(panel))
    print("CUSIPs:", panel[GROUP_COL].nunique())
    print("Date range:", panel[DATE_COL].min(), "to", panel[DATE_COL].max())
    print("Split counts:")
    print(panel[SPLIT_COL].value_counts())

    train = panel.loc[panel[SPLIT_COL].eq("train")].copy()
    eval_panel = panel.loc[panel[SPLIT_COL].isin(["validation", "test"])].copy()
    eval_months = sorted(eval_panel["_month_start"].dropna().unique())
    eval_months = [pd.Timestamp(x) for x in eval_months]

    contexts = build_expanding_contexts(panel, eval_months)

    expanding_predictions, expanding_coef = build_expanding_static_predictions(panel, contexts, eval_months)

    train_context = fit_fe_context(train, features=M4_FEATURES)
    global_init = fit_static_ols_standardised(train, train_context)

    print("\nInitial static train M4 residual variance:", global_init["residual_variance"])

    dynamic_specs = [
        {
            "model": "DLM_Kalman_M4_expandingFE_global_beta",
            "group_cols": [],
        },
        {
            "model": "DLM_Kalman_M4_expandingFE_issuer_beta",
            "group_cols": [ISSUER_COL],
        },
        {
            "model": "DLM_Kalman_M4_expandingFE_issuer_maturity_beta",
            "group_cols": [ISSUER_COL, MATURITY_BUCKET_COL],
        },
    ]

    candidate_rows = []
    best_specs: dict[str, dict[str, Any]] = {}

    for spec in dynamic_specs:
        model_name = spec["model"]
        print(f"\nTuning {model_name}")

        for discount_factor in DISCOUNT_FACTORS:
            for obs_mult in OBS_VARIANCE_MULTIPLIERS:
                print(f"  candidate discount={discount_factor}, obs_var_multiplier={obs_mult}")
                preds, _ = run_kalman_expanding_fe(
                    sample=panel,
                    contexts=contexts,
                    eval_months=[m for m in eval_months if m < VALIDATION_END_DATE],
                    group_cols=spec["group_cols"],
                    model_name=model_name,
                    discount_factor=discount_factor,
                    obs_variance_multiplier=obs_mult,
                    global_init=global_init,
                    train_context=train_context,
                    evaluation_splits=["validation"],
                    save_coefficients=False,
                )
                row = evaluate_candidate(preds, model_name, discount_factor, obs_mult)
                candidate_rows.append(row)

        cand = pd.DataFrame([r for r in candidate_rows if r["model"] == model_name])
        best = cand.sort_values(["validation_predictive_loglik", "validation_rmse"], ascending=[False, True], na_position="last").iloc[0]
        best_specs[model_name] = {
            "model": model_name,
            "group_cols": spec["group_cols"],
            "discount_factor": float(best["discount_factor"]),
            "obs_variance_multiplier": float(best["obs_variance_multiplier"]),
            "validation_rmse": float(best["validation_rmse"]),
            "validation_oos_r2_vs_expanding_m0": float(best["validation_oos_r2_vs_expanding_m0"]),
            "validation_predictive_loglik": float(best["validation_predictive_loglik"]),
            "validation_avg_loglik_per_obs": float(best["validation_avg_loglik_per_obs"]),
            "selection_criterion": "max_validation_predictive_loglik",
        }
        print("  selected:", best_specs[model_name])

    candidate_results = pd.DataFrame(candidate_rows)
    candidate_results.to_csv(OUTPUT_CANDIDATES, index=False)

    selected_prediction_parts = [expanding_predictions]
    selected_coef_parts = [expanding_coef]

    for model_name, spec in best_specs.items():
        preds, coefs = run_kalman_expanding_fe(
            sample=panel,
            contexts=contexts,
            eval_months=eval_months,
            group_cols=spec["group_cols"],
            model_name=model_name,
            discount_factor=spec["discount_factor"],
            obs_variance_multiplier=spec["obs_variance_multiplier"],
            global_init=global_init,
            train_context=train_context,
            evaluation_splits=["validation", "test"],
            save_coefficients=True,
        )
        selected_prediction_parts.append(preds)
        selected_coef_parts.append(coefs)

    predictions = pd.concat(selected_prediction_parts, ignore_index=True)
    coefficients = pd.concat(selected_coef_parts, ignore_index=True)

    model_order = [
        "Expanding_M0_FE",
        "Expanding_OLS_M4",
        "DLM_Kalman_M4_expandingFE_global_beta",
        "DLM_Kalman_M4_expandingFE_issuer_beta",
        "DLM_Kalman_M4_expandingFE_issuer_maturity_beta",
    ]

    selected_results = summarise_predictions(predictions, model_order)
    residual_diagnostics = build_residual_diagnostics(predictions, model_order)

    predictions.to_parquet(OUTPUT_PREDICTIONS, index=False)
    selected_results.to_csv(OUTPUT_SELECTED_RESULTS, index=False)
    residual_diagnostics.to_csv(OUTPUT_RESIDUAL_DIAGNOSTICS, index=False)
    coefficients.to_csv(OUTPUT_COEFFICIENTS, index=False)

    plot_validation_candidates(candidate_results)
    plot_test_oos_r2(selected_results)
    plot_global_coefficients(coefficients)
    plot_abs_residual_ecdf(predictions, model_order)

    manifest = {
        "script": "dynamic_state_space_extended_models.py",
        "input_panel": PANEL_PATH,
        "target": TARGET_COL,
        "features": M4_FEATURES,
        "split_dates": {
            "train_end_exclusive": TRAIN_END_DATE,
            "validation_end_exclusive": VALIDATION_END_DATE,
        },
        "fixed_effect_context": "monthly expanding CUSIP y-means and x-means using only observations before each evaluation month",
        "state_equation": "beta_t = beta_{t-1} + eta_t",
        "observation_equation": "final_vwap_return_it - ybar_i,month = (x_it - xbar_i,month)' beta_t + eps_it",
        "benchmark": "Expanding_M0_FE",
        "candidate_discount_factors": DISCOUNT_FACTORS,
        "candidate_obs_variance_multipliers": OBS_VARIANCE_MULTIPLIERS,
        "selection_criterion": "max_validation_predictive_loglik",
        "predictive_likelihood": "Gaussian one-step-ahead p(y_t | y_0:t-1, theta) computed before the same date update",
        "selected_specs": best_specs,
        "base_obs_variance_train_static_ols": global_init["residual_variance"],
        "initial_cov_multiplier": INITIAL_COV_MULTIPLIER,
        "min_obs_per_global_update_date": MIN_OBS_PER_GLOBAL_UPDATE_DATE,
        "min_obs_per_group_update_date": MIN_OBS_PER_GROUP_UPDATE_DATE,
        "min_group_train_obs": MIN_GROUP_TRAIN_OBS,
        "outputs": {
            "candidate_results": OUTPUT_CANDIDATES,
            "selected_results": OUTPUT_SELECTED_RESULTS,
            "residual_diagnostics": OUTPUT_RESIDUAL_DIAGNOSTICS,
            "coefficients_by_date": OUTPUT_COEFFICIENTS,
            "predictions": OUTPUT_PREDICTIONS,
            "fig_validation_rmse": FIG_VALIDATION_RMSE,
            "fig_test_oos_r2": FIG_TEST_OOS_R2,
            "fig_coefficients_global": FIG_COEFFICIENTS_GLOBAL,
            "fig_abs_residual_ecdf": FIG_ABS_RESIDUAL_ECDF,
        },
    }

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(json_safe(manifest), fh, indent=2)

    print("\nSaved extended dynamic state-space outputs")
    print("Candidate results:", OUTPUT_CANDIDATES)
    print("Selected results:", OUTPUT_SELECTED_RESULTS)
    print("Residual diagnostics:", OUTPUT_RESIDUAL_DIAGNOSTICS)
    print("Coefficients:", OUTPUT_COEFFICIENTS)
    print("Predictions:", OUTPUT_PREDICTIONS)
    print("Manifest:", OUTPUT_MANIFEST)
    print("\nSelected specs:")
    print(json.dumps(json_safe(best_specs), indent=2))
    print("\nSelected model results:")
    print(selected_results)


if __name__ == "__main__":
    main()
