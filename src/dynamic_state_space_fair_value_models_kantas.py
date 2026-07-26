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

TRAIN_END_DATE = pd.Timestamp("2023-01-01")
VALIDATION_END_DATE = pd.Timestamp("2024-01-01")

PANEL_PATH = REGRESSION_DIR / "regression_panel_gap5_with_peer_factors.parquet"

OUTPUT_CANDIDATES = TABLES_DIR / "dynamic_state_space_candidate_results.csv"
OUTPUT_SELECTED_RESULTS = TABLES_DIR / "dynamic_state_space_selected_model_results.csv"
OUTPUT_RESIDUAL_DIAGNOSTICS = TABLES_DIR / "dynamic_state_space_residual_diagnostics.csv"
OUTPUT_COEFFICIENTS = TABLES_DIR / "dynamic_state_space_coefficients_by_date.csv"
OUTPUT_MANIFEST = TABLES_DIR / "dynamic_state_space_manifest.json"
OUTPUT_PREDICTIONS = REGRESSION_DIR / "dynamic_state_space_gap5_model_predictions.parquet"

FIG_VALIDATION_RMSE = FIGURES_DIR / "dynamic_state_space_validation_rmse_by_discount.png"
FIG_TEST_OOS_R2 = FIGURES_DIR / "dynamic_state_space_test_oos_r2.png"
FIG_COEFFICIENTS = FIGURES_DIR / "dynamic_state_space_coefficients_over_time.png"
FIG_ABS_RESIDUAL_ECDF = FIGURES_DIR / "dynamic_state_space_abs_residual_ecdf.png"

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
MIN_OBS_PER_UPDATE_DATE = 25

SAVE_TRAIN_PREDICTIONS = True


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


def choose_log_price_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    out = df.copy()

    for col in ["final_log_vwap_price", "log_vwap_price"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            return out, col

    for col in ["final_vwap_price", "vwap_price"]:
        if col in out.columns:
            px = pd.to_numeric(out[col], errors="coerce")
            log_col = f"_log_{col}"
            out[log_col] = np.log(px.where(px > 0))
            return out, log_col

    return out, None


def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out[GROUP_COL] = out[GROUP_COL].astype(str).str.strip()

    for col in [TARGET_COL] + M4_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out, log_price_col = choose_log_price_column(out)
    if log_price_col is not None:
        out["_observed_log_price_for_dynamic_path"] = out[log_price_col]

    return assign_sample_split(out)


def load_model_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing peer-factor regression panel: {PANEL_PATH}\n"
            "Run peer_factor_models.py first."
        )

    panel = pd.read_parquet(PANEL_PATH)
    panel = prepare_panel(panel)

    required = [GROUP_COL, DATE_COL, SPLIT_COL, TARGET_COL] + M4_FEATURES
    assert_required_columns(panel, required, "dynamic state-space panel")

    complete = panel.dropna(subset=[TARGET_COL] + M4_FEATURES).copy()
    complete = complete.sort_values([DATE_COL, GROUP_COL]).reset_index(drop=True)
    complete["_sample_index"] = np.arange(len(complete))

    return complete


def fit_locked_fe_context(
    train: pd.DataFrame,
    features: list[str],
    target: str = TARGET_COL,
    group_col: str = GROUP_COL,
) -> dict[str, Any]:
    y_means = train.groupby(group_col)[target].mean()
    x_means = train.groupby(group_col)[features].mean()

    return {
        "features": features,
        "target": target,
        "group_col": group_col,
        "y_means": y_means,
        "x_means": x_means,
        "global_y_mean": float(train[target].mean()),
        "global_x_mean": train[features].mean(),
    }


def y_base(df: pd.DataFrame, context: dict[str, Any]) -> np.ndarray:
    group_col = context["group_col"]
    return (
        df[group_col]
        .map(context["y_means"])
        .fillna(context["global_y_mean"])
        .to_numpy(dtype=float)
    )


def center_x(df: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    features = context["features"]
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


def center_y(df: pd.DataFrame, context: dict[str, Any]) -> np.ndarray:
    y = pd.to_numeric(df[context["target"]], errors="coerce").to_numpy(dtype=float)
    return y - y_base(df, context)


def fit_static_ols(train: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
    features = context["features"]
    Xc = center_x(train, context)[features].to_numpy(dtype=float)
    yc = center_y(train, context)

    beta, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
    fitted_centered = Xc @ beta
    residual = yc - fitted_centered

    xpx = Xc.T @ Xc
    inv_xpx = np.linalg.pinv(xpx + RIDGE_EPS * np.eye(len(features)))
    resid_var = float(np.var(residual, ddof=len(features)))
    beta_cov = resid_var * inv_xpx

    return {
        "beta": pd.Series(beta, index=features, name="coefficient"),
        "residual_variance": resid_var,
        "beta_cov": beta_cov,
        "train_residual": residual,
    }


def predict_static_ols(df: pd.DataFrame, context: dict[str, Any], beta: pd.Series) -> np.ndarray:
    features = context["features"]
    return y_base(df, context) + center_x(df, context)[features].to_numpy(dtype=float) @ beta.loc[features].to_numpy(dtype=float)


def standardise_centered_design(
    df: pd.DataFrame,
    context: dict[str, Any],
    feature_scale: pd.Series,
) -> np.ndarray:
    Xc = center_x(df, context)[context["features"]]
    return (Xc / feature_scale).to_numpy(dtype=float)


def fit_static_ols_standardised(train: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
    features = context["features"]
    Xc_df = center_x(train, context)[features]
    yc = center_y(train, context)

    scale = Xc_df.std(axis=0, ddof=1).replace(0.0, np.nan)
    if scale.isna().any():
        bad = scale[scale.isna()].index.tolist()
        raise ValueError(f"Zero-variance centred features in training sample: {bad}")

    Xs = (Xc_df / scale).to_numpy(dtype=float)
    beta, _, _, _ = np.linalg.lstsq(Xs, yc, rcond=None)
    residual = yc - Xs @ beta

    p = len(features)
    xpx = Xs.T @ Xs
    inv_xpx = np.linalg.pinv(xpx + RIDGE_EPS * np.eye(p))
    resid_var = float(np.var(residual, ddof=p))
    beta_cov = resid_var * inv_xpx

    return {
        "beta_std": beta,
        "feature_scale": scale,
        "residual_variance": resid_var,
        "beta_cov_std": beta_cov,
        "train_residual": residual,
    }




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


def kalman_filter_batched_by_date(
    df: pd.DataFrame,
    context: dict[str, Any],
    beta0: np.ndarray,
    P0: np.ndarray,
    feature_scale: pd.Series,
    obs_variance: float,
    discount_factor: float,
    min_obs_per_update_date: int = MIN_OBS_PER_UPDATE_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not (0.0 < discount_factor <= 1.0):
        raise ValueError("discount_factor must lie in (0, 1].")

    features = context["features"]
    p = len(features)

    work = df.copy().sort_values([DATE_COL, GROUP_COL]).reset_index(drop=True)
    base = y_base(work, context)
    y = pd.to_numeric(work[context["target"]], errors="coerce").to_numpy(dtype=float)
    yc = y - base
    Xs = standardise_centered_design(work, context, feature_scale)

    beta = beta0.astype(float).copy()
    P = np.asarray(P0, dtype=float).copy()
    I = np.eye(p)

    fitted_centered = np.full(len(work), np.nan, dtype=float)
    state_rows: list[dict[str, Any]] = []

    date_codes = work[DATE_COL].to_numpy()
    unique_dates = pd.to_datetime(pd.Series(date_codes).drop_duplicates()).tolist()

    for current_date in unique_dates:
        mask = work[DATE_COL].eq(current_date).to_numpy()
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue

        P_prior = P / discount_factor
        P_prior = P_prior + COV_FLOOR * I
        beta_prior = beta.copy()

        H = Xs[idx, :]
        y_date = yc[idx]
        fitted_centered[idx] = H @ beta_prior

        valid = np.isfinite(y_date) & np.isfinite(H).all(axis=1)
        n_update = int(valid.sum())

        innovation_mean = np.nan
        innovation_rmse = np.nan
        predictive_loglik = np.nan
        predictive_loglik_per_obs = np.nan

        if n_update >= min_obs_per_update_date:
            H_valid = H[valid]
            y_valid = y_date[valid]
            innovation = y_valid - H_valid @ beta_prior
            innovation_mean = float(np.mean(innovation))
            innovation_rmse = float(np.sqrt(np.mean(innovation ** 2)))
            predictive_loglik = kalman_predictive_loglik_low_rank(
                beta_prior=beta_prior,
                cov_prior=P_prior,
                H=H_valid,
                y=y_valid,
                obs_var=obs_variance,
            )
            predictive_loglik_per_obs = float(predictive_loglik / n_update) if np.isfinite(predictive_loglik) else np.nan

            P_prior_inv = np.linalg.pinv(P_prior)
            precision = P_prior_inv + (H_valid.T @ H_valid) / obs_variance
            rhs = P_prior_inv @ beta_prior + (H_valid.T @ y_valid) / obs_variance

            try:
                beta = np.linalg.solve(precision, rhs)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(precision) @ rhs

            P = np.linalg.pinv(precision)
            P = 0.5 * (P + P.T)
            P = P + COV_FLOOR * I
        else:
            beta = beta_prior
            P = P_prior

        row = {
            DATE_COL: pd.Timestamp(current_date),
            "n_obs_on_date": int(idx.size),
            "n_obs_used_for_update": n_update,
            "discount_factor": float(discount_factor),
            "obs_variance": float(obs_variance),
            "innovation_mean": innovation_mean,
            "innovation_rmse": innovation_rmse,
            "predictive_loglik": predictive_loglik,
            "predictive_loglik_per_obs": predictive_loglik_per_obs,
            "state_trace_covariance": float(np.trace(P)),
        }

        for j, feature in enumerate(features):
            row[f"beta_prior__{feature}"] = float(beta_prior[j])
            row[f"beta_filtered__{feature}"] = float(beta[j])
            row[f"beta_filtered_original_units__{feature}"] = float(beta[j] / feature_scale.loc[feature])

        state_rows.append(row)

    out = work[["_sample_index", GROUP_COL, DATE_COL, SPLIT_COL]].copy()
    out["target_return"] = y
    out["base_return_locked_fe"] = base
    out["fitted_return"] = base + fitted_centered
    out["residual_return"] = out["target_return"] - out["fitted_return"]
    out["model"] = "DLM_Kalman_M4_lockedFE"
    out["model_family"] = "state_space_kalman"
    out["discount_factor"] = float(discount_factor)
    out["obs_variance"] = float(obs_variance)

    state_likelihood_cols = [DATE_COL, "predictive_loglik", "predictive_loglik_per_obs", "n_obs_used_for_update"]
    out = out.merge(pd.DataFrame(state_rows)[state_likelihood_cols], on=DATE_COL, how="left")

    if "_observed_log_price_for_dynamic_path" in work.columns:
        out["observed_log_price"] = work["_observed_log_price_for_dynamic_path"].to_numpy()

    states = pd.DataFrame(state_rows)
    return out, states


def prediction_frame(
    df: pd.DataFrame,
    model_name: str,
    model_family: str,
    fitted: np.ndarray,
    extra_cols: dict[str, Any] | None = None,
) -> pd.DataFrame:
    out = df[["_sample_index", GROUP_COL, DATE_COL, SPLIT_COL]].copy()
    out["model"] = model_name
    out["model_family"] = model_family
    out["target_return"] = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    out["fitted_return"] = fitted
    out["residual_return"] = out["target_return"] - out["fitted_return"]

    if "_observed_log_price_for_dynamic_path" in df.columns:
        out["observed_log_price"] = df["_observed_log_price_for_dynamic_path"].to_numpy()

    if extra_cols:
        for key, value in extra_cols.items():
            out[key] = value

    return out


def evaluate_predictions(preds: pd.DataFrame, m0_sse_by_split: dict[str, float]) -> pd.DataFrame:
    rows = []

    for (model, split), g in preds.groupby(["model", SPLIT_COL], sort=False):
        y = g["target_return"].to_numpy(dtype=float)
        yhat = g["fitted_return"].to_numpy(dtype=float)
        metrics = compute_metrics(y, yhat)
        metrics["model"] = model
        metrics[SPLIT_COL] = split
        metrics["model_family"] = g["model_family"].iloc[0]
        metrics["oos_r2_vs_locked_m0"] = oos_r2_from_sse(
            metrics["sse"],
            m0_sse_by_split.get(split, np.nan),
        )
        if "predictive_loglik" in g.columns:
            ll_blocks = g.dropna(subset=["predictive_loglik"]).drop_duplicates([DATE_COL])
            metrics["predictive_loglik"] = float(ll_blocks["predictive_loglik"].sum()) if not ll_blocks.empty else np.nan
            metrics["predictive_n_obs"] = int(ll_blocks["n_obs_used_for_update"].sum()) if "n_obs_used_for_update" in ll_blocks.columns and not ll_blocks.empty else 0
            metrics["avg_predictive_loglik_per_obs"] = (
                float(metrics["predictive_loglik"] / metrics["predictive_n_obs"])
                if metrics.get("predictive_n_obs", 0) > 0 and np.isfinite(metrics["predictive_loglik"])
                else np.nan
            )
        else:
            metrics["predictive_loglik"] = np.nan
            metrics["predictive_n_obs"] = 0
            metrics["avg_predictive_loglik_per_obs"] = np.nan

        rows.append(metrics)

    return pd.DataFrame(rows)[
        [
            "model",
            "model_family",
            SPLIT_COL,
            "n_obs",
            "rmse",
            "mae",
            "r2",
            "oos_r2_vs_locked_m0",
            "predictive_loglik",
            "predictive_n_obs",
            "avg_predictive_loglik_per_obs",
            "sse",
            "mean_residual",
            "std_residual",
            "p01_residual",
            "p05_residual",
            "p50_residual",
            "p95_residual",
            "p99_residual",
        ]
    ]


def make_residual_diagnostics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model, split), g in preds.groupby(["model", SPLIT_COL], sort=False):
        rows.append(
            {
                "model": model,
                SPLIT_COL: split,
                "model_family": g["model_family"].iloc[0],
                "n_obs": int(len(g)),
                "mean_abs_residual": float(g["residual_return"].abs().mean()),
                "median_abs_residual": float(g["residual_return"].abs().median()),
                "p95_abs_residual": float(g["residual_return"].abs().quantile(0.95)),
                "p99_abs_residual": float(g["residual_return"].abs().quantile(0.99)),
                "residual_std": float(g["residual_return"].std(ddof=1)),
                "residual_lag1_autocorr_by_cusip": residual_autocorr_by_cusip(g, "residual_return"),
            }
        )

    return pd.DataFrame(rows)


def add_fair_value_price_path(preds: pd.DataFrame) -> pd.DataFrame:

    if "observed_log_price" not in preds.columns:
        return preds

    out_parts = []

    for (model, cusip), g in preds.sort_values(["model", GROUP_COL, DATE_COL]).groupby(["model", GROUP_COL], sort=False):
        g = g.copy()
        obs = pd.to_numeric(g["observed_log_price"], errors="coerce").to_numpy(dtype=float)
        fitted = pd.to_numeric(g["fitted_return"], errors="coerce").to_numpy(dtype=float)

        fv = np.full(len(g), np.nan, dtype=float)
        valid_anchor = np.flatnonzero(np.isfinite(obs))

        if valid_anchor.size > 0:
            first = int(valid_anchor[0])
            fv[first] = obs[first]
            for k in range(first + 1, len(g)):
                if np.isfinite(fv[k - 1]) and np.isfinite(fitted[k]):
                    fv[k] = fv[k - 1] + fitted[k]

        g["fair_value_log_price_path"] = fv
        g["log_price_dislocation"] = g["observed_log_price"] - g["fair_value_log_price_path"]
        out_parts.append(g)

    return pd.concat(out_parts, ignore_index=True)


def plot_validation_grid(candidate_results: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    valid = candidate_results.loc[candidate_results[SPLIT_COL].eq("validation")].copy()
    if valid.empty:
        return

    plt.figure(figsize=(8, 4.5))
    for obs_mult, g in valid.groupby("obs_variance_multiplier"):
        g = g.sort_values("discount_factor")
        plt.plot(g["discount_factor"], g["rmse"], marker="o", label=f"R x {obs_mult:g}")

    plt.xlabel("Discount factor")
    plt.ylabel("Validation RMSE")
    plt.title("Dynamic state-space M4 validation RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_VALIDATION_RMSE, dpi=300, bbox_inches="tight")
    plt.close()


def plot_test_oos_r2(results: pd.DataFrame) -> None:
    test = results.loc[results[SPLIT_COL].eq("test")].copy()
    if test.empty:
        return

    test = test.sort_values("oos_r2_vs_locked_m0")

    plt.figure(figsize=(8, 4.5))
    plt.barh(test["model"], test["oos_r2_vs_locked_m0"])
    plt.xlabel("Test OOS R2 vs locked M0")
    plt.title("Dynamic state-space model comparison")
    plt.tight_layout()
    plt.savefig(FIG_TEST_OOS_R2, dpi=300, bbox_inches="tight")
    plt.close()


def plot_coefficients(states: pd.DataFrame, features: list[str]) -> None:
    if states.empty:
        return

    plt.figure(figsize=(10, 5.5))
    for feature in features:
        col = f"beta_filtered_original_units__{feature}"
        if col in states.columns:
            plt.plot(states[DATE_COL], states[col], label=feature)

    plt.xlabel("Date")
    plt.ylabel("Filtered coefficient, original feature units")
    plt.title("Kalman-filtered dynamic M4 coefficients")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_COEFFICIENTS, dpi=300, bbox_inches="tight")
    plt.close()


def plot_abs_residual_ecdf(preds: pd.DataFrame) -> None:
    test = preds.loc[preds[SPLIT_COL].eq("test")].copy()
    if test.empty:
        return

    plt.figure(figsize=(8, 4.5))
    for model, g in test.groupby("model", sort=False):
        x = np.sort(g["residual_return"].abs().dropna().to_numpy(dtype=float))
        if len(x) == 0:
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=model)

    plt.xlabel("Absolute test residual")
    plt.ylabel("Empirical CDF")
    plt.title("Test absolute residual distribution")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_ABS_RESIDUAL_ECDF, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_directories()

    panel = load_model_panel()

    train = panel.loc[panel[SPLIT_COL].eq("train")].copy()
    validation = panel.loc[panel[SPLIT_COL].eq("validation")].copy()
    test = panel.loc[panel[SPLIT_COL].eq("test")].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Expected non-empty train, validation and test samples. "
            f"Observed counts: {panel[SPLIT_COL].value_counts().to_dict()}"
        )

    print("Dynamic state-space panel")
    print("Rows:", len(panel))
    print("CUSIPs:", panel[GROUP_COL].nunique())
    print("Date range:", panel[DATE_COL].min(), "to", panel[DATE_COL].max())
    print("Split counts:")
    print(panel[SPLIT_COL].value_counts())

    context = fit_locked_fe_context(train, M4_FEATURES)

    m0_fitted = y_base(panel, context)
    static_std = fit_static_ols_standardised(train, context)
    static_original = fit_static_ols(train, context)
    static_m4_fitted = predict_static_ols(panel, context, static_original["beta"])

    m0_preds = prediction_frame(
        panel,
        model_name="Locked_M0_trainFE",
        model_family="locked_fixed_effects",
        fitted=m0_fitted,
    )
    static_m4_preds = prediction_frame(
        panel,
        model_name="Static_OLS_M4_trainFE",
        model_family="static_fixed_effects",
        fitted=static_m4_fitted,
    )

    m0_metrics = evaluate_predictions(m0_preds, {})
    m0_sse_by_split = (
        m0_metrics.set_index(SPLIT_COL)["sse"].to_dict()
        if not m0_metrics.empty
        else {}
    )

    candidate_frames = []
    candidate_state_frames = []
    candidate_rows = []

    beta0 = static_std["beta_std"]
    P0 = INITIAL_COV_MULTIPLIER * static_std["beta_cov_std"]
    P0 = 0.5 * (P0 + P0.T) + COV_FLOOR * np.eye(len(M4_FEATURES))
    feature_scale = static_std["feature_scale"]
    base_obs_variance = static_std["residual_variance"]

    for discount in DISCOUNT_FACTORS:
        for obs_mult in OBS_VARIANCE_MULTIPLIERS:
            obs_var = float(base_obs_variance * obs_mult)
            print(f"Filtering candidate discount={discount}, obs_var_multiplier={obs_mult}")

            candidate_preds, candidate_states = kalman_filter_batched_by_date(
                df=panel,
                context=context,
                beta0=beta0,
                P0=P0,
                feature_scale=feature_scale,
                obs_variance=obs_var,
                discount_factor=discount,
            )
            candidate_preds["obs_variance_multiplier"] = float(obs_mult)

            candidate_eval = evaluate_predictions(candidate_preds, m0_sse_by_split)
            candidate_eval["discount_factor"] = float(discount)
            candidate_eval["obs_variance_multiplier"] = float(obs_mult)
            candidate_eval["selected_by"] = "max_validation_predictive_loglik"
            candidate_rows.append(candidate_eval)

            candidate_frames.append(candidate_preds)
            candidate_state_frames.append(candidate_states)

    candidate_results = pd.concat(candidate_rows, ignore_index=True)
    candidate_results.to_csv(OUTPUT_CANDIDATES, index=False)
    plot_validation_grid(candidate_results)

    valid_candidates = candidate_results.loc[
        candidate_results[SPLIT_COL].eq("validation")
        & candidate_results["model"].eq("DLM_Kalman_M4_lockedFE")
    ].copy()

    best_row = valid_candidates.sort_values(["predictive_loglik", "rmse"], ascending=[False, True], na_position="last").iloc[0]
    best_discount = float(best_row["discount_factor"])
    best_obs_mult = float(best_row["obs_variance_multiplier"])
    best_obs_var = float(base_obs_variance * best_obs_mult)

    print("\nSelected dynamic model")
    print("Discount factor:", best_discount)
    print("Observation variance multiplier:", best_obs_mult)
    print("Validation predictive log-likelihood:", float(best_row["predictive_loglik"]))
    print("Validation RMSE:", float(best_row["rmse"]))

    selected_preds, selected_states = kalman_filter_batched_by_date(
        df=panel,
        context=context,
        beta0=beta0,
        P0=P0,
        feature_scale=feature_scale,
        obs_variance=best_obs_var,
        discount_factor=best_discount,
    )
    selected_preds["obs_variance_multiplier"] = best_obs_mult
    selected_states["obs_variance_multiplier"] = best_obs_mult

    all_preds = pd.concat([m0_preds, static_m4_preds, selected_preds], ignore_index=True)

    if not SAVE_TRAIN_PREDICTIONS:
        all_preds = all_preds.loc[~all_preds[SPLIT_COL].eq("train")].copy()

    all_preds = add_fair_value_price_path(all_preds)

    selected_results = evaluate_predictions(all_preds, m0_sse_by_split)
    selected_results.to_csv(OUTPUT_SELECTED_RESULTS, index=False)

    residual_diag = make_residual_diagnostics(all_preds)
    residual_diag.to_csv(OUTPUT_RESIDUAL_DIAGNOSTICS, index=False)

    selected_states.to_csv(OUTPUT_COEFFICIENTS, index=False)
    all_preds.to_parquet(OUTPUT_PREDICTIONS, index=False)

    plot_test_oos_r2(selected_results)
    plot_coefficients(selected_states, M4_FEATURES)
    plot_abs_residual_ecdf(all_preds)

    manifest = {
        "script": Path(__file__).name,
        "input_panel": PANEL_PATH,
        "output_predictions": OUTPUT_PREDICTIONS,
        "target": TARGET_COL,
        "features": M4_FEATURES,
        "split_dates": {
            "train_end_exclusive": TRAIN_END_DATE,
            "validation_end_exclusive": VALIDATION_END_DATE,
        },
        "fixed_effect_context": "locked CUSIP means and x-means from 2016-2022 train sample",
        "state_equation": "beta_t = beta_{t-1} + eta_t",
        "observation_equation": "final_vwap_return_it - ybar_i = (x_it - xbar_i)' beta_t + eps_it",
        "prediction_timing": "prequential by date: predict with beta prior, then update using same-date cross-section",
        "candidate_discount_factors": DISCOUNT_FACTORS,
        "candidate_obs_variance_multipliers": OBS_VARIANCE_MULTIPLIERS,
        "selection_criterion": "max_validation_predictive_loglik",
        "predictive_likelihood": "Gaussian one-step-ahead p(y_t | y_0:t-1, theta) computed before the same date update",
        "selected_discount_factor": best_discount,
        "selected_obs_variance_multiplier": best_obs_mult,
        "base_obs_variance_train_static_ols": base_obs_variance,
        "initial_cov_multiplier": INITIAL_COV_MULTIPLIER,
        "min_obs_per_update_date": MIN_OBS_PER_UPDATE_DATE,
        "outputs": {
            "candidate_results": OUTPUT_CANDIDATES,
            "selected_results": OUTPUT_SELECTED_RESULTS,
            "residual_diagnostics": OUTPUT_RESIDUAL_DIAGNOSTICS,
            "coefficients_by_date": OUTPUT_COEFFICIENTS,
            "predictions": OUTPUT_PREDICTIONS,
            "fig_validation_rmse": FIG_VALIDATION_RMSE,
            "fig_test_oos_r2": FIG_TEST_OOS_R2,
            "fig_coefficients": FIG_COEFFICIENTS,
            "fig_abs_residual_ecdf": FIG_ABS_RESIDUAL_ECDF,
        },
    }

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(json_safe(manifest), fh, indent=2)

    print("\nSaved dynamic state-space outputs")
    print("Candidate results:", OUTPUT_CANDIDATES)
    print("Selected results:", OUTPUT_SELECTED_RESULTS)
    print("Residual diagnostics:", OUTPUT_RESIDUAL_DIAGNOSTICS)
    print("Coefficients:", OUTPUT_COEFFICIENTS)
    print("Predictions:", OUTPUT_PREDICTIONS)
    print("Manifest:", OUTPUT_MANIFEST)
    print("\nSelected model results:")
    print(selected_results)


if __name__ == "__main__":
    main()
