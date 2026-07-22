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

OUTPUT_RESULTS = TABLES_DIR / "rolling_gap5_model_results.csv"
OUTPUT_WINDOW_SELECTION = TABLES_DIR / "rolling_gap5_window_selection.csv"
OUTPUT_COEFFICIENTS = TABLES_DIR / "rolling_gap5_coefficients_by_month.csv"
OUTPUT_RESIDUAL_DIAGNOSTICS = TABLES_DIR / "rolling_gap5_residual_diagnostics.csv"
OUTPUT_PREDICTIONS = REGRESSION_DIR / "rolling_gap5_model_predictions.parquet"
OUTPUT_MANIFEST = TABLES_DIR / "rolling_gap5_manifest.json"

FIG_OOS_R2 = FIGURES_DIR / "rolling_gap5_validation_test_oos_r2.png"
FIG_COEFFICIENTS = FIGURES_DIR / "rolling_gap5_coefficients_over_time.png"
FIG_RESIDUAL_ECDF = FIGURES_DIR / "rolling_gap5_test_abs_residual_ecdf.png"

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

ROLLING_SPECS = {
    "Rolling_OLS_M4_252d": 252,
    "Rolling_OLS_M4_504d": 504,
    "Rolling_OLS_M4_756d": 756,
    "Expanding_OLS_M4": None,
}

DYNAMIC_M0_BY_M4 = {
    "Rolling_OLS_M4_252d": "Rolling_M0_252d",
    "Rolling_OLS_M4_504d": "Rolling_M0_504d",
    "Rolling_OLS_M4_756d": "Rolling_M0_756d",
    "Expanding_OLS_M4": "Expanding_M0_FE",
}

LOCKED_TRAINVAL_MODELS = [
    "Locked_M0_trainval",
    "Locked_OLS_M4_trainval",
]

MIN_WINDOW_OBS = 20_000
MIN_WINDOW_CUSIPS = 200

SAVE_PREDICTIONS = True


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


def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out[GROUP_COL] = out[GROUP_COL].astype(str).str.strip()

    for col in [TARGET_COL] + M4_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

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


def fit_fe_ols(
    train: pd.DataFrame,
    features: list[str],
    target: str = TARGET_COL,
    group_col: str = GROUP_COL,
) -> dict[str, Any]:
    context = fit_fe_context(
        train=train,
        features=features,
        target=target,
        group_col=group_col,
    )

    if not features:
        coef = pd.Series(dtype=float)
        return {
            "context": context,
            "coef": coef,
            "features": features,
        }

    Xc = center_x(train, context)
    yc = center_y(train, context)

    beta, _, _, _ = np.linalg.lstsq(
        Xc[features].to_numpy(dtype=float),
        yc,
        rcond=None,
    )

    coef = pd.Series(beta, index=features, name="coefficient")

    return {
        "context": context,
        "coef": coef,
        "features": features,
    }


def predict_fe_ols(df: pd.DataFrame, fitted: dict[str, Any]) -> np.ndarray:
    context = fitted["context"]
    features = fitted["features"]

    base = y_base(df, context)

    if not features:
        return base

    Xc = center_x(df, context)
    beta = fitted["coef"].loc[features].to_numpy(dtype=float)

    return base + Xc[features].to_numpy(dtype=float) @ beta


def make_prediction_frame(
    df: pd.DataFrame,
    model_name: str,
    model_family: str,
    fitted_value: np.ndarray,
    refit_month: str,
    window_size_dates: int | None,
    window_start: str | None,
    window_end_exclusive: str | None,
    own_m0_fitted_value: np.ndarray | None = None,
    own_m0_model: str | None = None,
    evaluation_protocol: str | None = None,
) -> pd.DataFrame:
    out = df[["_sample_index", GROUP_COL, DATE_COL, SPLIT_COL]].copy()
    out["model"] = model_name
    out["model_family"] = model_family
    out["target_value"] = df[TARGET_COL].to_numpy(dtype=float)
    out["fitted_value"] = fitted_value
    out["residual"] = out["target_value"] - out["fitted_value"]

    if own_m0_fitted_value is None:
        out["own_m0_fitted_value"] = fitted_value
    else:
        out["own_m0_fitted_value"] = own_m0_fitted_value

    out["own_m0_model"] = own_m0_model if own_m0_model is not None else model_name
    out["evaluation_protocol"] = evaluation_protocol

    out["refit_month"] = refit_month
    out["window_size_dates"] = window_size_dates
    out["window_start"] = window_start
    out["window_end_exclusive"] = window_end_exclusive

    return out


def coefficient_rows(
    fitted: dict[str, Any],
    model_name: str,
    model_family: str,
    refit_month: str,
    window_size_dates: int | None,
    window_start: str | None,
    window_end_exclusive: str | None,
    n_obs: int,
    n_cusips: int,
) -> list[dict[str, Any]]:
    rows = []

    for feature, coef in fitted["coef"].items():
        rows.append(
            {
                "model": model_name,
                "model_family": model_family,
                "refit_month": refit_month,
                "window_size_dates": window_size_dates,
                "window_start": window_start,
                "window_end_exclusive": window_end_exclusive,
                "n_obs": int(n_obs),
                "n_cusips": int(n_cusips),
                "feature": feature,
                "coefficient": float(coef),
            }
        )

    return rows


def get_month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def get_estimation_window(
    sample: pd.DataFrame,
    month_start: pd.Timestamp,
    window_size_dates: int | None,
) -> tuple[pd.DataFrame, str | None, str]:
    prior = sample.loc[sample[DATE_COL] < month_start].copy()

    if prior.empty:
        return prior, None, str(month_start.date())

    if window_size_dates is None:
        window = prior
        window_start = str(window[DATE_COL].min().date())
        return window, window_start, str(month_start.date())

    unique_dates = np.array(sorted(prior[DATE_COL].dropna().unique()))

    if len(unique_dates) < window_size_dates:
        window_dates = unique_dates
    else:
        window_dates = unique_dates[-window_size_dates:]

    window_start_date = pd.Timestamp(window_dates[0])

    window = prior.loc[prior[DATE_COL] >= window_start_date].copy()

    return window, str(window_start_date.date()), str(month_start.date())


def fit_rolling_models(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_parts = []
    coef_rows_all = []

    eval_sample = sample.loc[sample[SPLIT_COL].isin(["validation", "test"])].copy()
    eval_sample["_eval_month"] = get_month_start(eval_sample[DATE_COL])

    eval_months = sorted(eval_sample["_eval_month"].dropna().unique())

    print("\nRolling / expanding monthly refits with protocol-matched M0:")
    print(f"Evaluation months: {len(eval_months)}")

    for m4_model_name, window_size in ROLLING_SPECS.items():
        m0_model_name = DYNAMIC_M0_BY_M4[m4_model_name]

        if window_size is None:
            model_family_m4 = "expanding_ols"
            model_family_m0 = "expanding_m0"
            evaluation_protocol = "walk_forward_expanding"
        else:
            model_family_m4 = "rolling_ols"
            model_family_m0 = "rolling_m0"
            evaluation_protocol = "walk_forward_rolling"

        print("\n" + "-" * 80)
        print(f"Model pair: {m0_model_name} / {m4_model_name}")
        print("-" * 80)

        for month_start_raw in eval_months:
            month_start = pd.Timestamp(month_start_raw)
            month_end = month_start + pd.offsets.MonthBegin(1)

            month_eval = eval_sample.loc[
                (eval_sample[DATE_COL] >= month_start)
                & (eval_sample[DATE_COL] < month_end)
            ].copy()

            if month_eval.empty:
                continue

            window_df, window_start, window_end_exclusive = get_estimation_window(
                sample=sample,
                month_start=month_start,
                window_size_dates=window_size,
            )

            n_obs = len(window_df)
            n_cusips = window_df[GROUP_COL].nunique()

            if n_obs < MIN_WINDOW_OBS or n_cusips < MIN_WINDOW_CUSIPS:
                print(
                    f"Skipping {m4_model_name} {month_start.date()}: "
                    f"n_obs={n_obs:,}, n_cusips={n_cusips:,}"
                )
                continue

            fitted_m0 = fit_fe_ols(window_df, features=[])
            fitted_m4 = fit_fe_ols(window_df, features=M4_FEATURES)

            pred_m0 = predict_fe_ols(month_eval, fitted_m0)
            pred_m4 = predict_fe_ols(month_eval, fitted_m4)

            prediction_parts.append(
                make_prediction_frame(
                    df=month_eval,
                    model_name=m0_model_name,
                    model_family=model_family_m0,
                    fitted_value=pred_m0,
                    refit_month=str(month_start.date()),
                    window_size_dates=window_size,
                    window_start=window_start,
                    window_end_exclusive=window_end_exclusive,
                    own_m0_fitted_value=pred_m0,
                    own_m0_model=m0_model_name,
                    evaluation_protocol=evaluation_protocol,
                )
            )

            prediction_parts.append(
                make_prediction_frame(
                    df=month_eval,
                    model_name=m4_model_name,
                    model_family=model_family_m4,
                    fitted_value=pred_m4,
                    refit_month=str(month_start.date()),
                    window_size_dates=window_size,
                    window_start=window_start,
                    window_end_exclusive=window_end_exclusive,
                    own_m0_fitted_value=pred_m0,
                    own_m0_model=m0_model_name,
                    evaluation_protocol=evaluation_protocol,
                )
            )

            coef_rows_all.extend(
                coefficient_rows(
                    fitted=fitted_m4,
                    model_name=m4_model_name,
                    model_family=model_family_m4,
                    refit_month=str(month_start.date()),
                    window_size_dates=window_size,
                    window_start=window_start,
                    window_end_exclusive=window_end_exclusive,
                    n_obs=n_obs,
                    n_cusips=n_cusips,
                )
            )

            print(
                f"{m4_model_name} {month_start.date()}: "
                f"window={window_start} to {window_end_exclusive}, "
                f"fit_rows={n_obs:,}, eval_rows={len(month_eval):,}"
            )

    rolling_predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame()
    )
    rolling_coefficients = pd.DataFrame(coef_rows_all)

    return rolling_predictions, rolling_coefficients


def fit_locked_trainval_models(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_parts = []
    coef_rows_all = []

    trainval = sample.loc[sample[DATE_COL] < VALIDATION_END_DATE].copy()
    test = sample.loc[sample[SPLIT_COL].eq("test")].copy()

    if trainval.empty or test.empty:
        raise ValueError("Cannot fit locked train+validation models: empty trainval or test sample.")

    print("\nLocked train+validation fixed-origin models:")
    print(f"Train+validation rows: {len(trainval):,}")
    print(f"Train+validation CUSIPs: {trainval[GROUP_COL].nunique():,}")
    print(f"Test rows: {len(test):,}")

    fitted_m0 = fit_fe_ols(trainval, features=[])
    fitted_m4 = fit_fe_ols(trainval, features=M4_FEATURES)

    pred_m0 = predict_fe_ols(test, fitted_m0)
    pred_m4 = predict_fe_ols(test, fitted_m4)

    window_start = str(trainval[DATE_COL].min().date())
    window_end_exclusive = str(VALIDATION_END_DATE.date())

    prediction_parts.append(
        make_prediction_frame(
            df=test,
            model_name="Locked_M0_trainval",
            model_family="locked_m0_trainval",
            fitted_value=pred_m0,
            refit_month="locked_trainval",
            window_size_dates=None,
            window_start=window_start,
            window_end_exclusive=window_end_exclusive,
            own_m0_fitted_value=pred_m0,
            own_m0_model="Locked_M0_trainval",
            evaluation_protocol="fixed_origin_trainval",
        )
    )

    prediction_parts.append(
        make_prediction_frame(
            df=test,
            model_name="Locked_OLS_M4_trainval",
            model_family="locked_ols_trainval",
            fitted_value=pred_m4,
            refit_month="locked_trainval",
            window_size_dates=None,
            window_start=window_start,
            window_end_exclusive=window_end_exclusive,
            own_m0_fitted_value=pred_m0,
            own_m0_model="Locked_M0_trainval",
            evaluation_protocol="fixed_origin_trainval",
        )
    )

    coef_rows_all.extend(
        coefficient_rows(
            fitted=fitted_m4,
            model_name="Locked_OLS_M4_trainval",
            model_family="locked_ols_trainval",
            refit_month="locked_trainval",
            window_size_dates=None,
            window_start=window_start,
            window_end_exclusive=window_end_exclusive,
            n_obs=len(trainval),
            n_cusips=trainval[GROUP_COL].nunique(),
        )
    )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    coefficients = pd.DataFrame(coef_rows_all)

    return predictions, coefficients


def evaluate_predictions(
    predictions: pd.DataFrame,
    benchmark_predictions: pd.DataFrame,
) -> pd.DataFrame:
    merged = predictions.merge(
        benchmark_predictions,
        on="_sample_index",
        how="left",
        validate="many_to_one",
    )

    if "own_m0_fitted_value" not in merged.columns:
        merged["own_m0_fitted_value"] = merged["static_m0_pred"]

    rows = []

    for (model_name, split_name), g in merged.groupby(["model", SPLIT_COL]):
        y_true = g["target_value"].to_numpy(dtype=float)
        y_pred = g["fitted_value"].to_numpy(dtype=float)

        own_m0_pred = g["own_m0_fitted_value"].to_numpy(dtype=float)
        static_m0_pred = g["static_m0_pred"].to_numpy(dtype=float)
        static_m4_pred = g["static_m4_pred"].to_numpy(dtype=float)

        metrics = compute_metrics(y_true, y_pred)

        own_m0_sse = float(np.sum((y_true - own_m0_pred) ** 2))
        static_m0_sse = float(np.sum((y_true - static_m0_pred) ** 2))
        static_m4_sse = float(np.sum((y_true - static_m4_pred) ** 2))

        rows.append(
            {
                "model": model_name,
                "model_family": g["model_family"].iloc[0],
                "evaluation_protocol": g["evaluation_protocol"].iloc[0],
                "own_m0_model": g["own_m0_model"].iloc[0],
                "split": split_name,
                "target": TARGET_COL,
                **metrics,
                "own_m0_sse_same_protocol": own_m0_sse,
                "static_m0_sse_same_sample": static_m0_sse,
                "static_m4_sse_same_sample": static_m4_sse,
                "oos_r2_vs_own_m0": oos_r2_from_sse(metrics["sse"], own_m0_sse),
                "oos_r2_vs_static_m0": oos_r2_from_sse(metrics["sse"], static_m0_sse),
                "incremental_r2_vs_static_m4": oos_r2_from_sse(metrics["sse"], static_m4_sse),
            }
        )

    return pd.DataFrame(rows)


def make_residual_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model_name, split_name), g in predictions.groupby(["model", SPLIT_COL]):
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "n_obs": int(len(g)),
                "residual_mean": float(g["residual"].mean()),
                "residual_std": float(g["residual"].std(ddof=1)),
                "residual_abs_mean": float(g["residual"].abs().mean()),
                "residual_p01": float(g["residual"].quantile(0.01)),
                "residual_p05": float(g["residual"].quantile(0.05)),
                "residual_p50": float(g["residual"].quantile(0.50)),
                "residual_p95": float(g["residual"].quantile(0.95)),
                "residual_p99": float(g["residual"].quantile(0.99)),
                "mean_cusip_lag1_autocorr": residual_autocorr_by_cusip(
                    g,
                    residual_col="residual",
                ),
            }
        )

    return pd.DataFrame(rows)


def make_window_selection(results: pd.DataFrame) -> pd.DataFrame:
    candidate_models = list(ROLLING_SPECS.keys())

    validation = results.loc[
        results["split"].eq("validation")
        & results["model"].isin(candidate_models)
    ].copy()

    if validation.empty:
        return pd.DataFrame()

    validation = validation.sort_values(["rmse", "model"]).reset_index(drop=True)
    validation["validation_rank_by_rmse"] = np.arange(1, len(validation) + 1)
    validation["selected_by_validation_rmse"] = validation["validation_rank_by_rmse"].eq(1)

    return validation[
        [
            "model",
            "model_family",
            "evaluation_protocol",
            "own_m0_model",
            "validation_rank_by_rmse",
            "selected_by_validation_rmse",
            "rmse",
            "mae",
            "r2",
            "oos_r2_vs_own_m0",
            "oos_r2_vs_static_m0",
            "incremental_r2_vs_static_m4",
            "n_obs",
        ]
    ].copy()


def make_figures(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    window_selection: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    m4_models_for_plot = [
        "Static_OLS_M4",
        "Locked_OLS_M4_trainval",
        "Rolling_OLS_M4_252d",
        "Rolling_OLS_M4_504d",
        "Rolling_OLS_M4_756d",
        "Expanding_OLS_M4",
    ]

    plot_df = results.loc[
        results["split"].isin(["validation", "test"])
        & results["model"].isin(m4_models_for_plot)
    ].copy()

    if not plot_df.empty:
        pivot = plot_df.pivot_table(
            index="model",
            columns="split",
            values="oos_r2_vs_own_m0",
            aggfunc="first",
        )

        sort_col = "test" if "test" in pivot.columns else "validation"
        pivot = pivot.sort_values(sort_col, ascending=False)

        ax = pivot.plot(kind="bar", figsize=(11, 5.5))
        ax.axhline(0.0, linewidth=1)
        ax.set_ylabel("OOS R2 versus protocol-matched M0")
        ax.set_title("Static, locked and walk-forward M4 models")
        ax.tick_params(axis="x", rotation=90)
        plt.tight_layout()
        plt.savefig(FIG_OOS_R2, dpi=300, bbox_inches="tight")
        plt.close()

    coeff_model = "Expanding_OLS_M4"

    coeff_df = coefficients.loc[
        coefficients["model"].eq(coeff_model)
        & coefficients["feature"].isin(M4_FEATURES)
    ].copy()

    if not coeff_df.empty:
        coeff_df["refit_month"] = pd.to_datetime(coeff_df["refit_month"], errors="coerce")

        pivot = coeff_df.pivot_table(
            index="refit_month",
            columns="feature",
            values="coefficient",
            aggfunc="first",
        ).sort_index()

        ax = pivot.plot(figsize=(11, 5.5))
        ax.axhline(0.0, linewidth=1)
        ax.set_ylabel("Coefficient")
        ax.set_title("Expanding-window M4 coefficients")
        plt.tight_layout()
        plt.savefig(FIG_COEFFICIENTS, dpi=300, bbox_inches="tight")
        plt.close()

    test_pred = predictions.loc[predictions[SPLIT_COL].eq("test")].copy()

    if not test_pred.empty:
        models_to_plot = ["Static_OLS_M4"]

        if not window_selection.empty:
            selected = window_selection.loc[
                window_selection["selected_by_validation_rmse"],
                "model",
            ]
            if len(selected) > 0:
                models_to_plot.append(selected.iloc[0])

        if "Locked_OLS_M4_trainval" in test_pred["model"].unique():
            models_to_plot.append("Locked_OLS_M4_trainval")

        models_to_plot = list(dict.fromkeys(models_to_plot))

        plt.figure(figsize=(8, 5.5))

        for model_name in models_to_plot:
            vals = (
                test_pred.loc[test_pred["model"].eq(model_name), "residual"]
                .abs()
                .dropna()
                .sort_values()
                .to_numpy(dtype=float)
            )

            if len(vals) == 0:
                continue

            ecdf = np.arange(1, len(vals) + 1) / len(vals)
            plt.plot(vals, ecdf, label=model_name)

        plt.xlabel("Absolute residual")
        plt.ylabel("Empirical CDF")
        plt.title("Test absolute residual ECDF")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_RESIDUAL_ECDF, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Saved: {FIG_OOS_R2}")
    print(f"Saved: {FIG_COEFFICIENTS}")
    print(f"Saved: {FIG_RESIDUAL_ECDF}")


def main() -> None:
    ensure_directories()

    print("\nROLLING / EXPANDING FAIR-VALUE MODELS")
    print(f"Reading panel: {PANEL_PATH}")

    panel = pd.read_parquet(PANEL_PATH)
    panel = prepare_panel(panel)
    panel = assign_sample_split(panel)
    panel["_sample_index"] = np.arange(len(panel))

    assert_required_columns(
        panel,
        [GROUP_COL, DATE_COL, TARGET_COL] + M4_FEATURES,
        context="rolling fair-value model panel",
    )

    required_cols = [GROUP_COL, DATE_COL, TARGET_COL, SPLIT_COL, "_sample_index"] + M4_FEATURES

    sample = (
        panel[required_cols]
        .dropna(subset=[TARGET_COL] + M4_FEATURES)
        .sort_values([GROUP_COL, DATE_COL])
        .reset_index(drop=True)
    )

    print("\nCommon M4 complete-case sample:")
    print("Rows:", f"{len(sample):,}")
    print("CUSIPs:", f"{sample[GROUP_COL].nunique():,}")
    print("Date range:", sample[DATE_COL].min(), "to", sample[DATE_COL].max())
    print("Split counts:")
    print(sample[SPLIT_COL].value_counts(dropna=False).sort_index())

    split_frames = {
        "train": sample.loc[sample[SPLIT_COL].eq("train")].copy(),
        "validation": sample.loc[sample[SPLIT_COL].eq("validation")].copy(),
        "test": sample.loc[sample[SPLIT_COL].eq("test")].copy(),
    }

    for split_name, split_df in split_frames.items():
        if split_df.empty:
            raise ValueError(f"Empty split: {split_name}")

    print("\nFitting static fixed-origin benchmarks...")

    train = split_frames["train"]

    static_m0_fit = fit_fe_ols(train, features=[])
    static_m4_fit = fit_fe_ols(train, features=M4_FEATURES)

    benchmark_predictions = sample[["_sample_index"]].copy()
    benchmark_predictions["static_m0_pred"] = predict_fe_ols(sample, static_m0_fit)
    benchmark_predictions["static_m4_pred"] = predict_fe_ols(sample, static_m4_fit)

    static_prediction_parts = []

    for split_name, split_df in split_frames.items():
        static_m0_pred = predict_fe_ols(split_df, static_m0_fit)
        static_m4_pred = predict_fe_ols(split_df, static_m4_fit)

        static_prediction_parts.append(
            make_prediction_frame(
                df=split_df,
                model_name="Static_M0_FE",
                model_family="static_m0",
                fitted_value=static_m0_pred,
                refit_month="static_train",
                window_size_dates=None,
                window_start=str(train[DATE_COL].min().date()),
                window_end_exclusive=str(TRAIN_END_DATE.date()),
                own_m0_fitted_value=static_m0_pred,
                own_m0_model="Static_M0_FE",
                evaluation_protocol="fixed_origin_train",
            )
        )

        static_prediction_parts.append(
            make_prediction_frame(
                df=split_df,
                model_name="Static_OLS_M4",
                model_family="static_ols",
                fitted_value=static_m4_pred,
                refit_month="static_train",
                window_size_dates=None,
                window_start=str(train[DATE_COL].min().date()),
                window_end_exclusive=str(TRAIN_END_DATE.date()),
                own_m0_fitted_value=static_m0_pred,
                own_m0_model="Static_M0_FE",
                evaluation_protocol="fixed_origin_train",
            )
        )

    static_predictions = pd.concat(static_prediction_parts, ignore_index=True)

    static_coef_rows = coefficient_rows(
        fitted=static_m4_fit,
        model_name="Static_OLS_M4",
        model_family="static_ols",
        refit_month="static_train",
        window_size_dates=None,
        window_start=str(train[DATE_COL].min().date()),
        window_end_exclusive=str(TRAIN_END_DATE.date()),
        n_obs=len(train),
        n_cusips=train[GROUP_COL].nunique(),
    )

    locked_predictions, locked_coefficients = fit_locked_trainval_models(sample)

    rolling_predictions, rolling_coefficients = fit_rolling_models(sample)

    all_predictions = pd.concat(
        [static_predictions, locked_predictions, rolling_predictions],
        ignore_index=True,
    )

    coefficient_frames = [
        pd.DataFrame(static_coef_rows),
        locked_coefficients,
        rolling_coefficients,
    ]
    coefficient_frames = [df for df in coefficient_frames if df is not None and not df.empty]

    all_coefficients = pd.concat(
        coefficient_frames,
        ignore_index=True,
    )

    results = evaluate_predictions(
        predictions=all_predictions,
        benchmark_predictions=benchmark_predictions,
    )

    residual_diagnostics = make_residual_diagnostics(all_predictions)
    window_selection = make_window_selection(results)

    print("\nSaving outputs...")

    results.to_csv(OUTPUT_RESULTS, index=False)
    window_selection.to_csv(OUTPUT_WINDOW_SELECTION, index=False)
    all_coefficients.to_csv(OUTPUT_COEFFICIENTS, index=False)
    residual_diagnostics.to_csv(OUTPUT_RESIDUAL_DIAGNOSTICS, index=False)

    if SAVE_PREDICTIONS:
        all_predictions.to_parquet(OUTPUT_PREDICTIONS, index=False)

    selected_dynamic_model = None
    if not window_selection.empty:
        selected = window_selection.loc[
            window_selection["selected_by_validation_rmse"],
            "model",
        ]
        if len(selected) > 0:
            selected_dynamic_model = selected.iloc[0]

    manifest = {
        "script": Path(__file__).name,
        "panel_path": str(PANEL_PATH),
        "target": TARGET_COL,
        "features": M4_FEATURES,
        "common_sample_rows": int(len(sample)),
        "common_sample_cusips": int(sample[GROUP_COL].nunique()),
        "split_counts": sample[SPLIT_COL].value_counts(dropna=False).to_dict(),
        "rolling_specs": ROLLING_SPECS,
        "dynamic_m0_by_m4": DYNAMIC_M0_BY_M4,
        "locked_trainval_models": LOCKED_TRAINVAL_MODELS,
        "monthly_refit": True,
        "rolling_window_unit": "unique observed panel dates before evaluation month",
        "walk_forward_rule": (
            "For each validation/test month, rolling and expanding coefficients are estimated only "
            "using observations strictly before the first day of that month."
        ),
        "fixed_origin_train_rule": (
            "Static_OLS_M4 is estimated once on 2016-2022 and evaluated on validation/test."
        ),
        "fixed_origin_trainval_rule": (
            "Locked_OLS_M4_trainval is estimated once on 2016-2023 and evaluated only on 2024-2025."
        ),
        "evaluation_metrics_note": (
            "oos_r2_vs_own_m0 compares each M4 model against the M0 benchmark estimated under "
            "the same protocol. oos_r2_vs_static_m0 compares all models against the original "
            "static train-only M0. incremental_r2_vs_static_m4 compares each model against "
            "Static_OLS_M4 on the same rows."
        ),
        "selected_dynamic_model_by_validation_rmse": selected_dynamic_model,
        "train_end_date": str(TRAIN_END_DATE.date()),
        "validation_end_date": str(VALIDATION_END_DATE.date()),
        "outputs": {
            "results": str(OUTPUT_RESULTS),
            "window_selection": str(OUTPUT_WINDOW_SELECTION),
            "coefficients": str(OUTPUT_COEFFICIENTS),
            "residual_diagnostics": str(OUTPUT_RESIDUAL_DIAGNOSTICS),
            "predictions": str(OUTPUT_PREDICTIONS) if SAVE_PREDICTIONS else None,
        },
    }

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(json_safe(manifest), fh, indent=2)

    print(f"Saved: {OUTPUT_RESULTS}")
    print(f"Saved: {OUTPUT_WINDOW_SELECTION}")
    print(f"Saved: {OUTPUT_COEFFICIENTS}")
    print(f"Saved: {OUTPUT_RESIDUAL_DIAGNOSTICS}")
    if SAVE_PREDICTIONS:
        print(f"Saved: {OUTPUT_PREDICTIONS}")
    print(f"Saved: {OUTPUT_MANIFEST}")

    make_figures(
        results=results,
        predictions=all_predictions,
        coefficients=all_coefficients,
        window_selection=window_selection,
    )

    print("\nVALIDATION WINDOW SELECTION")
    if window_selection.empty:
        print("No dynamic model selection table produced.")
    else:
        print(window_selection.to_string(index=False))

    print("\nFIXED-ORIGIN TEST COMPARISON")
    fixed_origin_models = [
        "Static_M0_FE",
        "Static_OLS_M4",
        "Locked_M0_trainval",
        "Locked_OLS_M4_trainval",
    ]

    fixed_test_perf = (
        results.loc[
            results["split"].eq("test")
            & results["model"].isin(fixed_origin_models)
        ]
        .sort_values("oos_r2_vs_own_m0", ascending=False)
        [
            [
                "model",
                "model_family",
                "evaluation_protocol",
                "own_m0_model",
                "rmse",
                "mae",
                "r2",
                "oos_r2_vs_own_m0",
                "oos_r2_vs_static_m0",
                "incremental_r2_vs_static_m4",
                "std_residual",
            ]
        ]
    )
    print(fixed_test_perf.to_string(index=False))

    print("\nWALK-FORWARD TEST PERFORMANCE")
    walk_forward_perf = (
        results.loc[
            results["split"].eq("test")
            & results["model"].isin(list(ROLLING_SPECS.keys()))
        ]
        .sort_values("oos_r2_vs_own_m0", ascending=False)
        [
            [
                "model",
                "model_family",
                "evaluation_protocol",
                "own_m0_model",
                "rmse",
                "mae",
                "r2",
                "oos_r2_vs_own_m0",
                "oos_r2_vs_static_m0",
                "incremental_r2_vs_static_m4",
                "std_residual",
            ]
        ]
    )
    print(walk_forward_perf.to_string(index=False))

    print("\nDONE.")


if __name__ == "__main__":
    main()