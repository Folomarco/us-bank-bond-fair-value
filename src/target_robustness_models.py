from __future__ import annotations


import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from config_institutional import (
    REGRESSION_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    ensure_directories,
)


GROUP_COL = "cusip_id"
DATE_COL = "date"
SPLIT_COL = "sample_split"

TRAIN_END_DATE = pd.Timestamp("2023-01-01")
VALIDATION_END_DATE = pd.Timestamp("2024-01-01")

PANEL_PATH = REGRESSION_DIR / "regression_panel_gap5_with_peer_factors.parquet"

OUTPUT_RESULTS = TABLES_DIR / "target_robustness_gap5_model_results.csv"
OUTPUT_COEFFICIENTS = TABLES_DIR / "target_robustness_gap5_model_coefficients.csv"
OUTPUT_RESIDUAL_DIAGNOSTICS = TABLES_DIR / "target_robustness_gap5_residual_diagnostics.csv"
OUTPUT_PREDICTIONS = REGRESSION_DIR / "target_robustness_gap5_model_predictions.parquet"
OUTPUT_MANIFEST = TABLES_DIR / "target_robustness_gap5_manifest.json"

FIG_OOS_R2 = FIGURES_DIR / "target_robustness_gap5_oos_r2.png"
FIG_RMSE = FIGURES_DIR / "target_robustness_gap5_test_rmse.png"

TARGET_SPECS = {
    "clean_vwap_return": {
        "column": "final_vwap_return",
        "family": "clean_price_return",
        "is_main": True,
    },
    "clean_median_price_return": {
        "column": "final_median_price_return",
        "family": "clean_price_return",
        "is_main": False,
    },
    "clean_last_price_return": {
        "column": "final_last_price_return",
        "family": "clean_price_return",
        "is_main": False,
    },
    "dirty_vwap_return": {
        "column": "final_dirty_vwap_return",
        "family": "dirty_total_return_proxy",
        "is_main": False,
    },
    "median_yield_change": {
        "column": "final_yield_change",
        "family": "yield_change",
        "is_main": False,
    },
}

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

MICROSTRUCTURE_FEATURES = [
    "log_n_trades",
    "log_total_volume",
    "single_trade_day",
    "price_dispersion_rel_filled",
    "price_range_rel_filled",
    "institutional_trade_share",
    "potential_agency_duplicate_share",
    "correction_report_share",
    "ats_trade_share",
    "buy_sell_imbalance",
    "business_gap_days",
]

MODEL_SPECS = {
    "M0_fixed_effects": [],
    "M1_rates": RATES_FEATURES,
    "M2_rates_equity": RATES_FEATURES + EQUITY_FEATURES,
    "M3_rates_equity_vix": RATES_FEATURES + EQUITY_FEATURES + VIX_FEATURES,
    "M4_rates_equity_vix_peer_raw": (
        RATES_FEATURES + EQUITY_FEATURES + VIX_FEATURES + PEER_FEATURES_RAW
    ),
    "M5_rates_equity_vix_peer_raw_microstructure_clean": (
        RATES_FEATURES + EQUITY_FEATURES + VIX_FEATURES + PEER_FEATURES_RAW + MICROSTRUCTURE_FEATURES
    ),
}

SAVE_PREDICTIONS = True
PREDICTION_SPLITS_TO_SAVE = ["validation", "test"]



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

    if "total_volume" in out.columns and "log_total_volume" not in out.columns:
        out["log_total_volume"] = np.log1p(
            pd.to_numeric(out["total_volume"], errors="coerce").where(
                pd.to_numeric(out["total_volume"], errors="coerce") >= 0
            )
        )

    if "n_trades" in out.columns:
        if "log_n_trades" not in out.columns:
            out["log_n_trades"] = np.log1p(
                pd.to_numeric(out["n_trades"], errors="coerce").where(
                    pd.to_numeric(out["n_trades"], errors="coerce") >= 0
                )
            )

        if "single_trade_day" not in out.columns:
            out["single_trade_day"] = (
                pd.to_numeric(out["n_trades"], errors="coerce").le(1).astype(float)
            )

    if "price_dispersion_rel" in out.columns and "price_dispersion_rel_filled" not in out.columns:
        out["price_dispersion_rel_filled"] = pd.to_numeric(
            out["price_dispersion_rel"], errors="coerce"
        ).fillna(0.0)

    if "price_range_rel" in out.columns and "price_range_rel_filled" not in out.columns:
        out["price_range_rel_filled"] = pd.to_numeric(
            out["price_range_rel"], errors="coerce"
        ).fillna(0.0)

    return out


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


def fit_fe_model(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    group_col: str = GROUP_COL,
) -> dict:
    train = train.copy()

    global_y_mean = float(train[target].mean())
    y_means = train.groupby(group_col)[target].mean()

    if len(features) == 0:
        return {
            "features": features,
            "coefficients": pd.Series(dtype=float),
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

    return {
        "features": features,
        "coefficients": coefficients,
        "y_means": y_means,
        "x_means": x_means,
        "global_y_mean": global_y_mean,
        "global_x_mean": global_x_mean,
    }


def predict_fe_model(
    df: pd.DataFrame,
    fitted: dict,
    group_col: str = GROUP_COL,
) -> np.ndarray:
    features = fitted["features"]
    coefficients = fitted["coefficients"]
    y_means = fitted["y_means"]
    x_means = fitted["x_means"]
    global_y_mean = fitted["global_y_mean"]
    global_x_mean = fitted["global_x_mean"]

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


def residual_autocorr_by_cusip(df: pd.DataFrame, residual_col: str) -> float:
    values = []

    for _, g in df.sort_values([GROUP_COL, DATE_COL]).groupby(GROUP_COL):
        r = g[residual_col].dropna()
        if len(r) >= 3:
            ac = r.autocorr(lag=1)
            if pd.notna(ac):
                values.append(ac)

    if not values:
        return np.nan

    return float(np.mean(values))


def available_feature_list(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [c for c in features if c in df.columns]


def assert_no_missing_features(df: pd.DataFrame, features: list[str], model_name: str) -> None:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features for {model_name}: {missing}")



def main() -> None:
    ensure_directories()

    print("\nTARGET ROBUSTNESS MODELS")
    print(f"Reading panel: {PANEL_PATH}")

    panel = pd.read_parquet(PANEL_PATH)
    panel = prepare_panel(panel)
    panel = assign_sample_split(panel)

    panel["_sample_index"] = np.arange(len(panel))

    print("Panel shape:", panel.shape)
    print("Date range:", panel[DATE_COL].min(), "to", panel[DATE_COL].max())
    print("Split counts:")
    print(panel[SPLIT_COL].value_counts(dropna=False).sort_index())

    available_targets = {
        name: spec
        for name, spec in TARGET_SPECS.items()
        if spec["column"] in panel.columns
    }

    missing_targets = {
        name: spec["column"]
        for name, spec in TARGET_SPECS.items()
        if spec["column"] not in panel.columns
    }

    if missing_targets:
        print("\nMissing targets, skipped:")
        for name, col in missing_targets.items():
            print(f" - {name}: {col}")

    if not available_targets:
        raise ValueError("No target robustness columns found.")

    for model_name, features in MODEL_SPECS.items():
        missing = [c for c in features if c not in panel.columns]
        if missing:
            print(f"\nWARNING: {model_name} has missing features and will use available subset:")
            for c in missing:
                print(f" - {c}")

    results_rows = []
    coef_rows = []
    residual_diag_rows = []
    prediction_parts = []

    for target_name, target_spec in available_targets.items():
        target_col = target_spec["column"]
        target_family = target_spec["family"]

        print("\n" + "=" * 80)
        print(f"Target: {target_name} ({target_col})")
        print("=" * 80)

        for model_name, raw_features in MODEL_SPECS.items():
            features = available_feature_list(panel, raw_features)

            if len(features) != len(raw_features):
                missing = sorted(set(raw_features) - set(features))
                print(f"Skipping missing features in {model_name}: {missing}")

            required_cols = [
                "_sample_index",
                GROUP_COL,
                DATE_COL,
                SPLIT_COL,
                target_col,
            ] + features

            sample = panel[required_cols].copy()

            sample[target_col] = pd.to_numeric(sample[target_col], errors="coerce")
            for col in features:
                sample[col] = pd.to_numeric(sample[col], errors="coerce")

            sample = sample.dropna(subset=[target_col] + features).copy()

            train = sample.loc[sample[SPLIT_COL].eq("train")].copy()
            validation = sample.loc[sample[SPLIT_COL].eq("validation")].copy()
            test = sample.loc[sample[SPLIT_COL].eq("test")].copy()

            if train.empty or validation.empty or test.empty:
                print(f"Skipping {target_name} / {model_name}: empty train/validation/test.")
                continue

            print(
                f"{target_name} / {model_name}: "
                f"train={len(train):,}, validation={len(validation):,}, test={len(test):,}, "
                f"features={len(features)}"
            )

            fitted_model = fit_fe_model(train=train, features=features, target=target_col)

            fitted_m0 = fit_fe_model(train=train, features=[], target=target_col)

            split_frames = {
                "train": train,
                "validation": validation,
                "test": test,
            }

            split_predictions = {}

            for split_name, split_df in split_frames.items():
                y_true = split_df[target_col].astype(float).to_numpy()
                y_pred = predict_fe_model(split_df, fitted_model)
                y_pred_m0 = predict_fe_model(split_df, fitted_m0)

                metrics = compute_metrics(y_true, y_pred)
                m0_metrics = compute_metrics(y_true, y_pred_m0)

                metrics_row = {
                    "target_name": target_name,
                    "target_col": target_col,
                    "target_family": target_family,
                    "model": model_name,
                    "split": split_name,
                    "n_features": len(features),
                    "features": "|".join(features),
                    **metrics,
                    "m0_sse_same_sample": m0_metrics["sse"],
                    "oos_r2_vs_m0_same_sample": oos_r2_from_sse(
                        metrics["sse"],
                        m0_metrics["sse"],
                    ),
                    "is_main_target": bool(target_spec["is_main"]),
                }

                results_rows.append(metrics_row)

                split_pred_df = split_df[["_sample_index", GROUP_COL, DATE_COL, SPLIT_COL]].copy()
                split_pred_df["target_name"] = target_name
                split_pred_df["target_col"] = target_col
                split_pred_df["target_family"] = target_family
                split_pred_df["model"] = model_name
                split_pred_df["target_value"] = y_true
                split_pred_df["fitted_value"] = y_pred
                split_pred_df["residual"] = y_true - y_pred

                split_predictions[split_name] = split_pred_df

                if SAVE_PREDICTIONS and split_name in PREDICTION_SPLITS_TO_SAVE:
                    prediction_parts.append(split_pred_df)

            for feature, coef in fitted_model["coefficients"].items():
                coef_rows.append(
                    {
                        "target_name": target_name,
                        "target_col": target_col,
                        "target_family": target_family,
                        "model": model_name,
                        "feature": feature,
                        "coefficient": float(coef),
                        "n_train": int(len(train)),
                    }
                )

            for split_name in ["validation", "test"]:
                pred_df = split_predictions[split_name].copy()

                residual_diag_rows.append(
                    {
                        "target_name": target_name,
                        "target_col": target_col,
                        "target_family": target_family,
                        "model": model_name,
                        "split": split_name,
                        "n_obs": int(len(pred_df)),
                        "residual_mean": float(pred_df["residual"].mean()),
                        "residual_std": float(pred_df["residual"].std()),
                        "residual_abs_mean": float(pred_df["residual"].abs().mean()),
                        "residual_p01": float(pred_df["residual"].quantile(0.01)),
                        "residual_p05": float(pred_df["residual"].quantile(0.05)),
                        "residual_p50": float(pred_df["residual"].quantile(0.50)),
                        "residual_p95": float(pred_df["residual"].quantile(0.95)),
                        "residual_p99": float(pred_df["residual"].quantile(0.99)),
                        "mean_cusip_lag1_autocorr": residual_autocorr_by_cusip(
                            pred_df,
                            residual_col="residual",
                        ),
                    }
                )

    results = pd.DataFrame(results_rows)
    coefficients = pd.DataFrame(coef_rows)
    residual_diagnostics = pd.DataFrame(residual_diag_rows)

    print("\nSaving outputs...")
    results.to_csv(OUTPUT_RESULTS, index=False)
    coefficients.to_csv(OUTPUT_COEFFICIENTS, index=False)
    residual_diagnostics.to_csv(OUTPUT_RESIDUAL_DIAGNOSTICS, index=False)

    if SAVE_PREDICTIONS and prediction_parts:
        predictions = pd.concat(prediction_parts, ignore_index=True)
        predictions.to_parquet(OUTPUT_PREDICTIONS, index=False)
    else:
        predictions = pd.DataFrame()

    manifest = {
        "script": Path(__file__).name,
        "panel_path": str(PANEL_PATH),
        "output_results": str(OUTPUT_RESULTS),
        "output_coefficients": str(OUTPUT_COEFFICIENTS),
        "output_residual_diagnostics": str(OUTPUT_RESIDUAL_DIAGNOSTICS),
        "output_predictions": str(OUTPUT_PREDICTIONS) if SAVE_PREDICTIONS else None,
        "n_panel_rows": int(len(panel)),
        "targets": TARGET_SPECS,
        "models": MODEL_SPECS,
        "train_end_date": str(TRAIN_END_DATE.date()),
        "validation_end_date": str(VALIDATION_END_DATE.date()),
        "note": (
            "Yield-change target has different units from return targets. "
            "Use OOS R2 and relative diagnostics rather than raw RMSE comparisons."
        ),
    }

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Saved: {OUTPUT_RESULTS}")
    print(f"Saved: {OUTPUT_COEFFICIENTS}")
    print(f"Saved: {OUTPUT_RESIDUAL_DIAGNOSTICS}")
    if SAVE_PREDICTIONS:
        print(f"Saved: {OUTPUT_PREDICTIONS}")
    print(f"Saved: {OUTPUT_MANIFEST}")

    make_figures(results)

    print("\nDONE.")


def make_figures(results: pd.DataFrame) -> None:
    if results.empty:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    test = results.loc[results["split"].eq("test")].copy()

    if test.empty:
        return

    plot_df = test.copy()
    plot_df["label"] = plot_df["target_name"] + "\n" + plot_df["model"].str.replace("_", " ")

    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(plot_df)), plot_df["oos_r2_vs_m0_same_sample"].to_numpy())
    plt.axhline(0.0, linewidth=1)
    plt.xticks(np.arange(len(plot_df)), plot_df["label"], rotation=90)
    plt.ylabel("Test OOS R2 versus M0 fixed effects, same sample")
    plt.title("Target robustness: test OOS R2")
    plt.tight_layout()
    plt.savefig(FIG_OOS_R2, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(plot_df)), plot_df["rmse"].to_numpy())
    plt.xticks(np.arange(len(plot_df)), plot_df["label"], rotation=90)
    plt.ylabel("Test RMSE")
    plt.title("Target robustness: test RMSE")
    plt.tight_layout()
    plt.savefig(FIG_RMSE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {FIG_OOS_R2}")
    print(f"Saved: {FIG_RMSE}")


if __name__ == "__main__":
    main()