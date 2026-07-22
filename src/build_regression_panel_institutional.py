from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config_institutional import (
    PROJECT_ROOT,
    TRACE_FINAL_BASELINE_PANEL_PATH,
    TRACE_MODEL_READY_PATH,
    FRED_PATH,
    EQUITY_PATH,
    TRACE_HOLIDAY_PATH,
    NYSE_HOLIDAY_PATH,
    REGRESSION_DIR,
    REGRESSION_PANEL_PATH,
    REGRESSION_PANEL_GAP5_PATH,
    FINAL_PANEL_INTEGRITY_REPORT_PATH,
    REGRESSION_PANEL_INTEGRITY_REPORT_PATH,
    RUN_MANIFEST_PATH,
    TABLES_DIR,
    FIGURES_DIR,
    GAP_THRESHOLDS,
    MODEL_READY_MAX_BUSINESS_GAP,
    EQUITY_ASOF_TOLERANCE_DAYS,
    EQUITY_ASOF_TOLERANCES_DAYS,
    ensure_directories,
)

from panel_integrity_audit import (
    assert_panel_integrity,
    write_run_manifest,
)

FRED_LEVEL_COLS = [
    "dgs2",
    "dgs5",
    "dgs10",
    "dgs30",
    "vix",
    "sp500",
    "moodys_aaa",
    "moodys_baa",
    "baa_10y_spread",
    "slope_10y_2y",
    "slope_30y_10y",
    "moodys_baa_aaa_spread",
]

EQUITY_LEVEL_COLS = [
    "adj_price",
    "ret",
    "retx",
    "ret_with_dlret",
    "log_market_cap",
    "d_log_market_cap",
    "equity_vol_20d",
    "equity_vol_60d",
]

FRED_SOURCE_DATE_COLS = [
    f"{col}_source_date"
    for col in FRED_LEVEL_COLS
]

MICROSTRUCTURE_COLS = [
    "n_trades",
    "volume_sample_n_trades",
    "total_volume",
    "price_sample_volume",
    "price_range_rel",
    "price_dispersion_rel",
    "institutional_trade_share",
    "potential_agency_duplicate_share",
    "correction_report_share",
    "ats_trade_share",
    "buy_sell_imbalance",
    "business_gap_days",
    "final_amihud_daily",
]


def prepare_trace(trace: pd.DataFrame) -> pd.DataFrame:
    trace = trace.copy()

    trace["date"] = pd.to_datetime(trace["date"])
    trace["prev_date"] = pd.to_datetime(trace["prev_date"])

    trace["trace_company_symbol"] = (
        trace["trace_company_symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    trace = trace.sort_values(["cusip_id", "date"]).reset_index(drop=True)
    trace["_row_id"] = np.arange(len(trace))

    return trace


def prepare_fred(fred: pd.DataFrame) -> pd.DataFrame:
    fred = fred.copy()
    fred["date"] = pd.to_datetime(fred["date"])
    fred = fred.sort_values("date").reset_index(drop=True)

    available_level_cols = [c for c in FRED_LEVEL_COLS if c in fred.columns]
    available_source_cols = [c for c in FRED_SOURCE_DATE_COLS if c in fred.columns]

    # Forward-fill levels so that holiday/missing FRED values do not break interval alignment.
    fred[available_level_cols] = fred[available_level_cols].ffill()

    for col in available_source_cols:
        fred[col] = pd.to_datetime(fred[col], errors="coerce")

    return fred[["date"] + available_level_cols + available_source_cols].copy()


def attach_fred_levels(
    panel: pd.DataFrame,
    fred: pd.DataFrame,
    left_date_col: str,
    suffix: str,
) -> pd.DataFrame:
    fred_cols = [c for c in fred.columns if c != "date"]

    right = fred.rename(
        columns={c: f"{c}_{suffix}" for c in fred_cols}
    ).copy()

    if left_date_col != "date":
        right = right.rename(columns={"date": left_date_col})

    left = panel.sort_values(left_date_col).copy()

    merged = pd.merge_asof(
        left,
        right.sort_values(left_date_col),
        on=left_date_col,
        direction="backward",
    )

    return merged.sort_values("_row_id").reset_index(drop=True)


def prepare_equity(equity: pd.DataFrame) -> pd.DataFrame:
    equity = equity.copy()

    equity["date"] = pd.to_datetime(equity["date"])
    equity["ticker"] = equity["ticker"].astype(str).str.upper().str.strip()

    if "adj_price" in equity.columns:
        equity["log_adj_price"] = np.log(equity["adj_price"].where(equity["adj_price"] > 0))

    cols = ["date", "ticker", "log_adj_price"]
    cols += [c for c in EQUITY_LEVEL_COLS if c in equity.columns and c != "adj_price"]

    equity = equity[cols].sort_values(["ticker", "date"]).reset_index(drop=True)

    return equity


def attach_equity_levels(
    panel: pd.DataFrame,
    equity: pd.DataFrame,
    left_date_col: str,
    suffix: str,
    equity_asof_tolerance_days: int = EQUITY_ASOF_TOLERANCE_DAYS,
) -> pd.DataFrame:
    output_parts = []

    equity_value_cols = [c for c in equity.columns if c not in ["date", "ticker"]]

    equity_date_col = f"equity_date_{suffix}"
    lag_col = f"issuer_equity_match_lag_days_{suffix}"

    for ticker, left_group in panel.groupby("trace_company_symbol", sort=False):
        left_group = left_group.copy().sort_values(left_date_col)

        right = equity.loc[equity["ticker"] == ticker].copy().sort_values("date")

        if right.empty:
            for col in equity_value_cols:
                left_group[f"issuer_{col}_{suffix}"] = np.nan

            left_group[equity_date_col] = pd.NaT
            left_group[lag_col] = np.nan

            output_parts.append(left_group)
            continue

        right = right.rename(
            columns={
                "date": equity_date_col,
                **{col: f"issuer_{col}_{suffix}" for col in equity_value_cols},
            }
        )

        merged = pd.merge_asof(
            left_group,
            right,
            left_on=left_date_col,
            right_on=equity_date_col,
            direction="backward",
            tolerance=pd.Timedelta(days=equity_asof_tolerance_days),
        )

        merged[lag_col] = (
            merged[left_date_col] - merged[equity_date_col]
        ).dt.days

        output_parts.append(merged)

    out = pd.concat(output_parts, ignore_index=True)
    out = out.sort_values("_row_id").reset_index(drop=True)

    return out


def add_interval_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()

    for col in FRED_LEVEL_COLS:
        t_col = f"{col}_t"
        p_col = f"{col}_prev"

        if t_col in panel.columns and p_col in panel.columns:
            panel[f"d_{col}_interval"] = panel[t_col] - panel[p_col]

        src_t_col = f"{col}_source_date_t"
        src_p_col = f"{col}_source_date_prev"

        if src_t_col in panel.columns:
            panel[src_t_col] = pd.to_datetime(panel[src_t_col], errors="coerce")
            panel[f"{col}_age_days_t"] = (
                panel["date"] - panel[src_t_col]
            ).dt.days

        if src_p_col in panel.columns:
            panel[src_p_col] = pd.to_datetime(panel[src_p_col], errors="coerce")
            panel[f"{col}_age_days_prev"] = (
                panel["prev_date"] - panel[src_p_col]
            ).dt.days

        if src_t_col in panel.columns and src_p_col in panel.columns:
            panel[f"{col}_max_age_days_interval"] = panel[
                [f"{col}_age_days_t", f"{col}_age_days_prev"]
            ].max(axis=1)

    if {"sp500_t", "sp500_prev"}.issubset(panel.columns):
        sp500_t = pd.to_numeric(panel["sp500_t"], errors="coerce")
        sp500_prev = pd.to_numeric(panel["sp500_prev"], errors="coerce")

        panel["sp500_log_return_interval"] = (
                np.log(sp500_t.where(sp500_t > 0))
                - np.log(sp500_prev.where(sp500_prev > 0))
        )

        panel["sp500_simple_return_interval"] = (
                np.exp(panel["sp500_log_return_interval"]) - 1.0
        )

    if {
        "issuer_log_adj_price_t",
        "issuer_log_adj_price_prev",
    }.issubset(panel.columns):
        panel["issuer_equity_log_return_interval"] = (
            panel["issuer_log_adj_price_t"]
            - panel["issuer_log_adj_price_prev"]
        )
        panel["issuer_equity_simple_return_interval"] = (
            np.exp(panel["issuer_equity_log_return_interval"]) - 1.0
        )

    if {
        "issuer_equity_log_return_interval",
        "sp500_log_return_interval",
    }.issubset(panel.columns):
        panel["issuer_equity_excess_sp500_log_return_interval"] = (
            panel["issuer_equity_log_return_interval"]
            - panel["sp500_log_return_interval"]
        )

        panel["issuer_equity_excess_sp500_simple_return_interval"] = (
            np.exp(panel["issuer_equity_excess_sp500_log_return_interval"]) - 1.0
        )

    if {
        "issuer_log_market_cap_t",
        "issuer_log_market_cap_prev",
    }.issubset(panel.columns):
        panel["issuer_d_log_market_cap_interval"] = (
            panel["issuer_log_market_cap_t"]
            - panel["issuer_log_market_cap_prev"]
        )

    rename_map = {
        "issuer_equity_vol_20d_t": "issuer_equity_vol_20d",
        "issuer_equity_vol_60d_t": "issuer_equity_vol_60d",
        "issuer_log_market_cap_t": "issuer_log_market_cap",
    }

    panel = panel.rename(
        columns={old: new for old, new in rename_map.items() if old in panel.columns}
    )

    return panel


def write_diagnostics(panel: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": len(panel),
        "cusips": panel["cusip_id"].nunique(),
        "first_date": panel["date"].min(),
        "last_date": panel["date"].max(),
        "issuers": panel["trace_company_symbol"].nunique(),
        "missing_final_vwap_return": panel["final_vwap_return"].isna().sum()
        if "final_vwap_return" in panel.columns
        else np.nan,
        "missing_final_dirty_vwap_return": panel["final_dirty_vwap_return"].isna().sum()
        if "final_dirty_vwap_return" in panel.columns
        else np.nan,
        "missing_dirty_clean_vwap_return_diff": panel["dirty_clean_vwap_return_diff"].isna().sum()
        if "dirty_clean_vwap_return_diff" in panel.columns
        else np.nan,
        "missing_d_dgs10_interval": panel["d_dgs10_interval"].isna().sum()
        if "d_dgs10_interval" in panel.columns
        else np.nan,
        "missing_d_baa_10y_spread_interval": panel["d_baa_10y_spread_interval"].isna().sum()
        if "d_baa_10y_spread_interval" in panel.columns
        else np.nan,
        "missing_issuer_equity_return_interval": panel["issuer_equity_log_return_interval"].isna().sum()
        if "issuer_equity_log_return_interval" in panel.columns
        else np.nan,
        "missing_sp500_log_return_interval": panel["sp500_log_return_interval"].isna().sum()
        if "sp500_log_return_interval" in panel.columns
        else np.nan,
        "missing_issuer_equity_excess_sp500_log_return_interval": panel[
            "issuer_equity_excess_sp500_log_return_interval"].isna().sum()
        if "issuer_equity_excess_sp500_log_return_interval" in panel.columns
        else np.nan,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(TABLES_DIR / "regression_panel_summary.csv", index=False)

    issuer_summary = (
        panel.groupby("trace_company_symbol")
        .agg(
            rows=("cusip_id", "size"),
            cusips=("cusip_id", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            mean_years_to_maturity=("years_to_maturity", "mean")
            if "years_to_maturity" in panel.columns
            else ("cusip_id", "size"),
            mean_total_volume=("total_volume", "mean")
            if "total_volume" in panel.columns
            else ("cusip_id", "size"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    issuer_summary.to_csv(TABLES_DIR / "regression_panel_issuer_summary.csv", index=False)

    missing_summary = (
        panel.isna()
        .sum()
        .rename("missing")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    missing_summary["missing_share"] = missing_summary["missing"] / len(panel)
    missing_summary = missing_summary.sort_values("missing", ascending=False)

    missing_summary.to_csv(TABLES_DIR / "regression_panel_missing_summary.csv", index=False)

    available_micro_cols = [c for c in MICROSTRUCTURE_COLS if c in panel.columns]
    if available_micro_cols:
        micro_summary = (
            panel[available_micro_cols]
            .describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
            .T
            .reset_index()
            .rename(columns={"index": "variable"})
        )
        micro_summary.to_csv(TABLES_DIR / "regression_panel_microstructure_summary.csv", index=False)

    target_cols = [
        c for c in [
            "final_vwap_return",
            "final_dirty_vwap_return",
            "dirty_clean_vwap_return_diff",
            "final_median_price_return",
            "final_last_price_return",
            "final_yield_change",
        ]
        if c in panel.columns
    ]
    if target_cols:
        target_summary = (
            panel[target_cols]
            .describe(percentiles=[0.001, 0.005, 0.01, 0.5, 0.99, 0.995, 0.999])
            .T
            .reset_index()
            .rename(columns={"index": "target"})
        )
        target_summary.to_csv(TABLES_DIR / "regression_panel_target_robustness_summary.csv", index=False)

    print("\nREGRESSION PANEL SUMMARY")
    print(summary_df.T)

    print("\nISSUER SUMMARY")
    print(issuer_summary)

    print("\nTOP MISSING VARIABLES")
    print(missing_summary.head(20))

def write_gap_sensitivity_regression_diagnostics(panel: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    target_col = "final_vwap_return"

    for gap in GAP_THRESHOLDS:
        flag = f"valid_return_gap_{gap}bd"

        if flag not in panel.columns:
            raise ValueError(f"Missing required gap flag in regression panel: {flag}")

        df_gap = panel.loc[panel[flag]].copy()

        rows.append(
            {
                "gap_threshold_bd": gap,
                "rows": len(df_gap),
                "cusips": df_gap["cusip_id"].nunique(),
                "issuers": df_gap["trace_company_symbol"].nunique()
                if "trace_company_symbol" in df_gap.columns
                else np.nan,
                "first_date": df_gap["date"].min(),
                "last_date": df_gap["date"].max(),
                "mean_business_gap": df_gap["business_gap_days"].mean()
                if "business_gap_days" in df_gap.columns
                else np.nan,
                "median_business_gap": df_gap["business_gap_days"].median()
                if "business_gap_days" in df_gap.columns
                else np.nan,
                "p95_business_gap": df_gap["business_gap_days"].quantile(0.95)
                if "business_gap_days" in df_gap.columns
                else np.nan,
                "mean_abs_return": df_gap[target_col].abs().mean()
                if target_col in df_gap.columns
                else np.nan,
                                "p99_abs_return": df_gap[target_col].abs().quantile(0.99)
                if target_col in df_gap.columns
                else np.nan,
                "mean_abs_dirty_vwap_return": df_gap["final_dirty_vwap_return"].abs().mean()
                if "final_dirty_vwap_return" in df_gap.columns
                else np.nan,
                "p99_abs_dirty_vwap_return": df_gap["final_dirty_vwap_return"].abs().quantile(0.99)
                if "final_dirty_vwap_return" in df_gap.columns
                else np.nan,
                "median_abs_dirty_clean_diff_bp": (
                    10_000.0 * df_gap["dirty_clean_vwap_return_diff"].abs().median()
                )
                if "dirty_clean_vwap_return_diff" in df_gap.columns
                else np.nan,
                "p99_abs_dirty_clean_diff_bp": (
                    10_000.0 * df_gap["dirty_clean_vwap_return_diff"].abs().quantile(0.99)
                )
                if "dirty_clean_vwap_return_diff" in df_gap.columns
                else np.nan,
                "missing_final_dirty_vwap_return": df_gap["final_dirty_vwap_return"].isna().sum()
                if "final_dirty_vwap_return" in df_gap.columns
                else np.nan,
                "missing_d_dgs10_interval": df_gap["d_dgs10_interval"].isna().sum()
                if "d_dgs10_interval" in df_gap.columns
                else np.nan,
                "missing_d_baa_10y_spread_interval": df_gap["d_baa_10y_spread_interval"].isna().sum()
                if "d_baa_10y_spread_interval" in df_gap.columns
                else np.nan,
                "missing_d_vix_interval": df_gap["d_vix_interval"].isna().sum()
                if "d_vix_interval" in df_gap.columns
                else np.nan,
                "missing_issuer_equity_log_return_interval": df_gap["issuer_equity_log_return_interval"].isna().sum()
                if "issuer_equity_log_return_interval" in df_gap.columns
                else np.nan,
                "missing_sp500_log_return_interval": df_gap["sp500_log_return_interval"].isna().sum()
                if "sp500_log_return_interval" in df_gap.columns
                else np.nan,
                "missing_issuer_equity_excess_sp500_log_return_interval": df_gap[
                    "issuer_equity_excess_sp500_log_return_interval"].isna().sum()
                if "issuer_equity_excess_sp500_log_return_interval" in df_gap.columns
                else np.nan,
            }
        )

    gap_diag = pd.DataFrame(rows)

    gap_diag_path = TABLES_DIR / "regression_panel_gap_sensitivity_summary.csv"
    gap_diag.to_csv(gap_diag_path, index=False)

    print("\nREGRESSION PANEL GAP SENSITIVITY SUMMARY")
    print(gap_diag)
    print(f"Saved: {gap_diag_path}")

def write_fred_freshness_diagnostics(panel: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    pooled_ages = []

    for col in FRED_LEVEL_COLS:
        for suffix, left_date_col in [("t", "date"), ("prev", "prev_date")]:
            age_col = f"{col}_age_days_{suffix}"
            source_col = f"{col}_source_date_{suffix}"

            if age_col not in panel.columns or source_col not in panel.columns:
                continue

            age = pd.to_numeric(panel[age_col], errors="coerce")
            age_valid = age.dropna()

            if age_valid.empty:
                continue

            pooled_ages.append(
                pd.DataFrame(
                    {
                        "variable": col,
                        "side": suffix,
                        "age_days": age_valid,
                    }
                )
            )

            rows.append(
                {
                    "variable": col,
                    "side": suffix,
                    "n_obs": int(age_valid.shape[0]),
                    "mean_age_days": float(age_valid.mean()),
                    "median_age_days": float(age_valid.median()),
                    "p95_age_days": float(age_valid.quantile(0.95)),
                    "p99_age_days": float(age_valid.quantile(0.99)),
                    "max_age_days": float(age_valid.max()),
                    "share_age_gt_0d": float(age_valid.gt(0).mean()),
                    "share_age_gt_1d": float(age_valid.gt(1).mean()),
                    "share_age_gt_3d": float(age_valid.gt(3).mean()),
                    "share_age_gt_5d": float(age_valid.gt(5).mean()),
                }
            )

    summary = pd.DataFrame(rows)

    summary_path = TABLES_DIR / "fred_asof_age_summary.csv"
    summary.to_csv(summary_path, index=False)

    if pooled_ages:
        ages = pd.concat(pooled_ages, ignore_index=True)

        x = np.sort(ages["age_days"].dropna().to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)

        plt.figure(figsize=(7, 4.5))
        plt.plot(x, y)
        plt.xlabel("FRED matched-value age in days")
        plt.ylabel("Empirical CDF")
        plt.title("FRED as-of freshness diagnostic")
        plt.tight_layout()

        fig_path = FIGURES_DIR / "fred_match_age_ecdf.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved FRED age ECDF to: {fig_path}")

    print("\nFRED AS-OF AGE SUMMARY")
    print(summary)
    print(f"Saved: {summary_path}")

def write_sp500_missing_diagnostics(panel: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if "sp500_log_return_interval" not in panel.columns:
        return

    missing = panel.loc[
        panel["sp500_log_return_interval"].isna(),
        [
            "cusip_id",
            "date",
            "prev_date",
            "trace_company_symbol",
            "sp500_t",
            "sp500_prev",
            "sp500_source_date_t",
            "sp500_source_date_prev",
        ]
    ].copy()

    missing_path = TABLES_DIR / "sp500_missing_interval_diagnostics.csv"
    missing.to_csv(missing_path, index=False)

    if not missing.empty:
        by_year = (
            missing.assign(year=missing["date"].dt.year)
            .groupby("year")
            .agg(
                missing_rows=("cusip_id", "size"),
                first_date=("date", "min"),
                last_date=("date", "max"),
            )
            .reset_index()
        )
    else:
        by_year = pd.DataFrame(
            columns=["year", "missing_rows", "first_date", "last_date"]
        )

    by_year_path = TABLES_DIR / "sp500_missing_interval_by_year.csv"
    by_year.to_csv(by_year_path, index=False)

    print("\nSP500 MISSING INTERVAL DIAGNOSTICS")
    print(by_year)
    print(f"Saved: {missing_path}")
    print(f"Saved: {by_year_path}")

def write_hard_integrity_audits(
    regression_panel: pd.DataFrame,
    regression_panel_gap5: pd.DataFrame,
) -> None:
    if not TRACE_FINAL_BASELINE_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing final baseline panel for integrity audit: "
            f"{TRACE_FINAL_BASELINE_PANEL_PATH}"
        )

    final_panel = pd.read_parquet(TRACE_FINAL_BASELINE_PANEL_PATH)

    final_required_columns = [
        "cusip_id",
        "date",
        "trace_company_symbol",
        "vwap_price",
        "final_vwap_return",
        "prev_date",
        "calendar_gap_days",
        "business_gap_days",
        "n_trades",
        "total_volume",
    ]

    final_nonnegative_cols = [
        c for c in [
            "n_trades",
            "volume_sample_n_trades",
            "total_volume",
            "price_sample_volume",
            "vwap_price",
            "median_price",
            "last_price",
            "min_price",
            "max_price",
            "price_range",
            "price_range_rel",
            "price_dispersion_rel",
            "years_to_maturity",
            "calendar_gap_days",
            "business_gap_days",
            "dirty_vwap_price",
            "vwap_accrued_interest",
        ]
        if c in final_panel.columns
    ]

    final_expected_dtypes = {
        "cusip_id": "string",
        "date": "datetime",
        "prev_date": "datetime",
        "trace_company_symbol": "string",
        "n_trades": "numeric",
        "total_volume": "numeric",
        "vwap_price": "numeric",
        "final_vwap_return": "numeric",
        "business_gap_days": "numeric",
    }

    final_report = assert_panel_integrity(
        df=final_panel,
        panel_name="final_baseline_panel",
        key_cols=["cusip_id", "date"],
        date_col="date",
        required_columns=final_required_columns,
        required_nonmissing=[
            "cusip_id",
            "date",
            "trace_company_symbol",
            "vwap_price",
            "n_trades",
            "total_volume",
        ],
        required_nonnegative=final_nonnegative_cols,
        expected_dtypes=final_expected_dtypes,
        group_col="cusip_id",
        prev_date_col="prev_date",
        forbid_weekends=True,
        fail_fast=True,
    )

    FINAL_PANEL_INTEGRITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_report.to_csv(FINAL_PANEL_INTEGRITY_REPORT_PATH, index=False)

    regression_required_columns = [
                                      "cusip_id",
                                      "date",
                                      "prev_date",
                                      "trace_company_symbol",
                                      "final_vwap_return",
                                      "final_dirty_vwap_return",
                                      "dirty_clean_vwap_return_diff",
                                      "business_gap_days",
                                      "d_dgs2_interval",
                                      "d_dgs5_interval",
                                      "d_dgs10_interval",
                                      "d_dgs30_interval",
                                      "d_baa_10y_spread_interval",
                                      "d_vix_interval",
                                      "issuer_equity_log_return_interval",
                                      "sp500_log_return_interval",
                                      "issuer_equity_excess_sp500_log_return_interval",
                                  ] + [f"valid_return_gap_{gap}bd" for gap in GAP_THRESHOLDS]

    regression_nonnegative_cols = [
        c for c in [
            "n_trades",
            "volume_sample_n_trades",
            "total_volume",
            "price_sample_volume",
            "vwap_price",
            "median_price",
            "last_price",
            "min_price",
            "max_price",
            "price_range",
            "price_range_rel",
            "price_dispersion_rel",
            "years_to_maturity",
            "calendar_gap_days",
            "business_gap_days",
            "dirty_vwap_price",
            "vwap_accrued_interest",
            "issuer_equity_match_lag_days_t",
            "issuer_equity_match_lag_days_prev",
        ]
        if c in regression_panel.columns
    ]

    regression_expected_dtypes = {
        "cusip_id": "string",
        "date": "datetime",
        "prev_date": "datetime",
        "trace_company_symbol": "string",
        "final_vwap_return": "numeric",
        "final_dirty_vwap_return": "numeric",
        "dirty_clean_vwap_return_diff": "numeric",
        "d_dgs10_interval": "numeric",
        "d_baa_10y_spread_interval": "numeric",
        "d_vix_interval": "numeric",
        "issuer_equity_log_return_interval": "numeric",
        "sp500_log_return_interval": "numeric",
        "issuer_equity_excess_sp500_log_return_interval": "numeric",
        "business_gap_days": "numeric",
    }

    for gap in GAP_THRESHOLDS:
        regression_expected_dtypes[f"valid_return_gap_{gap}bd"] = "bool"

    regression_report = assert_panel_integrity(
        df=regression_panel,
        panel_name="regression_gap_sensitivity_panel",
        key_cols=["cusip_id", "date"],
        date_col="date",
        required_columns=regression_required_columns,
        required_nonmissing=[
            "cusip_id",
            "date",
            "prev_date",
            "trace_company_symbol",
            "final_vwap_return",
            "final_dirty_vwap_return",
            "dirty_clean_vwap_return_diff",
            "d_dgs2_interval",
            "d_dgs5_interval",
            "d_dgs10_interval",
            "d_dgs30_interval",
            "d_baa_10y_spread_interval",
            "d_vix_interval",
            "issuer_equity_log_return_interval",
        ],
        required_nonnegative=regression_nonnegative_cols,
        expected_dtypes=regression_expected_dtypes,
        group_col="cusip_id",
        prev_date_col="prev_date",
        forbid_weekends=True,
        fail_fast=True,
    )

    REGRESSION_PANEL_INTEGRITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    regression_report.to_csv(REGRESSION_PANEL_INTEGRITY_REPORT_PATH, index=False)

    baseline_flag = f"valid_return_gap_{MODEL_READY_MAX_BUSINESS_GAP}bd"

    expected_gap5_rows = int(regression_panel[baseline_flag].sum())
    actual_gap5_rows = len(regression_panel_gap5)

    if expected_gap5_rows != actual_gap5_rows:
        raise ValueError(
            "Regression gap5 panel row count does not match full-panel gap5 flag: "
            f"expected {expected_gap5_rows}, got {actual_gap5_rows}"
        )

    print("\nHARD PANEL INTEGRITY AUDITS")
    print(f"Saved final-panel audit to: {FINAL_PANEL_INTEGRITY_REPORT_PATH}")
    print(f"Saved regression-panel audit to: {REGRESSION_PANEL_INTEGRITY_REPORT_PATH}")

def write_pipeline_run_manifest() -> None:
    config = {
        "project_root": PROJECT_ROOT,
        "gap_thresholds": GAP_THRESHOLDS,
        "model_ready_max_business_gap": MODEL_READY_MAX_BUSINESS_GAP,
        "equity_asof_tolerance_days": EQUITY_ASOF_TOLERANCE_DAYS,
        "equity_asof_tolerances_days": EQUITY_ASOF_TOLERANCES_DAYS,
        "trace_holiday_path": TRACE_HOLIDAY_PATH,
        "nyse_holiday_path": NYSE_HOLIDAY_PATH,
    }

    input_paths = [
        TRACE_FINAL_BASELINE_PANEL_PATH,
        TRACE_MODEL_READY_PATH,
        FRED_PATH,
        EQUITY_PATH,
        TRACE_HOLIDAY_PATH,
        NYSE_HOLIDAY_PATH,
    ]

    output_paths = [
        REGRESSION_PANEL_PATH,
        REGRESSION_PANEL_GAP5_PATH,
        FINAL_PANEL_INTEGRITY_REPORT_PATH,
        REGRESSION_PANEL_INTEGRITY_REPORT_PATH,
        TABLES_DIR / "regression_panel_summary.csv",
        TABLES_DIR / "regression_panel_gap_sensitivity_summary.csv",
        TABLES_DIR / "equity_asof_tolerance_summary.csv",
        TABLES_DIR / "fred_asof_age_summary.csv",
        FIGURES_DIR / "fred_match_age_ecdf.png",
    ]

    manifest = write_run_manifest(
        manifest_path=RUN_MANIFEST_PATH,
        config=config,
        input_paths=input_paths,
        output_paths=output_paths,
        hash_inputs=True,
        hash_outputs=False,
    )

    print("\nRUN MANIFEST")
    print(f"Saved: {RUN_MANIFEST_PATH}")
    print("Input files fingerprinted:", len(manifest["inputs"]))
    print("Output files recorded:", len(manifest["outputs"]))

def build_regression_panel(
    equity_asof_tolerance_days: int = EQUITY_ASOF_TOLERANCE_DAYS,
    save_outputs: bool = True,
) -> pd.DataFrame:
    ensure_directories()
    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nEquity as-of tolerance: {equity_asof_tolerance_days} calendar days")

    print("Loading TRACE:", TRACE_MODEL_READY_PATH)
    print("Loading FRED:", FRED_PATH)
    print("Loading EQUITY:", EQUITY_PATH)

    trace = pd.read_parquet(TRACE_MODEL_READY_PATH)
    fred = pd.read_parquet(FRED_PATH)
    equity = pd.read_parquet(EQUITY_PATH)

    print("TRACE:", trace.shape)
    print("FRED:", fred.shape)
    print("EQUITY:", equity.shape)

    panel = prepare_trace(trace)
    fred = prepare_fred(fred)
    equity = prepare_equity(equity)

    print("\nAttaching FRED levels at current bond date...")
    panel = attach_fred_levels(panel, fred, left_date_col="date", suffix="t")

    print("Attaching FRED levels at previous bond date...")
    panel = attach_fred_levels(panel, fred, left_date_col="prev_date", suffix="prev")

    print("Attaching issuer equity variables at current bond date...")
    panel = attach_equity_levels(
        panel,
        equity,
        left_date_col="date",
        suffix="t",
        equity_asof_tolerance_days=equity_asof_tolerance_days,
    )

    print("Attaching issuer equity variables at previous bond date...")
    panel = attach_equity_levels(
        panel,
        equity,
        left_date_col="prev_date",
        suffix="prev",
        equity_asof_tolerance_days=equity_asof_tolerance_days,
    )

    print("Adding interval-aligned features...")
    panel = add_interval_features(panel)

    panel = panel.sort_values(["cusip_id", "date"]).reset_index(drop=True)

    if "_row_id" in panel.columns:
        panel = panel.drop(columns=["_row_id"])

    required_gap_flags = [f"valid_return_gap_{gap}bd" for gap in GAP_THRESHOLDS]
    missing_gap_flags = [flag for flag in required_gap_flags if flag not in panel.columns]

    if missing_gap_flags:
        raise ValueError(
            "The TRACE input panel does not contain the required gap flags: "
            f"{missing_gap_flags}. Re-run trace_bond_data_institutional_cleaner.py first."
        )

    baseline_flag = f"valid_return_gap_{MODEL_READY_MAX_BUSINESS_GAP}bd"

    if save_outputs:
        print("\nSaving regression gap-sensitivity panel:", REGRESSION_PANEL_PATH)
        panel.to_parquet(REGRESSION_PANEL_PATH, index=False)

        regression_panel_gap5 = panel.loc[panel[baseline_flag]].copy()

        print("\nSaving baseline gap5 regression panel:", REGRESSION_PANEL_GAP5_PATH)
        regression_panel_gap5.to_parquet(REGRESSION_PANEL_GAP5_PATH, index=False)

        write_diagnostics(panel)
        write_gap_sensitivity_regression_diagnostics(panel)
        write_fred_freshness_diagnostics(panel)
        write_sp500_missing_diagnostics(panel)
        write_hard_integrity_audits(
            regression_panel=panel,
            regression_panel_gap5=regression_panel_gap5,
        )

    return panel

def run_equity_asof_tolerance_stress_test() -> pd.DataFrame:
    rows = []

    for tol in EQUITY_ASOF_TOLERANCES_DAYS:
        print(f"\nRunning equity as-of tolerance stress test: {tol} calendar days")

        panel_tol = build_regression_panel(
            equity_asof_tolerance_days=tol,
            save_outputs=False,
        )

        out_path = REGRESSION_DIR / f"regression_panel_equity_tol{tol}.parquet"
        panel_tol.to_parquet(out_path, index=False)

        baseline_flag = f"valid_return_gap_{MODEL_READY_MAX_BUSINESS_GAP}bd"

        if baseline_flag in panel_tol.columns:
            panel_gap5 = panel_tol.loc[panel_tol[baseline_flag]].copy()
        else:
            panel_gap5 = panel_tol.copy()

        row = {
            "equity_asof_tolerance_days": tol,
            "rows_total": len(panel_tol),
            "rows_gap5": len(panel_gap5),
            "cusips_total": panel_tol["cusip_id"].nunique(),
            "cusips_gap5": panel_gap5["cusip_id"].nunique(),
            "issuers_total": panel_tol["trace_company_symbol"].nunique()
            if "trace_company_symbol" in panel_tol.columns
            else np.nan,
            "missing_issuer_equity_log_return_interval_total": panel_tol["issuer_equity_log_return_interval"].isna().sum()
            if "issuer_equity_log_return_interval" in panel_tol.columns
            else np.nan,
            "missing_issuer_equity_log_return_interval_gap5": panel_gap5["issuer_equity_log_return_interval"].isna().sum()
            if "issuer_equity_log_return_interval" in panel_gap5.columns
            else np.nan,
            "missing_share_issuer_equity_log_return_interval_gap5": panel_gap5["issuer_equity_log_return_interval"].isna().mean()
            if "issuer_equity_log_return_interval" in panel_gap5.columns and len(panel_gap5) > 0
            else np.nan,
            "mean_abs_issuer_equity_log_return_interval_gap5": panel_gap5["issuer_equity_log_return_interval"].abs().mean()
            if "issuer_equity_log_return_interval" in panel_gap5.columns
            else np.nan,
            "p99_abs_issuer_equity_log_return_interval_gap5": panel_gap5["issuer_equity_log_return_interval"].abs().quantile(0.99)
            if "issuer_equity_log_return_interval" in panel_gap5.columns
            else np.nan,
        }

        for col, prefix in [
            ("issuer_equity_match_lag_days_t", "current"),
            ("issuer_equity_match_lag_days_prev", "prev"),
        ]:
            if col in panel_gap5.columns:
                lag = panel_gap5[col]

                row.update(
                    {
                        f"mean_{prefix}_equity_lag_days_gap5": lag.mean(),
                        f"median_{prefix}_equity_lag_days_gap5": lag.median(),
                        f"p95_{prefix}_equity_lag_days_gap5": lag.quantile(0.95),
                        f"max_{prefix}_equity_lag_days_gap5": lag.max(),
                        f"share_{prefix}_lag_gt_1d_gap5": lag.gt(1).mean(),
                        f"share_{prefix}_lag_gt_3d_gap5": lag.gt(3).mean(),
                        f"share_{prefix}_lag_gt_5d_gap5": lag.gt(5).mean(),
                    }
                )

        rows.append(row)

        print(f"Saved tolerance panel: {out_path}")

    summary = pd.DataFrame(rows)

    summary_path = TABLES_DIR / "equity_asof_tolerance_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nEQUITY AS-OF TOLERANCE SUMMARY")
    print(summary)
    print(f"Saved: {summary_path}")

    return summary

def main() -> None:
    ensure_directories()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)

    build_regression_panel(
        equity_asof_tolerance_days=EQUITY_ASOF_TOLERANCE_DAYS,
        save_outputs=True,
    )

    run_equity_asof_tolerance_stress_test()

    write_pipeline_run_manifest()


if __name__ == "__main__":
    main()