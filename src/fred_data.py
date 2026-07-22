from functools import reduce
import time
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd


from config_institutional import DRIVERS_DIR, ensure_directories

ensure_directories()

OUT_DIR = DRIVERS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2016-01-01"
END_DATE = "2025-12-31"

FORCE_REBUILD_FRED = False

drivers_path = OUT_DIR / "fred_driver_levels_and_changes.parquet"
drivers_csv_path = OUT_DIR / "fred_driver_levels_and_changes.csv"
coverage_path = OUT_DIR / "fred_driver_coverage.csv"
fred_baseline_path = OUT_DIR / "fred_baseline_driver_levels_and_changes.parquet"
fred_baseline_csv_path = OUT_DIR / "fred_baseline_driver_levels_and_changes.csv"



fred_series = {

    "dgs2": "DGS2",
    "dgs5": "DGS5",
    "dgs10": "DGS10",
    "dgs30": "DGS30",

    "vix": "VIXCLS",

    "sp500": "SP500",

    "moodys_aaa": "DAAA",
    "moodys_baa": "DBAA",
    "baa_10y_spread": "BAA10Y",
}
RAW_FRED_LEVEL_COLS = list(fred_series.keys())

CONSTRUCTED_FRED_LEVEL_COLS = [
    "slope_10y_2y",
    "slope_30y_10y",
    "moodys_baa_aaa_spread",
]

FRED_LEVEL_COLS = RAW_FRED_LEVEL_COLS + CONSTRUCTED_FRED_LEVEL_COLS


def _conservative_source_date(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    source_cols = [f"{col}_source_date" for col in cols if f"{col}_source_date" in df.columns]

    if not source_cols:
        return pd.Series(pd.NaT, index=df.index)

    return df[source_cols].min(axis=1)


def prepare_fred_with_source_dates(fred: pd.DataFrame) -> pd.DataFrame:
    fred = fred.copy()
    fred["date"] = pd.to_datetime(fred["date"], errors="coerce")
    fred = fred.sort_values("date").reset_index(drop=True)

    for col in RAW_FRED_LEVEL_COLS:
        if col not in fred.columns:
            continue

        fred[col] = pd.to_numeric(fred[col].replace(".", np.nan), errors="coerce")

        source_col = f"{col}_source_date"
        fred[source_col] = fred["date"].where(fred[col].notna()).ffill()

        fred[col] = fred[col].ffill()

    if {"dgs10", "dgs2"}.issubset(fred.columns):
        fred["slope_10y_2y"] = fred["dgs10"] - fred["dgs2"]
        fred["slope_10y_2y_source_date"] = _conservative_source_date(
            fred,
            ["dgs10", "dgs2"],
        )

    if {"dgs30", "dgs10"}.issubset(fred.columns):
        fred["slope_30y_10y"] = fred["dgs30"] - fred["dgs10"]
        fred["slope_30y_10y_source_date"] = _conservative_source_date(
            fred,
            ["dgs30", "dgs10"],
        )

    if {"moodys_baa", "moodys_aaa"}.issubset(fred.columns):
        fred["moodys_baa_aaa_spread"] = fred["moodys_baa"] - fred["moodys_aaa"]
        fred["moodys_baa_aaa_spread_source_date"] = _conservative_source_date(
            fred,
            ["moodys_baa", "moodys_aaa"],
        )

    for col in FRED_LEVEL_COLS:
        if col in fred.columns:
            fred[f"d_{col}"] = fred[col].diff()

    return fred


def download_fred_series(
    name: str,
    fred_id: str,
    max_attempts: int = 5,
    sleep_seconds: int = 10,
) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            df = pd.read_csv(url)

            df.columns = ["date", name]

            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            df[name] = pd.to_numeric(
                df[name].replace(".", np.nan),
                errors="coerce",
            )

            df = df[
                (df["date"] >= pd.Timestamp(START_DATE))
                & (df["date"] <= pd.Timestamp(END_DATE))
            ].copy()

            return df

        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            wait = sleep_seconds * attempt

            print(
                f"FRED download failed for {name} ({fred_id}), "
                f"attempt {attempt}/{max_attempts}: {exc}"
            )

            if attempt < max_attempts:
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)

    raise RuntimeError(
        f"Could not download FRED series {name} ({fred_id}) "
        f"after {max_attempts} attempts."
    ) from last_error


if drivers_path.exists() and not FORCE_REBUILD_FRED:
    print(f"Loading existing FRED drivers from: {drivers_path}")
    drivers = pd.read_parquet(drivers_path)
    drivers["date"] = pd.to_datetime(drivers["date"], errors="coerce")

    missing_raw_series = [
        name for name in RAW_FRED_LEVEL_COLS
        if name not in drivers.columns
    ]

    if missing_raw_series:
        print("\nMissing raw FRED series in cached file:")
        print(missing_raw_series)

        for name in missing_raw_series:
            fred_id = fred_series[name]
            print(f"Downloading missing FRED series {name}: {fred_id}")
            tmp = download_fred_series(name, fred_id)
            drivers = drivers.merge(tmp, on="date", how="outer")

        drivers = drivers.sort_values("date").reset_index(drop=True)
        drivers = drivers[drivers["date"].dt.weekday < 5].copy()

        if {"dgs10", "dgs2"}.issubset(drivers.columns):
            drivers["slope_10y_2y"] = drivers["dgs10"] - drivers["dgs2"]

        if {"dgs30", "dgs10"}.issubset(drivers.columns):
            drivers["slope_30y_10y"] = drivers["dgs30"] - drivers["dgs10"]

        if {"moodys_baa", "moodys_aaa"}.issubset(drivers.columns):
            drivers["moodys_baa_aaa_spread"] = (
                drivers["moodys_baa"] - drivers["moodys_aaa"]
            )

        for col in FRED_LEVEL_COLS:
            if col in drivers.columns:
                drivers[f"d_{col}"] = drivers[col].diff()

else:
    fred_dfs = []

    for name, fred_id in fred_series.items():
        print(f"Downloading {name}: {fred_id}")
        tmp = download_fred_series(name, fred_id)
        fred_dfs.append(tmp)

    drivers = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="outer"),
        fred_dfs
    )

    drivers = drivers.sort_values("date").reset_index(drop=True)

    drivers = drivers[drivers["date"].dt.weekday < 5].copy()


    drivers["slope_10y_2y"] = drivers["dgs10"] - drivers["dgs2"]
    drivers["slope_30y_10y"] = drivers["dgs30"] - drivers["dgs10"]

    drivers["moodys_baa_aaa_spread"] = (
        drivers["moodys_baa"] - drivers["moodys_aaa"]
    )


    level_cols = [col for col in drivers.columns if col != "date"]

    for col in level_cols:
        drivers[f"d_{col}"] = drivers[col].diff()


    coverage_rows = []

    for col in level_cols:
        non_missing = drivers[col].notna()

        coverage_rows.append({
            "variable": col,
            "fred_id": next(
                (sid for k, sid in fred_series.items() if k == col),
                "constructed"
            ),
            "n_obs": int(non_missing.sum()),
            "first_date": drivers.loc[non_missing, "date"].min() if non_missing.any() else pd.NaT,
            "last_date": drivers.loc[non_missing, "date"].max() if non_missing.any() else pd.NaT,
            "missing": int(drivers[col].isna().sum()),
            "missing_share": float(drivers[col].isna().mean()),
        })

    coverage = pd.DataFrame(coverage_rows)
    coverage = coverage.sort_values(["first_date", "variable"])

    print("\nFRED DRIVER COVERAGE")
    print(coverage)

    print("\nFRED DRIVERS HEAD")
    print(drivers.head())

    print("\nFRED DRIVERS TAIL")
    print(drivers.tail())



    DIAG_DIR = OUT_DIR / "diagnostics"
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    baseline_level_cols = [
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

    baseline_level_cols = [col for col in baseline_level_cols if col in drivers.columns]
    baseline_change_cols = [f"d_{col}" for col in baseline_level_cols if f"d_{col}" in drivers.columns]


    missing_details = []

    for col in baseline_level_cols:
        missing_dates = drivers.loc[drivers[col].isna(), "date"]

        missing_details.append({
            "variable": col,
            "n_missing": int(missing_dates.shape[0]),
            "missing_share": float(drivers[col].isna().mean()),
            "first_missing_date": missing_dates.min() if len(missing_dates) > 0 else pd.NaT,
            "last_missing_date": missing_dates.max() if len(missing_dates) > 0 else pd.NaT,
        })

    missing_details = pd.DataFrame(missing_details)
    missing_details_path = DIAG_DIR / "fred_missing_details.csv"
    missing_details.to_csv(missing_details_path, index=False)

    print("\nFRED MISSING DETAILS")
    print(missing_details)
    print(f"Saved: {missing_details_path}")


    level_stats = drivers[baseline_level_cols].describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T

    level_stats_path = DIAG_DIR / "fred_level_descriptive_stats.csv"
    level_stats.to_csv(level_stats_path)

    print("\nFRED LEVEL DESCRIPTIVE STATS")
    print(level_stats)
    print(f"Saved: {level_stats_path}")


    change_stats = drivers[baseline_change_cols].describe(
        percentiles=[0.001, 0.005, 0.01, 0.05, 0.50, 0.95, 0.99, 0.995, 0.999]
    ).T

    change_stats_path = DIAG_DIR / "fred_change_descriptive_stats.csv"
    change_stats.to_csv(change_stats_path)

    print("\nFRED CHANGE DESCRIPTIVE STATS")
    print(change_stats)
    print(f"Saved: {change_stats_path}")


    sanity_checks = []

    for col in baseline_level_cols:
        x = drivers[col]

        sanity_checks.append({
            "variable": col,
            "min": x.min(),
            "max": x.max(),
            "negative_values": int((x < 0).sum()),
            "zero_values": int((x == 0).sum()),
            "n_obs": int(x.notna().sum()),
        })

    sanity_checks = pd.DataFrame(sanity_checks)
    sanity_checks_path = DIAG_DIR / "fred_level_sanity_checks.csv"
    sanity_checks.to_csv(sanity_checks_path, index=False)

    print("\nFRED LEVEL SANITY CHECKS")
    print(sanity_checks)
    print(f"Saved: {sanity_checks_path}")


    outlier_rules = {

        "d_dgs2": 0.25,
        "d_dgs5": 0.25,
        "d_dgs10": 0.25,
        "d_dgs30": 0.25,
        "d_moodys_aaa": 0.25,
        "d_moodys_baa": 0.25,
        "d_baa_10y_spread": 0.25,
        "d_moodys_baa_aaa_spread": 0.25,

        "d_vix": 10.0,
    }

    outlier_rows = []

    for col, threshold in outlier_rules.items():
        if col not in drivers.columns:
            continue

        temp = drivers.loc[
            drivers[col].abs() > threshold,
            ["date", col]
        ].copy()

        temp["variable"] = col
        temp["threshold"] = threshold
        temp = temp.rename(columns={col: "change_value"})

        outlier_rows.append(temp)

    if outlier_rows:
        fred_outliers = pd.concat(outlier_rows, ignore_index=True)
    else:
        fred_outliers = pd.DataFrame(columns=["date", "change_value", "variable", "threshold"])

    fred_outliers_path = DIAG_DIR / "fred_large_daily_changes.csv"
    fred_outliers.to_csv(fred_outliers_path, index=False)

    print("\nFRED LARGE DAILY CHANGES")
    print(fred_outliers.sort_values(["variable", "date"]).head(50))
    print(f"Number of large-change observations: {len(fred_outliers)}")
    print(f"Saved: {fred_outliers_path}")


    change_corr = drivers[baseline_change_cols].corr()

    change_corr_path = DIAG_DIR / "fred_change_correlation_matrix.csv"
    change_corr.to_csv(change_corr_path)

    print("\nFRED CHANGE CORRELATION MATRIX")
    print(change_corr)
    print(f"Saved: {change_corr_path}")


    baseline_cols_to_save = ["date"] + baseline_level_cols + baseline_change_cols

    fred_baseline = drivers[baseline_cols_to_save].copy()

    fred_baseline_path = OUT_DIR / "fred_baseline_driver_levels_and_changes.parquet"
    fred_baseline_csv_path = OUT_DIR / "fred_baseline_driver_levels_and_changes.csv"

    fred_baseline.to_parquet(fred_baseline_path, index=False)
    fred_baseline.to_csv(fred_baseline_csv_path, index=False)

    print(f"\nSaved baseline FRED drivers to: {fred_baseline_path}")
    print(f"Saved baseline FRED drivers CSV to: {fred_baseline_csv_path}")


    drivers.to_parquet(drivers_path, index=False)
    drivers.to_csv(drivers_csv_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    print(f"\nSaved FRED drivers to: {drivers_path}")
    print(f"Saved FRED drivers CSV to: {drivers_csv_path}")
    print(f"Saved coverage diagnostics to: {coverage_path}")


drivers = prepare_fred_with_source_dates(drivers)

baseline_level_cols = [col for col in FRED_LEVEL_COLS if col in drivers.columns]
baseline_source_date_cols = [
    f"{col}_source_date"
    for col in baseline_level_cols
    if f"{col}_source_date" in drivers.columns
]
baseline_change_cols = [
    f"d_{col}"
    for col in baseline_level_cols
    if f"d_{col}" in drivers.columns
]

baseline_cols_to_save = (
        ["date"]
        + baseline_level_cols
        + baseline_source_date_cols
        + baseline_change_cols
)

fred_baseline = drivers[baseline_cols_to_save].copy()

fred_baseline.to_parquet(fred_baseline_path, index=False)
fred_baseline.to_csv(fred_baseline_csv_path, index=False)


drivers.to_parquet(drivers_path, index=False)
drivers.to_csv(drivers_csv_path, index=False)

print("\nFRED SOURCE-DATE BASELINE EXPORT")
print(f"Saved baseline FRED drivers with source dates to: {fred_baseline_path}")
print(f"Saved baseline FRED drivers CSV with source dates to: {fred_baseline_csv_path}")
print(f"Saved full FRED drivers with source dates to: {drivers_path}")