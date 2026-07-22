from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


from config_institutional import (
    EQUITY_RAW_ZIP,
    EQUITY_DIR,
    NYSE_HOLIDAY_PATH,
    ensure_directories,
)

from calendar_utils import (
    business_dates_custom,
    load_market_holidays,
)

ensure_directories()

RAW_ZIP = EQUITY_RAW_ZIP

OUT_DIR = EQUITY_DIR
DIAG_DIR = OUT_DIR / "diagnostics"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)
NYSE_BUSINESS_CALENDAR = load_market_holidays(
    NYSE_HOLIDAY_PATH,
    calendar_name="NYSE",
)

EXPECTED_TICKERS = {"BAC", "GS", "JPM", "MS", "WFC"}

START_DATE = pd.Timestamp("2016-01-01")
END_DATE = pd.Timestamp("2025-12-31")


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def read_wrds_zip(zip_path: Path) -> pd.DataFrame:
    if not zip_path.exists():
        raise FileNotFoundError(f"File not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        members = [
            name for name in zf.namelist()
            if not name.endswith("/")
            and name.lower().endswith((".csv", ".txt"))
        ]

        if not members:
            raise ValueError(f"No CSV/TXT file found inside {zip_path.name}")

        print("\nFiles inside ZIP:")
        for m in members:
            print(f" - {m}")

        member = members[0]
        print(f"\nReading: {member}")

        with zf.open(member) as fh:
            df = pd.read_csv(fh, low_memory=False)

    return normalise_columns(df)


def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.strip().replace(
            {
                "": np.nan,
                ".": np.nan,
                "nan": np.nan,
                "NaN": np.nan,
                "None": np.nan,
                "B": np.nan,
                "C": np.nan,
            }
        ),
        errors="coerce"
    )



stocks_raw = read_wrds_zip(RAW_ZIP)

print("\nRAW CRSP STOCK DATA")
print("Shape:", stocks_raw.shape)
print("Columns:")
print(stocks_raw.columns.tolist())
print("\nHead:")
print(stocks_raw.head())


date_col = pick_col(stocks_raw, ["date", "dlycaldt", "yyyymmdd", "datadate", "caldt"])
ticker_col = pick_col(stocks_raw, ["ticker", "tradingsymbol", "tic", "tsymbol"])
permno_col = pick_col(stocks_raw, ["permno"])
permco_col = pick_col(stocks_raw, ["permco"])

company_col = pick_col(stocks_raw, ["issuernm", "securitynm", "comnam", "company_name", "conm"])

price_col = pick_col(stocks_raw, ["prc", "dlyprc", "price", "dlyclose"])
ret_col = pick_col(stocks_raw, ["ret", "dlyret"])
retx_col = pick_col(stocks_raw, ["retx", "dlyretx"])
dlret_col = pick_col(stocks_raw, ["dlret"])

volume_col = pick_col(stocks_raw, ["vol", "dlyvol", "volume"])
shrout_col = pick_col(stocks_raw, ["shrout"])
market_cap_col = pick_col(stocks_raw, ["dlycap", "market_cap", "mktcap"])

cfacpr_col = pick_col(stocks_raw, ["cfacpr", "dlyfacprc"])
cfacshr_col = pick_col(stocks_raw, ["cfacshr", "disfacshr"])

exchange_col = pick_col(stocks_raw, ["exchcd", "primaryexch"])
share_code_col = pick_col(stocks_raw, ["shrcd", "sharetype"])

required = {
    "date": date_col,
    "ticker": ticker_col,
    "price": price_col,
    "ret": ret_col,
}

missing_required = [k for k, v in required.items() if v is None]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}. Available columns: {stocks_raw.columns.tolist()}")

print("\nDETECTED COLUMNS")
detected_cols = {
    "date_col": date_col,
    "ticker_col": ticker_col,
    "permno_col": permno_col,
    "permco_col": permco_col,
    "company_col": company_col,
    "price_col": price_col,
    "ret_col": ret_col,
    "retx_col": retx_col,
    "dlret_col": dlret_col,
    "volume_col": volume_col,
    "shrout_col": shrout_col,
    "market_cap_col": market_cap_col,
    "cfacpr_col": cfacpr_col,
    "cfacshr_col": cfacshr_col,
    "exchange_col": exchange_col,
    "share_code_col": share_code_col,
}
for k, v in detected_cols.items():
    print(f"{k}: {v}")


stocks = stocks_raw.copy()

stocks["date"] = pd.to_datetime(stocks[date_col], errors="coerce")
stocks["ticker"] = stocks[ticker_col].astype(str).str.upper().str.strip()

if permno_col:
    stocks["permno"] = to_numeric_safe(stocks[permno_col]).astype("Int64")
else:
    stocks["permno"] = pd.NA

if permco_col:
    stocks["permco"] = to_numeric_safe(stocks[permco_col]).astype("Int64")
else:
    stocks["permco"] = pd.NA

if company_col:
    stocks["company_name"] = stocks[company_col].astype(str).str.strip()
else:
    stocks["company_name"] = np.nan

stocks["raw_price"] = to_numeric_safe(stocks[price_col])
stocks["price"] = stocks["raw_price"].abs()

stocks["ret"] = to_numeric_safe(stocks[ret_col])
if retx_col:
    stocks["retx"] = to_numeric_safe(stocks[retx_col])
else:
    stocks["retx"] = np.nan

if dlret_col:
    stocks["dlret"] = to_numeric_safe(stocks[dlret_col])
    stocks["ret_with_dlret"] = (1.0 + stocks["ret"].fillna(0.0)) * (1.0 + stocks["dlret"].fillna(0.0)) - 1.0
    stocks.loc[stocks["ret"].isna() & stocks["dlret"].isna(), "ret_with_dlret"] = np.nan
else:
    stocks["dlret"] = np.nan
    stocks["ret_with_dlret"] = stocks["ret"]

if volume_col:
    stocks["volume"] = to_numeric_safe(stocks[volume_col])
else:
    stocks["volume"] = np.nan

if shrout_col:
    stocks["shrout"] = to_numeric_safe(stocks[shrout_col])
else:
    stocks["shrout"] = np.nan

if cfacpr_col:
    stocks["cfacpr"] = to_numeric_safe(stocks[cfacpr_col])
else:
    stocks["cfacpr"] = np.nan

if cfacshr_col:
    stocks["cfacshr"] = to_numeric_safe(stocks[cfacshr_col])
else:
    stocks["cfacshr"] = np.nan

if exchange_col:
    stocks["exchcd"] = stocks[exchange_col].astype(str).str.upper().str.strip()
else:
    stocks["exchcd"] = np.nan

if share_code_col:
    stocks["shrcd"] = stocks[share_code_col].astype(str).str.upper().str.strip()
else:
    stocks["shrcd"] = np.nan

stocks["adj_price"] = np.where(
    stocks["cfacpr"].notna() & (stocks["cfacpr"] != 0),
    stocks["price"] / stocks["cfacpr"],
    stocks["price"]
)

if market_cap_col:
    stocks["market_cap_raw"] = to_numeric_safe(stocks[market_cap_col])

    stocks["market_cap"] = stocks["market_cap_raw"]

    stocks["market_cap_usd_mn"] = stocks["market_cap_raw"] / 1_000_000.0

else:
    stocks["market_cap_raw"] = np.nan

    stocks["market_cap"] = np.where(
        stocks["price"].notna() & stocks["shrout"].notna(),
        stocks["price"] * stocks["shrout"],
        np.nan
    )

    stocks["market_cap_usd_mn"] = stocks["market_cap"] / 1_000_000.0

stocks["log_market_cap"] = np.log(stocks["market_cap"].where(stocks["market_cap"] > 0))

stocks = stocks[
    (stocks["date"] >= START_DATE)
    & (stocks["date"] <= END_DATE)
].copy()

stocks = stocks.sort_values(["ticker", "permno", "date"])


print("\nCLEANED STOCK DATA")
print("Shape:", stocks.shape)
print("Date range:", stocks["date"].min(), "to", stocks["date"].max())
print("Tickers:", sorted(stocks["ticker"].dropna().unique()))

unexpected_tickers = sorted(set(stocks["ticker"].dropna()) - EXPECTED_TICKERS)
missing_tickers = sorted(EXPECTED_TICKERS - set(stocks["ticker"].dropna()))

print("\nTICKER CHECK")
print("Expected tickers:", sorted(EXPECTED_TICKERS))
print("Unexpected tickers:", unexpected_tickers)
print("Missing expected tickers:", missing_tickers)

ticker_check = pd.DataFrame({
    "expected_ticker": sorted(EXPECTED_TICKERS),
    "present": [t in set(stocks["ticker"].dropna()) for t in sorted(EXPECTED_TICKERS)]
})
ticker_check_path = DIAG_DIR / "crsp_expected_ticker_check.csv"
ticker_check.to_csv(ticker_check_path, index=False)


coverage_cols = ["ticker"]

if "permno" in stocks.columns:
    coverage_cols.append("permno")

coverage = (
    stocks
    .groupby(coverage_cols, dropna=False)
    .agg(
        n_obs=("date", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_dates=("date", "nunique"),
        missing_ret=("ret", lambda x: x.isna().sum()),
        missing_price=("price", lambda x: x.isna().sum()),
        missing_volume=("volume", lambda x: x.isna().sum()),
        missing_shrout=("shrout", lambda x: x.isna().sum()),
        negative_raw_price=("raw_price", lambda x: (x < 0).sum()),
        min_ret=("ret", "min"),
        max_ret=("ret", "max"),
        mean_ret=("ret", "mean"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        mean_market_cap=("market_cap", "mean"),
    )
    .reset_index()
    .sort_values(["ticker", "first_date"])
)

coverage_path = DIAG_DIR / "crsp_equity_coverage_by_ticker_permno.csv"
coverage.to_csv(coverage_path, index=False)

print("\nCOVERAGE BY TICKER/PERMNO")
print(coverage)
print(f"Saved: {coverage_path}")


duplicates_ticker_date = stocks.duplicated(["ticker", "date"], keep=False)
duplicates_permno_date = stocks.duplicated(["permno", "date"], keep=False) if stocks["permno"].notna().any() else pd.Series(False, index=stocks.index)

dup_ticker_date_df = stocks.loc[
    duplicates_ticker_date,
    ["date", "ticker", "permno", "company_name", "price", "ret", "volume", "market_cap"]
].sort_values(["ticker", "date"])

dup_permno_date_df = stocks.loc[
    duplicates_permno_date,
    ["date", "ticker", "permno", "company_name", "price", "ret", "volume", "market_cap"]
].sort_values(["permno", "date"])

dup_ticker_date_path = DIAG_DIR / "crsp_duplicate_ticker_date_rows.csv"
dup_permno_date_path = DIAG_DIR / "crsp_duplicate_permno_date_rows.csv"

dup_ticker_date_df.to_csv(dup_ticker_date_path, index=False)
dup_permno_date_df.to_csv(dup_permno_date_path, index=False)

print("\nDUPLICATE CHECKS")
print("Duplicate ticker-date rows:", len(dup_ticker_date_df))
print("Duplicate permno-date rows:", len(dup_permno_date_df))
print(f"Saved: {dup_ticker_date_path}")
print(f"Saved: {dup_permno_date_path}")


business_coverage_rows = []

for ticker, g in stocks.groupby("ticker"):
    start = g["date"].min()
    end = g["date"].max()

    expected_bdays_nyse = business_dates_custom(
        start=start,
        end=end,
        calendar=NYSE_BUSINESS_CALENDAR,
    )

    expected_bdays_pandas = pd.bdate_range(start, end)

    observed_dates = pd.DatetimeIndex(
        pd.to_datetime(g["date"].dropna().unique())
    ).normalize().sort_values()

    missing_bdays_nyse = expected_bdays_nyse.difference(observed_dates)
    missing_bdays_pandas = expected_bdays_pandas.difference(observed_dates)

    business_coverage_rows.append(
        {
            "ticker": ticker,
            "calendar": "NYSE",
            "first_date": start,
            "last_date": end,
            "expected_business_days_nyse": len(expected_bdays_nyse),
            "expected_business_days_pandas_bday": len(expected_bdays_pandas),
            "observed_dates": len(observed_dates),
            "missing_business_days_nyse": len(missing_bdays_nyse),
            "missing_business_days_pandas_bday": len(missing_bdays_pandas),
            "first_missing_business_day_nyse": (
                missing_bdays_nyse.min()
                if len(missing_bdays_nyse) > 0
                else pd.NaT
            ),
            "last_missing_business_day_nyse": (
                missing_bdays_nyse.max()
                if len(missing_bdays_nyse) > 0
                else pd.NaT
            ),
        }
    )

business_coverage = pd.DataFrame(business_coverage_rows)

business_coverage_path = DIAG_DIR / "crsp_calendar_coverage_nyse.csv"
business_coverage.to_csv(business_coverage_path, index=False)

legacy_business_coverage_path = DIAG_DIR / "crsp_business_day_coverage_by_ticker.csv"
business_coverage.to_csv(legacy_business_coverage_path, index=False)

print("\nNYSE CALENDAR COVERAGE BY TICKER")
print(business_coverage)
print(f"Saved: {business_coverage_path}")
print(f"Saved legacy copy: {legacy_business_coverage_path}")


return_quantiles = (
    stocks
    .groupby("ticker")["ret"]
    .quantile([0.001, 0.005, 0.01, 0.05, 0.50, 0.95, 0.99, 0.995, 0.999])
    .rename("ret_quantile")
    .reset_index()
    .rename(columns={"level_1": "quantile"})
)

return_quantiles_path = DIAG_DIR / "crsp_return_quantiles_by_ticker.csv"
return_quantiles.to_csv(return_quantiles_path, index=False)

extreme_returns = stocks.loc[
    stocks["ret"].abs() > 0.10,
    [
        "date", "ticker", "permno", "company_name",
        "price", "adj_price", "ret", "volume", "market_cap"
    ]
].sort_values(["date", "ticker"])

extreme_returns_path = DIAG_DIR / "crsp_extreme_returns_abs_gt_10pct.csv"
extreme_returns.to_csv(extreme_returns_path, index=False)

large_price_moves = stocks.copy()
large_price_moves["log_adj_price"] = np.log(large_price_moves["adj_price"].where(large_price_moves["adj_price"] > 0))
large_price_moves["log_adj_price_change"] = (
    large_price_moves
    .groupby("ticker")["log_adj_price"]
    .diff()
)

large_price_moves_df = large_price_moves.loc[
    large_price_moves["log_adj_price_change"].abs() > 0.10,
    [
        "date", "ticker", "permno", "company_name",
        "adj_price", "log_adj_price_change", "ret", "volume", "market_cap"
    ]
].sort_values(["date", "ticker"])

large_price_moves_path = DIAG_DIR / "crsp_large_adjusted_price_moves_abs_gt_10pct.csv"
large_price_moves_df.to_csv(large_price_moves_path, index=False)

print("\nRETURN OUTLIERS")
print("Rows with abs(ret) > 10%:", len(extreme_returns))
print(extreme_returns.head(30))
print(f"Saved: {extreme_returns_path}")

print("\nLARGE ADJUSTED PRICE MOVES")
print("Rows with abs(log adjusted price change) > 10%:", len(large_price_moves_df))
print(f"Saved: {large_price_moves_path}")


ticker_permno_map = (
    stocks
    .groupby(["ticker", "permno"], dropna=False)
    .agg(
        n_obs=("date", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        company_name=("company_name", "last"),
    )
    .reset_index()
    .sort_values(["ticker", "n_obs"], ascending=[True, False])
)

ticker_permno_map_path = DIAG_DIR / "crsp_ticker_permno_map.csv"
ticker_permno_map.to_csv(ticker_permno_map_path, index=False)

permno_ticker_counts = (
    ticker_permno_map
    .groupby("ticker")
    .agg(
        n_permno=("permno", "nunique"),
        total_obs=("n_obs", "sum"),
    )
    .reset_index()
)

permno_ticker_counts_path = DIAG_DIR / "crsp_permno_count_by_ticker.csv"
permno_ticker_counts.to_csv(permno_ticker_counts_path, index=False)

print("\nTICKER / PERMNO MAP")
print(ticker_permno_map)
print(f"Saved: {ticker_permno_map_path}")

print("\nPERMNO COUNT BY TICKER")
print(permno_ticker_counts)
print(f"Saved: {permno_ticker_counts_path}")


stocks_clean = (
    stocks
    .sort_values(["ticker", "date", "market_cap"], ascending=[True, True, False])
    .drop_duplicates(["ticker", "date"], keep="first")
    .copy()
)

stocks_clean = stocks_clean.sort_values(["ticker", "date"])


stocks_clean["equity_vol_20d"] = (
    stocks_clean
    .groupby("ticker")["ret"]
    .transform(lambda x: x.rolling(window=20, min_periods=15).std() * np.sqrt(252))
)

stocks_clean["equity_vol_60d"] = (
    stocks_clean
    .groupby("ticker")["ret"]
    .transform(lambda x: x.rolling(window=60, min_periods=40).std() * np.sqrt(252))
)

stocks_clean["d_log_market_cap"] = (
    stocks_clean
    .groupby("ticker")["log_market_cap"]
    .diff()
)

equity_cols = [
    "date",
    "ticker",
    "permno",
    "permco",
    "company_name",
    "price",
    "adj_price",
    "ret",
    "retx",
    "ret_with_dlret",
    "volume",
    "shrout",
    "market_cap_raw",
    "market_cap",
    "market_cap_usd_mn",
    "log_market_cap",
    "d_log_market_cap",
    "equity_vol_20d",
    "equity_vol_60d",
    "exchcd",
    "shrcd",
]

equity_cols = [col for col in equity_cols if col in stocks_clean.columns]

equity_daily = stocks_clean[equity_cols].copy()

equity_daily_path = OUT_DIR / "crsp_us_banks_equity_daily.parquet"
equity_daily_csv_path = OUT_DIR / "crsp_us_banks_equity_daily.csv"

equity_daily.to_parquet(equity_daily_path, index=False)
equity_daily.to_csv(equity_daily_csv_path, index=False)

print("\nCLEAN EQUITY DAILY DATASET")
print(equity_daily.shape)
print(equity_daily.head())
print(f"Saved: {equity_daily_path}")
print(f"Saved CSV: {equity_daily_csv_path}")


wide_ret = equity_daily.pivot(index="date", columns="ticker", values="ret")
wide_ret.columns = [f"eq_ret_{c.lower()}" for c in wide_ret.columns]

wide_vol20 = equity_daily.pivot(index="date", columns="ticker", values="equity_vol_20d")
wide_vol20.columns = [f"eq_vol20_{c.lower()}" for c in wide_vol20.columns]

wide_mcap = equity_daily.pivot(index="date", columns="ticker", values="log_market_cap")
wide_mcap.columns = [f"log_mcap_{c.lower()}" for c in wide_mcap.columns]

equity_wide = (
    wide_ret
    .join(wide_vol20, how="outer")
    .join(wide_mcap, how="outer")
    .reset_index()
    .sort_values("date")
)

equity_wide_path = OUT_DIR / "crsp_us_banks_equity_wide.parquet"
equity_wide_csv_path = OUT_DIR / "crsp_us_banks_equity_wide.csv"

equity_wide.to_parquet(equity_wide_path, index=False)
equity_wide.to_csv(equity_wide_csv_path, index=False)

print("\nWIDE EQUITY DRIVER DATASET")
print(equity_wide.shape)
print(equity_wide.head())
print(f"Saved: {equity_wide_path}")
print(f"Saved CSV: {equity_wide_csv_path}")


final_summary = {
    "raw_rows": len(stocks_raw),
    "clean_rows": len(stocks),
    "equity_daily_rows": len(equity_daily),
    "wide_dates": len(equity_wide),
    "tickers_found": ", ".join(sorted(stocks["ticker"].dropna().unique())),
    "first_date": stocks["date"].min(),
    "last_date": stocks["date"].max(),
    "duplicate_ticker_date_rows": len(dup_ticker_date_df),
    "extreme_return_rows_abs_gt_10pct": len(extreme_returns),
}

final_summary_df = pd.DataFrame([final_summary])
final_summary_path = DIAG_DIR / "crsp_equity_final_summary.csv"
final_summary_df.to_csv(final_summary_path, index=False)

print("\nFINAL CRSP EQUITY SUMMARY")
print(final_summary_df.T)
print(f"Saved: {final_summary_path}")

print("\nCRSP EQUITY DIAGNOSTICS COMPLETE.")