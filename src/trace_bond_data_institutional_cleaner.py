from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from bond_cashflow_utils import add_dirty_price_columns

from config_institutional import (
    DATA_DIR,
    PROCESSED_DIR,
    TRACE_PARQUET_DIR,
    BOND_DAY_DIR,
    TRACE_MASTER_ZIP,
    TRACE_RAW_PATTERN,
    TRACE_HOLIDAY_PATH,
    FIGURES_DIR,
    TRACE_MODEL_READY_DIRTY_PATH,
    TRACE_MODEL_READY_GAP5_DIRTY_PATH,
    ensure_directories,
)

from calendar_utils import (
    business_gap_custom,
    business_gap_pandas_bday,
    load_market_holidays,
)

try:
    from config_institutional import TRACE_CLEANED_TRADE_DIR, TRACE_CLEANING_DIAG_DIR
except ImportError:
    TRACE_CLEANED_TRADE_DIR = PROCESSED_DIR / "trace_cleaned_trades"
    TRACE_CLEANING_DIAG_DIR = PROCESSED_DIR / "diagnostics" / "trace_cleaning"

print("TRACE_CLEANED_TRADE_DIR:", TRACE_CLEANED_TRADE_DIR)
print("TRACE_CLEANING_DIAG_DIR:", TRACE_CLEANING_DIAG_DIR)


ensure_directories()
TRACE_BUSINESS_CALENDAR = load_market_holidays(
    TRACE_HOLIDAY_PATH,
    calendar_name="TRACE OTC / SIFMA",
)
FORCE_REBUILD = False
FORCE_REBUILD_CLEANED_TRADES = False
FILE_PATTERN = TRACE_RAW_PATTERN
CHUNKSIZE = 500_000

RAW_DIR = DATA_DIR
RAW_FILES = sorted(RAW_DIR.glob(FILE_PATTERN))

PARQUET_DIR = TRACE_PARQUET_DIR
DIAG_DIR = PROCESSED_DIR / "diagnostics"

for path in [PARQUET_DIR, BOND_DAY_DIR, DIAG_DIR, TRACE_CLEANED_TRADE_DIR, TRACE_CLEANING_DIAG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

if not RAW_FILES:
    print("Script location:", Path(__file__).resolve())
    print("Current working directory:", Path.cwd())
    print("RAW_DIR checked:", RAW_DIR.resolve())
    print("Files in RAW_DIR:", [p.name for p in RAW_DIR.glob("*")])
    raise FileNotFoundError(f"No {FILE_PATTERN} files found in {RAW_DIR}")

print("Script location:", Path(__file__).resolve())
print("Current working directory:", Path.cwd())
print("Using RAW_DIR:", RAW_DIR.resolve())
print("Zip files found:", [p.name for p in RAW_FILES])

REGULAR_STATUS_VALUES = {
    "T", "TRADE", "NEW", "N", "M", "O", "RPT", "REPORT", "NORMAL", "A", "ACCEPTED"
}
REGULAR_FRMT_VALUES = {
    "A", "E"
}
CANCEL_STATUS_VALUES = {
    "C", "X", "CANCEL", "CANCELLED", "CANCELED", "DELETE", "DELETED", "VOID"
}
CORRECTION_STATUS_VALUES = {
    "W", "Y", "CORR", "CORRECT", "CORRECTED", "CORRECTION", "AMEND", "AMENDED"
}
REVERSAL_STATUS_VALUES = {
    "R", "REV", "REVERSE", "REVERSED", "REVERSAL"
}

SPECIAL_FRMT_VALUES = {
    "CANCEL", "CANCELLED", "CANCELED", "CORR", "CORRECT", "CORRECTION",
    "REV", "REVERSAL", "REVERSED", "SPECIAL", "ASOF", "AS-OF", "WHENISSUED",
    "WHEN_ISSUED", "WI", "HALT", "HALTED", "INVALID", "DELETE", "DELETED"
}

MIN_PRICE = 20.0
MAX_PRICE = 200.0
MIN_YIELD = -5.0
MAX_YIELD = 30.0
INSTITUTIONAL_TRADE_SIZE = 100_000.0

MIN_ACTIVE_DAYS = 100
MIN_TOTAL_TRADES = 250
MODEL_READY_MAX_BUSINESS_GAP = 5
GAP_THRESHOLDS = [1, 3, 5, 10]
MAX_SENSITIVITY_GAP = max(GAP_THRESHOLDS)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, ".": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan}),
        errors="coerce",
    )


def clean_string_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.upper().str.strip()
    out = out.replace(
        {
            "": np.nan,
            ".": np.nan,
            "NAN": np.nan,
            "NA": np.nan,
            "NONE": np.nan,
            "<NA>": np.nan,
        }
    )
    return out


def clean_id_series(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    raw = raw.replace(
        {
            "": np.nan,
            ".": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "None": np.nan,
            "<NA>": np.nan,
        }
    )

    numeric = pd.to_numeric(raw, errors="coerce")
    integer_like = numeric.notna() & np.isclose(numeric % 1, 0)

    out = raw.copy()
    if integer_like.any():
        out.loc[integer_like] = numeric.loc[integer_like].astype("Int64").astype(str)

    return out


def business_day_gap(start: pd.Series, end: pd.Series) -> pd.Series:
    return business_gap_custom(
        start=start,
        end=end,
        calendar=TRACE_BUSINESS_CALENDAR,
    )

def _value_to_str(x) -> str:
    if pd.isna(x):
        return ""

    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d")

    return str(x)


def _compact_unique_values(s: pd.Series, max_values: int = 8) -> str:
    values = pd.Series(s.dropna().unique())

    if values.empty:
        return ""

    values = values.sort_values() if not pd.api.types.is_object_dtype(values) else values.astype(str).sort_values()
    shown = values.head(max_values).map(_value_to_str).tolist()

    suffix = "" if len(values) <= max_values else f" | ... +{len(values) - max_values} more"
    return " | ".join(shown) + suffix


def audit_trace_volume_bunching(
    df: pd.DataFrame,
    vol_col: str,
    sample_name: str,
    knots: list[float] | None = None,
    round_units: list[float] | None = None,
    hist_sample_size: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if knots is None:
        knots = [1e5, 5e5, 1e6, 5e6]

    if round_units is None:
        round_units = [1e3, 5e3, 1e4, 2.5e4, 5e4, 1e5, 5e5, 1e6]

    if vol_col not in df.columns:
        summary = pd.DataFrame(
            [
                {
                    "sample": sample_name,
                    "volume_column": vol_col,
                    "n_rows": len(df),
                    "n_valid_volume": 0,
                    "note": "volume column not found",
                }
            ]
        )
        return summary, pd.DataFrame(), np.array([])

    raw = pd.to_numeric(df[vol_col], errors="coerce")
    valid = raw.loc[raw.notna() & raw.gt(0)].astype(float)

    row = {
        "sample": sample_name,
        "volume_column": vol_col,
        "n_rows": int(len(df)),
        "n_missing_or_nonpositive_volume": int((raw.isna() | raw.le(0)).sum()),
        "n_valid_volume": int(len(valid)),
        "share_valid_volume": float(len(valid) / len(df)) if len(df) else np.nan,
        "min_volume": float(valid.min()) if len(valid) else np.nan,
        "median_volume": float(valid.median()) if len(valid) else np.nan,
        "mean_volume": float(valid.mean()) if len(valid) else np.nan,
        "p95_volume": float(valid.quantile(0.95)) if len(valid) else np.nan,
        "p99_volume": float(valid.quantile(0.99)) if len(valid) else np.nan,
        "max_volume": float(valid.max()) if len(valid) else np.nan,
    }

    for knot in knots:
        label = int(knot)
        if len(valid):
            exact = np.isclose(valid.to_numpy(), knot, rtol=0.0, atol=1e-9)
            near_1pct = np.abs(valid.to_numpy() - knot) <= 0.01 * knot
            near_5pct = np.abs(valid.to_numpy() - knot) <= 0.05 * knot

            row[f"n_exact_{label}"] = int(exact.sum())
            row[f"share_exact_{label}"] = float(exact.mean())
            row[f"share_within_1pct_{label}"] = float(near_1pct.mean())
            row[f"share_within_5pct_{label}"] = float(near_5pct.mean())
        else:
            row[f"n_exact_{label}"] = 0
            row[f"share_exact_{label}"] = np.nan
            row[f"share_within_1pct_{label}"] = np.nan
            row[f"share_within_5pct_{label}"] = np.nan

    round_rows = []

    for unit in round_units:
        if len(valid):
            x = valid.to_numpy()
            multiple = np.isclose(np.mod(x, unit), 0.0, rtol=0.0, atol=1e-9)

            round_rows.append(
                {
                    "sample": sample_name,
                    "volume_column": vol_col,
                    "round_unit": float(unit),
                    "n_valid_volume": int(len(valid)),
                    "n_multiple": int(multiple.sum()),
                    "share_multiple": float(multiple.mean()),
                }
            )

    summary = pd.DataFrame([row])
    round_table = pd.DataFrame(round_rows)

    if len(valid) > hist_sample_size:
        hist_sample = valid.sample(
            n=hist_sample_size,
            random_state=42,
            replace=False,
        ).to_numpy()
    else:
        hist_sample = valid.to_numpy()

    hist_values = np.log10(hist_sample[hist_sample > 0]) if len(hist_sample) else np.array([])

    return summary, round_table, hist_values


def write_trace_volume_bunching_outputs(
    summary_parts: list[pd.DataFrame],
    round_table_parts: list[pd.DataFrame],
    hist_parts: list[np.ndarray],
) -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if summary_parts:
        summary = pd.concat(summary_parts, ignore_index=True)
    else:
        summary = pd.DataFrame()

    if round_table_parts:
        round_table = pd.concat(round_table_parts, ignore_index=True)
    else:
        round_table = pd.DataFrame()

    summary_path = DIAG_DIR / "trace_volume_bunching_summary.csv"
    round_path = DIAG_DIR / "trace_volume_round_number_table.csv"

    summary.to_csv(summary_path, index=False)
    round_table.to_csv(round_path, index=False)

    hist_values = [
        values
        for values in hist_parts
        if values is not None and len(values) > 0
    ]

    if hist_values:
        x = np.concatenate(hist_values)

        plt.figure(figsize=(7, 4.5))
        plt.hist(x, bins=80)
        plt.xlabel("log10 reported TRACE trade volume")
        plt.ylabel("Frequency")
        plt.title("TRACE reported-volume distribution")
        plt.tight_layout()

        fig_path = FIGURES_DIR / "trace_log_volume_histogram.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved TRACE log-volume histogram to: {fig_path}")

    print("\nTRACE VOLUME BUNCHING AUDIT")
    print(summary.head(20))
    print(f"Saved volume bunching summary to: {summary_path}")
    print(f"Saved round-number table to: {round_path}")


def audit_master_pit_consistency(
    master: pd.DataFrame,
    pit_panel: pd.DataFrame,
    static_cols: list[str],
    maturity_col: str | None = None,
    output_prefix: str = "master_pit",
) -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    master = master.copy()
    pit_panel = pit_panel.copy()

    master["cusip_id"] = master["cusip_id"].astype(str).str.strip()
    master["stdt_effective"] = pd.to_datetime(master["stdt_effective"], errors="coerce")
    master["enddt_effective"] = pd.to_datetime(master["enddt_effective"], errors="coerce")


    invalid_interval = (
        master["stdt_effective"].isna()
        | master["enddt_effective"].isna()
        | master["stdt_effective"].gt(master["enddt_effective"])
    )

    invalid_cols = [
        c
        for c in [
            "cusip_id",
            "stdt",
            "enddt",
            "stdt_effective",
            "enddt_effective",
            maturity_col,
            "cpn_rt",
            "cpn_type_cd",
            "debt_type_cd",
            "scrty_type_cd",
            "scrty_sbtp_cd",
            "sub_prdct_type",
        ]
        if c and c in master.columns
    ]

    invalid_rows = master.loc[invalid_interval, invalid_cols].copy()

    invalid_path = DIAG_DIR / f"{output_prefix}_invalid_interval_rows.csv"
    invalid_rows.to_csv(invalid_path, index=False)


    m = master.sort_values(
        ["cusip_id", "stdt_effective", "enddt_effective"]
    ).copy()

    m["prev_stdt_effective"] = m.groupby("cusip_id")["stdt_effective"].shift(1)
    m["prev_enddt_effective"] = m.groupby("cusip_id")["enddt_effective"].shift(1)

    overlap_mask = (
        m["prev_enddt_effective"].notna()
        & m["stdt_effective"].le(m["prev_enddt_effective"])
    )

    overlap_cols = [
        c
        for c in [
            "cusip_id",
            "prev_stdt_effective",
            "prev_enddt_effective",
            "stdt_effective",
            "enddt_effective",
            maturity_col,
            "cpn_rt",
            "cpn_type_cd",
            "debt_type_cd",
            "scrty_type_cd",
            "scrty_sbtp_cd",
            "sub_prdct_type",
        ]
        if c and c in m.columns
    ]

    overlaps = m.loc[overlap_mask, overlap_cols].copy()

    overlap_path = DIAG_DIR / f"{output_prefix}_overlap_by_cusip.csv"
    overlaps.to_csv(overlap_path, index=False)


    pit_panel["date"] = pd.to_datetime(pit_panel["date"], errors="coerce")
    pit_panel["cusip_id"] = pit_panel["cusip_id"].astype(str).str.strip()

    audit_rows = []

    for col in static_cols:
        if col not in pit_panel.columns:
            continue

        temp = pit_panel[["cusip_id", "date", col]].dropna(subset=[col]).copy()

        for cusip, g in temp.groupby("cusip_id", sort=False):
            n_unique = int(g[col].nunique(dropna=True))

            if n_unique <= 1:
                continue

            audit_rows.append(
                {
                    "cusip_id": cusip,
                    "field": col,
                    "n_unique_nonmissing_values": n_unique,
                    "n_rows": int(len(g)),
                    "first_date": g["date"].min(),
                    "last_date": g["date"].max(),
                    "values": _compact_unique_values(g[col]),
                }
            )

    static_audit = pd.DataFrame(audit_rows)

    static_path = DIAG_DIR / f"{output_prefix}_static_field_change_audit.csv"
    static_audit.to_csv(static_path, index=False)


    ytm_path = DIAG_DIR / f"{output_prefix}_years_to_maturity_nonmonotonic_rows.csv"

    if "years_to_maturity" in pit_panel.columns:
        y = pit_panel.sort_values(["cusip_id", "date"]).copy()
        y["years_to_maturity_change"] = y.groupby("cusip_id")["years_to_maturity"].diff()

        nonmonotonic = y["years_to_maturity_change"].gt(1e-6).fillna(False)

        ytm_cols = [
            c
            for c in [
                "cusip_id",
                "date",
                "prev_date",
                "trace_company_symbol",
                "stdt_effective",
                "enddt_effective",
                maturity_col,
                "years_to_maturity",
                "years_to_maturity_change",
                "cpn_rt",
                "cpn_type_cd",
                "debt_type_cd",
                "scrty_type_cd",
                "scrty_sbtp_cd",
            ]
            if c and c in y.columns
        ]

        y.loc[nonmonotonic, ytm_cols].to_csv(ytm_path, index=False)
        n_nonmonotonic = int(nonmonotonic.sum())
    else:
        pd.DataFrame().to_csv(ytm_path, index=False)
        n_nonmonotonic = 0

    print("\nMASTER PIT CONSISTENCY AUDIT")
    print("Invalid interval rows:", len(invalid_rows))
    print("Overlapping interval rows:", len(overlaps))
    print("Static-field CUSIP-field changes:", len(static_audit))
    print("Non-monotonic years_to_maturity rows:", n_nonmonotonic)
    print(f"Saved invalid intervals to: {invalid_path}")
    print(f"Saved overlaps to: {overlap_path}")
    print(f"Saved static-field audit to: {static_path}")
    print(f"Saved years-to-maturity monotonicity audit to: {ytm_path}")

def iter_csv_from_zip(zip_path: Path, chunksize: int = CHUNKSIZE):
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            name for name in zf.namelist()
            if not name.endswith("/") and name.lower().endswith((".csv", ".txt"))
        ]

        if not members:
            raise ValueError(f"No CSV/TXT file found inside {zip_path.name}")

        with zf.open(members[0]) as fh:
            for chunk in pd.read_csv(fh, chunksize=chunksize, low_memory=False):
                yield normalise_columns(chunk)


def read_sample(zip_path: Path, nrows: int = 5) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            name for name in zf.namelist()
            if not name.endswith("/") and name.lower().endswith((".csv", ".txt"))
        ]

        if not members:
            raise ValueError(f"No CSV/TXT file found inside {zip_path.name}")

        with zf.open(members[0]) as fh:
            return normalise_columns(pd.read_csv(fh, nrows=nrows, low_memory=False))


def read_wrds_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            name for name in zf.namelist()
            if not name.endswith("/") and name.lower().endswith((".csv", ".txt"))
        ]

        if not members:
            raise ValueError(f"No CSV/TXT file found inside {zip_path.name}")

        member = members[0]
        print(f"Reading {member} from {zip_path.name}")
        with zf.open(member) as fh:
            df = pd.read_csv(fh, low_memory=False)

    return normalise_columns(df)


def clean_trade_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = normalise_columns(df)

    date_col = pick_col(df, ["trd_exctn_dt"])
    time_col = pick_col(df, ["trd_exctn_tm"])
    price_col = pick_col(df, ["rptd_pr", "rptd_prc", "price"])
    volume_col = pick_col(df, ["entrd_vol_qt", "ascii_rptd_vol_tx", "rptd_vol_qt"])
    yield_col = pick_col(df, ["yld_pt", "yield", "yld"])

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")

    if time_col:
        df[time_col] = df[time_col].astype(str).str.strip()
        if date_col:
            df["execution_ts"] = pd.to_datetime(
                df[date_col].dt.strftime("%Y-%m-%d") + " " + df[time_col],
                errors="coerce",
            )
    elif date_col:
        df["execution_ts"] = df[date_col]

    for col in [price_col, volume_col, yield_col]:
        if col and col in df.columns:
            df[col] = to_numeric_safe(df[col])

    for col in ["trc_st", "frmt_cd", "side", "rptg_party_type", "contra_party_type", "ats_indicator", "sub_prd_type", "sub_prdct_type", "company_symbol"]:
        if col in df.columns:
            df[col] = clean_string_series(df[col])

    for col in ["msg_seq_nb", "orig_msg_seq_nb"]:
        if col in df.columns:
            df[col] = clean_id_series(df[col])

    if "cusip_id" in df.columns:
        df["cusip_id"] = df["cusip_id"].astype(str).str.strip()

    return df


def classify_status_values(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for field, regular, cancel, correction, reversal, special in [
        ("trc_st", REGULAR_STATUS_VALUES, CANCEL_STATUS_VALUES, CORRECTION_STATUS_VALUES, REVERSAL_STATUS_VALUES, set()),
        ("frmt_cd", REGULAR_FRMT_VALUES, CANCEL_STATUS_VALUES, CORRECTION_STATUS_VALUES, REVERSAL_STATUS_VALUES, SPECIAL_FRMT_VALUES),
    ]:
        if field not in df.columns:
            continue

        vc = df[field].astype("object").where(df[field].notna(), "<MISSING>").value_counts(dropna=False)
        for value, count in vc.items():
            value_str = str(value).upper().strip()
            if value_str in cancel:
                action = "drop_report_and_referenced_original"
            elif value_str in reversal:
                action = "drop_report_and_referenced_original"
            elif value_str in correction:
                action = "drop_referenced_original_keep_correcting_report_if_valid"
            elif value_str in special:
                action = "drop_special_condition_from_price_sample"
            elif value_str in regular:
                action = "regular_or_new_trade"
            elif value_str == "<MISSING>":
                action = "missing_value_no_action_by_itself"
            else:
                action = "unmapped_diagnose_before_tightening"

            rows.append({"field": field, "value": value, "count": int(count), "cleaner_action": action})

    return pd.DataFrame(rows)


def apply_trace_cleaner(df: pd.DataFrame, file_name: str = "") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    cusip_col = pick_col(df, ["cusip_id"])
    date_col = pick_col(df, ["trd_exctn_dt"])
    time_col = pick_col(df, ["trd_exctn_tm"])
    price_col = pick_col(df, ["rptd_pr", "rptd_prc", "price"])
    volume_col = pick_col(df, ["entrd_vol_qt", "ascii_rptd_vol_tx", "rptd_vol_qt"])
    yield_col = pick_col(df, ["yld_pt", "yield", "yld"])
    subprd_col = pick_col(df, ["sub_prd_type", "sub_prdct_type", "sub_product_type"])

    required = [cusip_col, date_col, price_col, volume_col]
    if any(col is None for col in required):
        raise ValueError(f"Missing required TRACE columns in {file_name}. Available columns: {df.columns.tolist()}")

    status = df["trc_st"] if "trc_st" in df.columns else pd.Series(np.nan, index=df.index)
    frmt = df["frmt_cd"] if "frmt_cd" in df.columns else pd.Series(np.nan, index=df.index)

    df["is_cancel_report"] = status.isin(CANCEL_STATUS_VALUES) | frmt.isin(CANCEL_STATUS_VALUES)
    df["is_correction_report"] = status.isin(CORRECTION_STATUS_VALUES) | frmt.isin(CORRECTION_STATUS_VALUES)
    df["is_reversal_report"] = status.isin(REVERSAL_STATUS_VALUES) | frmt.isin(REVERSAL_STATUS_VALUES)
    df["is_special_condition"] = frmt.isin(SPECIAL_FRMT_VALUES)

    if "orig_msg_seq_nb" in df.columns:
        df["has_orig_msg_seq_nb"] = df["orig_msg_seq_nb"].notna()
        referenced_ids = set(df.loc[df["has_orig_msg_seq_nb"], "orig_msg_seq_nb"].dropna().astype(str))
    else:
        df["has_orig_msg_seq_nb"] = False
        referenced_ids = set()

    if "msg_seq_nb" in df.columns and referenced_ids:
        df["is_referenced_original"] = df["msg_seq_nb"].astype(str).isin(referenced_ids)
    else:
        df["is_referenced_original"] = False

    df["valid_date"] = df[date_col].notna()
    df["valid_cusip"] = df[cusip_col].notna() & df[cusip_col].astype(str).str.strip().ne("")
    df["valid_price"] = df[price_col].between(MIN_PRICE, MAX_PRICE)
    df["valid_volume"] = df[volume_col].gt(0)

    if yield_col:
        df["valid_yield"] = df[yield_col].isna() | df[yield_col].between(MIN_YIELD, MAX_YIELD)
    else:
        df["valid_yield"] = True

    if subprd_col:
        df["is_corporate_trade"] = df[subprd_col].astype(str).str.upper().str.strip().eq("CORP")
    else:
        df["is_corporate_trade"] = True

    duplicate_key = [
        c for c in [cusip_col, date_col, time_col, "execution_ts", price_col, volume_col, "side", "rptg_party_type", "contra_party_type", "msg_seq_nb"]
        if c is not None and c in df.columns
    ]
    df["is_exact_duplicate"] = df.duplicated(duplicate_key, keep="first") if duplicate_key else False

    agency_key = [c for c in [cusip_col, "execution_ts", price_col, volume_col] if c is not None and c in df.columns]
    df["is_potential_agency_duplicate"] = df.duplicated(agency_key, keep=False) if agency_key else False

    df["basic_valid_trade"] = (
        df["valid_date"]
        & df["valid_cusip"]
        & df["is_corporate_trade"]
        & df["valid_price"]
        & df["valid_volume"]
        & df["valid_yield"]
    )

    df["clean_price_sample"] = (
        df["basic_valid_trade"]
        & ~df["is_cancel_report"]
        & ~df["is_reversal_report"]
        & ~df["is_referenced_original"]
        & ~df["is_exact_duplicate"]
        & ~df["is_special_condition"]
    )

    df["clean_volume_sample"] = (
        df["valid_date"]
        & df["valid_cusip"]
        & df["is_corporate_trade"]
        & df["valid_volume"]
        & ~df["is_cancel_report"]
        & ~df["is_reversal_report"]
        & ~df["is_referenced_original"]
        & ~df["is_exact_duplicate"]
    )

    df["clean_institutional_price_sample"] = (
        df["clean_price_sample"] & df[volume_col].ge(INSTITUTIONAL_TRADE_SIZE)
    )
    df["clean_price_no_agency_sample"] = (
        df["clean_price_sample"] & ~df["is_potential_agency_duplicate"]
    )

    waterfall = build_trade_cleaning_waterfall(df, file_name=file_name)
    mapping = classify_status_values(df)
    if not mapping.empty:
        mapping.insert(0, "file", file_name)

    return df, waterfall, mapping


def build_trade_cleaning_waterfall(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    steps: list[tuple[str, pd.Series]] = [
        ("raw_rows", pd.Series(True, index=df.index)),
        ("valid_date_and_cusip", df["valid_date"] & df["valid_cusip"]),
        ("corporate_trades", df["valid_date"] & df["valid_cusip"] & df["is_corporate_trade"]),
        ("valid_price_volume_yield", df["basic_valid_trade"]),
        ("drop_cancellations_reversals", df["basic_valid_trade"] & ~df["is_cancel_report"] & ~df["is_reversal_report"]),
        ("drop_referenced_originals", df["basic_valid_trade"] & ~df["is_cancel_report"] & ~df["is_reversal_report"] & ~df["is_referenced_original"]),
        ("drop_exact_duplicates", df["basic_valid_trade"] & ~df["is_cancel_report"] & ~df["is_reversal_report"] & ~df["is_referenced_original"] & ~df["is_exact_duplicate"]),
        ("clean_price_sample", df["clean_price_sample"]),
        ("clean_volume_sample", df["clean_volume_sample"]),
        ("institutional_price_sample", df["clean_institutional_price_sample"]),
        ("price_no_agency_sample", df["clean_price_no_agency_sample"]),
    ]

    rows = []
    previous_count = None
    previous_cusips = None
    for step, mask in steps:
        mask = mask.fillna(False)
        n_rows = int(mask.sum())
        n_cusips = int(df.loc[mask, "cusip_id"].nunique()) if "cusip_id" in df.columns else np.nan
        rows.append(
            {
                "file": file_name,
                "step": step,
                "rows_remaining": n_rows,
                "cusips_remaining": n_cusips,
                "rows_lost_since_previous_step": 0 if previous_count is None else previous_count - n_rows,
                "cusips_lost_since_previous_step": 0 if previous_cusips is None else previous_cusips - n_cusips,
                "share_of_raw_rows": n_rows / len(df) if len(df) else np.nan,
            }
        )
        previous_count = n_rows
        previous_cusips = n_cusips

    return pd.DataFrame(rows)


def add_trade_side_fields(df: pd.DataFrame, volume_col: str) -> pd.DataFrame:
    df = df.copy()

    if "side" in df.columns:
        side = df["side"].astype(str).str.upper()
        df["_buy_volume"] = np.where(side.str.startswith("B"), df[volume_col], 0.0)
        df["_sell_volume"] = np.where(side.str.startswith("S"), df[volume_col], 0.0)
    else:
        df["_buy_volume"] = np.nan
        df["_sell_volume"] = np.nan

    if "ats_indicator" in df.columns:
        ats = df["ats_indicator"].astype(str).str.upper()
        df["_is_ats"] = np.where(ats.isin(["Y", "YES", "1", "TRUE", "T"]), 1.0, 0.0)
    else:
        df["_is_ats"] = np.nan

    return df


def aggregate_cleaned_year(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    cusip_col = pick_col(df, ["cusip_id"])
    date_col = pick_col(df, ["trd_exctn_dt"])
    price_col = pick_col(df, ["rptd_pr", "rptd_prc", "price"])
    volume_col = pick_col(df, ["entrd_vol_qt", "ascii_rptd_vol_tx", "rptd_vol_qt"])
    yield_col = pick_col(df, ["yld_pt", "yield", "yld"])

    df_price = df.loc[df["clean_price_sample"]].copy()
    df_volume = df.loc[df["clean_volume_sample"]].copy()

    if df_price.empty:
        print(f"WARNING: no clean price trades in {file_name}.")
        return pd.DataFrame()

    for temp in [df_price, df_volume]:
        temp.sort_values([cusip_col, date_col, "execution_ts"], inplace=True)

    df_price["_px_vol"] = df_price[price_col] * df_price[volume_col]
    df_price = add_trade_side_fields(df_price, volume_col)
    df_volume = add_trade_side_fields(df_volume, volume_col)

    group_cols = [cusip_col, date_col]

    price_agg = {
        "company_symbol": ("company_symbol", "first") if "company_symbol" in df_price.columns else (cusip_col, "size"),
        "bond_sym_id": ("bond_sym_id", "first") if "bond_sym_id" in df_price.columns else (cusip_col, "size"),
        "n_trades": (price_col, "size"),
        "price_sample_volume": (volume_col, "sum"),
        "median_price": (price_col, "median"),
        "first_price": (price_col, "first"),
        "last_price": (price_col, "last"),
        "min_price": (price_col, "min"),
        "max_price": (price_col, "max"),
        "price_std": (price_col, "std"),
        "px_vol_sum": ("_px_vol", "sum"),
        "institutional_price_trades": ("clean_institutional_price_sample", "sum"),
        "no_agency_price_trades": ("clean_price_no_agency_sample", "sum"),
        "potential_agency_duplicate_trades": ("is_potential_agency_duplicate", "sum"),
        "correction_report_trades": ("is_correction_report", "sum"),
    }

    if yield_col:
        price_agg.update(
            {
                "median_yield": (yield_col, "median"),
                "first_yield": (yield_col, "first"),
                "last_yield": (yield_col, "last"),
                "yield_std": (yield_col, "std"),
            }
        )

    daily_price = df_price.groupby(group_cols).agg(**price_agg).reset_index()
    daily_price["vwap_price"] = daily_price["px_vol_sum"] / daily_price["price_sample_volume"]
    daily_price = daily_price.drop(columns=["px_vol_sum"])

    volume_agg = {
        "volume_sample_n_trades": (volume_col, "size"),
        "total_volume": (volume_col, "sum"),
        "buy_volume": ("_buy_volume", "sum"),
        "sell_volume": ("_sell_volume", "sum"),
        "ats_trade_share": ("_is_ats", "mean"),
    }
    daily_volume = df_volume.groupby(group_cols).agg(**volume_agg).reset_index()

    daily = daily_price.merge(daily_volume, on=group_cols, how="left")
    daily["total_volume"] = daily["total_volume"].fillna(daily["price_sample_volume"])
    daily["volume_sample_n_trades"] = daily["volume_sample_n_trades"].fillna(daily["n_trades"])

    denom = daily["buy_volume"] + daily["sell_volume"]
    daily["buy_sell_imbalance"] = np.where(
        denom.gt(0),
        (daily["buy_volume"] - daily["sell_volume"]) / denom,
        np.nan,
    )

    daily["price_range"] = daily["max_price"] - daily["min_price"]
    daily["price_range_rel"] = daily["price_range"] / daily["vwap_price"]
    daily["price_dispersion_rel"] = daily["price_std"] / daily["vwap_price"]
    daily["institutional_trade_share"] = daily["institutional_price_trades"] / daily["n_trades"]
    daily["potential_agency_duplicate_share"] = daily["potential_agency_duplicate_trades"] / daily["n_trades"]
    daily["correction_report_share"] = daily["correction_report_trades"] / daily["n_trades"]

    return daily


def add_bond_day_returns(bond_day: pd.DataFrame) -> pd.DataFrame:
    bond_day = bond_day.copy()
    bond_day = bond_day.sort_values(["cusip_id", "date"])

    for price_col, return_col in [
        ("vwap_price", "vwap_return"),
        ("median_price", "median_price_return"),
        ("last_price", "last_price_return"),
    ]:
        if price_col in bond_day.columns:
            log_col = f"log_{price_col}"
            bond_day[log_col] = np.log(bond_day[price_col].where(bond_day[price_col] > 0))
            bond_day[return_col] = bond_day.groupby("cusip_id")[log_col].diff()

    if "median_yield" in bond_day.columns:
        bond_day["yield_change"] = bond_day.groupby("cusip_id")["median_yield"].diff()

    if {"vwap_return", "total_volume"}.issubset(bond_day.columns):
        bond_day["amihud_daily"] = np.where(
            bond_day["total_volume"].gt(0),
            bond_day["vwap_return"].abs() / bond_day["total_volume"],
            np.nan,
        )

    return bond_day


def convert_raw_zips_to_parquet() -> list[Path]:
    print("\nRAW FILES FOUND:")
    for file in RAW_FILES:
        print(f"- {file.name}")

    print("\nSAMPLE FROM FIRST FILE:")
    sample = read_sample(RAW_FILES[0], nrows=5)
    print(sample.head())
    print("\nColumns:")
    print(sample.columns.tolist())

    for file in RAW_FILES:
        out_file = PARQUET_DIR / f"{file.stem}.parquet"

        if out_file.exists() and not FORCE_REBUILD:
            print(f"\nSkipping {file.name}: parquet already exists.")
            continue

        if out_file.exists() and FORCE_REBUILD:
            print(f"\nDeleting existing parquet: {out_file.name}")
            out_file.unlink()

        print(f"\nProcessing raw file: {file.name}")
        parts = []
        for chunk in iter_csv_from_zip(file, chunksize=CHUNKSIZE):
            parts.append(clean_trade_chunk(chunk))

        df_year = pd.concat(parts, ignore_index=True)
        df_year.to_parquet(out_file, index=False)
        print(f"Saved: {out_file}")
        print(f"Shape: {df_year.shape}")

    return sorted(PARQUET_DIR.glob("us_banks_[0-9][0-9].parquet"))


def write_parquet_summary(parquet_files: Iterable[Path]) -> None:
    summary = []
    for file in parquet_files:
        print(f"\nSummarising {file.name}")
        df = pd.read_parquet(file)
        date_col = pick_col(df, ["trd_exctn_dt"])
        summary.append(
            {
                "file": file.name,
                "rows": len(df),
                "first_date": df[date_col].min() if date_col else None,
                "last_date": df[date_col].max() if date_col else None,
                "n_cusip": df["cusip_id"].nunique() if "cusip_id" in df.columns else None,
                "n_bond_sym": df["bond_sym_id"].nunique() if "bond_sym_id" in df.columns else None,
                "n_issuer": df["company_symbol"].nunique() if "company_symbol" in df.columns else None,
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_path = PROCESSED_DIR / "trace_file_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\nTRACE FILE SUMMARY:")
    print(summary_df)
    print(f"\nSaved summary to: {summary_path}")


def build_clean_bond_day(parquet_files: Iterable[Path]) -> pd.DataFrame:
    daily_parts = []
    waterfalls = []
    mappings = []
    flag_summaries = []
    volume_audit_summaries = []
    volume_round_tables = []
    volume_hist_parts = []

    for file in parquet_files:
        print(f"\nCleaning and aggregating to bond-day: {file.name}")
        cleaned_path = TRACE_CLEANED_TRADE_DIR / f"{file.stem}_cleaned_trades.parquet"

        if cleaned_path.exists() and not FORCE_REBUILD_CLEANED_TRADES:
            print(f"Loading existing cleaned trades: {cleaned_path}")
            df = pd.read_parquet(cleaned_path)
            waterfall = build_trade_cleaning_waterfall(df, file.name)
            mapping = classify_status_values(df)
            if not mapping.empty:
                mapping.insert(0, "file", file.name)
        else:
            df = pd.read_parquet(file)
            df = clean_trade_chunk(df)
            df, waterfall, mapping = apply_trace_cleaner(df, file_name=file.name)
            df.to_parquet(cleaned_path, index=False)
            print(f"Saved cleaned trade-level file: {cleaned_path}")

        volume_col = pick_col(df, ["entrd_vol_qt", "ascii_rptd_vol_tx", "rptd_vol_qt"])

        if volume_col:
            vol_summary, vol_round, vol_hist = audit_trace_volume_bunching(
                df=df,
                vol_col=volume_col,
                sample_name=file.stem,
                knots=[1e5, 5e5, 1e6, 5e6],
            )
            volume_audit_summaries.append(vol_summary)
            volume_round_tables.append(vol_round)

            if vol_hist is not None and len(vol_hist) > 0:
                volume_hist_parts.append(vol_hist)

        waterfalls.append(waterfall)
        if mapping is not None and not mapping.empty:
            mappings.append(mapping)

        flag_cols = [
            "is_cancel_report", "is_correction_report", "is_reversal_report",
            "is_referenced_original", "is_exact_duplicate", "is_special_condition",
            "is_potential_agency_duplicate", "clean_price_sample", "clean_volume_sample",
            "clean_institutional_price_sample", "clean_price_no_agency_sample",
        ]
        flag_summaries.append(
            {
                "file": file.name,
                "rows": len(df),
                **{col: int(df[col].sum()) for col in flag_cols if col in df.columns},
            }
        )

        daily = aggregate_cleaned_year(df, file.name)
        if not daily.empty:
            daily_parts.append(daily)

    if not daily_parts:
        raise ValueError("No daily bond observations were created from clean price sample.")

    trade_waterfall = pd.concat(waterfalls, ignore_index=True)
    trade_waterfall.to_csv(TRACE_CLEANING_DIAG_DIR / "trace_trade_cleaning_waterfall_by_year.csv", index=False)

    if mappings:
        status_mapping = pd.concat(mappings, ignore_index=True)
        status_mapping.to_csv(TRACE_CLEANING_DIAG_DIR / "trace_status_action_mapping.csv", index=False)

    flag_summary = pd.DataFrame(flag_summaries)
    flag_summary.to_csv(TRACE_CLEANING_DIAG_DIR / "trace_cleaning_flags_summary.csv", index=False)

    write_trace_volume_bunching_outputs(
        summary_parts=volume_audit_summaries,
        round_table_parts=volume_round_tables,
        hist_parts=volume_hist_parts,
    )

    bond_day = pd.concat(daily_parts, ignore_index=True)
    bond_day = bond_day.rename(columns={"trd_exctn_dt": "date"})
    bond_day["date"] = pd.to_datetime(bond_day["date"])
    bond_day["cusip_id"] = bond_day["cusip_id"].astype(str).str.strip()

    weekend_mask = bond_day["date"].dt.weekday >= 5
    print("Weekend bond-day rows removed:", int(weekend_mask.sum()))
    bond_day = bond_day.loc[~weekend_mask].copy()

    bond_day = add_bond_day_returns(bond_day)

    bond_day_path = BOND_DAY_DIR / "trace_banks_bond_day.parquet"
    bond_day.to_parquet(bond_day_path, index=False)

    print("\nBOND-DAY DATASET:")
    print(bond_day.shape)
    print(bond_day.head())
    print(f"\nSaved bond-day dataset to: {bond_day_path}")

    return bond_day


def write_bond_day_diagnostics(bond_day: pd.DataFrame) -> pd.DataFrame:
    liq_summary = (
        bond_day.groupby("cusip_id")
        .agg(
            company_symbol=("company_symbol", "first") if "company_symbol" in bond_day.columns else ("cusip_id", "size"),
            bond_sym_id=("bond_sym_id", "first") if "bond_sym_id" in bond_day.columns else ("cusip_id", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            active_days=("date", "nunique"),
            total_trades=("n_trades", "sum"),
            avg_trades_per_day=("n_trades", "mean"),
            total_volume=("total_volume", "sum"),
            avg_daily_volume=("total_volume", "mean"),
            median_daily_volume=("total_volume", "median"),
            avg_price_dispersion_rel=("price_dispersion_rel", "mean"),
            avg_price_range_rel=("price_range_rel", "mean"),
        )
        .reset_index()
        .sort_values(["total_trades", "active_days"], ascending=False)
    )

    liq_path = PROCESSED_DIR / "bond_liquidity_summary.csv"
    liq_summary.to_csv(liq_path, index=False)
    print("\nLIQUIDITY / TRADING-QUALITY SUMMARY:")
    print(liq_summary.head(20))
    print(f"\nSaved summary to: {liq_path}")

    cusip_path = PROCESSED_DIR / "bank_cusips.txt"
    bond_day["cusip_id"].dropna().astype(str).drop_duplicates().sort_values().to_csv(cusip_path, index=False, header=False)
    print(f"Saved CUSIP list to: {cusip_path}")

    liquid_cusips = liq_summary.loc[
        (liq_summary["active_days"] >= MIN_ACTIVE_DAYS)
        & (liq_summary["total_trades"] >= MIN_TOTAL_TRADES),
        "cusip_id",
    ]
    bond_day_liquid = bond_day[bond_day["cusip_id"].isin(liquid_cusips)].copy()

    liquid_path = BOND_DAY_DIR / "trace_banks_bond_day_liquid.parquet"
    bond_day_liquid.to_parquet(liquid_path, index=False)

    liquid_cusip_path = PROCESSED_DIR / "bank_liquid_cusips.txt"
    bond_day_liquid["cusip_id"].dropna().astype(str).drop_duplicates().sort_values().to_csv(
        liquid_cusip_path, index=False, header=False
    )

    print("\nLIQUID BOND-DAY DATASET:")
    print(bond_day_liquid.shape)
    print(f"Number of liquid CUSIPs: {bond_day_liquid['cusip_id'].nunique()}")
    print(f"Saved liquid dataset to: {liquid_path}")

    daily_market_activity = (
        bond_day.groupby("date")
        .agg(
            n_bond_days=("cusip_id", "size"),
            n_cusip=("cusip_id", "nunique"),
            total_trades=("n_trades", "sum"),
            total_volume=("total_volume", "sum"),
        )
        .reset_index()
        .sort_values("date")
    )
    daily_market_activity.to_csv(PROCESSED_DIR / "daily_market_activity.csv", index=False)

    issuer_year = (
        bond_day.assign(year=bond_day["date"].dt.year)
        .groupby(["year", "company_symbol"])
        .agg(
            n_cusip=("cusip_id", "nunique"),
            n_bond_days=("cusip_id", "size"),
            total_trades=("n_trades", "sum"),
            total_volume=("total_volume", "sum"),
        )
        .reset_index()
        .sort_values(["year", "company_symbol"])
    )
    issuer_year.to_csv(PROCESSED_DIR / "issuer_year_coverage.csv", index=False)

    return bond_day_liquid


def build_final_panel_with_master() -> pd.DataFrame:
    if not TRACE_MASTER_ZIP.exists():
        raise FileNotFoundError(f"Master file not found: {TRACE_MASTER_ZIP}")

    master = read_wrds_zip(TRACE_MASTER_ZIP)
    print("\nMASTER FILE SHAPE:")
    print(master.shape)

    cusip_col = pick_col(master, ["cusip_id", "cusip", "cusip_full"])
    ticker_col = pick_col(master, ["company_symbol", "ticker"])
    issuer_col = pick_col(master, ["issuer_nm", "issuer_name", "issuer"])
    coupon_col = pick_col(master, ["cpn_rt", "coupon", "coupon_rate"])
    maturity_col = pick_col(master, ["mtrty_dt", "maturity_dt", "maturity_date"])
    debt_type_col = pick_col(master, ["debt_type_cd"])
    security_type_col = pick_col(master, ["scrty_type_cd"])
    security_subtype_col = pick_col(master, ["scrty_sbtp_cd", "scrty_sbtyp_cd"])
    sub_product_col = pick_col(master, ["sub_prdct_type", "sub_prd_type", "sub_product_type"])
    callable_col = pick_col(master, ["callable_flg", "callable", "call_ind"])
    coupon_type_col = pick_col(master, ["cpn_type_cd", "coupon_type"])
    amount_col = pick_col(master, ["offering_amt", "principal_amt", "amount_outstanding"])
    issue_date_col = pick_col(
        master,
        ["issue_dt", "issue_date", "dated_dt", "dated_date", "offering_date"]
    )

    coupon_freq_col = pick_col(
        master,
        ["cpn_freq", "coupon_freq", "coupon_frequency", "pmt_freq", "payment_frequency"]
    )

    day_count_col = pick_col(
        master,
        ["day_count", "day_count_basis", "dcc", "day_cnt"]
    )

    if cusip_col is None:
        raise ValueError("No CUSIP column found in Master File.")

    master = master.rename(columns={cusip_col: "cusip_id"})
    master["cusip_id"] = master["cusip_id"].astype(str).str.strip()

    if "stdt" not in master.columns or "enddt" not in master.columns:
        raise ValueError("Master File must contain stdt and enddt for point-in-time merge.")

    master["stdt"] = pd.to_datetime(master["stdt"], errors="coerce")
    master["enddt"] = pd.to_datetime(master["enddt"], errors="coerce")
    master["stdt_effective"] = master["stdt"].fillna(pd.Timestamp("1900-01-01"))
    master["enddt_effective"] = master["enddt"].fillna(pd.Timestamp("2099-12-31"))

    if maturity_col and maturity_col in master.columns:
        master[maturity_col] = pd.to_datetime(master[maturity_col], errors="coerce")
    if issue_date_col and issue_date_col in master.columns:
        master[issue_date_col] = pd.to_datetime(master[issue_date_col], errors="coerce")
    if coupon_col and coupon_col in master.columns:
        master[coupon_col] = pd.to_numeric(master[coupon_col], errors="coerce")
    if amount_col and amount_col in master.columns:
        master[amount_col] = pd.to_numeric(master[amount_col], errors="coerce")

    master["_non_missing_fields"] = master.notna().sum(axis=1)
    master_latest = (
        master.sort_values(["cusip_id", "stdt_effective", "_non_missing_fields"], ascending=[True, True, True])
        .drop_duplicates("cusip_id", keep="last")
        .drop(columns=["_non_missing_fields"])
        .copy()
    )

    latest_path = PROCESSED_DIR / "master_liquid_latest_diagnostic.parquet"
    latest_csv_path = PROCESSED_DIR / "master_liquid_latest_diagnostic.csv"
    master_latest.to_parquet(latest_path, index=False)
    master_latest.to_csv(latest_csv_path, index=False)

    master_diag_rows = []
    for col in [ticker_col, issuer_col, sub_product_col, debt_type_col, security_type_col, security_subtype_col, coupon_type_col, callable_col, "ind_144a", "cnvrb_fl", "dissem"]:
        if col and col in master_latest.columns:
            vc = master_latest[col].astype("object").where(master_latest[col].notna(), "<MISSING>").value_counts(dropna=False).head(50)
            for value, count in vc.items():
                master_diag_rows.append({"field": col, "value": value, "count": int(count)})
    pd.DataFrame(master_diag_rows).to_csv(DIAG_DIR / "master_latest_value_counts.csv", index=False)

    bond_day_liquid = pd.read_parquet(BOND_DAY_DIR / "trace_banks_bond_day_liquid.parquet")
    bond_day_liquid["date"] = pd.to_datetime(bond_day_liquid["date"])
    bond_day_liquid["cusip_id"] = bond_day_liquid["cusip_id"].astype(str).str.strip()

    if "company_symbol" in bond_day_liquid.columns and "trace_company_symbol" not in bond_day_liquid.columns:
        bond_day_liquid = bond_day_liquid.rename(columns={"company_symbol": "trace_company_symbol"})
    if "bond_sym_id" in bond_day_liquid.columns and "trace_bond_sym_id" not in bond_day_liquid.columns:
        bond_day_liquid = bond_day_liquid.rename(columns={"bond_sym_id": "trace_bond_sym_id"})

    master_cusips = set(master["cusip_id"].dropna())
    liquid_set = set(bond_day_liquid["cusip_id"].dropna())
    missing_master = sorted(liquid_set - master_cusips)
    pd.DataFrame({"cusip_id": missing_master}).to_csv(PROCESSED_DIR / "liquid_cusips_missing_master.csv", index=False)

    dup_counts = master.groupby("cusip_id").size().rename("n_master_rows").reset_index()
    dup_counts[dup_counts["n_master_rows"] > 1].sort_values("n_master_rows", ascending=False).to_csv(
        PROCESSED_DIR / "master_duplicate_cusips.csv", index=False
    )

    master["_non_missing_fields"] = master.notna().sum(axis=1)
    master_pit = (
        master.sort_values(["cusip_id", "stdt_effective", "_non_missing_fields"], ascending=[True, True, True])
        .drop_duplicates(["cusip_id", "stdt_effective"], keep="last")
        .drop(columns=["_non_missing_fields"])
        .copy()
    )

    master_data_cols = [col for col in master_pit.columns if col != "cusip_id"]
    merged_parts = []

    print("\nRunning point-in-time merge by CUSIP...")
    for cusip, left_group in bond_day_liquid.groupby("cusip_id", sort=False):
        right_group = master_pit.loc[master_pit["cusip_id"] == cusip].copy()
        left_group = left_group.sort_values("date").copy()

        if right_group.empty:
            temp = left_group.copy()
            for col in master_data_cols:
                temp[col] = np.nan
        else:
            right_group = right_group.drop(columns=["cusip_id"]).sort_values("stdt_effective").copy()
            temp = pd.merge_asof(
                left_group,
                right_group,
                left_on="date",
                right_on="stdt_effective",
                direction="backward",
            )
        merged_parts.append(temp)

    bond_day_with_master = pd.concat(merged_parts, ignore_index=True)

    valid_master_row = (
        bond_day_with_master["stdt_effective"].notna()
        & (bond_day_with_master["date"] <= bond_day_with_master["enddt_effective"])
    )
    print("\nPOINT-IN-TIME MERGE DIAGNOSTICS:")
    print("Merged panel shape:", bond_day_with_master.shape)
    print("Rows without valid point-in-time master row:", int((~valid_master_row).sum()))
    print("Share without valid point-in-time master row:", float((~valid_master_row).mean()))

    for col in master_data_cols:
        bond_day_with_master.loc[~valid_master_row, col] = np.nan

    if maturity_col and maturity_col in bond_day_with_master.columns:
        bond_day_with_master[maturity_col] = pd.to_datetime(bond_day_with_master[maturity_col], errors="coerce")
        bond_day_with_master["years_to_maturity"] = (
            bond_day_with_master[maturity_col] - bond_day_with_master["date"]
        ).dt.days / 365.25

    if coupon_col and coupon_col in bond_day_with_master.columns:
        bond_day_with_master[coupon_col] = pd.to_numeric(bond_day_with_master[coupon_col], errors="coerce")

    master_static_cols = [
        col
        for col in [
            coupon_col,
            maturity_col,
            coupon_type_col,
            debt_type_col,
            security_type_col,
            security_subtype_col,
            sub_product_col,
            callable_col,
            amount_col,
            issue_date_col,
        ]
        if col and col in bond_day_with_master.columns
    ]

    audit_master_pit_consistency(
        master=master_pit,
        pit_panel=bond_day_with_master,
        static_cols=master_static_cols,
        maturity_col=maturity_col,
        output_prefix="master_pit_liquid",
    )

    pit_panel_path = BOND_DAY_DIR / "trace_banks_bond_day_liquid_with_master_pit.parquet"
    bond_day_with_master.to_parquet(pit_panel_path, index=False)

    final_panel = apply_final_bond_filters(
        bond_day_with_master,
        maturity_col=maturity_col,
        coupon_col=coupon_col,
        issue_date_col=issue_date_col,
        coupon_freq_col=coupon_freq_col,
        day_count_col=day_count_col,
    )

    final_master_static_cols = [
        col
        for col in [
            coupon_col,
            maturity_col,
            coupon_type_col,
            debt_type_col,
            security_type_col,
            security_subtype_col,
            sub_product_col,
            callable_col,
            amount_col,
            issue_date_col,
        ]
        if col and col in final_panel.columns
    ]

    audit_master_pit_consistency(
        master=master_pit,
        pit_panel=final_panel,
        static_cols=final_master_static_cols,
        maturity_col=maturity_col,
        output_prefix="master_pit_final_baseline",
    )

    final_panel_path = BOND_DAY_DIR / "trace_banks_final_baseline_panel.parquet"
    final_panel.to_parquet(final_panel_path, index=False)

    final_summary = (
        final_panel.groupby("trace_company_symbol")
        .agg(
            n_cusip=("cusip_id", "nunique"),
            n_bond_days=("cusip_id", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            total_trades=("n_trades", "sum"),
            total_volume=("total_volume", "sum"),
            avg_years_to_maturity=("years_to_maturity", "mean") if "years_to_maturity" in final_panel.columns else ("cusip_id", "size"),
            avg_price_dispersion_rel=("price_dispersion_rel", "mean") if "price_dispersion_rel" in final_panel.columns else ("cusip_id", "size"),
        )
        .reset_index()
        .rename(columns={"trace_company_symbol": "company_symbol"})
    )
    final_summary.to_csv(PROCESSED_DIR / "trace_banks_final_baseline_summary.csv", index=False)

    print("\nFINAL BASELINE PANEL:")
    print("Rows:", len(final_panel))
    print("CUSIPs:", final_panel["cusip_id"].nunique())
    print("First date:", final_panel["date"].min())
    print("Last date:", final_panel["date"].max())
    print(f"Saved final baseline panel to: {final_panel_path}")

    build_additional_diagnostics(bond_day_with_master, final_panel, maturity_col, coupon_col)
    build_model_ready_panel(final_panel)

    return final_panel


def apply_final_bond_filters(
    panel: pd.DataFrame,
    maturity_col: str | None,
    coupon_col: str | None,
    issue_date_col: str | None = None,
    coupon_freq_col: str | None = None,
    day_count_col: str | None = None,
) -> pd.DataFrame:
    final_panel = panel.copy()

    for col in ["sub_prdct_type", "sub_prd_type", "ind_144a", "cnvrb_fl", "dissem", "cpn_type_cd"]:
        if col in final_panel.columns:
            final_panel[col] = clean_string_series(final_panel[col])

    baseline_conditions = pd.Series(True, index=final_panel.index)

    if "sub_prdct_type" in final_panel.columns:
        baseline_conditions &= final_panel["sub_prdct_type"].eq("CORP")
    elif "sub_prd_type" in final_panel.columns:
        baseline_conditions &= final_panel["sub_prd_type"].eq("CORP")

    if "ind_144a" in final_panel.columns:
        baseline_conditions &= final_panel["ind_144a"].eq("N")
    if "cnvrb_fl" in final_panel.columns:
        baseline_conditions &= final_panel["cnvrb_fl"].eq("N")
    if "dissem" in final_panel.columns:
        baseline_conditions &= final_panel["dissem"].eq("Y")
    if "cpn_type_cd" in final_panel.columns:
        baseline_conditions &= final_panel["cpn_type_cd"].eq("FXPV")
    if maturity_col and maturity_col in final_panel.columns:
        baseline_conditions &= final_panel[maturity_col].notna()
    if coupon_col and coupon_col in final_panel.columns:
        baseline_conditions &= final_panel[coupon_col].notna()
    if "years_to_maturity" in final_panel.columns:
        baseline_conditions &= final_panel["years_to_maturity"].gt(1.0)

    final_panel = final_panel.loc[baseline_conditions].copy()
    final_panel = final_panel.sort_values(["cusip_id", "date"])

    for price_col, return_col in [
        ("vwap_price", "final_vwap_return"),
        ("median_price", "final_median_price_return"),
        ("last_price", "final_last_price_return"),
    ]:
        if price_col in final_panel.columns:
            log_col = f"final_log_{price_col}"
            final_panel[log_col] = np.log(final_panel[price_col].where(final_panel[price_col] > 0))
            final_panel[return_col] = final_panel.groupby("cusip_id")[log_col].diff()

    if (
        coupon_col
        and maturity_col
        and "vwap_price" in final_panel.columns
        and coupon_col in final_panel.columns
        and maturity_col in final_panel.columns
    ):
        final_panel = add_dirty_price_columns(
            final_panel,
            clean_price_col="vwap_price",
            coupon_rate_col=coupon_col,
            date_col="date",
            maturity_col=maturity_col,
            issue_date_col=issue_date_col if issue_date_col in final_panel.columns else None,
            coupon_freq_col=coupon_freq_col if coupon_freq_col in final_panel.columns else None,
            day_count_col=day_count_col if day_count_col in final_panel.columns else None,
            default_coupon_freq=2,
            default_day_count="30_360_US",
            group_col="cusip_id",
            prefix="vwap",
            return_col="final_dirty_vwap_return",
        )

        if "final_vwap_return" in final_panel.columns:
            final_panel["dirty_clean_vwap_return_diff"] = (
                final_panel["final_dirty_vwap_return"]
                - final_panel["final_vwap_return"]
            )
            final_panel["abs_dirty_clean_vwap_return_diff"] = (
                final_panel["dirty_clean_vwap_return_diff"].abs()
            )

    if "median_yield" in final_panel.columns:
        final_panel["final_yield_change"] = final_panel.groupby("cusip_id")["median_yield"].diff()

    final_panel["prev_date"] = final_panel.groupby("cusip_id")["date"].shift(1)
    final_panel["calendar_gap_days"] = (final_panel["date"] - final_panel["prev_date"]).dt.days
    final_panel["business_gap_days"] = business_day_gap(final_panel["prev_date"], final_panel["date"])

    write_gap_calendar_diagnostics(final_panel)

    if {"final_vwap_return", "total_volume"}.issubset(final_panel.columns):
        final_panel["final_amihud_daily"] = np.where(
            final_panel["total_volume"].gt(0),
            final_panel["final_vwap_return"].abs() / final_panel["total_volume"],
            np.nan,
        )

    return final_panel


def build_additional_diagnostics(pit_panel: pd.DataFrame, final_panel: pd.DataFrame, maturity_col: str | None, coupon_col: str | None) -> None:
    print("\nADDITIONAL DIAGNOSTICS")
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    integrity_rows = []
    for name, df in [("pit_panel", pit_panel), ("final_panel", final_panel)]:
        integrity_rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "n_cusip": df["cusip_id"].nunique(),
                "first_date": df["date"].min(),
                "last_date": df["date"].max(),
                "duplicate_cusip_date_rows": int(df.duplicated(["cusip_id", "date"]).sum()),
                "weekend_rows": int((df["date"].dt.weekday >= 5).sum()),
            }
        )
    pd.DataFrame(integrity_rows).to_csv(DIAG_DIR / "dataset_integrity_checks.csv", index=False)

    fp = final_panel.copy()
    if "years_to_maturity" in fp.columns:
        fp["maturity_bucket"] = pd.cut(
            fp["years_to_maturity"],
            bins=[1, 3, 5, 7, 10, 15, 30, np.inf],
            labels=["1-3y", "3-5y", "5-7y", "7-10y", "10-15y", "15-30y", "30y+"],
            right=False,
        )
    fp["year"] = fp["date"].dt.year

    composition_cols = ["year"]
    if "trace_company_symbol" in fp.columns:
        composition_cols.append("trace_company_symbol")
    if "maturity_bucket" in fp.columns:
        composition_cols.append("maturity_bucket")

    (
        fp.groupby(composition_cols, observed=True)
        .agg(
            rows=("cusip_id", "size"),
            n_cusip=("cusip_id", "nunique"),
            total_trades=("n_trades", "sum"),
            total_volume=("total_volume", "sum"),
            avg_years_to_maturity=("years_to_maturity", "mean") if "years_to_maturity" in fp.columns else ("cusip_id", "size"),
            avg_price_dispersion_rel=("price_dispersion_rel", "mean") if "price_dispersion_rel" in fp.columns else ("cusip_id", "size"),
        )
        .reset_index()
        .to_csv(DIAG_DIR / "final_panel_composition_year_issuer_maturity.csv", index=False)
    )

    quantile_cols = [
        c for c in [
            "final_vwap_return",
            "final_dirty_vwap_return",
            "dirty_clean_vwap_return_diff",
            "abs_dirty_clean_vwap_return_diff",
            "final_median_price_return",
            "final_last_price_return",
            "final_yield_change",
            "price_dispersion_rel",
            "price_range_rel",
            "business_gap_days",
        ]
        if c in fp.columns
    ]
    if quantile_cols:
        q = fp[quantile_cols].quantile([0.001, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 0.999])
        q.to_csv(DIAG_DIR / "return_yield_microstructure_quantiles.csv")

    extreme_mask = pd.Series(False, index=fp.index)
    if "final_vwap_return" in fp.columns:
        extreme_mask |= fp["final_vwap_return"].abs().gt(0.05)
    if "final_yield_change" in fp.columns:
        extreme_mask |= fp["final_yield_change"].abs().gt(1.0)
    if "business_gap_days" in fp.columns:
        extreme_mask |= fp["business_gap_days"].gt(20)
    if "price_dispersion_rel" in fp.columns:
        extreme_mask |= fp["price_dispersion_rel"].gt(fp["price_dispersion_rel"].quantile(0.999))

    extreme_cols = [
        c for c in [
            "cusip_id", "date", "prev_date", "calendar_gap_days", "business_gap_days", "trace_company_symbol",
            "vwap_price", "dirty_vwap_price", "vwap_accrued_interest", "vwap_accrual_fraction",
            "vwap_prev_coupon_date", "vwap_next_coupon_date", "vwap_ai_method",
            "median_price", "last_price", "median_yield",
            "final_vwap_return", "final_dirty_vwap_return", "dirty_clean_vwap_return_diff",
            "final_median_price_return", "final_last_price_return", "final_yield_change", "n_trades",
            "total_volume", "price_dispersion_rel", "price_range_rel", "years_to_maturity", coupon_col, maturity_col,
        ]
        if c and c in fp.columns
    ]
    fp.loc[extreme_mask, extreme_cols].to_csv(DIAG_DIR / "extreme_return_yield_gap_microstructure_observations.csv", index=False)

    gap_summary = (
        fp.groupby("cusip_id")
        .agg(
            n_obs=("date", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            median_business_gap=("business_gap_days", "median"),
            p95_business_gap=("business_gap_days", lambda x: x.quantile(0.95)),
            max_business_gap=("business_gap_days", "max"),
            share_gaps_gt_5bd=("business_gap_days", lambda x: (x > 5).mean()),
            share_gaps_gt_20bd=("business_gap_days", lambda x: (x > 20).mean()),
            avg_price_dispersion_rel=("price_dispersion_rel", "mean") if "price_dispersion_rel" in fp.columns else ("cusip_id", "size"),
        )
        .reset_index()
        .sort_values(["share_gaps_gt_20bd", "max_business_gap"], ascending=False)
    )
    if "trace_company_symbol" in fp.columns:
        gap_summary["trace_company_symbol"] = gap_summary["cusip_id"].map(fp.groupby("cusip_id")["trace_company_symbol"].first())
    gap_summary.to_csv(DIAG_DIR / "final_panel_gap_summary_by_cusip.csv", index=False)

def write_dirty_price_diagnostics(panel: pd.DataFrame) -> None:
    print("\nDIRTY-PRICE ROBUSTNESS DIAGNOSTICS")

    required_cols = {
        "final_vwap_return",
        "final_dirty_vwap_return",
        "dirty_clean_vwap_return_diff",
    }

    if not required_cols.issubset(panel.columns):
        print("Dirty-price diagnostics skipped: required dirty-price columns not found.")
        return

    df = panel.copy()

    gap_flag = f"valid_return_gap_{MODEL_READY_MAX_BUSINESS_GAP}bd"
    if gap_flag in df.columns:
        df = df.loc[df[gap_flag]].copy()

    df = df.dropna(
        subset=[
            "final_vwap_return",
            "final_dirty_vwap_return",
            "dirty_clean_vwap_return_diff",
        ]
    ).copy()

    if df.empty:
        print("Dirty-price diagnostics skipped: no valid dirty-return observations.")
        return

    df["abs_dirty_clean_vwap_return_diff"] = df["dirty_clean_vwap_return_diff"].abs()
    df["dirty_clean_diff_bp"] = 10_000.0 * df["dirty_clean_vwap_return_diff"]
    df["abs_dirty_clean_diff_bp"] = 10_000.0 * df["abs_dirty_clean_vwap_return_diff"]

    if "years_to_maturity" in df.columns:
        df["dirty_diag_maturity_bucket"] = pd.cut(
            df["years_to_maturity"],
            bins=[1, 3, 5, 7, 10, 15, 30, np.inf],
            labels=["1-3y", "3-5y", "5-7y", "7-10y", "10-15y", "15-30y", "30y+"],
            right=False,
        )

    coupon_candidates = [c for c in ["cpn_rt", "coupon", "coupon_rate"] if c in df.columns]
    coupon_col = coupon_candidates[0] if coupon_candidates else None

    if coupon_col:
        df["dirty_diag_coupon_bucket"] = pd.cut(
            pd.to_numeric(df[coupon_col], errors="coerce"),
            bins=[0, 1, 2, 3, 4, 5, 6, 8, 10, np.inf],
            labels=["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-8", "8-10", "10+"],
            right=False,
        )

    def summarise(temp: pd.DataFrame, breakdown: str, bucket: str) -> dict:
        x = temp["dirty_clean_diff_bp"]
        ax = temp["abs_dirty_clean_diff_bp"]

        return {
            "breakdown": breakdown,
            "bucket": bucket,
            "n_obs": int(len(temp)),
            "mean_diff_bp": float(x.mean()),
            "median_diff_bp": float(x.median()),
            "mean_abs_diff_bp": float(ax.mean()),
            "median_abs_diff_bp": float(ax.median()),
            "p95_abs_diff_bp": float(ax.quantile(0.95)),
            "p99_abs_diff_bp": float(ax.quantile(0.99)),
            "max_abs_diff_bp": float(ax.max()),
            "share_abs_diff_gt_1bp": float((ax > 1.0).mean()),
            "share_abs_diff_gt_5bp": float((ax > 5.0).mean()),
        }

    rows = [summarise(df, "all", "all")]

    if "trace_company_symbol" in df.columns:
        for issuer, g in df.groupby("trace_company_symbol"):
            rows.append(summarise(g, "issuer", str(issuer)))

    if "dirty_diag_maturity_bucket" in df.columns:
        for bucket, g in df.groupby("dirty_diag_maturity_bucket", observed=True):
            rows.append(summarise(g, "maturity_bucket", str(bucket)))

    if "dirty_diag_coupon_bucket" in df.columns:
        for bucket, g in df.groupby("dirty_diag_coupon_bucket", observed=True):
            rows.append(summarise(g, "coupon_bucket", str(bucket)))

    summary = pd.DataFrame(rows)
    summary_path = DIAG_DIR / "dirty_vs_clean_return_summary.csv"
    summary.to_csv(summary_path, index=False)

    top_cols = [
        c for c in [
            "cusip_id",
            "date",
            "prev_date",
            "trace_company_symbol",
            "vwap_price",
            "dirty_vwap_price",
            "vwap_accrued_interest",
            "vwap_accrual_fraction",
            "vwap_prev_coupon_date",
            "vwap_next_coupon_date",
            "vwap_ai_method",
            "final_vwap_return",
            "final_dirty_vwap_return",
            "dirty_clean_vwap_return_diff",
            "dirty_clean_diff_bp",
            "abs_dirty_clean_diff_bp",
            "business_gap_days",
            "years_to_maturity",
            coupon_col,
        ]
        if c and c in df.columns
    ]

    top_path = DIAG_DIR / "dirty_vs_clean_top_differences.csv"
    (
        df.sort_values("abs_dirty_clean_diff_bp", ascending=False)
        .head(100)
        .loc[:, top_cols]
        .to_csv(top_path, index=False)
    )

    ecdf_path = FIGURES_DIR / "dirty_vs_clean_return_diff_ecdf.png"

    x = np.sort(df["abs_dirty_clean_diff_bp"].dropna().to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y)
    plt.xlabel("Absolute difference between dirty and clean VWAP returns (bp)")
    plt.ylabel("Empirical CDF")
    plt.title("Dirty-price robustness: clean versus dirty return difference")
    plt.tight_layout()
    plt.savefig(ecdf_path, dpi=200)
    plt.close()

    print(f"Saved dirty-price summary to: {summary_path}")
    print(f"Saved dirty-price top differences to: {top_path}")
    print(f"Saved dirty-price ECDF to: {ecdf_path}")

def add_return_outlier_flags(
    df: pd.DataFrame,
    group_col: str = "cusip_id",
    date_col: str = "date",
    ret_col: str = "final_vwap_return",
    window: int = 60,
    min_periods: int = 20,
    mad_k: float = 8.0,
    abs_ret_hard: float = 0.15,
) -> pd.DataFrame:
    out = df.copy()

    if ret_col not in out.columns:
        raise ValueError(f"Missing return column for outlier flags: {ret_col}")

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values([group_col, date_col]).copy()

    out[ret_col] = pd.to_numeric(out[ret_col], errors="coerce")

    g = out.groupby(group_col, sort=False)[ret_col]

    roll_med = g.transform(
        lambda x: x.shift(1).rolling(
            window=window,
            min_periods=min_periods,
        ).median()
    )

    abs_dev = (out[ret_col] - roll_med).abs()

    roll_mad = abs_dev.groupby(out[group_col], sort=False).transform(
        lambda x: x.shift(1).rolling(
            window=window,
            min_periods=min_periods,
        ).median()
    )

    robust_scale = 1.4826 * roll_mad.replace(0.0, np.nan)
    robust_z = (out[ret_col] - roll_med) / robust_scale

    out["return_outlier_rolling_median"] = roll_med
    out["return_outlier_rolling_mad"] = roll_mad
    out["return_outlier_robust_z"] = robust_z

    out["flag_return_outlier_mad"] = (
        robust_z.abs().gt(mad_k).fillna(False)
    )

    out["flag_return_outlier_abs"] = (
        out[ret_col].abs().gt(abs_ret_hard).fillna(False)
    )

    out["flag_return_outlier_any"] = (
        out["flag_return_outlier_mad"]
        | out["flag_return_outlier_abs"]
    )

    out["return_outlier_rule"] = np.select(
        [
            out["flag_return_outlier_mad"] & out["flag_return_outlier_abs"],
            out["flag_return_outlier_mad"],
            out["flag_return_outlier_abs"],
        ],
        [
            "mad_and_abs",
            "mad_only",
            "abs_only",
        ],
        default="none",
    )

    return out


def flag_two_day_reversals(
    df: pd.DataFrame,
    group_col: str = "cusip_id",
    date_col: str = "date",
    ret_col: str = "final_vwap_return",
    min_abs_ret: float = 0.03,
    reversal_ratio: float = 0.25,
    max_next_gap: int | None = 5,
) -> pd.DataFrame:
    out = df.copy()

    if ret_col not in out.columns:
        raise ValueError(f"Missing return column for reversal flags: {ret_col}")

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values([group_col, date_col]).copy()

    out[ret_col] = pd.to_numeric(out[ret_col], errors="coerce")

    g = out.groupby(group_col, sort=False)

    out["next_date_for_reversal"] = g[date_col].shift(-1)
    out["next_return_for_reversal"] = g[ret_col].shift(-1)

    if "business_gap_days" in out.columns:
        out["next_business_gap_days_for_reversal"] = g["business_gap_days"].shift(-1)
    else:
        out["next_business_gap_days_for_reversal"] = np.nan

    current_ret = out[ret_col]
    next_ret = out["next_return_for_reversal"]

    abs_current = current_ret.abs()
    abs_next = next_ret.abs()

    out["two_day_reversal_ratio"] = abs_next / abs_current.replace(0.0, np.nan)

    valid_pair = current_ret.notna() & next_ret.notna()

    opposite_sign = (current_ret * next_ret).lt(0)

    large_start = abs_current.ge(min_abs_ret)

    meaningful_reversal = abs_next.ge(reversal_ratio * abs_current)

    flag_start = (
        valid_pair
        & opposite_sign
        & large_start
        & meaningful_reversal
    )

    if max_next_gap is not None and "next_business_gap_days_for_reversal" in out.columns:
        flag_start &= out["next_business_gap_days_for_reversal"].le(max_next_gap).fillna(False)

    out["flag_two_day_reversal_start"] = flag_start.fillna(False)

    out["flag_two_day_reversal_followthrough"] = (
        g["flag_two_day_reversal_start"]
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    out["flag_two_day_reversal_pair"] = (
        out["flag_two_day_reversal_start"]
        | out["flag_two_day_reversal_followthrough"]
    )

    return out


def write_return_outlier_diagnostics(panel: pd.DataFrame) -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    required = {
        "cusip_id",
        "date",
        "final_vwap_return",
        "flag_return_outlier_any",
        "flag_two_day_reversal_pair",
    }

    if not required.issubset(panel.columns):
        missing = sorted(required.difference(panel.columns))
        raise ValueError(f"Missing columns for return-outlier diagnostics: {missing}")

    df = panel.copy()

    if "has_valid_return" in df.columns:
        df = df.loc[df["has_valid_return"]].copy()
    else:
        df = df.loc[df["final_vwap_return"].notna()].copy()

    if "business_gap_days" in df.columns:
        df = df.loc[df["business_gap_days"].notna()].copy()

    df["flag_outlier_or_reversal_candidate"] = (
        df["flag_return_outlier_any"]
        | df["flag_two_day_reversal_pair"]
    )

    def summarise(temp: pd.DataFrame, sample: str, breakdown: str, bucket: str) -> dict:
        out = {
            "sample": sample,
            "breakdown": breakdown,
            "bucket": bucket,
            "n_obs": int(len(temp)),
            "n_cusips": int(temp["cusip_id"].nunique()) if "cusip_id" in temp.columns else np.nan,
            "mean_abs_return": float(temp["final_vwap_return"].abs().mean()) if len(temp) else np.nan,
            "p99_abs_return": float(temp["final_vwap_return"].abs().quantile(0.99)) if len(temp) else np.nan,
            "return_outlier_mad_n": int(temp["flag_return_outlier_mad"].sum()) if "flag_return_outlier_mad" in temp.columns else 0,
            "return_outlier_abs_n": int(temp["flag_return_outlier_abs"].sum()) if "flag_return_outlier_abs" in temp.columns else 0,
            "return_outlier_any_n": int(temp["flag_return_outlier_any"].sum()),
            "two_day_reversal_start_n": int(temp["flag_two_day_reversal_start"].sum()) if "flag_two_day_reversal_start" in temp.columns else 0,
            "two_day_reversal_pair_n": int(temp["flag_two_day_reversal_pair"].sum()),
            "outlier_or_reversal_candidate_n": int(temp["flag_outlier_or_reversal_candidate"].sum()),
        }

        if len(temp) > 0:
            out["return_outlier_any_share"] = out["return_outlier_any_n"] / len(temp)
            out["two_day_reversal_pair_share"] = out["two_day_reversal_pair_n"] / len(temp)
            out["outlier_or_reversal_candidate_share"] = out["outlier_or_reversal_candidate_n"] / len(temp)
        else:
            out["return_outlier_any_share"] = np.nan
            out["two_day_reversal_pair_share"] = np.nan
            out["outlier_or_reversal_candidate_share"] = np.nan

        for col in [
            "n_trades",
            "total_volume",
            "price_dispersion_rel",
            "price_range_rel",
            "business_gap_days",
        ]:
            if col in temp.columns and len(temp) > 0:
                out[f"median_{col}"] = float(pd.to_numeric(temp[col], errors="coerce").median())
                out[f"mean_{col}"] = float(pd.to_numeric(temp[col], errors="coerce").mean())

        return out

    summary_rows = []

    samples = [("all_valid_returns", df)]

    for gap in GAP_THRESHOLDS:
        flag = f"valid_return_gap_{gap}bd"
        if flag in df.columns:
            samples.append((f"gap{gap}", df.loc[df[flag]].copy()))

    for sample_name, sample_df in samples:
        summary_rows.append(
            summarise(sample_df, sample_name, "all", "all")
        )

        if "trace_company_symbol" in sample_df.columns:
            for issuer, g in sample_df.groupby("trace_company_symbol"):
                summary_rows.append(
                    summarise(g, sample_name, "issuer", str(issuer))
                )

    summary = pd.DataFrame(summary_rows)

    summary_path = DIAG_DIR / "return_outlier_flag_summary.csv"
    summary.to_csv(summary_path, index=False)

    candidate_cols = [
        c for c in [
            "cusip_id",
            "trace_company_symbol",
            "date",
            "prev_date",
            "next_date_for_reversal",
            "business_gap_days",
            "next_business_gap_days_for_reversal",
            "final_vwap_return",
            "next_return_for_reversal",
            "two_day_reversal_ratio",
            "return_outlier_robust_z",
            "return_outlier_rule",
            "flag_return_outlier_mad",
            "flag_return_outlier_abs",
            "flag_return_outlier_any",
            "flag_two_day_reversal_start",
            "flag_two_day_reversal_pair",
            "n_trades",
            "total_volume",
            "price_dispersion_rel",
            "price_range_rel",
            "vwap_price",
            "median_price",
            "last_price",
            "years_to_maturity",
        ]
        if c in df.columns
    ]

    reversal_path = DIAG_DIR / "two_day_reversal_candidates.csv"

    (
        df.loc[df["flag_two_day_reversal_start"], candidate_cols]
        .sort_values(["date", "cusip_id"])
        .to_csv(reversal_path, index=False)
    )

    overlap_rows = []

    flag_defs = {
        "return_outlier_mad": "flag_return_outlier_mad",
        "return_outlier_abs": "flag_return_outlier_abs",
        "return_outlier_any": "flag_return_outlier_any",
        "two_day_reversal_pair": "flag_two_day_reversal_pair",
        "outlier_or_reversal_candidate": "flag_outlier_or_reversal_candidate",
    }

    for flag_name, flag_col in flag_defs.items():
        if flag_col not in df.columns:
            continue

        temp = df.loc[df[flag_col]].copy()

        row = {
            "flag": flag_name,
            "n_flagged": int(len(temp)),
            "share_flagged": float(len(temp) / len(df)) if len(df) else np.nan,
            "n_cusips_flagged": int(temp["cusip_id"].nunique()) if "cusip_id" in temp.columns else np.nan,
            "mean_abs_return": float(temp["final_vwap_return"].abs().mean()) if len(temp) else np.nan,
            "p99_abs_return": float(temp["final_vwap_return"].abs().quantile(0.99)) if len(temp) else np.nan,
        }

        for col in [
            "n_trades",
            "total_volume",
            "price_dispersion_rel",
            "price_range_rel",
            "business_gap_days",
            "institutional_trade_share",
            "potential_agency_duplicate_share",
            "correction_report_share",
            "ats_trade_share",
        ]:
            if col in temp.columns and len(temp) > 0:
                x = pd.to_numeric(temp[col], errors="coerce")
                row[f"median_{col}"] = float(x.median())
                row[f"mean_{col}"] = float(x.mean())

        overlap_rows.append(row)

    overlap = pd.DataFrame(overlap_rows)

    overlap_path = DIAG_DIR / "outlier_overlap_with_microstructure.csv"
    overlap.to_csv(overlap_path, index=False)

    top_outliers_path = DIAG_DIR / "top_return_outlier_candidates.csv"

    top_cols = [
        c for c in [
            "cusip_id",
            "trace_company_symbol",
            "date",
            "prev_date",
            "calendar_gap_days",
            "business_gap_days",
            "final_vwap_return",
            "return_outlier_robust_z",
            "return_outlier_rule",
            "flag_return_outlier_any",
            "flag_two_day_reversal_pair",
            "n_trades",
            "total_volume",
            "price_dispersion_rel",
            "price_range_rel",
            "vwap_price",
            "median_price",
            "last_price",
            "years_to_maturity",
        ]
        if c in df.columns
    ]

    (
        df.loc[df["flag_outlier_or_reversal_candidate"], top_cols]
        .assign(abs_return=lambda x: x["final_vwap_return"].abs())
        .sort_values("abs_return", ascending=False)
        .head(200)
        .drop(columns=["abs_return"])
        .to_csv(top_outliers_path, index=False)
    )

    print("\nRETURN OUTLIER AND REVERSAL DIAGNOSTICS")
    print(summary.head(20))
    print(f"Saved return-outlier summary to: {summary_path}")
    print(f"Saved two-day reversal candidates to: {reversal_path}")
    print(f"Saved microstructure overlap diagnostics to: {overlap_path}")
    print(f"Saved top return outlier candidates to: {top_outliers_path}")

def write_gap_calendar_diagnostics(final_panel: pd.DataFrame) -> None:
    required = {"cusip_id", "date", "prev_date", "business_gap_days"}

    if not required.issubset(final_panel.columns):
        missing = sorted(required.difference(final_panel.columns))
        raise ValueError(f"Missing columns for gap-calendar diagnostics: {missing}")

    diag = final_panel.copy()

    diag["business_gap_days_pandas_bday"] = business_gap_pandas_bday(
        diag["prev_date"],
        diag["date"],
    )

    diag["business_gap_calendar_diff"] = (
        diag["business_gap_days"]
        - diag["business_gap_days_pandas_bday"]
    )

    if "final_vwap_return" in diag.columns:
        has_valid_return = diag["final_vwap_return"].notna()
    else:
        has_valid_return = diag["prev_date"].notna()

    summary_rows = []

    for gap in GAP_THRESHOLDS:
        old_flag_col = f"valid_return_gap_{gap}bd_pandas_bday"
        new_flag_col = f"valid_return_gap_{gap}bd_holiday_calendar"
        changed_col = f"changed_valid_return_gap_{gap}bd"

        diag[old_flag_col] = (
            has_valid_return
            & diag["business_gap_days_pandas_bday"].notna()
            & diag["business_gap_days_pandas_bday"].le(gap)
        )

        diag[new_flag_col] = (
            has_valid_return
            & diag["business_gap_days"].notna()
            & diag["business_gap_days"].le(gap)
        )

        diag[changed_col] = diag[old_flag_col] != diag[new_flag_col]

        eligible = has_valid_return & diag["business_gap_days"].notna()

        summary_rows.append(
            {
                "gap_threshold_bd": gap,
                "eligible_return_rows": int(eligible.sum()),
                "rows_old_pandas_bday": int(diag.loc[eligible, old_flag_col].sum()),
                "rows_new_holiday_calendar": int(diag.loc[eligible, new_flag_col].sum()),
                "rows_changing_classification": int(diag.loc[eligible, changed_col].sum()),
                "share_changing_classification": (
                    diag.loc[eligible, changed_col].mean()
                    if eligible.any()
                    else np.nan
                ),
                "mean_old_pandas_bday_gap": diag.loc[eligible, "business_gap_days_pandas_bday"].mean(),
                "mean_new_holiday_calendar_gap": diag.loc[eligible, "business_gap_days"].mean(),
                "p99_abs_gap_difference": diag.loc[
                    eligible,
                    "business_gap_calendar_diff",
                ].abs().quantile(0.99),
                "max_abs_gap_difference": diag.loc[
                    eligible,
                    "business_gap_calendar_diff",
                ].abs().max(),
            }
        )

    comparison_summary = pd.DataFrame(summary_rows)
    comparison_path = DIAG_DIR / "gap_calendar_comparison_summary.csv"
    comparison_summary.to_csv(comparison_path, index=False)

    changed_cols = [
        f"changed_valid_return_gap_{gap}bd"
        for gap in GAP_THRESHOLDS
        if f"changed_valid_return_gap_{gap}bd" in diag.columns
    ]

    if changed_cols:
        changed_any = diag[changed_cols].any(axis=1)
    else:
        changed_any = pd.Series(False, index=diag.index)

    export_cols = [
        c
        for c in [
            "cusip_id",
            "trace_company_symbol",
            "date",
            "prev_date",
            "calendar_gap_days",
            "business_gap_days_pandas_bday",
            "business_gap_days",
            "business_gap_calendar_diff",
            "final_vwap_return",
            "n_trades",
            "total_volume",
            "price_dispersion_rel",
            "price_range_rel",
        ]
        if c in diag.columns
    ]

    for gap in GAP_THRESHOLDS:
        for c in [
            f"valid_return_gap_{gap}bd_pandas_bday",
            f"valid_return_gap_{gap}bd_holiday_calendar",
            f"changed_valid_return_gap_{gap}bd",
        ]:
            if c in diag.columns:
                export_cols.append(c)

    changed_path = DIAG_DIR / "holiday_reclassified_gap_rows.csv"

    diag.loc[changed_any, export_cols].sort_values(
        ["date", "cusip_id"]
    ).to_csv(changed_path, index=False)

    print("\nGAP CALENDAR COMPARISON")
    print(comparison_summary)
    print(f"Saved gap-calendar comparison to: {comparison_path}")
    print(f"Saved holiday-reclassified rows to: {changed_path}")

def build_model_ready_panel(final_panel: pd.DataFrame) -> pd.DataFrame:
    print("\nMODEL-READY PANEL WITH GAP SENSITIVITY")
    model_panel = final_panel.copy()

    return_col = "final_vwap_return"
    yield_change_col = "final_yield_change" if "final_yield_change" in model_panel.columns else None

    if return_col not in model_panel.columns:
        raise ValueError(f"Required return column not found: {return_col}")

    if "business_gap_days" not in model_panel.columns:
        raise ValueError("Required column not found: business_gap_days")

    model_panel["has_valid_return"] = model_panel[return_col].notna()

    for gap in GAP_THRESHOLDS:
        model_panel[f"valid_return_gap_{gap}bd"] = (
                model_panel["has_valid_return"]
                & model_panel["business_gap_days"].notna()
                & model_panel["business_gap_days"].le(gap)
        )

    write_dirty_price_diagnostics(model_panel)

    model_panel = add_return_outlier_flags(
        model_panel,
        group_col="cusip_id",
        date_col="date",
        ret_col=return_col,
        window=60,
        min_periods=20,
        mad_k=8.0,
        abs_ret_hard=0.15,
    )

    model_panel = flag_two_day_reversals(
        model_panel,
        group_col="cusip_id",
        date_col="date",
        ret_col=return_col,
        min_abs_ret=0.03,
        reversal_ratio=0.25,
        max_next_gap=MODEL_READY_MAX_BUSINESS_GAP,
    )

    model_panel["flag_outlier_or_reversal_candidate"] = (
            model_panel["flag_return_outlier_any"]
            | model_panel["flag_two_day_reversal_pair"]
    )

    write_return_outlier_diagnostics(model_panel)
    all_returns_panel = model_panel.loc[
        model_panel["has_valid_return"]
        & model_panel["business_gap_days"].notna()
    ].copy()

    all_returns_path = BOND_DAY_DIR / "trace_banks_final_model_ready_all_returns.parquet"
    all_returns_panel.to_parquet(all_returns_path, index=False)

    sensitivity_panel = model_panel.loc[
        model_panel[f"valid_return_gap_{MAX_SENSITIVITY_GAP}bd"]
    ].copy()

    sensitivity_path = BOND_DAY_DIR / "trace_banks_final_model_ready_gap_sensitivity.parquet"
    sensitivity_panel.to_parquet(sensitivity_path, index=False)
    robust_sensitivity_panel = sensitivity_panel.loc[
        ~sensitivity_panel["flag_outlier_or_reversal_candidate"]
    ].copy()

    robust_sensitivity_path = (
            BOND_DAY_DIR / "trace_banks_final_model_ready_gap_sensitivity_robust.parquet"
    )

    robust_sensitivity_panel.to_parquet(
        robust_sensitivity_path,
        index=False,
    )

    print(f"Saved robust sensitivity panel excluding flagged outlier/reversal candidates to: {robust_sensitivity_path}")
    if "final_dirty_vwap_return" in sensitivity_panel.columns:
        dirty_sensitivity_panel = sensitivity_panel.loc[
            sensitivity_panel["final_dirty_vwap_return"].notna()
        ].copy()

        for col in ["final_dirty_vwap_return", "dirty_clean_vwap_return_diff"]:
            if col in dirty_sensitivity_panel.columns:
                lo = dirty_sensitivity_panel[col].quantile(0.005)
                hi = dirty_sensitivity_panel[col].quantile(0.995)
                dirty_sensitivity_panel[f"{col}_winsor_005_995"] = (
                    dirty_sensitivity_panel[col].clip(lo, hi)
                )

        dirty_sensitivity_panel.to_parquet(
            TRACE_MODEL_READY_DIRTY_PATH,
            index=False,
        )
    gap_summary_rows = []

    for gap in GAP_THRESHOLDS:
        gap_panel = model_panel.loc[
            model_panel[f"valid_return_gap_{gap}bd"]
        ].copy()

        for col in [
            return_col,
            "final_dirty_vwap_return",
            "dirty_clean_vwap_return_diff",
            yield_change_col,
            "final_median_price_return",
            "final_last_price_return",
        ]:
            if col and col in gap_panel.columns:
                lo = gap_panel[col].quantile(0.005)
                hi = gap_panel[col].quantile(0.995)
                gap_panel[f"{col}_winsor_005_995"] = gap_panel[col].clip(lo, hi)

        gap_path = BOND_DAY_DIR / f"trace_banks_final_model_ready_gap{gap}.parquet"
        gap_panel.to_parquet(gap_path, index=False)

        gap_summary_rows.append(
            {
                "gap_threshold_bd": gap,
                "rows": len(gap_panel),
                "cusips": gap_panel["cusip_id"].nunique(),
                "first_date": gap_panel["date"].min(),
                "last_date": gap_panel["date"].max(),
                "share_of_final_panel": len(gap_panel) / len(final_panel) if len(final_panel) else np.nan,
                "share_of_all_return_rows": len(gap_panel) / len(all_returns_panel) if len(
                    all_returns_panel) else np.nan,
                "mean_business_gap": gap_panel["business_gap_days"].mean(),
                "median_business_gap": gap_panel["business_gap_days"].median(),
                "p95_business_gap": gap_panel["business_gap_days"].quantile(0.95),
                "mean_abs_vwap_return": gap_panel[return_col].abs().mean(),
                "p99_abs_vwap_return": gap_panel[return_col].abs().quantile(0.99),
                "return_outlier_any_n": int(gap_panel["flag_return_outlier_any"].sum())
                if "flag_return_outlier_any" in gap_panel.columns
                else np.nan,
                "return_outlier_any_share": float(gap_panel["flag_return_outlier_any"].mean())
                if "flag_return_outlier_any" in gap_panel.columns and len(gap_panel) > 0
                else np.nan,
                "two_day_reversal_pair_n": int(gap_panel["flag_two_day_reversal_pair"].sum())
                if "flag_two_day_reversal_pair" in gap_panel.columns
                else np.nan,
                "two_day_reversal_pair_share": float(gap_panel["flag_two_day_reversal_pair"].mean())
                if "flag_two_day_reversal_pair" in gap_panel.columns and len(gap_panel) > 0
                else np.nan,
                "outlier_or_reversal_candidate_n": int(gap_panel["flag_outlier_or_reversal_candidate"].sum())
                if "flag_outlier_or_reversal_candidate" in gap_panel.columns
                else np.nan,
                "outlier_or_reversal_candidate_share": float(gap_panel["flag_outlier_or_reversal_candidate"].mean())
                if "flag_outlier_or_reversal_candidate" in gap_panel.columns and len(gap_panel) > 0
                else np.nan,
            }
        )

    gap_summary = pd.DataFrame(gap_summary_rows)
    gap_summary_path = DIAG_DIR / "model_ready_gap_sensitivity_summary.csv"
    gap_summary.to_csv(gap_summary_path, index=False)

    baseline_panel = model_panel.loc[
        model_panel[f"valid_return_gap_{MODEL_READY_MAX_BUSINESS_GAP}bd"]
    ].copy()

    for col in [
        return_col,
        "final_dirty_vwap_return",
        "dirty_clean_vwap_return_diff",
        yield_change_col,
        "final_median_price_return",
        "final_last_price_return",
    ]:
        if col and col in baseline_panel.columns:
            lo = baseline_panel[col].quantile(0.005)
            hi = baseline_panel[col].quantile(0.995)
            baseline_panel[f"{col}_winsor_005_995"] = baseline_panel[col].clip(lo, hi)

    baseline_path = BOND_DAY_DIR / "trace_banks_final_model_ready_gap5.parquet"
    baseline_panel.to_parquet(baseline_path, index=False)

    if "final_dirty_vwap_return" in baseline_panel.columns:
        dirty_baseline_panel = baseline_panel.loc[
            baseline_panel["final_dirty_vwap_return"].notna()
        ].copy()

        dirty_baseline_panel.to_parquet(
            TRACE_MODEL_READY_GAP5_DIRTY_PATH,
            index=False,
        )

    baseline_summary = {
        "baseline_gap_threshold_bd": MODEL_READY_MAX_BUSINESS_GAP,
        "rows": len(baseline_panel),
        "cusips": baseline_panel["cusip_id"].nunique(),
        "first_date": baseline_panel["date"].min(),
        "last_date": baseline_panel["date"].max(),
        "share_of_final_panel": len(baseline_panel) / len(final_panel) if len(final_panel) else np.nan,
        "share_of_all_return_rows": len(baseline_panel) / len(all_returns_panel) if len(all_returns_panel) else np.nan,
    }

    pd.DataFrame([baseline_summary]).to_csv(DIAG_DIR / "model_ready_gap5_summary.csv", index=False)

    print("\nGap sensitivity summary:")
    print(gap_summary)

    print(f"\nSaved all-return panel to: {all_returns_path}")
    print(f"Saved sensitivity panel to: {sensitivity_path}")
    print(f"Saved baseline gap5 panel to: {baseline_path}")
    if "final_dirty_vwap_return" in sensitivity_panel.columns:
        print(f"Saved dirty sensitivity panel to: {TRACE_MODEL_READY_DIRTY_PATH}")
        print(f"Saved dirty gap5 panel to: {TRACE_MODEL_READY_GAP5_DIRTY_PATH}")
    print(f"Saved gap summary to: {gap_summary_path}")

    return sensitivity_panel


def main() -> None:
    parquet_files = convert_raw_zips_to_parquet()
    write_parquet_summary(parquet_files)
    bond_day = build_clean_bond_day(parquet_files)
    write_bond_day_diagnostics(bond_day)
    build_final_panel_with_master()
    print("\nTRACE institutional cleaning pipeline complete.")


if __name__ == "__main__":
    main()
