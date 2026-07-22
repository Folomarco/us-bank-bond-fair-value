from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_market_holidays(path: Path, calendar_name: str) -> np.busdaycalendar:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {calendar_name} holiday calendar: {path}. "
            "Create this CSV before running the pipeline."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{calendar_name} holiday calendar is empty: {path}")

    date_col = "date" if "date" in df.columns else df.columns[0]

    holidays = pd.to_datetime(df[date_col], errors="coerce").dropna()

    if holidays.empty:
        raise ValueError(
            f"No valid holiday dates found in {calendar_name} calendar: {path}"
        )

    holidays = pd.DatetimeIndex(holidays.dt.normalize().unique())
    holidays_np = holidays.values.astype("datetime64[D]")

    return np.busdaycalendar(
        weekmask="1111100",
        holidays=holidays_np,
    )


def business_gap_custom(
    start: pd.Series,
    end: pd.Series,
    calendar: np.busdaycalendar,
) -> pd.Series:
    start_dt = pd.to_datetime(start, errors="coerce")
    end_dt = pd.to_datetime(end, errors="coerce")

    out = pd.Series(np.nan, index=start.index, dtype="float64")
    valid = start_dt.notna() & end_dt.notna()

    if valid.any():
        start_days = start_dt.loc[valid].dt.normalize().values.astype("datetime64[D]")
        end_days = end_dt.loc[valid].dt.normalize().values.astype("datetime64[D]")

        out.loc[valid] = np.busday_count(
            start_days,
            end_days,
            busdaycal=calendar,
        ).astype(float)

    return out


def business_gap_pandas_bday(start: pd.Series, end: pd.Series) -> pd.Series:
    start_dt = pd.to_datetime(start, errors="coerce")
    end_dt = pd.to_datetime(end, errors="coerce")

    out = pd.Series(np.nan, index=start.index, dtype="float64")
    valid = start_dt.notna() & end_dt.notna()

    out.loc[valid] = [
        len(pd.bdate_range(start=s, end=e)) - 1
        for s, e in zip(start_dt.loc[valid], end_dt.loc[valid])
    ]

    return out


def business_dates_custom(
    start,
    end,
    calendar: np.busdaycalendar,
) -> pd.DatetimeIndex:
    if pd.isna(start) or pd.isna(end):
        return pd.DatetimeIndex([])

    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    dates_np = dates.values.astype("datetime64[D]")

    mask = np.is_busday(dates_np, busdaycal=calendar)

    return pd.DatetimeIndex(dates[mask])