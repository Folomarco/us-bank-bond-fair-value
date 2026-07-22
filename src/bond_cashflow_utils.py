from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset


def _coupon_rate_to_decimal(x) -> float:
    if pd.isna(x):
        return np.nan
    return float(x) / 100.0


def _coupon_frequency_to_int(x, default_coupon_freq: int = 2) -> int:
    if pd.isna(x):
        return default_coupon_freq

    value = str(x).strip().upper()
    mapping = {
        "1": 1,
        "A": 1,
        "ANNUAL": 1,
        "ANNUALLY": 1,
        "2": 2,
        "S": 2,
        "SA": 2,
        "SEMI": 2,
        "SEMIANNUAL": 2,
        "SEMI-ANNUAL": 2,
        "4": 4,
        "Q": 4,
        "QUARTERLY": 4,
        "12": 12,
        "M": 12,
        "MONTHLY": 12,
    }

    if value in mapping:
        return mapping[value]

    try:
        out = int(float(value))
        if out > 0:
            return out
    except ValueError:
        pass

    return default_coupon_freq


def _day_count_30_360_us(start: pd.Timestamp, end: pd.Timestamp) -> float:
    if pd.isna(start) or pd.isna(end):
        return np.nan

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    d1 = start.day
    d2 = end.day

    if d1 == 31:
        d1 = 30

    if d2 == 31 and d1 in [30, 31]:
        d2 = 30

    return (
        360 * (end.year - start.year)
        + 30 * (end.month - start.month)
        + (d2 - d1)
    )


def _infer_coupon_window(
    obs_date: pd.Timestamp,
    maturity_date: pd.Timestamp,
    coupon_freq: int,
    issue_date: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if pd.isna(obs_date) or pd.isna(maturity_date):
        return pd.NaT, pd.NaT

    obs_date = pd.Timestamp(obs_date)
    maturity_date = pd.Timestamp(maturity_date)

    if obs_date >= maturity_date:
        return pd.NaT, pd.NaT

    coupon_freq = int(coupon_freq)
    if coupon_freq <= 0:
        coupon_freq = 2

    months_step = int(round(12 / coupon_freq))
    months_step = max(months_step, 1)

    months_diff = (
        (maturity_date.year - obs_date.year) * 12
        + (maturity_date.month - obs_date.month)
    )

    k = max(0, months_diff // months_step)
    next_coupon = maturity_date - DateOffset(months=int(k * months_step))

    while next_coupon <= obs_date and k > 0:
        k -= 1
        next_coupon = maturity_date - DateOffset(months=int(k * months_step))

    while next_coupon - DateOffset(months=months_step) > obs_date:
        k += 1
        next_coupon = maturity_date - DateOffset(months=int(k * months_step))

    prev_coupon = next_coupon - DateOffset(months=months_step)

    if issue_date is not None and pd.notna(issue_date):
        issue_date = pd.Timestamp(issue_date)
        if prev_coupon < issue_date <= obs_date:
            prev_coupon = issue_date

    return pd.Timestamp(prev_coupon), pd.Timestamp(next_coupon)


def add_dirty_price_columns(
    df: pd.DataFrame,
    clean_price_col: str,
    coupon_rate_col: str,
    date_col: str,
    maturity_col: str,
    issue_date_col: str | None = None,
    coupon_freq_col: str | None = None,
    day_count_col: str | None = None,
    default_coupon_freq: int = 2,
    default_day_count: str = "30_360_US",
    group_col: str = "cusip_id",
    prefix: str = "vwap",
    return_col: str = "final_dirty_vwap_return",
) -> pd.DataFrame:
    out = df.copy()

    required = [clean_price_col, coupon_rate_col, date_col, maturity_col]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for dirty-price construction: {missing}")

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[maturity_col] = pd.to_datetime(out[maturity_col], errors="coerce")

    if issue_date_col and issue_date_col in out.columns:
        out[issue_date_col] = pd.to_datetime(out[issue_date_col], errors="coerce")
    else:
        issue_date_col = None

    clean_price = pd.to_numeric(out[clean_price_col], errors="coerce")
    coupon_decimal = out[coupon_rate_col].map(_coupon_rate_to_decimal)

    if coupon_freq_col and coupon_freq_col in out.columns:
        coupon_freq = out[coupon_freq_col].map(
            lambda x: _coupon_frequency_to_int(x, default_coupon_freq)
        )
    else:
        coupon_freq = pd.Series(default_coupon_freq, index=out.index)

    prev_coupon_dates = []
    next_coupon_dates = []

    for idx, row in out.iterrows():
        issue_date = row[issue_date_col] if issue_date_col else None

        prev_coupon, next_coupon = _infer_coupon_window(
            obs_date=row[date_col],
            maturity_date=row[maturity_col],
            coupon_freq=int(coupon_freq.loc[idx]),
            issue_date=issue_date,
        )

        prev_coupon_dates.append(prev_coupon)
        next_coupon_dates.append(next_coupon)

    prev_coupon_col = f"{prefix}_prev_coupon_date"
    next_coupon_col = f"{prefix}_next_coupon_date"
    accrual_fraction_col = f"{prefix}_accrual_fraction"
    accrued_interest_col = f"{prefix}_accrued_interest"
    dirty_price_col = f"dirty_{prefix}_price"
    dirty_log_col = f"final_log_dirty_{prefix}_price"
    method_col = f"{prefix}_ai_method"

    out[prev_coupon_col] = pd.to_datetime(prev_coupon_dates)
    out[next_coupon_col] = pd.to_datetime(next_coupon_dates)
    numerators = [
        _day_count_30_360_us(start, end)
        for start, end in zip(out[prev_coupon_col], out[date_col])
    ]
    denominators = [
        _day_count_30_360_us(start, end)
        for start, end in zip(out[prev_coupon_col], out[next_coupon_col])
    ]
    numerator = pd.Series(numerators, index=out.index, dtype="float64")
    denominator = pd.Series(denominators, index=out.index, dtype="float64")
    accrual_fraction = numerator / denominator.replace(0, np.nan)
    accrual_fraction = accrual_fraction.clip(lower=0.0, upper=1.0)
    out[accrual_fraction_col] = accrual_fraction
    out[accrued_interest_col] = (
        100.0
        * coupon_decimal
        / coupon_freq.astype(float).replace(0, np.nan)
        * out[accrual_fraction_col]
    )
    out[dirty_price_col] = clean_price + out[accrued_interest_col]
    out[dirty_log_col] = np.log(out[dirty_price_col].where(out[dirty_price_col] > 0))
    coupon_cashflow_col = f"{prefix}_coupon_cashflow"
    coupon_paid_flag_col = f"{prefix}_coupon_paid_flag"
    prev_accrual_fraction_col = f"{prefix}_prev_accrual_fraction"
    dirty_simple_return_col = f"final_dirty_{prefix}_simple_return"
    coupon_payment = (
            100.0
            * coupon_decimal
            / coupon_freq.astype(float).replace(0, np.nan)
    )
    if group_col in out.columns:
        prev_dirty_price = out.groupby(group_col)[dirty_price_col].shift(1)
        prev_obs_date = out.groupby(group_col)[date_col].shift(1)
        prev_accrual_fraction = out.groupby(group_col)[accrual_fraction_col].shift(1)
    else:
        prev_dirty_price = out[dirty_price_col].shift(1)
        prev_obs_date = out[date_col].shift(1)
        prev_accrual_fraction = out[accrual_fraction_col].shift(1)
    out[prev_accrual_fraction_col] = prev_accrual_fraction
    coupon_date_crossed = (
            prev_obs_date.notna()
            & out[prev_coupon_col].notna()
            & out[date_col].notna()
            & (prev_obs_date < out[prev_coupon_col])
            & (out[prev_coupon_col] <= out[date_col])
    )
    accrual_reset_detected = (
            prev_accrual_fraction.notna()
            & out[accrual_fraction_col].notna()
            & (out[accrual_fraction_col] + 1e-8 < prev_accrual_fraction)
    )
    coupon_paid_in_interval = coupon_date_crossed | accrual_reset_detected
    out[coupon_paid_flag_col] = coupon_paid_in_interval.astype(int)
    out[coupon_cashflow_col] = np.where(
        coupon_paid_in_interval,
        coupon_payment,
        0.0,
    )
    out[dirty_simple_return_col] = (
            (out[dirty_price_col] + out[coupon_cashflow_col])
            / prev_dirty_price
            - 1.0
    )
    out[return_col] = np.log1p(
        out[dirty_simple_return_col].where(out[dirty_simple_return_col] > -1.0)
    )
    valid_ai = (
        out[accrued_interest_col].notna()
        & out[dirty_price_col].notna()
        & out[prev_coupon_col].notna()
        & out[next_coupon_col].notna()
    )
    out[method_col] = np.where(
        valid_ai,
        f"proxy_{default_day_count.lower()}_freq{default_coupon_freq}",
        "unavailable",
    )

    return out