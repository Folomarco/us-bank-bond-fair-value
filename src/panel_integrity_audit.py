from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def _dtype_matches(series: pd.Series, expected: str) -> bool:
    expected = str(expected).lower().strip()

    if expected in {"datetime", "datetime64", "datetime64[ns]"}:
        return pd.api.types.is_datetime64_any_dtype(series)

    if expected in {"numeric", "number"}:
        return pd.api.types.is_numeric_dtype(series)

    if expected in {"float", "floating"}:
        return pd.api.types.is_float_dtype(series) or pd.api.types.is_numeric_dtype(series)

    if expected in {"integer", "int"}:
        return pd.api.types.is_integer_dtype(series) or pd.api.types.is_numeric_dtype(series)

    if expected in {"bool", "boolean"}:
        return pd.api.types.is_bool_dtype(series)

    if expected in {"string", "object", "str"}:
        return (
            pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
        )

    return str(series.dtype).lower() == expected


def assert_panel_integrity(
    df: pd.DataFrame,
    panel_name: str,
    key_cols: list[str],
    date_col: str,
    required_columns: list[str] | None = None,
    required_nonmissing: list[str] | None = None,
    required_nonnegative: list[str] | None = None,
    expected_dtypes: dict[str, str] | None = None,
    group_col: str | None = "cusip_id",
    prev_date_col: str | None = "prev_date",
    forbid_weekends: bool = True,
    fail_fast: bool = True,
) -> pd.DataFrame:

    required_columns = required_columns or []
    required_nonmissing = required_nonmissing or []
    required_nonnegative = required_nonnegative or []
    expected_dtypes = expected_dtypes or {}

    checks: list[dict[str, Any]] = []

    def add_check(
        check: str,
        value: Any,
        expected: Any,
        passed: bool,
        severity: str = "error",
        detail: str = "",
    ) -> None:
        checks.append(
            {
                "panel": panel_name,
                "check": check,
                "value": _json_safe(value),
                "expected": _json_safe(expected),
                "passed": bool(passed),
                "severity": severity,
                "detail": detail,
            }
        )

    add_check("row_count_positive", len(df), "> 0", len(df) > 0)

    all_required_cols = sorted(set(key_cols + [date_col] + required_columns))
    missing_required_cols = [c for c in all_required_cols if c not in df.columns]

    add_check(
        "missing_required_columns",
        len(missing_required_cols),
        0,
        len(missing_required_cols) == 0,
        detail=", ".join(missing_required_cols),
    )

    if missing_required_cols:
        report = pd.DataFrame(checks)
        if fail_fast:
            raise ValueError(
                f"Panel integrity failed for {panel_name}: missing columns "
                f"{missing_required_cols}"
            )
        return report

    duplicate_keys = int(df.duplicated(key_cols).sum())
    add_check("duplicate_key_rows", duplicate_keys, 0, duplicate_keys == 0)

    null_in_keys = int(df[key_cols].isna().any(axis=1).sum())
    add_check("null_in_key_rows", null_in_keys, 0, null_in_keys == 0)

    date_values = pd.to_datetime(df[date_col], errors="coerce")

    null_dates = int(date_values.isna().sum())
    add_check("null_dates", null_dates, 0, null_dates == 0)

    if forbid_weekends:
        weekend_rows = int(date_values.dt.weekday.ge(5).fillna(False).sum())
        add_check("weekend_rows", weekend_rows, 0, weekend_rows == 0)

    if group_col and group_col in df.columns:

        date_diff = date_values.groupby(df[group_col]).diff()
        non_monotonic = int(date_diff.lt(pd.Timedelta(0)).fillna(False).sum())

        add_check(
            "non_monotonic_dates_within_group",
            non_monotonic,
            0,
            non_monotonic == 0,
            detail=f"group_col={group_col}",
        )

    if prev_date_col and prev_date_col in df.columns:
        prev_dates = pd.to_datetime(df[prev_date_col], errors="coerce")
        prev_ge_date = int((prev_dates >= date_values).fillna(False).sum())

        add_check(
            "prev_date_ge_date",
            prev_ge_date,
            0,
            prev_ge_date == 0,
        )


    for col in required_nonmissing:
        if col not in df.columns:
            add_check(
                f"missing_required_nonmissing_col__{col}",
                1,
                0,
                False,
                detail="Column not found",
            )
            continue

        missing = int(df[col].isna().sum())
        add_check(
            f"missing_values__{col}",
            missing,
            0,
            missing == 0,
        )

    for col in required_nonnegative:
        if col not in df.columns:
            add_check(
                f"missing_required_nonnegative_col__{col}",
                1,
                0,
                False,
                detail="Column not found",
            )
            continue

        x = pd.to_numeric(df[col], errors="coerce")
        negative = int(x.lt(0).fillna(False).sum())

        add_check(
            f"negative_values__{col}",
            negative,
            0,
            negative == 0,
        )

    for col, expected_dtype in expected_dtypes.items():
        if col not in df.columns:
            add_check(
                f"dtype_missing_col__{col}",
                "missing",
                expected_dtype,
                False,
            )
            continue

        observed_dtype = str(df[col].dtype)
        passed = _dtype_matches(df[col], expected_dtype)

        add_check(
            f"dtype__{col}",
            observed_dtype,
            expected_dtype,
            passed,
        )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        inf_count = int(np.isinf(df[numeric_cols].to_numpy(dtype=float, copy=False)).sum())
    else:
        inf_count = 0

    add_check("infinite_numeric_values", inf_count, 0, inf_count == 0)


    if "business_gap_days" in df.columns:
        gap = pd.to_numeric(df["business_gap_days"], errors="coerce")

        negative_gap = int(gap.lt(0).fillna(False).sum())
        add_check("negative_business_gap_days", negative_gap, 0, negative_gap == 0)

        if "final_vwap_return" in df.columns:
            has_valid_return = df["final_vwap_return"].notna()
        else:
            has_valid_return = gap.notna()

        for col in df.columns:
            if (
                col.startswith("valid_return_gap_")
                and col.endswith("bd")
                and col.replace("valid_return_gap_", "").replace("bd", "").isdigit()
            ):
                threshold = int(
                    col.replace("valid_return_gap_", "").replace("bd", "")
                )

                expected_flag = (
                    has_valid_return
                    & gap.notna()
                    & gap.le(threshold)
                )

                observed_flag = df[col].fillna(False).astype(bool)
                mismatch = int((observed_flag != expected_flag).sum())

                add_check(
                    f"gap_flag_consistency__{col}",
                    mismatch,
                    0,
                    mismatch == 0,
                    detail=f"threshold={threshold}",
                )

    report = pd.DataFrame(checks)

    hard_failures = report.loc[
        report["severity"].eq("error") & ~report["passed"]
    ]

    if fail_fast and not hard_failures.empty:
        failed = hard_failures[["check", "value", "expected", "detail"]]
        raise ValueError(
            f"Panel integrity failed for {panel_name}:\n"
            f"{failed.to_string(index=False)}"
        )

    return report


def file_fingerprint(path: Path, hash_file: bool = True) -> dict[str, Any]:
    path = Path(path)

    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": None,
        "modified_utc": None,
        "sha256": None,
    }

    if not path.exists():
        return out

    stat = path.stat()
    out["size_bytes"] = int(stat.st_size)
    out["modified_utc"] = datetime.fromtimestamp(
        stat.st_mtime,
        tz=timezone.utc,
    ).replace(microsecond=0).isoformat()

    if hash_file:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        out["sha256"] = h.hexdigest()

    return out


def write_run_manifest(
    manifest_path: Path,
    config: dict[str, Any],
    input_paths: list[Path],
    output_paths: list[Path],
    hash_inputs: bool = True,
    hash_outputs: bool = False,
) -> dict[str, Any]:
    manifest = {
        "created_utc": _now_utc_iso(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "libraries": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "config": _json_safe(config),
        "inputs": [
            file_fingerprint(Path(path), hash_file=hash_inputs)
            for path in input_paths
        ],
        "outputs": [
            file_fingerprint(Path(path), hash_file=hash_outputs)
            for path in output_paths
        ],
    }

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    return manifest