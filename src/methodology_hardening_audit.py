from __future__ import annotations


import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config_institutional import (
    DIAGNOSTICS_DIR,
    REGRESSION_DIR,
    TABLES_DIR,
    TRACE_CONVERGENCE_PRICE_PATH,
    TRACE_FINAL_BASELINE_PANEL_PATH,
    ensure_directories,
)


GROUP_COL = "cusip_id"
DATE_COL = "date"
PREV_DATE_COL = "prev_date"
TARGET_COL = "final_vwap_return"
SPLIT_COL = "sample_split"

PEER_PANEL_PATH = REGRESSION_DIR / "regression_panel_gap5_with_peer_factors.parquet"
PREDICTIONS_PATH = REGRESSION_DIR / "peer_baseline_gap5_model_predictions.parquet"
SIGNALS_PATH = REGRESSION_DIR / "dislocation_signals_gap5_m4_m5.parquet"
EVENT_TRADES_PATH = TABLES_DIR / "dislocation_event_driven_strategy_trades.csv"
TRACE_REFERENCE_AUDIT_PATH = (
    DIAGNOSTICS_DIR / "trace_cleaning" / "trace_message_reference_resolution.csv"
)
TRACE_REFERENCE_SUMMARY_PATH = (
    DIAGNOSTICS_DIR / "trace_cleaning" / "trace_message_reference_summary.csv"
)
TRACE_STATUS_AUDIT_PATH = (
    DIAGNOSTICS_DIR / "trace_cleaning" / "trace_status_schema_detection.csv"
)
TRACE_CLEANING_FLAGS_PATH = (
    DIAGNOSTICS_DIR / "trace_cleaning" / "trace_cleaning_flags_summary.csv"
)
MASTER_ACTIVE_MATCH_SUMMARY_PATH = (
    DIAGNOSTICS_DIR / "master_pit_active_match_summary.csv"
)
OUTPUT_AUDIT = DIAGNOSTICS_DIR / "methodology_hardening_post_run_audit.csv"
OUTPUT_MANIFEST = DIAGNOSTICS_DIR / "methodology_hardening_post_run_manifest.json"

MATURITY_BINS = [1, 3, 5, 7, 10, 15, 30, np.inf]
MATURITY_LABELS = ["1-3y", "3-5y", "5-7y", "7-10y", "10-15y", "15-30y", "30y+"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    ensure_directories()
    rows: list[dict[str, Any]] = []

    def record(
        check: str,
        passed: bool,
        value: Any,
        expected: Any,
        severity: str = "error",
        detail: str = "",
    ) -> None:
        rows.append(
            {
                "check": check,
                "passed": bool(passed),
                "value": _json_safe(value),
                "expected": _json_safe(expected),
                "severity": severity,
                "detail": detail,
            }
        )

    required_paths = {
        "final_baseline_panel": TRACE_FINAL_BASELINE_PANEL_PATH,
        "peer_panel": PEER_PANEL_PATH,
        "baseline_predictions": PREDICTIONS_PATH,
        "signals": SIGNALS_PATH,
        "trace_reference_audit": TRACE_REFERENCE_AUDIT_PATH,
        "trace_reference_summary": TRACE_REFERENCE_SUMMARY_PATH,
        "trace_status_audit": TRACE_STATUS_AUDIT_PATH,
        "trace_cleaning_flags": TRACE_CLEANING_FLAGS_PATH,
        "master_active_match_summary": MASTER_ACTIVE_MATCH_SUMMARY_PATH,
    }
    for name, path in required_paths.items():
        record(
            f"required_path__{name}",
            path.exists(),
            path.exists(),
            True,
            detail=str(path),
        )


    if TRACE_STATUS_AUDIT_PATH.exists():
        status_audit = pd.read_csv(TRACE_STATUS_AUDIT_PATH)
        expected_actions = {"M": "regular", "N": "cancel", "O": "correction"}
        observed_schema = set(status_audit.get("detected_schema", pd.Series(dtype=str)).dropna().astype(str))
        if observed_schema == {"wrds_mno"}:
            actual_actions = {
                str(row.trc_st): str(row.mapped_action)
                for row in status_audit.itertuples(index=False)
            }
            mismatches = {
                code: {"expected": action, "actual": actual_actions.get(code)}
                for code, action in expected_actions.items()
                if actual_actions.get(code) != action
            }
            record(
                "wrds_mno_status_action_mapping",
                len(mismatches) == 0,
                json.dumps(mismatches, sort_keys=True),
                json.dumps(expected_actions, sort_keys=True),
            )

    if TRACE_REFERENCE_SUMMARY_PATH.exists():
        reference_summary = pd.read_csv(TRACE_REFERENCE_SUMMARY_PATH)
        nonexact_resolved = int(
            pd.to_numeric(
                reference_summary.get("nonexact_resolved_matches", 0),
                errors="coerce",
            )
            .fillna(0)
            .max()
        )
        record(
            "trace_reference_resolutions_require_exact_execution_date",
            nonexact_resolved == 0,
            nonexact_resolved,
            0,
        )

    if TRACE_CLEANING_FLAGS_PATH.exists():
        flags = pd.read_csv(TRACE_CLEANING_FLAGS_PATH)
        cancel_rows = int(pd.to_numeric(flags.get("is_cancel_report", 0), errors="coerce").fillna(0).sum())
        correction_rows = int(pd.to_numeric(flags.get("is_correction_report", 0), errors="coerce").fillna(0).sum())
        clean_cancel_rows = int(pd.to_numeric(flags.get("clean_price_cancel_rows", 0), errors="coerce").fillna(0).sum())
        clean_correction_rows = int(pd.to_numeric(flags.get("clean_price_correction_rows", 0), errors="coerce").fillna(0).sum())
        clean_unresolved_correction_rows = int(
            pd.to_numeric(
                flags.get("clean_price_unresolved_correction_rows", 0),
                errors="coerce",
            ).fillna(0).sum()
        )
        record("trace_cancel_messages_detected", cancel_rows > 0, cancel_rows, "> 0")
        record("trace_correction_messages_detected", correction_rows > 0, correction_rows, "> 0")
        record("clean_price_contains_no_cancel_messages", clean_cancel_rows == 0, clean_cancel_rows, 0)
        record("clean_price_retains_corrected_messages", clean_correction_rows > 0, clean_correction_rows, "> 0")
        record(
            "clean_price_contains_no_unresolved_corrections",
            clean_unresolved_correction_rows == 0,
            clean_unresolved_correction_rows,
            0,
        )

    if MASTER_ACTIVE_MATCH_SUMMARY_PATH.exists():
        master_summary = pd.read_csv(MASTER_ACTIVE_MATCH_SUMMARY_PATH)
        if not master_summary.empty:
            row = master_summary.iloc[0]
            max_matches = int(row.get("max_active_master_matches", 0))
            fallback_rows = int(row.get("rows_falling_back_to_earlier_active_interval", 0))
            record(
                "master_interval_join_reports_active_match_counts",
                max_matches >= 1,
                max_matches,
                ">= 1",
                detail=f"fallback_rows={fallback_rows}",
            )

    if TRACE_FINAL_BASELINE_PANEL_PATH.exists():
        final_panel = pd.read_parquet(TRACE_FINAL_BASELINE_PANEL_PATH)
        final_panel[DATE_COL] = pd.to_datetime(final_panel[DATE_COL], errors="coerce")
        duplicate_keys = int(final_panel.duplicated([GROUP_COL, DATE_COL]).sum())
        weekend_rows = int(final_panel[DATE_COL].dt.weekday.ge(5).fillna(False).sum())
        nonpositive_price = int(
            pd.to_numeric(final_panel.get("vwap_price"), errors="coerce").le(0).fillna(False).sum()
        )
        record("final_panel_duplicate_cusip_date", duplicate_keys == 0, duplicate_keys, 0)
        record("final_panel_weekend_rows", weekend_rows == 0, weekend_rows, 0)
        record("final_panel_nonpositive_vwap", nonpositive_price == 0, nonpositive_price, 0)
        if "master_active_match_count" in final_panel.columns:
            missing_active_master = int(
                pd.to_numeric(final_panel["master_active_match_count"], errors="coerce")
                .fillna(0)
                .le(0)
                .sum()
            )
            record(
                "final_panel_has_active_master_match",
                missing_active_master == 0,
                missing_active_master,
                0,
            )

    if PEER_PANEL_PATH.exists():
        peer = pd.read_parquet(PEER_PANEL_PATH)
        peer[DATE_COL] = pd.to_datetime(peer[DATE_COL], errors="coerce")
        expected_bucket = pd.cut(
            pd.to_numeric(peer["years_to_maturity"], errors="coerce"),
            bins=MATURITY_BINS,
            labels=MATURITY_LABELS,
            right=False,
        ).astype("string")
        actual_bucket = peer["peer_maturity_bucket"].astype("string")
        bucket_mismatch = int(
            (expected_bucket.notna() & actual_bucket.notna() & expected_bucket.ne(actual_bucket)).sum()
        )
        record("point_in_time_maturity_bucket_mismatch", bucket_mismatch == 0, bucket_mismatch, 0)

        for base in [
            "same_issuer_maturity",
            "other_bank_maturity",
            "bank_sector_maturity",
        ]:
            factor_col = f"peer_raw_{base}"
            n_col = f"peer_n_{base}"
            if factor_col in peer.columns and n_col in peer.columns:
                bad_min_peer = int(
                    (peer[factor_col].notna() & pd.to_numeric(peer[n_col], errors="coerce").lt(3)).sum()
                )
                record(
                    f"minimum_peer_count__{base}",
                    bad_min_peer == 0,
                    bad_min_peer,
                    0,
                )

        source_cols = [c for c in peer.columns if c.endswith("_source_date_t")]
        future_count = 0
        for col in source_cols:
            src = pd.to_datetime(peer[col], errors="coerce")
            future_count += int((src.notna() & peer[DATE_COL].notna() & src.gt(peer[DATE_COL])).sum())
        record("peer_panel_future_source_dates", future_count == 0, future_count, 0)

    if PREDICTIONS_PATH.exists():
        preds = pd.read_parquet(PREDICTIONS_PATH)
        preds[DATE_COL] = pd.to_datetime(preds[DATE_COL], errors="coerce")
        expected_split = np.select(
            [
                preds[DATE_COL] < pd.Timestamp("2023-01-01"),
                (preds[DATE_COL] >= pd.Timestamp("2023-01-01"))
                & (preds[DATE_COL] < pd.Timestamp("2024-01-01")),
                preds[DATE_COL] >= pd.Timestamp("2024-01-01"),
            ],
            ["train", "validation", "test"],
            default="excluded",
        )
        split_mismatch = int((preds[SPLIT_COL].astype(str).to_numpy() != expected_split).sum())
        record("prediction_split_mismatch", split_mismatch == 0, split_mismatch, 0)

    if SIGNALS_PATH.exists():
        signals = pd.read_parquet(SIGNALS_PATH)
        signals[DATE_COL] = pd.to_datetime(signals[DATE_COL], errors="coerce")
        train_candidates = int(
            signals.loc[signals[SPLIT_COL].eq("train"), "m4_candidate_flag"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        record("candidate_events_in_training", train_candidates == 0, train_candidates, 0)

        if TRACE_CONVERGENCE_PRICE_PATH.exists():
            price = pd.read_parquet(TRACE_CONVERGENCE_PRICE_PATH)
            price[DATE_COL] = pd.to_datetime(price[DATE_COL], errors="coerce")
            price_col = "vwap_price" if "vwap_price" in price.columns else "final_vwap_price"
            price["_log_price"] = np.log(
                pd.to_numeric(price[price_col], errors="coerce").where(
                    pd.to_numeric(price[price_col], errors="coerce") > 0
                )
            )
            lookup = price.dropna(subset=[GROUP_COL, DATE_COL, "_log_price"]).set_index(
                [GROUP_COL, DATE_COL]
            )["_log_price"]
            checked = 0
            mismatches = 0
            for h in [1, 3, 5]:
                ret_col = f"future_return_{h}obs"
                exit_col = f"future_exit_date_{h}obs"
                if ret_col not in signals.columns or exit_col not in signals.columns:
                    continue
                sample = signals.dropna(subset=[ret_col, exit_col]).head(5000)
                for row in sample.itertuples(index=False):
                    entry_key = (str(getattr(row, GROUP_COL)), pd.Timestamp(getattr(row, DATE_COL)))
                    exit_key = (str(getattr(row, GROUP_COL)), pd.Timestamp(getattr(row, exit_col)))
                    if entry_key not in lookup.index or exit_key not in lookup.index:
                        continue
                    direct = float(lookup.loc[exit_key] - lookup.loc[entry_key])
                    reported = float(getattr(row, ret_col))
                    checked += 1
                    if not np.isclose(direct, reported, rtol=0.0, atol=1e-12):
                        mismatches += 1
            record(
                "fixed_horizon_direct_price_identity",
                checked > 0 and mismatches == 0,
                mismatches,
                0,
                detail=f"checked={checked}",
            )

    if EVENT_TRADES_PATH.exists():
        trades = pd.read_csv(EVENT_TRADES_PATH)
        direct_mismatch = 0
        checked = 0
        if {
            "entry_log_vwap_price",
            "exit_log_vwap_price",
            "direction",
            "gross_bond_payoff",
        }.issubset(trades.columns):
            valid = trades.dropna(
                subset=[
                    "entry_log_vwap_price",
                    "exit_log_vwap_price",
                    "direction",
                    "gross_bond_payoff",
                ]
            )
            direct = pd.to_numeric(valid["direction"], errors="coerce") * (
                pd.to_numeric(valid["exit_log_vwap_price"], errors="coerce")
                - pd.to_numeric(valid["entry_log_vwap_price"], errors="coerce")
            )
            reported = pd.to_numeric(valid["gross_bond_payoff"], errors="coerce")
            checked = int(len(valid))
            direct_mismatch = int((~np.isclose(direct, reported, rtol=0.0, atol=1e-12)).sum())
        record(
            "event_driven_direct_price_identity",
            checked > 0 and direct_mismatch == 0,
            direct_mismatch,
            0,
            detail=f"checked={checked}",
        )
    else:
        record(
            "event_driven_trade_file_present",
            False,
            False,
            True,
            severity="warning",
            detail=str(EVENT_TRADES_PATH),
        )

    audit = pd.DataFrame(rows)
    OUTPUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_AUDIT, index=False)

    manifest = {
        "audit_file": str(OUTPUT_AUDIT),
        "n_checks": int(len(audit)),
        "n_failed_errors": int((~audit["passed"] & audit["severity"].eq("error")).sum()),
        "n_failed_warnings": int((~audit["passed"] & audit["severity"].eq("warning")).sum()),
        "paths": {name: str(path) for name, path in required_paths.items()},
    }
    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(audit.to_string(index=False))
    print(f"\nSaved audit: {OUTPUT_AUDIT}")
    print(f"Saved manifest: {OUTPUT_MANIFEST}")

    failed = audit.loc[(~audit["passed"]) & audit["severity"].eq("error")]
    if not failed.empty:
        raise RuntimeError(
            "Methodology hardening audit failed:\n"
            + failed[["check", "value", "expected", "detail"]].to_string(index=False)
        )


if __name__ == "__main__":
    main()
