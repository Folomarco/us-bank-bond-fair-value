from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from config_institutional import REGRESSION_DIR, TABLES_DIR, ensure_directories


GROUP_COL = "cusip_id"
DATE_COL = "date"
SPLIT_COL = "sample_split"
ISSUER_COL = "trace_company_symbol"
MATURITY_BUCKET_COL = "peer_maturity_bucket"
PANEL_PATH = REGRESSION_DIR / "regression_panel_gap5_with_peer_factors.parquet"
OUTPUT_TABLE = TABLES_DIR / "residual_model_comparison_test.csv"
OUTPUT_LATEX = TABLES_DIR / "residual_model_comparison_test_latex.tex"
OUTPUT_MANIFEST = TABLES_DIR / "residual_model_comparison_manifest.json"

MIN_GROUP_HISTORY_OBS = 100
MIN_CUSIP_HISTORY_OBS = 20
CUSIP_ROLLING_WINDOW_OBS = 60
MIN_GLOBAL_HISTORY_OBS = 500

MODEL_SPECS = [
    {
        "label": "Static OLS M4 full",
        "source": "peer_baseline_gap5_model_predictions.parquet",
        "path": REGRESSION_DIR / "peer_baseline_gap5_model_predictions.parquet",
        "model": "M4_rates_equity_vix_peer_raw",
        "peer_variant": "raw",
        "residual_col": "residual_return",
        "target_col": "final_vwap_return",
        "fitted_col": "fitted_return",
        "residual_family": "static_full_peer",
    },
    {
        "label": "Static OLS M4 no-sector",
        "source": "peer_baseline_gap5_model_predictions.parquet",
        "path": REGRESSION_DIR / "peer_baseline_gap5_model_predictions.parquet",
        "model": "M4d_rates_equity_vix_peer_same_other_raw",
        "peer_variant": "raw",
        "residual_col": "residual_return",
        "target_col": "final_vwap_return",
        "fitted_col": "fitted_return",
        "residual_family": "static_no_sector_ablation",
    },
    {
        "label": "Expanding OLS M4 full",
        "source": "rolling_gap5_model_predictions.parquet",
        "path": REGRESSION_DIR / "rolling_gap5_model_predictions.parquet",
        "model": "Expanding_OLS_M4",
        "residual_col": "residual",
        "target_col": "target_value",
        "fitted_col": "fitted_value",
        "residual_family": "expanding_full_peer",
    },
    {
        "label": "DLM M4 full, issuer-maturity beta",
        "source": "dynamic_state_space_extended_gap5_model_predictions.parquet",
        "path": REGRESSION_DIR / "dynamic_state_space_extended_gap5_model_predictions.parquet",
        "model": "DLM_Kalman_M4_expandingFE_issuer_maturity_beta",
        "residual_col": "residual",
        "target_col": "target_value",
        "fitted_col": "fitted_value",
        "residual_family": "dlm_full_peer",
    },
    {
        "label": "Expanding OLS M4 no-sector",
        "source": "dynamic_state_space_extended_no_sector_gap5_model_predictions.parquet",
        "path": REGRESSION_DIR / "dynamic_state_space_extended_no_sector_gap5_model_predictions.parquet",
        "model": "Expanding_OLS_M4NoSector",
        "residual_col": "residual",
        "target_col": "target_value",
        "fitted_col": "fitted_value",
        "residual_family": "expanding_no_sector",
    },
    {
        "label": "DLM M4 no-sector, issuer-maturity beta",
        "source": "dynamic_state_space_extended_no_sector_gap5_model_predictions.parquet",
        "path": REGRESSION_DIR / "dynamic_state_space_extended_no_sector_gap5_model_predictions.parquet",
        "model": "DLM_Kalman_M4NoSector_expandingFE_issuer_maturity_beta",
        "residual_col": "residual",
        "target_col": "target_value",
        "fitted_col": "fitted_value",
        "residual_family": "dlm_no_sector",
    },
]

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

def read_parquet_required(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            "Run the upstream model scripts before residual_model_comparison.py."
        )
    return pd.read_parquet(path)


def load_metadata() -> pd.DataFrame:
    panel = read_parquet_required(PANEL_PATH, "M4 peer regression panel")
    panel = panel.copy()
    panel[DATE_COL] = pd.to_datetime(panel[DATE_COL], errors="coerce")
    keep = [c for c in ["_sample_index", GROUP_COL, DATE_COL, ISSUER_COL, MATURITY_BUCKET_COL] if c in panel.columns]
    meta = panel[keep].drop_duplicates().copy()
    return meta


def merge_metadata_if_needed(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    need_cols = [c for c in [ISSUER_COL, MATURITY_BUCKET_COL] if c not in out.columns]
    if not need_cols:
        return out
    if "_sample_index" in out.columns and "_sample_index" in meta.columns:
        add_cols = ["_sample_index"] + [c for c in [ISSUER_COL, MATURITY_BUCKET_COL] if c in meta.columns]
        out = out.merge(meta[add_cols].drop_duplicates("_sample_index"), on="_sample_index", how="left")
    else:
        add_cols = [GROUP_COL, DATE_COL] + [c for c in [ISSUER_COL, MATURITY_BUCKET_COL] if c in meta.columns]
        out = out.merge(meta[add_cols].drop_duplicates([GROUP_COL, DATE_COL]), on=[GROUP_COL, DATE_COL], how="left")

    return out


def standardise_prediction_block(spec: dict[str, Any], meta: pd.DataFrame) -> pd.DataFrame:
    raw = read_parquet_required(spec["path"], spec["source"])
    raw = raw.copy()

    if DATE_COL not in raw.columns:
        raise ValueError(f"{spec['source']} does not contain {DATE_COL}.")
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce")

    if "model" not in raw.columns:
        raise ValueError(f"{spec['source']} does not contain a model column.")

    df = raw.loc[raw["model"].eq(spec["model"])].copy()

    if "peer_variant" in spec and "peer_variant" in df.columns:
        df = df.loc[df["peer_variant"].eq(spec["peer_variant"])].copy()

    if df.empty:
        available = sorted(raw["model"].dropna().astype(str).unique())
        raise ValueError(
            f"Model {spec['model']} not found in {spec['source']}.\n"
            f"Available models include: {available[:30]}"
        )

    required = [GROUP_COL, DATE_COL, SPLIT_COL, spec["residual_col"], spec["target_col"], spec["fitted_col"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {spec['label']}: {missing}")

    out = df[[c for c in df.columns if c in {
        "_sample_index", GROUP_COL, DATE_COL, SPLIT_COL, ISSUER_COL, MATURITY_BUCKET_COL,
        spec["residual_col"], spec["target_col"], spec["fitted_col"],
        "own_m0_fitted_value", "own_m0_model", "evaluation_protocol"
    }]].copy()

    out = out.rename(
        columns={
            spec["residual_col"]: "residual",
            spec["target_col"]: "target_value",
            spec["fitted_col"]: "fitted_value",
        }
    )

    out["residual"] = pd.to_numeric(out["residual"], errors="coerce")
    out["target_value"] = pd.to_numeric(out["target_value"], errors="coerce")
    out["fitted_value"] = pd.to_numeric(out["fitted_value"], errors="coerce")
    out[GROUP_COL] = out[GROUP_COL].astype(str).str.strip()

    out = merge_metadata_if_needed(out, meta)
    out["residual_source"] = spec["label"]
    out["residual_family"] = spec["residual_family"]
    out["source_file"] = spec["source"]
    out["source_model"] = spec["model"]

    return out


def residual_autocorr_by_cusip(df: pd.DataFrame) -> float:
    values: list[float] = []

    for _, g in df.sort_values([GROUP_COL, DATE_COL]).groupby(GROUP_COL, sort=False):
        r = pd.to_numeric(g["residual"], errors="coerce").dropna()
        if len(r) < 3:
            continue
        if float(r.std(ddof=1)) == 0.0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ac = r.autocorr(lag=1)
        if pd.notna(ac) and np.isfinite(ac):
            values.append(float(ac))

    return float(np.mean(values)) if values else np.nan


def _cusip_rolling_zscore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    def past_mean(x: pd.Series) -> pd.Series:
        return x.shift(1).rolling(window=CUSIP_ROLLING_WINDOW_OBS, min_periods=MIN_CUSIP_HISTORY_OBS).mean()

    def past_std(x: pd.Series) -> pd.Series:
        return x.shift(1).rolling(window=CUSIP_ROLLING_WINDOW_OBS, min_periods=MIN_CUSIP_HISTORY_OBS).std(ddof=1)

    out["z_cusip_mean"] = out.groupby(GROUP_COL, group_keys=False)["residual"].apply(past_mean)
    out["z_cusip_std"] = out.groupby(GROUP_COL, group_keys=False)["residual"].apply(past_std)
    out["z_cusip"] = np.where(
        out["z_cusip_std"].gt(0),
        (out["residual"] - out["z_cusip_mean"]) / out["z_cusip_std"],
        np.nan,
    )
    return out


def _past_expanding_by_group_date(
    df: pd.DataFrame,
    group_cols: list[str],
    prefix: str,
    min_obs: int,
) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")

    if group_cols:
        for col in group_cols:
            if col not in out.columns:
                out[col] = "missing"
            out[col] = out[col].astype("object").where(out[col].notna(), "missing")
        agg_cols = group_cols + [DATE_COL]
    else:
        out["_global_group"] = "all"
        group_cols = ["_global_group"]
        agg_cols = group_cols + [DATE_COL]

    temp = out[agg_cols + ["residual"]].dropna(subset=[DATE_COL, "residual"]).copy()
    temp["_count"] = 1
    temp["_sum"] = temp["residual"]
    temp["_sumsq"] = temp["residual"] ** 2

    daily = (
        temp.groupby(agg_cols, dropna=False)
        .agg(_daily_count=("_count", "sum"), _daily_sum=("_sum", "sum"), _daily_sumsq=("_sumsq", "sum"))
        .reset_index()
        .sort_values(group_cols + [DATE_COL])
    )

    gb = daily.groupby(group_cols, dropna=False)
    daily["_past_count"] = gb["_daily_count"].cumsum().groupby([daily[c] for c in group_cols], dropna=False).shift(1)
    daily["_past_sum"] = gb["_daily_sum"].cumsum().groupby([daily[c] for c in group_cols], dropna=False).shift(1)
    daily["_past_sumsq"] = gb["_daily_sumsq"].cumsum().groupby([daily[c] for c in group_cols], dropna=False).shift(1)

    count = daily["_past_count"]
    mean = daily["_past_sum"] / count
    var = (daily["_past_sumsq"] - (daily["_past_sum"] ** 2) / count) / (count - 1)
    var = var.where(var > 0)

    daily[f"{prefix}_past_count"] = count
    daily[f"{prefix}_past_mean"] = mean
    daily[f"{prefix}_past_std"] = np.sqrt(var)

    stats = daily[agg_cols + [f"{prefix}_past_count", f"{prefix}_past_mean", f"{prefix}_past_std"]]
    out = out.merge(stats, on=agg_cols, how="left")
    out[f"{prefix}_z"] = np.where(
        out[f"{prefix}_past_count"].ge(min_obs) & out[f"{prefix}_past_std"].gt(0),
        (out["residual"] - out[f"{prefix}_past_mean"]) / out[f"{prefix}_past_std"],
        np.nan,
    )

    if "_global_group" in out.columns:
        out = out.drop(columns=["_global_group"])

    return out


def add_comparable_past_zscores(df: pd.DataFrame) -> pd.DataFrame:
    parts = []

    for source, g in df.groupby("residual_source", sort=False):
        h = g.copy().sort_values([DATE_COL, GROUP_COL]).reset_index(drop=True)
        h = _cusip_rolling_zscore(h)

        group_cols = [c for c in [ISSUER_COL, MATURITY_BUCKET_COL] if c in h.columns]
        if group_cols:
            h = _past_expanding_by_group_date(
                h,
                group_cols=group_cols,
                prefix="z_issuer_maturity",
                min_obs=MIN_GROUP_HISTORY_OBS,
            )
        else:
            h["z_issuer_maturity_z"] = np.nan
            h["z_issuer_maturity_past_count"] = np.nan

        h = _past_expanding_by_group_date(
            h,
            group_cols=[],
            prefix="z_global",
            min_obs=MIN_GLOBAL_HISTORY_OBS,
        )

        h["z_main"] = h["z_issuer_maturity_z"]
        h["z_source"] = np.where(h["z_main"].notna(), "issuer_maturity_past", pd.NA)

        use_cusip = h["z_main"].isna() & h["z_cusip"].notna()
        h.loc[use_cusip, "z_main"] = h.loc[use_cusip, "z_cusip"]
        h.loc[use_cusip, "z_source"] = "cusip_rolling"

        use_global = h["z_main"].isna() & h["z_global_z"].notna()
        h.loc[use_global, "z_main"] = h.loc[use_global, "z_global_z"]
        h.loc[use_global, "z_source"] = "global_past"

        h["abs_z_main"] = h["z_main"].abs()
        parts.append(h)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarise_residual_source(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for source, g_all in df.groupby("residual_source", sort=False):
        g = g_all.loc[g_all[SPLIT_COL].eq("test")].copy()
        if g.empty:
            continue

        r = pd.to_numeric(g["residual"], errors="coerce").dropna()
        abs_r = r.abs()
        z = pd.to_numeric(g.get("abs_z_main"), errors="coerce") if "abs_z_main" in g.columns else pd.Series(dtype=float)
        z_valid = z.dropna()

        rows.append(
            {
                "residual_source": source,
                "residual_family": g["residual_family"].iloc[0],
                "n_obs": int(len(r)),
                "mae": float(abs_r.mean()),
                "rmse": float(np.sqrt(np.mean(r ** 2))),
                "p95_abs_residual": float(abs_r.quantile(0.95)),
                "p99_abs_residual": float(abs_r.quantile(0.99)),
                "residual_lag1_autocorr_by_cusip": residual_autocorr_by_cusip(g),
                "n_z_available": int(len(z_valid)),
                "abs_z_ge_2_rate": float((z_valid >= 2.0).mean()) if len(z_valid) else np.nan,
                "abs_z_ge_3_rate": float((z_valid >= 3.0).mean()) if len(z_valid) else np.nan,
                "issuer_maturity_z_share": float(g["z_source"].eq("issuer_maturity_past").mean()) if "z_source" in g.columns else np.nan,
                "cusip_rolling_z_share": float(g["z_source"].eq("cusip_rolling").mean()) if "z_source" in g.columns else np.nan,
                "global_z_share": float(g["z_source"].eq("global_past").mean()) if "z_source" in g.columns else np.nan,
            }
        )

    return pd.DataFrame(rows)


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def write_latex_table(summary: pd.DataFrame) -> None:
    df = summary.copy()
    order = [spec["label"] for spec in MODEL_SPECS]
    df["_order"] = df["residual_source"].map({name: i for i, name in enumerate(order)})
    df = df.sort_values("_order")

    newline = r"\\"
    lines = []
    lines.append("% Auto-generated by residual_model_comparison.py")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Test-sample residual comparison across static, expanding and dynamic fair-value models}")
    lines.append(r"\label{tab:residual_model_comparison}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Residual source & MAE & RMSE & 95\% $|u|$ & 99\% $|u|$ & Lag-1 AC & $|z|\geq 2$ rate " + newline)
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        lines.append(
            f"{latex_escape(str(row['residual_source']))} & "
            f"{row['mae']:.6f} & "
            f"{row['rmse']:.6f} & "
            f"{row['p95_abs_residual']:.6f} & "
            f"{row['p99_abs_residual']:.6f} & "
            f"{row['residual_lag1_autocorr_by_cusip']:.3f} & "
            f"{100.0 * row['abs_z_ge_2_rate']:.2f}\\% " + newline
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append("}")
    lines.append(r"\end{table}")

    OUTPUT_LATEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_directories()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    meta = load_metadata()

    blocks = []
    for spec in MODEL_SPECS:
        print(f"Loading {spec['label']} from {spec['source']} ...")
        blocks.append(standardise_prediction_block(spec, meta))

    predictions = pd.concat(blocks, ignore_index=True)
    predictions = predictions.loc[predictions[SPLIT_COL].isin(["validation", "test"])].copy()
    predictions = predictions.dropna(subset=["residual", GROUP_COL, DATE_COL, SPLIT_COL])

    predictions = add_comparable_past_zscores(predictions)
    summary = summarise_residual_source(predictions)

    order = {spec["label"]: i for i, spec in enumerate(MODEL_SPECS)}
    summary["_order"] = summary["residual_source"].map(order)
    summary = summary.sort_values("_order").drop(columns=["_order"])

    summary.to_csv(OUTPUT_TABLE, index=False)
    write_latex_table(summary)

    manifest = {
        "script": "residual_model_comparison.py",
        "purpose": "Residual comparison across static M4, expanding OLS M4 and dynamic state-space models.",
        "inputs": [json_safe(spec["path"]) for spec in MODEL_SPECS] + [json_safe(PANEL_PATH)],
        "outputs": {
            "summary_csv": OUTPUT_TABLE,
            "latex_table": OUTPUT_LATEX,
            "manifest": OUTPUT_MANIFEST,
        },
        "z_score_note": (
            "Comparable |z| event rates are computed from past-only residual scaling "
            "using validation+test predictions. They are residual diagnostics, not the "
            "official M4 dislocation signal produced by dislocation_signal_engine_v3.py."
        ),
        "model_sources": MODEL_SPECS,
        "n_rows_used_validation_test": int(len(predictions)),
    }
    OUTPUT_MANIFEST.write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")

    print("\nRESIDUAL MODEL COMPARISON")
    print(summary)
    print(f"\nSaved CSV: {OUTPUT_TABLE}")
    print(f"Saved LaTeX: {OUTPUT_LATEX}")
    print(f"Saved manifest: {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
