from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from config_institutional import DIAGNOSTICS_DIR, FIGURES_DIR


SUMMARY_PATH = DIAGNOSTICS_DIR / "model_ready_gap_sensitivity_summary.csv"


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing gap-sensitivity summary: {SUMMARY_PATH}"
        )

    summary = (
        pd.read_csv(SUMMARY_PATH)
        .sort_values("gap_threshold_bd")
        .reset_index(drop=True)
    )

    required = {
        "gap_threshold_bd",
        "rows",
        "p99_abs_vwap_return",
    }
    missing = required.difference(summary.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {SUMMARY_PATH.name}: {sorted(missing)}"
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Sample coverage
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        summary["gap_threshold_bd"],
        summary["rows"],
        marker="o",
    )
    ax.set_title("Sample coverage by trading-gap threshold")
    ax.set_xlabel("Maximum business-day gap")
    ax.set_ylabel("Number of observations")
    ax.set_xticks(summary["gap_threshold_bd"])
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.tight_layout()

    coverage_path = FIGURES_DIR / "gap_sensitivity_n_obs.png"
    fig.savefig(coverage_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Return-tail magnitude
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        summary["gap_threshold_bd"],
        summary["p99_abs_vwap_return"],
        marker="o",
    )
    ax.set_title("Return tail magnitude by trading-gap threshold")
    ax.set_xlabel("Maximum business-day gap")
    ax.set_ylabel("99th percentile absolute return")
    ax.set_xticks(summary["gap_threshold_bd"])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()

    tail_path = FIGURES_DIR / "gap_sensitivity_p99_abs_return.png"
    fig.savefig(tail_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {coverage_path}")
    print(f"Saved: {tail_path}")


if __name__ == "__main__":
    main()