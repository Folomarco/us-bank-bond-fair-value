from pathlib import Path

# PROJECT STRUCTURE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
CODE_DIR = SRC_DIR
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"


# RAW DATA

TRACE_RAW_PATTERN = "us_banks_[0-9][0-9].zip"
TRACE_MASTER_ZIP = DATA_DIR / "liquid_us_banks_bonds.zip"
EQUITY_RAW_ZIP = DATA_DIR / "us_banks_stocks.zip"
TRACE_HOLIDAY_PATH = REFERENCE_DIR / "market_holidays_trace_otc_2016_2025.csv"
NYSE_HOLIDAY_PATH = REFERENCE_DIR / "market_holidays_nyse_2016_2025.csv"


# PROCESSED DATA DIRECTORIES

TRACE_PARQUET_DIR = PROCESSED_DIR / "trace_parquet"
TRACE_CLEANED_TRADE_DIR = PROCESSED_DIR / "trace_cleaned_trades"
BOND_DAY_DIR = PROCESSED_DIR / "bond_day"
DRIVERS_DIR = PROCESSED_DIR / "drivers"
EQUITY_DIR = PROCESSED_DIR / "equity"
DIAGNOSTICS_DIR = PROCESSED_DIR / "diagnostics"
TRACE_CLEANING_DIAG_DIR = DIAGNOSTICS_DIR / "trace_cleaning"
REGRESSION_DIR = PROCESSED_DIR / "regression"


# GAP SENSITIVITY SETTINGS

EQUITY_ASOF_TOLERANCE_DAYS = 3
EQUITY_ASOF_TOLERANCES_DAYS = [0, 1, 3, 5, 10]
MODEL_READY_MAX_BUSINESS_GAP = 5
GAP_THRESHOLDS = [1, 3, 5, 10]
MAX_SENSITIVITY_GAP = max(GAP_THRESHOLDS)


# MAIN PROCESSED FILES

TRACE_FINAL_BASELINE_PANEL_PATH = BOND_DAY_DIR / "trace_banks_final_baseline_panel.parquet"
TRACE_MODEL_READY_PATH = BOND_DAY_DIR / "trace_banks_final_model_ready_gap_sensitivity.parquet"
TRACE_MODEL_READY_GAP5_PATH = BOND_DAY_DIR / "trace_banks_final_model_ready_gap5.parquet"
TRACE_MODEL_READY_DIRTY_PATH = (
    BOND_DAY_DIR / "trace_banks_final_model_ready_gap_sensitivity_dirty.parquet"
)
TRACE_MODEL_READY_GAP5_DIRTY_PATH = (
    BOND_DAY_DIR / "trace_banks_final_model_ready_gap5_dirty.parquet"
)
FRED_PATH = DRIVERS_DIR / "fred_baseline_driver_levels_and_changes.parquet"
EQUITY_PATH = EQUITY_DIR / "crsp_us_banks_equity_daily.parquet"
REGRESSION_PANEL_PATH = REGRESSION_DIR / "regression_panel_gap_sensitivity.parquet"
REGRESSION_PANEL_GAP5_PATH = REGRESSION_DIR / "regression_panel_gap5.parquet"
FINAL_PANEL_INTEGRITY_REPORT_PATH = DIAGNOSTICS_DIR / "final_panel_integrity_report.csv"
REGRESSION_PANEL_INTEGRITY_REPORT_PATH = REGRESSION_DIR / "regression_panel_integrity_report.csv"


# PROJECT OUTPUTS

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUN_MANIFEST_PATH = OUTPUTS_DIR / "run_manifest.json"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"


# HELPERS

def ensure_directories() -> None:
    for path in [
        REFERENCE_DIR,
        PROCESSED_DIR,
        TRACE_PARQUET_DIR,
        TRACE_CLEANED_TRADE_DIR,
        BOND_DAY_DIR,
        DRIVERS_DIR,
        EQUITY_DIR,
        DIAGNOSTICS_DIR,
        TRACE_CLEANING_DIAG_DIR,
        REGRESSION_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_DIR:", DATA_DIR)
    print("PROCESSED_DIR:", PROCESSED_DIR)
    print("TRACE_MODEL_READY_PATH:", TRACE_MODEL_READY_PATH)
    print("FRED_PATH:", FRED_PATH)
    print("EQUITY_PATH:", EQUITY_PATH)
    print("REGRESSION_PANEL_PATH:", REGRESSION_PANEL_PATH)