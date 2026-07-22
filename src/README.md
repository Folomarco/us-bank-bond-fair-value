# Source code

The scripts are organised in the approximate execution order shown below.

## Data preparation

1. `create_market_holiday_files.py`
   - creates the TRACE and NYSE holiday calendars.

2. `trace_bond_data_institutional_cleaner.py`
   - cleans TRACE transaction reports;
   - constructs bond-day prices and returns;
   - applies the point-in-time TRACE Master File filters.

3. `fred_data.py`
   - downloads and prepares FRED market variables.

4. `crsp_equity_data.py`
   - prepares issuer-equity variables from the authorised CRSP extract.

5. `build_regression_panel_institutional.py`
   - aligns market and issuer variables to bond-return intervals;
   - builds the model-ready regression panels.

## Fair-value models

6. `peer_factor_models.py`
   - constructs leave-one-out peer returns;
   - estimates the static M0--M5 model sequence and peer ablations.

7. `target_robustness_models.py`
   - estimates the alternative return-target specifications.

8. `regularized_fair_value_models.py`
   - estimates Ridge, Lasso, Elastic Net and PCA robustness models.

9. `rolling_fair_value_models.py`
   - estimates locked, rolling and expanding fixed-effect models.

10. `dynamic_state_space_fair_value_models_kantas.py`
    - estimates the initial dynamic linear model specifications.

11. `dynamic_state_space_extended_models_kantas.py`
    - estimates the full-peer global, issuer and issuer--maturity state models.

12. `dynamic_state_space_extended_models_no_sector_kantas.py`
    - estimates the corresponding no-sector state-space robustness models.

## Residual analysis

13. `residual_model_comparison.py`
    - compares residual diagnostics across static, expanding and dynamic models.

14. `dislocation_signal_engine_v3.py`
    - constructs residual dislocation scores;
    - applies the price-quality classification;
    - produces uncertainty intervals and convergence diagnostics.

## Supporting modules

- `config_institutional.py`: project paths and shared settings;
- `calendar_utils.py`: market-calendar functions;
- `bond_cashflow_utils.py`: dirty-price and accrued-interest functions;
- `panel_integrity_audit.py`: panel integrity checks.

The scripts require authorised local data in the repository's `data/`
directory. They should be run from the repository root or with `src/` included
in the Python path.
