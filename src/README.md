# Source code

The scripts should be run manually from the repository root in the order shown
below. Restricted TRACE, TRACE Master File and CRSP inputs must be supplied
locally in `data/`.

## Data preparation

1. `create_market_holiday_files.py`
2. `trace_bond_data_institutional_cleaner.py`
3. `refresh_report_gap_figures.py`
4. `crsp_equity_data.py`
5. `fred_data.py`
6. `build_regression_panel_institutional.py`

The cleaning stage resolves TRACE cancellations and corrections, constructs
bond-date VWAP prices, applies the training-frozen activity rule and performs
the point-in-time Master File merge. The regression-panel stage aligns market
and issuer variables to each bond-specific return interval.

## Fair-value models

7. `peer_factor_models.py`
8. `target_robustness_models.py`
9. `regularized_fair_value_models.py`
10. `rolling_fair_value_models.py`
11. `dynamic_state_space_models.py`
12. `dynamic_state_space_models_no_sector.py`

The peer script constructs point-in-time leave-one-out peer returns and
estimates the static M0--M5 models. The two dynamic scripts estimate global,
issuer and issuer--maturity state-space models for the full-peer and no-sector
specifications.

## Residual and convergence analysis

13. `residual_model_comparison.py`
14. `dislocation_signal_engine.py`
15. `methodology_hardening_audit.py`

The dislocation engine constructs past-only residual scores, applies the M5
price-quality classification, estimates fair-value intervals and calculates
fixed-horizon, skip-first-mark and event-driven convergence payoffs from direct
VWAP entry and exit prices. Event-driven sensitivities use absolute z-score
thresholds of 2.0, 2.5 and 3.0.

The final audit checks the methodological conditions used in the submitted
analysis and should complete without failed hard checks.

## Supporting modules

- `config_institutional.py`: project paths and shared settings
- `calendar_utils.py`: TRACE and NYSE calendar functions
- `bond_cashflow_utils.py`: accrued-interest and dirty-price utilities
- `panel_integrity_audit.py`: panel and point-in-time integrity checks

Full numerical reproduction requires authorised access to the restricted WRDS
inputs. Generated tables and model-ready datasets are not distributed.
