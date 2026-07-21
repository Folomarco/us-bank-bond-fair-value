# Fair Value, Residual Convergence and Dislocation Signals in US Bank Corporate Bonds

This repository contains the Python code developed for the MSc dissertation
“Fair Value, Residual Convergence and Dislocation Signals in US Bank Corporate Bonds”
at Imperial College London.

## Project overview

The project studies whether interval-aligned market, issuer and peer-bond
variables can improve fair-value estimates for individual US bank corporate
bonds, and whether large model residuals are followed by subsequent convergence.

The empirical analysis covers fixed-rate bonds issued by Bank of America,
Goldman Sachs, JPMorgan Chase, Morgan Stanley and Wells Fargo over 2016–2025.

The modelling framework includes:

- bond fixed-effect fair-value models;
- leave-one-out peer-bond factors;
- Ridge, Lasso, Elastic Net and principal-component models;
- rolling and expanding fixed-effect estimation;
- dynamic linear models estimated with Kalman filtering;
- residual-dislocation classification and convergence diagnostics.

## Data

The analysis uses:

- TRACE corporate-bond transactions;
- TRACE Master File bond characteristics;
- CRSP issuer-equity data;
- FRED Treasury, volatility and credit-market series.

TRACE, TRACE Master File and CRSP data were accessed through WRDS and cannot
be redistributed. This repository therefore contains no proprietary data or
processed data derived from restricted WRDS files.

Authorised users must supply their own local copies of the required inputs.

## Repository structure

```text
src/       Python source code
data/      Description of the required input data
outputs/   Description of the generated model outputs
figures/   Figures generated for the dissertation
