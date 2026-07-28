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
be redistributed. The repository contains therefore no raw, row-level or reusable processed datasets
derived from restricted WRDS files. It contains only source code,
documentation and aggregate figures reproduced in the dissertation.

Authorised users must supply their own local copies of the required inputs.

## Repository structure

```text
src/       Python source code
data/      Description of the required input data
outputs/   Description of the generated model outputs
figures/   Figures generated for the dissertation
```
## Environment

The code was developed and tested with Python 3.10.2.

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Execution

No wrapper scripts are included. Run the Python files manually from the
`src/` directory in the order documented in
[`src/README.md`](src/README.md).

Project-relative paths, input filenames and output directories are defined in
`src/config_institutional.py`.

## Reproducibility

The repository contains the final methodology-hardened source code used for the
submitted dissertation, including the training-frozen sample construction,
point-in-time peer factors, static and dynamic fair-value models, direct VWAP
convergence diagnostics and the final post-run audit.

The reported tables and figures are generated from saved model outputs rather
than edited manually. Full numerical reproduction requires authorised access
to the restricted TRACE, TRACE Master File and CRSP inputs described above.

## Use of generative AI

I acknowledge the use of ChatGPT, specifically GPT-5.5 and GPT-5.6 Sol
(OpenAI, https://chatgpt.com), to discuss technical concepts, review Python
and LaTeX, and improve clarity. All suggestions were checked independently.
The submitted work, including the final text, code, modelling choices,
analysis and interpretation, is my own.

## Author

Marco Folonaro  
MSc Machine Learning and Data Science  
Imperial College London
