# Data inputs

This repository does not redistribute restricted WRDS data. Full numerical reproduction requires authorised access to the TRACE, TRACE Master File and CRSP inputs used in the dissertation. Restricted inputs must be placed locally in this `data/` directory before running the pipeline.

## TRACE transaction data

The dissertation uses TRACE corporate-bond transaction extracts covering
2016–2025 for bonds issued by Bank of America, Goldman Sachs, JPMorgan Chase,
Morgan Stanley and Wells Fargo. The pipeline searches for annual TRACE ZIP archives matching:

```text
us_banks_[0-9][0-9].zip
```

For the 2016–2025 analysis period, the corresponding filenames are:

```text
us_banks_16.zip
us_banks_17.zip
us_banks_18.zip
us_banks_19.zip
us_banks_20.zip
us_banks_21.zip
us_banks_22.zip
us_banks_23.zip
us_banks_24.zip
us_banks_25.zip
```

Each ZIP archive must contain at least one CSV or TXT export. The TRACE cleaning code requires the transaction extract to contain a CUSIP identifier, execution date, reported price and reported volume. The code accepts the WRDS field names used in the original extracts and performs additional validation of lifecycle, correction and transaction-quality fields when constructing the cleaned samples. The exact input filename pattern and all project-relative paths are defined
in `src/config_institutional.py`.

## TRACE Master File

The TRACE Master File extract must be supplied as:

```text
liquid_us_banks_bonds.zip
```

The ZIP archive must contain at least one CSV or TXT export. The point-in-time Master File merge requires `cusip_id`, `stdt` and `enddt`, together with the bond-characteristic fields used by the analysis, including coupon, maturity and security-type information. The code uses the validity interval defined by `stdt` and `enddt` to ensure that bond characteristics are matched using information active on the corresponding bond date.

## CRSP issuer-equity data

The CRSP issuer-equity extract must be supplied as:

```text
us_banks_stocks.zip
```

The ZIP archive must contain at least one CSV or TXT export. The analysis expects observations for the following issuer tickers:

```text
BAC
GS
JPM
MS
WFC
```

The minimum required CRSP information is:

- date;
- ticker;
- price;
- total return.

The code accepts the alternative WRDS/CRSP column names explicitly listed in `src/crsp_equity_data.py`. The empirical analysis uses observations between 2016-01-01 and 2025-12-31.

## FRED market data

No local FRED input file is required. Public market series are downloaded
directly by `src/fred_data.py`. The series used by the pipeline are:

```text
DGS2
DGS5
DGS10
DGS30
VIXCLS
SP500
DAAA
DBAA
BAA10Y
```

The FRED download period is 2016-01-01 to 2025-12-31.

## Market calendars

The TRACE and NYSE market-holiday files required by the pipeline are generated
locally by:

```text
src/create_market_holiday_files.py
```

No market-calendar input files need to be supplied manually.

## Data restrictions

TRACE, TRACE Master File and CRSP data were accessed through WRDS and cannot
be redistributed. Proprietary data, processed extracts and model-ready panels
derived from the restricted inputs are therefore not included in this
repository.

No WRDS credentials or restricted data are stored in the repository.
Authorised users must provide their own copies of the required inputs.
