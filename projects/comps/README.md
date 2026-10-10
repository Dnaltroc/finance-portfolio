# Comps Builder

Give it a ticker list, get a formatted comparable-company workbook: EV/Revenue, EV/EBITDA, P/E, margins, growth, with median and mean rows and a source for every input.

```bash
# Offline demo on fixtures (no keys)
FINLIB_OFFLINE=1 python -m projects.comps run --tickers AAPL,MSFT,GOOGL --peerset demo

# Live (needs .env with SEC_IDENTITY and FINNHUB_KEY, and pip install -e ".[live]")
python -m projects.comps run --tickers AAPL,MSFT,GOOGL,META,AMZN --peerset megacap

# Quick terminal view
FINLIB_OFFLINE=1 python -m projects.comps show --tickers AAPL,MSFT,GOOGL
```

Output: `output/comps_<peerset>_<date>.xlsx` with three sheets.

| Sheet | Contents |
| --- | --- |
| Comps | One row per company; multiples formatted as `x`, margins as `%`, `NM` on negative bases; Median and Mean rows; color scale on EV/EBITDA and P/E |
| Inputs | Every pulled value (blue) with its basis (LTM period or spot) and source (XBRL tag, API, or fixture) |
| Notes | Methodology and the exact quarters used per ticker |

## How the numbers are built

- **LTM** = sum of the four most recent quarterly values. Growth compares to the prior four quarters.
- **EBITDA** = operating income + D&A. If D&A is missing from XBRL, EBITDA is blank and flagged rather than silently using EBIT.
- **EV** = market cap + total debt − cash. Minority interest, preferred and leases are not included; add them to `XBRL_TAGS` in `finlib/providers/edgar.py` if a peer set needs them.
- **XBRL tag fallbacks:** each concept tries 2–3 tags in order (e.g. `RevenueFromContractWithCustomerExcludingAssessedTax` → `Revenues`). Companies tag differently; the tag actually used is written to the Inputs sheet.

## Known limits

- Live XBRL parsing depends on the `edgartools` DataFrame columns, which have changed between versions. `_quarterly_series` in `edgar.py` is defensive but should be validated against your installed version before a live demo.
- Fiscal year-end mismatches across a peer set are not aligned; the Notes sheet shows each ticker's LTM window so the reader can judge.
- Fixture values in `tests/fixtures/comps/` are illustrative, not sourced from filings.

## Tests

```bash
FINLIB_OFFLINE=1 pytest tests/test_ratios.py tests/test_comps_workbook.py
```

Covers: LTM math ties to quarters, EV and multiple math ties to inputs, negative-EBITDA and missing-concept paths produce `NM` and flags, median/mean skip `NM`, workbook structure and source citations.
