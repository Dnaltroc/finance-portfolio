# Comps Builder

Give it a ticker list, get a formatted comparable-company workbook: EV/Revenue, EV/EBITDA, P/E, margins, growth, with median and mean rows and a source for every input.

## Resume line

> Built a Python comps builder that pulls LTM financials from SEC XBRL companyfacts and writes a sourced Excel comps table; tied every AAPL, MSFT and GOOGL input to the companies' earnings releases.

```bash
# Offline demo on fixtures (no keys)
FINLIB_OFFLINE=1 python -m projects.comps run --tickers AAPL,MSFT,GOOGL --peerset demo

# Live: needs SEC_IDENTITY in .env (free, no key). Prices come from Finnhub if FINNHUB_KEY is
# set, else yfinance, else pass them by hand. Shares come from the SEC filing.
python -m projects.comps run --tickers AAPL,MSFT,GOOGL,META,AMZN --peerset megacap
python -m projects.comps run --tickers AAPL,MSFT --prices "AAPL=336.64,MSFT=535.07"

# Quick terminal view
FINLIB_OFFLINE=1 python -m projects.comps show --tickers AAPL,MSFT,GOOGL
```

Output: `output/comps_<peerset>_<date>.xlsx` with three sheets.

| Sheet | Contents |
| --- | --- |
| Comps | One row per company; multiples formatted as `x`, margins as `%`, `NM` on negative bases; Median and Mean rows; color scale on EV/EBITDA and P/E |
| Inputs | Every pulled value (blue) with its basis (LTM period or spot) and source (XBRL tag, API, or fixture) |
| Notes | Methodology and the exact quarters used per ticker |

## How the numbers are built (live mode, verified on real filings)

Data comes straight from the SEC's [companyfacts API](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json), the same facts the company tagged in its 10-K and 10-Q.

| Number | How it is built | What it represents |
| --- | --- | --- |
| LTM revenue, COGS, operating income, D&A, net income | If the latest period is a 10-K: the 12-month value. If a 10-Q: **prior fiscal year + year-to-date this year − year-to-date last year**. Companies do not tag Q4 as a 3-month value, so summing four quarters fails for most tickers; this method works for all. | Trailing twelve months of GAAP results, as reported, not adjusted |
| Prior LTM | Same method one year earlier | Basis for revenue growth |
| Staleness flag | `stale LTM` when the LTM ends more than 200 days before the company's newest tagged fact (the revenue tag fell behind) | A warning in the Flags column, never a silent old number |
| Cash | `CashCashEquivalentsAndShortTermInvestments` if tagged, else cash + current marketable securities | Cash and short-term investments at the LTM balance-sheet date |
| Total debt | Long-term debt + current portion + commercial paper + short-term borrowings, whichever are tagged. **Zero** when the balance sheet at the LTM end tags no borrowing line at all (a debt-free issuer); still missing if any other borrowing tag (convertible notes, credit line) is there | Interest-bearing borrowings. No leases, preferred or minority interest |
| Shares | `dei:EntityCommonStockSharesOutstanding` (cover page), else `us-gaap:CommonStockSharesOutstanding` for multi-class issuers such as Alphabet | Basic shares outstanding from the filing, not diluted |
| Price | Finnhub, else yfinance, else `--prices` | Spot close on the run date |
| Market cap | price × shares | |
| EV | market cap + total debt − cash | Enterprise value, street-simple |
| EBITDA | operating income + D&A | D&A total tag when present, else depreciation + amortization of intangibles (Microsoft, Alphabet). The Inputs sheet shows which. |
| EV/Revenue, EV/EBITDA | EV ÷ LTM | NM on a non-positive base |
| P/E | market cap ÷ LTM GAAP net income | Includes non-operating items (Alphabet's 2026 net income carries large investment gains, so its P/E reads low) |
| Gross margin, EBITDA margin | ÷ LTM revenue | |
| Growth | LTM ÷ prior LTM − 1 | |

Every LTM cell on the Inputs sheet carries the arithmetic that produced it, e.g. `FY 2024-09-29..2025-09-27 416,161,000,000 + YTD 2025-09-28..2026-06-27 364,357,000,000 - prior YTD ... 313,695,000,000`, so a reviewer can tie each number to the filing.

**Verified Oct 10, 2026:** for AAPL, MSFT and GOOGL every component (FY, YTD, prior YTD, cash, debt) matched the figures in the companies' own earnings releases, and share counts matched the market feed to the share.

## Known limits

- Stale tags are ignored: a balance-sheet tag is used only if dated within 45 days of the LTM end (Alphabet's `DebtCurrent` last appeared in 2018 and is correctly skipped).
- Fiscal year-ends are not aligned across a peer set; the Notes sheet shows each ticker's LTM window.
- A company that reports D&A only inside a combined "depreciation, amortization and other" line will show the split components instead; the difference is small (Microsoft: 39.0B vs 38.5B).
- The Finnhub price path is written to its documented API but was exercised only with yfinance in testing.
- Fixture values in `tests/fixtures/comps/` are illustrative, not sourced from filings.

## Tests

```bash
FINLIB_OFFLINE=1 pytest tests/test_ratios.py tests/test_comps_workbook.py
```

Covers: LTM math ties to quarters, EV and multiple math ties to inputs, negative-EBITDA and missing-concept paths produce `NM` and flags, median/mean skip `NM`, workbook structure and source citations.
