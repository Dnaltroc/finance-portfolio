# finance-portfolio

Five finance automation projects built on one shared library. Each project ends in a working tool and a one-command demo that runs on fixtures with no API keys.

| # | Project | Status | Run |
| --- | --- | --- | --- |
| 1 | [Comps Builder](projects/comps/README.md) — SEC XBRL + market data → Excel comps table | **Working** | `make demo-comps` |
| 2 | Variance Report Generator — actuals vs budget → Excel pack + commentary | Planned | `make run-variance` |
| 3 | LBO Returns Engine — Python LBO → live-formula Excel with scenarios | Planned | `make run-lbo` |
| 4 | Deal Sourcing Pipeline — daily SEC Form D → scored digest | Planned | `make run-sourcing` |
| 5 | Portfolio KPI Dashboard — monthly KPIs → hosted page | Planned | `make run-dashboard` |

## Setup

```bash
pip install -e ".[dev]"          # core + pytest/ruff
pip install -e ".[live]"         # edgartools + yfinance for live data
cp .env.example .env             # add SEC_IDENTITY and FINNHUB_KEY
make test                        # all tests run offline on fixtures
```

## Layout

```
finlib/
  providers/     edgar.py (SEC XBRL via edgartools), market.py (Finnhub → yfinance fallback)
  models/        ratios.py — pure valuation math, no I/O
  excel_writer/  styles.py, tables.py — house style, openpyxl
  store/         cache.py — SQLite JSON cache so demos never hit a rate limit twice
projects/        one package per project, each with cli.py and a README
tests/           pytest; fixtures in tests/fixtures/
```

Every provider honors `FINLIB_OFFLINE=1` and reads from `tests/fixtures/` instead of the network. That is how CI runs and how demos run without keys.

## Conventions

- Hardcoded inputs are blue in every workbook; computed cells are black.
- Multiples on a zero or negative base show `NM` and are excluded from medians.
- Every pulled number carries its source (XBRL tag, API, or fixture) on the Inputs sheet.
