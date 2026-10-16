# AGENTS.md

Guardrails for every session in this repo. Read before changing anything.

## What this is

Six finance tools on one shared library (`finlib/`). The engines in `finlib/models/` and the data providers in `finlib/providers/` were verified against real SEC filings and independent hand recomputation on 2026-10-10 (see `projects/comps/README.md` and each project README). They are the product; everything else is a view on them.

## Rules

1. **`finlib/models/` and `finlib/providers/` are frozen.** Do not change math, tag lists, LTM logic, scoring, or sign conventions. If you believe a number is wrong, write a failing test in `tests/` that proves it, then stop and ask.
2. **Run `FINLIB_OFFLINE=1 pytest -q` before and after any change.** All tests must pass (96 + 2 that need LibreOffice, which CI installs; the Filing Tone torch tests skip without the `[ml]` extra and run in `ml.yml`). On Windows PowerShell: `$env:FINLIB_OFFLINE="1"; pytest -q`.
3. **New calculations go in `finlib/models/` with a test**, never inline in a page, CLI, or notebook.
4. **Never fabricate a figure.** If live data is unavailable, say so and fall back to fixtures labeled "Sample data (synthetic)".
5. **No paid APIs, no database server, no auth, no new backend service** without asking. SQLite files and free APIs (SEC, yfinance) are fine.
6. **Excel conventions:** blue font = input, black = formula or computed; negatives in parentheses; `NM` where a ratio has no meaning; every pulled value carries its source.
7. **The app is a view.** Pages call finlib and the `build_*` / `load_all` / `payload_from_frame` builders in `projects/`; `tests/test_app.py` asserts each page's headline numbers equal a direct finlib call.
8. **Keep the CLIs working.** `make demo-*` targets are the regression check and the README demos.
9. Commit per logical unit with a clear message. Do not commit `.env`, `output/`, `dist/`, or `*.sqlite`.

## Environment

- Python 3.11+. `pip install -e ".[dev]"`; add `.[live]` for yfinance; `.[app]` for the Streamlit app (`streamlit run app/Home.py`).
- `.env` holds `SEC_IDENTITY` (required for any live SEC call; format `"Name email"`) and optionally `FINNHUB_KEY`.
- `FINLIB_OFFLINE=1` makes every provider read `tests/fixtures/` instead of the network.
- LibreOffice (`soffice`) is optional; two LBO tests that recalc Excel formulas skip without it.

## Where things are

| Path | Holds |
| --- | --- |
| `finlib/providers/edgar.py` | SEC companyfacts: LTM = FY + YTD − prior YTD; cash/debt as total-or-components |
| `finlib/providers/market.py` | Prices: Finnhub → yfinance → `--prices` manual |
| `finlib/providers/formd.py` | SEC Form D daily index + XML parser, 8 req/s cap |
| `finlib/models/ratios.py` | EV, multiples, margins, growth; `None` → NM |
| `finlib/models/variance.py` | Budget vs actual; F/U by account type; rollup on contribution |
| `finlib/models/lbo.py` | Operating build, senior-first sweep, IRR/MOIC, sensitivity |
| `finlib/models/sourcing.py` | Fund filter and 0–100 score |
| `finlib/models/kpi.py` | Margin, burn, runway, calendar-month growth |
| `finlib/models/dcf.py` | UFCF DCF, perpetuity / exit-multiple terminal value, reverse DCF (revenue scale, exit multiple) |
| `finlib/nlp/` | Filing Tone (PyTorch): PhraseBank loader, tokenizer, TextCNN, DistilBERT wrapper, training loop, hand-written metrics, 10-K section/sentence extraction. Torch is an optional `[ml]` extra; never import it under `app/` |
| `projects/filing_tone/` | CLI (`train`, `snapshot`, `results`), committed `tone_snapshot.json` + `results.json` the page reads, `README.md` results block written by code |
| `finlib/excel_writer/` | openpyxl writers; LBO workbook is live formulas |
| `projects/<name>/cli.py` | Typer CLI per tool; `README.md` per tool has the spec and resume line |
| `projects/pitch/` | Equity pitch: `aehr.yaml` assumptions + committed `aehr_snapshot.json` -> memo PDF in `docs/pitch/` |
| `projects/powerbi/` | Power BI export: star-schema CSVs in `data/` (committed) + build guide and DAX in `README.md` |
| `Dockerfile`, `compose.yaml`, `deploy/k8s/` | Container image and Kubernetes manifests for the same app; `deploy/README.md`. CI deploys to kind on every push |
| `app/` | Streamlit app: `Home.py`, `pages/`, `fp_ui.py` (formatting, mode switch, chart style). A view only; no math. |
| `tests/fixtures/` | Offline data: comps JSON (illustrative), Form D day (synthetic), sample CSVs |

## Design System
Always read DESIGN.md before making any visual or UI decisions.
All font choices, colors, spacing, and aesthetic direction are defined there.
Do not deviate without explicit user approval.
