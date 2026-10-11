# Phase 0 review

Reviewed 2026-10-11 on Windows 11, Python 3.14.3, in a local `.venv`. Nothing under `finlib/` or `projects/` has been changed.

## Baseline

| Check | Result |
| --- | --- |
| `FINLIB_OFFLINE=1 pytest -q` | **42 passed, 2 skipped**. The two skips are the LBO Excel-recalc tests, which need LibreOffice (`soffice`). It isn't installed on this machine, so "Excel IRR = Python IRR" is unverified here. Every other test passes. |
| `make` | Not installed on Windows. I ran each target's command directly (same env vars, same args). |
| `pip install -e ".[dev]"` | **Fails.** setuptools refuses flat-layout auto-discovery because `finlib/`, `projects/` and `tests/` are all top-level packages. Tests still pass because pytest sets `pythonpath = ["."]`. Fix: add `[tool.setuptools.packages.find] include = ["finlib*", "projects*"]`. This is needed anyway for Streamlit Cloud. **The same bug likely breaks `.github/workflows/sourcing.yml`**, which runs `pip install -e ".[dev]"` as its first step. |
| CI coverage | The only workflow is the Form D digest. It runs `tests/test_sourcing.py` alone. No CI runs the full suite, and none installs LibreOffice. |
| All five demos | Ran clean. Outputs are described below. |

## What each tool does and produces

| Tool | Engine | Demo output | Headline (sample data) |
| --- | --- | --- | --- |
| Comps | `providers/edgar.py` (companyfacts, LTM = FY + YTD − prior YTD), `providers/market.py`, `models/ratios.py` | `comps_demo_<date>.xlsx`. **Comps**: 3 rows plus Median and Mean, color scale on EV/EBITDA and P/E. **Inputs**: every value in blue with basis and source. **Notes**: methodology and periods. | Median EV/EBITDA 20.8x, P/E 31.4x |
| Variance | `models/variance.py` | `variance_2026-09.xlsx`. **Summary**: Revenue / Expenses / Net plus contribution by department. **Flags**: 4 rows with an empty Driver column. **Detail**: 17 lines, filterable. **Inputs**: both files as loaded. Plus `commentary_2026-09.md` with 4 bullets ending "driven by ___". | Net $(362)K vs $(172)K budget, $(190)K U |
| LBO | `models/lbo.py`, `excel_writer/lbo_workbook.py` | `lbo_Gulf_Coast_Services_Co.xlsx` with 5 sheets. Every computed cell is a formula (spot-checked Operating, Debt, Returns). The scenario dropdown drives `INDEX/MATCH`. **Sensitivity** holds static values from Python. | Base 2.55x / 20.6%; Upside 3.96x / 31.7%; Downside 1.02x / 0.4% |
| Sourcing | `providers/formd.py`, `models/sourcing.py`, `store/sourcing_db.py` | `digest_2026-10-09.md`: ranked markdown table with SEC links. It also upserts `sourcing.sqlite` in the working directory. | 36 filings, 25 kept, 11 dropped |
| Dashboard | `models/kpi.py` | `dist/index.html` (31 KB, self-contained, Plotly from cdnjs) | 8 companies, Sep-26 revenue $8.1M, +42% YoY, 2 critical / 1 warning |

I hand-checked the LBO base case. Equity = 720 + 14.4 fees + 10 min cash − 440 debt = 304.4. Exit equity 777.1 / 304.4 = 2.553x, and 2.553^(1/5) − 1 = 20.6%. Both tie.

## What I'd distrust

None of these is a wrong number under the engine's own stated rules. Items 1–3 can put a misleading figure on screen in the app. Items 4–8 are documentation mismatches or limits worth labeling.

1. **LBO accepts debt larger than the purchase price.** With 10x leverage on a 9x entry, `sponsor_equity` = −55.6 and the engine returns **MOIC −5.13x** (IRR is NaN). On a slider-driven page this is one drag away. Nothing validates `sponsor_equity > 0`.
2. **LBO cash can go deeply negative with no revolver.** Mandatory amortization is paid even when there's no cash. In a stressed case, cash runs 10 → (34) → … → (177). The engine flags `cash_shortfall_years`, which the CLI prints, but exit net debt still nets off negative cash. The red band the spec asks for covers this visually. The returns number in that case should carry a warning.
3. **Manual prices are silently ignored offline.** `MarketProvider.quote` checks `offline()` before overrides, so `AAPL=999` returns the fixture's 232.5. The page must disable manual prices in Offline mode, or say so.
4. **Comps Notes sheet describes the wrong method.** `excel_writer/tables.py` hardcodes "LTM = sum of the four most recent reported quarters". That's true for fixtures but false for live mode, which uses FY + YTD − prior YTD. The Inputs sheet carries the correct basis per value. The writer isn't in the frozen folders, but the prompt says not to rewrite it, so I want your OK before changing that one line.
5. **Portfolio YoY isn't same-store.** `portfolio_totals` compares this month's total revenue across all current companies against the total 12 months earlier. Adding a company with only 3 months of history moves YoY from 42.4% to 60.0%. It also finds the year-ago month by list position, not by calendar (unlike `compute_kpis`), and the same goes for `t3m_growth`. The sample data has every company in all 24 months, so the demo is unaffected. Uploaded data could break this.
6. **Form D amount score has a high floor.** It is log10(sold) / log10(ceiling), so the $250K minimum already earns 31.6 of 45 points, $1M earns 35.1 and $50M earns 45. As a result amount barely separates deals, and state and industry dominate the ranking. The weight sliders on the page will make this visible.
7. **Fund filter doesn't match its README.** The README says fund-shaped names are dropped "when the industry is blank or Other". The code applies strong patterns (`Fund`, `LP`, `SPV`…) regardless of industry, and weak ones (`Investors`, `Holdings`) only to LP and Trust entities. So "Summit Coast Investors LLC" (industry "Other") ranks #10 as an operating company.
8. **Fixtures are illustrative.** The comps fixtures aren't from filings (the README says so), and the Form D day is synthetic. Every offline view gets the "Sample data (synthetic)" label.

**Live paths I could not run here:** no `.env`, so SEC live comps and live Form D are untested on this machine. The Finnhub path is documented as never exercised.

## Plan, adjusted for what I found

The stack, pages and design rules are as in your prompt. These are the changes:

**Setup**
- Fix packaging (`packages.find`), add the `app` extra (`streamlit`, `plotly`), and add `requirements.txt` for Streamlit Community Cloud. Cloud installs from that file. Add `.venv/` to `.gitignore`.
- Read `SEC_IDENTITY` and `FINNHUB_KEY` from `st.secrets` or `.env` and set them as env vars before the providers are built. If they're missing, the page shows a message and falls back to fixtures.
- Toggle offline mode per page by setting `FINLIB_OFFLINE` before each provider call. `finlib.offline()` reads the env var on every call, so no engine change is needed.
- Create providers inside each run, never with `st.cache_resource`. `JsonCache` holds a `sqlite3` connection, which can't cross Streamlit's threads.

**No logic in pages: lift the CLI orchestration into importable functions**

The CLIs hold real logic the app must not copy:
- comps overrides shares with the SEC count
- variance filters to one period before `pnl_summary`, which would otherwise sum across all periods
- sourcing writes the database inside `run_day`

I'll split each into a pure `build_*()` that both the CLI and the page call, plus a `write_*()` that saves files. CLI output stays byte-for-byte the same. The existing `make demo-*` commands are the regression check, and I'll diff the outputs before and after.

**Additive functions in `finlib/models` (new code only, each with a test; existing functions unchanged)**
- `sourcing.funnel(rows)`: counts for filings → funds dropped → amendments dropped → below-minimum dropped → kept, from `drop_reason`.
- `variance.waterfall(detail)`: department `favorable_amt` steps from budget net to actual net. The test asserts the steps sum exactly to the Net variance.
- LBO tranche-level series already exist on `LBOResult`, so no new LBO math is needed.

**Per-page handling of the risks above**
- LBO: if sponsor equity ≤ 0, the returns tile reads "n/a: debt exceeds purchase price" in red and MOIC is not shown. If cash falls below the minimum, a warning on the tile names the years.
- Comps: hide the manual-price input in Offline mode.
- Sourcing: Offline mode limits the date picker to the fixture day, 2026-10-09. Live mode caches each day's fetch with `st.cache_data`. A full day is a few hundred requests at 8/s, so about 30–60 s behind a spinner. Weight sliders re-score from the cached filings. The page **does not write** `sourcing.sqlite`. It reads it if it exists, and the first-time flag is labeled "not seen in local database" (empty on Streamlit Cloud).
- Dashboard: same-store caveat in the YoY tile's tooltip. The static `dist/index.html` build stays as is.

**Testing**
- `tests/test_app.py` with `AppTest`: each page renders on sample data, the headline figures equal a direct finlib call, and downloads are non-empty.
- I'll run Ruff, and existing tests must still pass. I'll make one commit per page.
- Screenshots: Playwright with headless Chromium into `docs/screenshots/`.

## Decisions I need from you before "go"

1. **Variance commentary with drivers.** `commentary()` ignores the `driver` column. Option A (recommended): change `commentary()` to write the driver when one is filled and `___` when it's blank. CLI output is identical because the CLI's drivers are always blank. Add a test. This touches a frozen file, hence the ask. Option B: a new `commentary_with_drivers()` beside it, leaving the original untouched.
2. **LBO guard.** Should the model reject `sponsor_equity <= 0`? That needs a failing test plus a validator, which changes frozen math. Or should the page handle it alone? My recommendation is page-only, as above, since the engine's math is correct and the input is the problem.
3. **Comps Notes line (item 4).** Should I fix the one hardcoded methodology line so it states the basis actually used?
4. **LibreOffice and CI.** No CI runs the Excel IRR tie-out today, and it can't run here. Should I install LibreOffice locally (`winget install TheDocumentFoundation.LibreOffice`, about 350 MB)? Should I also add a `test.yml` workflow that runs the full suite plus `tests/test_app.py` on Ubuntu with LibreOffice installed via `apt`? I recommend the workflow. It's free on GitHub and proves the tie-out on every push.

Say "go", along with answers to 1–4, and I'll start Phase 1.

## Decisions taken (2026-10-11)

The owner said to proceed on the recommendations:
1. `commentary()` writes the analyst's driver when one is filled in; blank drivers still render `___`, so CLI output is unchanged. A failing test came first.
2. The LBO negative-equity guard lives on the page only. The engine math is untouched.
3. The comps Notes methodology line now states the basis actually used (fixture quarters vs SEC FY + YTD − prior YTD).
4. Added a `test.yml` workflow that runs the full suite with LibreOffice on Ubuntu. No local LibreOffice install.
