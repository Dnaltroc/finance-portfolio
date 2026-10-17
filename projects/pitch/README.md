# Equity Pitch: Aehr Test Systems (AEHR)

A two-page sell-side style memo on Aehr Test Systems, a small-cap semiconductor test company. Every figure in it is computed from a dated data snapshot by this repo's engines. Nothing in it is typed by hand.

**[Read the memo (PDF)](../../docs/pitch/aehr_memo.pdf)**

## Resume line

> Wrote a two-page equity research memo on Aehr Test Systems (AEHR) with a reverse DCF showing the share price required a revenue path 6.4 times guidance-based forecasts; built the DCF engine, comps, and memo generator in Python from SEC filings.

## The call

| | |
| --- | --- |
| Rating | Sell |
| Price (Oct 9, 2026) | $84.64 |
| 12-month target | $52 (peer median EV / revenue on FY28 revenue), 39% downside |
| DCF value, base / bull / bear | $16.77 / $28.63 / $8.74 |
| What the price implies | A revenue path 6.4x our base case (about $1.8B by FY31), or 70x FY36 EBITDA |

Aehr guided fiscal 2027 revenue up about 180% as AI processors move burn-in to the wafer. The thesis is not that the business is weak. It is that the price already pays for several more wins of the same size.

## How it is built

| Step | Where | What it does |
| --- | --- | --- |
| Snapshot | `python -m projects.pitch snapshot` | Pulls AEHR and six peers through the comps engine (SEC companyfacts + price), six years of AEHR 10-K history, peer betas, and the 10-year Treasury yield. Writes `aehr_snapshot.json`, which is committed. |
| Assumptions | `aehr.yaml` | Every judgment call with its reason: scenarios, margins, working capital, equity risk premium. Memo prose lives here too, with `{placeholders}` for numbers. |
| Engine | `finlib/models/dcf.py` | UFCF build, perpetuity and exit-multiple terminal values, mid-year convention, sensitivity grids, and two reverse-DCF solves: the revenue scale and the exit multiple the price implies. |
| Memo | `python -m projects.pitch build` | Renders HTML and prints it to PDF with Playwright. |

```bash
python -m projects.pitch summary              # headline numbers from the committed snapshot
python -m projects.pitch build                # docs/pitch/aehr_memo.html and .pdf
SEC_IDENTITY="Name email" python -m projects.pitch snapshot   # refresh the data (live)
```

## Choices to be ready to defend

- **Discount rate.** CAPM with the peer median beta (1.57), not AEHR's own 3.5, which reflects a stock that went from $19 to $147 in a year rather than its business risk. 5.24% risk-free rate, 4.5% equity risk premium (an assumption), no debt: 12.3%. The grids span 10.3% to 15.3%.
- **Working capital.** 40% of new revenue, against Aehr's own 58% to 100% of revenue over FY23 to FY26. A higher figure lowers every value.
- **Margins.** The base case reaches a 25% EBIT margin by FY31, above the 20.6% FY23 peak, and the bull case reaches 30%. These are generous on purpose, so the call does not rest on a stingy forecast.
- **Why the reverse DCF solves for revenue scale, not one growth rate.** With working capital tied to growth, very high growth makes free cash flow negative, so value is not monotonic in a single growth rate. Scaling the whole revenue path is monotonic and easy to explain.
- **Peers with NM.** Onto and Camtek carry convertible notes that the comps engine does not sum. Their EV is left blank rather than understated, and they drop out of the medians, leaving four peers (median 8.8x EV / revenue). Teradyne repaid its revolver and is in the median at 14.0x.
- **What changed since the first draft.** The first draft had Teradyne as NM because the engine read a $0 borrowing line and an undrawn credit facility as debt it could not sum. Fixed with a test (`tests/test_edgar_live_shapes.py`); the target moved from $35 to $52. The DCF and reverse DCF did not change.
- **Signals to watch.** Q1 FY27 results on Nov 2; a pilot order from the second AI processor maker (the main risk to the call); share sales under the July 27 automatic shelf; insider selling (about $17M in August, no purchases); timing of the lead hyperscaler's second Sonoma device, now a calendar 2027 ramp.

## Live tracker

The memo is priced on one day. The **AEHR live** page in the app (and `python -m projects.pitch live` in a terminal) follows what has happened since: last price and day move, the move since the memo price and the move still needed to reach the target, intraday and one-year charts with the memo price and target drawn in, the latest headlines, and the SEC filings feed with insider filings (Forms 3, 4, 144) split from company filings. That covers most of the signals to watch above.

| Piece | Where |
| --- | --- |
| Feeds | `finlib/live/feeds.py`: Yahoo Finance prices via yfinance, Yahoo Finance and Google News RSS headlines (de-duplicated), SEC EDGAR submissions. Free, no key. |
| Math | `finlib/models/tracker.py`: day move, move since the memo, move to target. |
| Builder | `projects/pitch/tracker.py`: fetches each section on its own; a failed feed falls back to the synthetic fixture, labeled, with the error. |
| Page | `app/pages/7_AEHR_Live.py`: refreshes every 60 seconds. |

```bash
python -m projects.pitch live                      # live
FINLIB_OFFLINE=1 python -m projects.pitch live     # synthetic fixture (make demo-live)
```

## Tests

`tests/test_pitch.py` checks the WACC against CAPM done by hand and the base case against an independently built DCF. It also confirms that both reverse-DCF answers reproduce the share price, and that the memo prints the computed figures with no unfilled placeholders. `tests/test_dcf.py` checks the engine against hand calculations. `tests/test_live.py` checks the feed parsers on fixture RSS and SEC JSON, the tracker math by hand, and that a failed feed falls back to labeled sample data.

Student research, prepared for a portfolio. Not investment advice.
