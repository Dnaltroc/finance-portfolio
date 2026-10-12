# Power BI Report: Variance, Portfolio KPIs, Peer Comps

A three-page Power BI report built on this repo's engines. Python does the finance math (`finlib`) and exports a star schema of CSV tables; Power BI does the data model, the DAX measures and the visuals.

## Resume line

> Built a three-page Power BI report (budget vs actual variance, portfolio KPIs, peer comps) on a star-schema model exported from a tested Python finance library; wrote the DAX measures for contribution variance, year-over-year growth and runway, and reconciled every card to the source engine.

## The data

```bash
python -m projects.powerbi export     # or: make export-powerbi
```

This writes `projects/powerbi/data/` and prints the check values the report must match. The CSVs are committed, and `tests/test_powerbi.py` fails if they drift from a fresh export.

| Table | Grain | Comes from |
| --- | --- | --- |
| `variance_fact` | period x department x account | `finlib.models.variance.compute_variance` on the sample GL (synthetic) |
| `dim_department`, `dim_account` | one row per department / account | same |
| `kpi_fact` | company x month, 24 months | `finlib.models.kpi.compute_kpis` on the sample portfolio (synthetic) |
| `dim_company` | one row per company | `latest_snapshot` (latest runway, status, trailing-3-month growth) |
| `calendar` | one row per month, no gaps | shared by both fact tables |
| `comps` | one row per ticker | the dated AEHR pitch snapshot (SEC filings, Yahoo prices, Oct 9 2026) |
| `aehr_price` | weekly close | same snapshot |

Variance and KPI data are synthetic and every row says so in `data_source`. Comps are real and carry `price_source`, `debt_source` and `snapshot_pulled_at`. Blank multiples are NM (not meaningful), not zero.

## Build it in Power BI Desktop

Power BI Desktop is free (Microsoft Store, Windows only).

### 1. Load the tables

1. **Home > Get data > Text/CSV**, pick `variance_fact.csv`, click **Transform Data** (not Load).
2. In Power Query, **Home > New Source > Text/CSV** for each of the other seven files.
3. Check the types on each query (click the icon left of each column header):
   - `month` columns: **Text**. `calendar[month_start]` and `aehr_price[date]`: **Date**.
   - Money, ratios and multiples: **Decimal Number**. `flag`, `is_revenue`: **True/False**.
4. **Close & Apply**.

Tip: if you move the folder, **Transform data > Data source settings > Change Source** updates the path.

### 2. Model view: relationships

Drag each fact column onto the dimension column. All are many-to-one, single direction, dimension filters fact.

| From (many) | To (one) |
| --- | --- |
| `variance_fact[month]` | `calendar[month]` |
| `variance_fact[department]` | `dim_department[department]` |
| `variance_fact[account]` | `dim_account[account]` |
| `kpi_fact[month]` | `calendar[month]` |
| `kpi_fact[company]` | `dim_company[company]` |

Delete any relationship Power BI auto-detected that is not in this table. `comps` and `aehr_price` stand alone.

In `calendar`, select `month_name` and set **Column tools > Sort by column > month_index**, so months sort in date order, not alphabetically.

### 3. Measures

Create a blank table to hold them: **Home > Enter data**, name it `_Measures`, load. Then **New measure** for each block below.

**Variance.** Revenue lines count positive and expense lines negative, the same "contribution" rule the Python rollup uses, so a department with both nets correctly.

```DAX
Actual = SUM ( variance_fact[actual] )
Budget = SUM ( variance_fact[budget] )

Contribution Actual =
SUMX ( variance_fact, IF ( variance_fact[is_revenue], variance_fact[actual], - variance_fact[actual] ) )

Contribution Budget =
SUMX ( variance_fact, IF ( variance_fact[is_revenue], variance_fact[budget], - variance_fact[budget] ) )

Favorable Variance = SUM ( variance_fact[favorable_amt] )

Favorable Variance % = DIVIDE ( [Favorable Variance], ABS ( [Contribution Budget] ) )

F/U = IF ( [Favorable Variance] > 0, "F", IF ( [Favorable Variance] < 0, "U", "-" ) )

Flagged Lines = CALCULATE ( COUNTROWS ( variance_fact ), variance_fact[flag] = TRUE () )
```

`Favorable Variance` is positive when the result is good for the business: revenue above budget, or costs below it. It always equals `Contribution Actual - Contribution Budget`.

**Portfolio KPIs.**

```DAX
Revenue = SUM ( kpi_fact[revenue] )
Gross Margin = DIVIDE ( SUM ( kpi_fact[gross_profit] ), [Revenue] )
Net Burn = SUM ( kpi_fact[net_burn] )
Cash = SUM ( kpi_fact[cash] )
Headcount = SUM ( kpi_fact[headcount] )

Runway (months) = DIVIDE ( [Cash], [Net Burn] )

Revenue YoY % =
VAR m = MAX ( 'calendar'[month_index] )
VAR prior = CALCULATE ( [Revenue], REMOVEFILTERS ( 'calendar' ), 'calendar'[month_index] = m - 12 )
RETURN DIVIDE ( [Revenue] - prior, prior )

Latest Month = CALCULATE ( MAX ( kpi_fact[month] ), REMOVEFILTERS ( kpi_fact ), REMOVEFILTERS ( 'calendar' ) )

Revenue (Latest Month) =
VAR m = [Latest Month]
RETURN CALCULATE ( [Revenue], REMOVEFILTERS ( 'calendar' ), 'calendar'[month] = m )

Cash (Latest Month) =
VAR m = [Latest Month]
RETURN CALCULATE ( [Cash], REMOVEFILTERS ( 'calendar' ), 'calendar'[month] = m )

Net Burn (Latest Month) =
VAR m = [Latest Month]
RETURN CALCULATE ( [Net Burn], REMOVEFILTERS ( 'calendar' ), 'calendar'[month] = m )

Companies Critical = CALCULATE ( COUNTROWS ( dim_company ), dim_company[runway_status] = "critical" )
```

`Revenue YoY %` steps back 12 rows of a gap-free calendar instead of using `DATEADD`, which needs a daily date table. `Runway` is blank when a company is profitable (no burn), matching the Python engine.

**Comps.** `MEDIAN` skips blanks, so NM peers drop out of the median on their own, as they do in the memo.

```DAX
Peer Median EV/Revenue = CALCULATE ( MEDIAN ( comps[ev_revenue] ), comps[role] = "Peer" )
Peer Median EV/EBITDA = CALCULATE ( MEDIAN ( comps[ev_ebitda] ), comps[role] = "Peer" )
AEHR EV/Revenue = CALCULATE ( MAX ( comps[ev_revenue] ), comps[ticker] = "AEHR" )
AEHR Premium to Peers = DIVIDE ( [AEHR EV/Revenue], [Peer Median EV/Revenue] ) - 1
```

Format: money measures as Currency with 0 decimals, `%` measures as Percentage with 1 decimal, multiples as Decimal with 1 decimal and the suffix `x` (custom format `0.0"x"`).

### 4. Pages

**Page 1: Budget vs Actual.**
- Slicer: `calendar[month_name]`, single select, set to Sep 2026.
- Cards: `Contribution Actual` (title "Actual net"), `Contribution Budget`, `Favorable Variance`, `Flagged Lines`.
- Waterfall: Category `dim_department[department]`, Y-axis `Favorable Variance`. Shows how each department moved net income versus budget.
- Matrix: rows `dim_department[department]` then `dim_account[account]`; values `Actual`, `Budget`, `Favorable Variance`, `Favorable Variance %`, `F/U`. Conditional formatting on `Favorable Variance`: font color by rules, red below 0, green above.
- Text box: "Sample data (synthetic). Flag rule: |variance| >= $25,000 or >= 10%."

**Page 2: Portfolio KPIs.**
- Cards: `Revenue (Latest Month)`, `Revenue YoY %` (with the slicer on Sep 2026), `Cash (Latest Month)`, `Net Burn (Latest Month)`, `Companies Critical`.
- Line chart: X `calendar[month_name]`, Y `Revenue`, Legend `dim_company[company]`.
- Table: `dim_company[company]`, `dim_company[latest_runway_months]`, `dim_company[runway_status]`, `dim_company[t3m_growth]`. Conditional formatting on status: critical red, warning amber.
- Slicer: `dim_company[company]`.

**Page 3: AEHR vs Peers.**
- Clustered bar: Y `comps[ticker]`, X `comps[ev_revenue]`, sorted descending. Add a constant line at `Peer Median EV/Revenue` (Analytics pane).
- Cards: `AEHR EV/Revenue`, `Peer Median EV/Revenue`, `AEHR Premium to Peers`.
- Line chart: `aehr_price[date]`, `aehr_price[close]`.
- Text box: "SEC companyfacts and Yahoo Finance prices as of Oct 9, 2026. Blank = NM: Onto and Camtek carry convertible notes the engine does not sum."

### 5. Check it against the engine

These come from `python -m projects.powerbi export`. If a card does not match, the model or a measure is wrong, not the data.

| Card | Filter | Should read |
| --- | --- | --- |
| Contribution Actual (net) | Sep 2026 | ($362,288) |
| Contribution Budget (net) | Sep 2026 | ($172,000) |
| Favorable Variance | Sep 2026 | ($190,288) |
| Favorable Variance | Aug 2026 | $57,528 |
| Flagged Lines | Sep 2026 | 4 |
| Revenue (Latest Month) | none | $8,106,743 |
| Revenue YoY % | Sep 2026 | 42.4% |
| Cash (Latest Month) | none | $44,499,130 |
| Net Burn (Latest Month) | none | $2,380,350 |
| Companies Critical | none | 2 |
| Peer Median EV/Revenue | none | 8.8x |
| Peer Median EV/EBITDA | none | 55.0x |

### 6. Save and share

- Save as `projects/powerbi/finance_portfolio.pbix` and commit it.
- **File > Export > Export to PDF** to `docs/powerbi/finance_portfolio.pdf`, and screenshot each page into `docs/screenshots/`.
- A live "Publish to web" link needs a Power BI Pro license, which costs money. The `.pbix`, the PDF and the screenshots work without one.
