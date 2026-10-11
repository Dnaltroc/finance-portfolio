# Portfolio KPI Dashboard

Monthly KPIs for a set of companies in, one self-contained interactive page out. Summary tiles, a revenue small-multiples grid, a sortable table with red/amber/green runway status, and a per-company detail view. Hostable on GitHub Pages, S3, or as a Claude artifact; no server.

## Resume line

> Built a portfolio KPI dashboard (revenue, margin, burn, runway with red/amber/green status) as a static page and an interactive Streamlit app.

```bash
python -m projects.dashboard build                                     # sample data -> dist/index.html
python -m projects.dashboard build --kpis fund.csv --title "Fund II"   # your data
python -m projects.dashboard build --db kpis.sqlite                    # also store computed series
python -m projects.dashboard.make_sample_data                          # regenerate the synthetic file
```

## Input

`kpis.csv` with one row per company-month: `company, month (YYYY-MM), revenue, cogs, opex, cash, headcount`. Every row is validated; failures name the row and the reason.

## Metrics

| Metric | Formula |
| --- | --- |
| Gross margin | (revenue − cogs) / revenue |
| Net income | revenue − cogs − opex |
| Net burn | max(0, −net income) |
| Runway | cash / net burn in months; blank when profitable; 0 when cash is gone |
| Growth | MoM and YoY on revenue; trailing 3-month in the detail view |
| Revenue per head | revenue / headcount |

Runway status: **critical** under 6 months, **warning** 6 to 12, **good** 12+ or profitable.

## Page

- Tiles: portfolio revenue with YoY, blended gross margin, net burn, cash and headcount, count needing attention.
- Small multiples: one revenue line per company with an area fill and an emphasized last point; click a panel to open it in detail.
- Table: sortable on every column; status pills carry a label, not color alone.
- Detail: revenue vs gross profit, cash vs net burn, with hover tooltips. The page opens on the first critical company.
- Light and dark theme from the viewer's system setting; works at phone width.

Plotly loads from cdnjs; everything else is inline. Data is embedded at build time, so the page is a static artifact of the CSV it was built from.

## Tests

`pytest tests/test_kpi.py`: metric math, runway bands, zero-runway on negative cash, validation messages, portfolio totals tying to the snapshot, and the built page embedding the exact payload.

Sample data is synthetic with planted stories: two companies near the end of their runway, one flat, three profitable, one post-Series-B.
