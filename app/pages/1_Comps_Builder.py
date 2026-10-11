"""Comps Builder: SEC XBRL fundamentals + market prices -> comparable-company table."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fp_ui as ui
from finlib.excel_writer import write_comps_workbook
from finlib.models.ratios import summary_rows
from finlib.providers.edgar import FIXTURE_DIR
from projects.comps.cli import load_all

ui.page_setup("Comps Builder",
              "Ticker list in, comparable-company table out: EV/Revenue, EV/EBITDA, P/E, margins and "
              "growth from SEC filings, with the source of every input.", ":material/table_chart:")

FIXTURE_TICKERS = sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))

# ---------------------------------------------------------------- inputs ------------
sb = st.sidebar
sb.header("Inputs")
offline = ui.mode_picker("comps_mode", "Live pulls SEC companyfacts and a market price.")
tickers_text = sb.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL",
                             help=f"Offline fixtures exist for {', '.join(FIXTURE_TICKERS)}.")
peerset = sb.text_input("Peer set label", "demo" if offline else "peerset",
                        help="Used in the workbook title and file name.")
prices = ""
if offline:
    sb.caption("Manual prices apply in Live mode only; offline runs use the fixture price.")
else:
    prices = sb.text_input("Manual prices (optional)", "", placeholder="AAPL=232.50,MSFT=428.00",
                           help="Overrides the market feed. Shares still come from the SEC filing.")

tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
if not tickers:
    st.warning("Enter at least one ticker.")
    st.stop()


@st.cache_data(ttl=3600, show_spinner=False)
def run(tickers: tuple[str, ...], offline: bool, prices: str):
    with ui.finlib_mode(offline):
        return load_all(list(tickers), prices or None)


with st.spinner("Pulling filings and prices..." if not offline else "Reading fixtures..."):
    try:
        fundamentals, quotes, rows = run(tuple(tickers), offline, prices)
    except FileNotFoundError:
        missing = [t for t in tickers if t not in FIXTURE_TICKERS]
        st.error(f"No offline fixture for {', '.join(missing)}. Offline mode covers "
                 f"{', '.join(FIXTURE_TICKERS)}; switch to Live for other tickers.")
        st.stop()
    except Exception as e:  # network, unknown ticker, missing price
        st.error(f"Could not build comps: {e}")
        st.stop()

if offline:
    ui.sample_badge("Fixture financials are illustrative, not sourced from filings; prices are fixed.")
else:
    ui.live_badge("Fundamentals from SEC companyfacts (XBRL); price per the source column below.")

summary = {s["company"]: s for s in summary_rows(rows)}
med = summary["Median"]

# ---------------------------------------------------------------- headline ----------
c = st.columns(4)
c[0].metric("Median EV / EBITDA", ui.fmt_x(med["ev_ebitda"]))
c[1].metric("Median EV / Revenue", ui.fmt_x(med["ev_revenue"]))
c[2].metric("Median P / E", ui.fmt_x(med["pe"]))
c[3].metric("Median EBITDA margin", ui.fmt_pct(med["ebitda_margin"]))

# ---------------------------------------------------------------- table -------------
st.subheader("Comps table")
st.caption("$ in millions except per-share. NM = not meaningful (zero or negative base); "
           "NM values are excluded from the median and mean.")
table = pd.DataFrame([{
    "Ticker": r.ticker, "Company": r.company, "Price": f"${r.price:,.2f}",
    "Mkt cap": ui.fmt_musd(r.market_cap_m, 0), "EV": ui.fmt_musd(r.ev_m, 0),
    "LTM revenue": ui.fmt_musd(r.revenue_m, 0), "LTM EBITDA": ui.fmt_musd(r.ebitda_m, 0),
    "LTM net income": ui.fmt_musd(r.net_income_m, 0),
    "EV / Revenue": ui.fmt_x(r.ev_revenue), "EV / EBITDA": ui.fmt_x(r.ev_ebitda), "P / E": ui.fmt_x(r.pe),
    "Gross margin": ui.fmt_pct(r.gross_margin), "EBITDA margin": ui.fmt_pct(r.ebitda_margin),
    "Revenue growth": ui.fmt_pct(r.revenue_growth), "Flags": r.flags,
} for r in rows] + [{
    "Ticker": "", "Company": label, "EV / Revenue": ui.fmt_x(s["ev_revenue"]),
    "EV / EBITDA": ui.fmt_x(s["ev_ebitda"]), "P / E": ui.fmt_x(s["pe"]),
    "Gross margin": ui.fmt_pct(s["gross_margin"]), "EBITDA margin": ui.fmt_pct(s["ebitda_margin"]),
    "Revenue growth": ui.fmt_pct(s["revenue_growth"]),
} for label, s in summary.items()]).fillna("")
st.dataframe(table, hide_index=True, width="stretch")

# ---------------------------------------------------------------- charts ------------
left, right = st.columns(2)
with left:
    valid = [r for r in rows if r.ev_ebitda is not None]
    nm = [r.ticker for r in rows if r.ev_ebitda is None]
    fig = go.Figure(go.Bar(
        x=[r.ticker for r in valid], y=[r.ev_ebitda for r in valid], marker_color=ui.BLUE,
        marker_cornerradius=4, text=[ui.fmt_x(r.ev_ebitda) for r in valid], textposition="outside",
        customdata=[r.company for r in valid], hovertemplate="%{customdata}<br>EV / EBITDA %{y:.1f}x<extra></extra>",
    ))
    if med["ev_ebitda"] is not None:
        fig.add_hline(y=med["ev_ebitda"], line=dict(color=ui.INK_2, width=1.5, dash="dash"),
                      annotation_text=f"Median {ui.fmt_x(med['ev_ebitda'])}", annotation_position="top right")
    fig.update_layout(title=dict(text="EV / EBITDA by company", font_size=14), bargap=0.45)
    ui.show(ui.style(fig, y_title="EV / EBITDA (x)", y_format=".0f"))
    if nm:
        st.caption(f"Not shown (NM): {', '.join(nm)}.")

with right:
    pts = [r for r in rows if r.ebitda_margin is not None and r.revenue_growth is not None]
    fig = go.Figure(go.Scatter(
        x=[r.revenue_growth for r in pts], y=[r.ebitda_margin for r in pts], mode="markers+text",
        marker=dict(size=14, color=ui.BLUE, line=dict(color="white", width=2)),
        text=[r.ticker for r in pts], textposition="top center",
        customdata=[[r.company, ui.fmt_pct(r.revenue_growth), ui.fmt_pct(r.ebitda_margin)] for r in pts],
        hovertemplate="%{customdata[0]}<br>Revenue growth %{customdata[1]}<br>"
                      "EBITDA margin %{customdata[2]}<extra></extra>",
    ))
    fig.update_layout(title=dict(text="EBITDA margin vs revenue growth (LTM)", font_size=14))
    fig.update_traces(cliponaxis=False)
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(title="Revenue growth, LTM vs prior LTM", tickformat=".0%")
    fig = ui.style(fig, y_title="EBITDA margin", y_format=".0%")
    fig.update_layout(margin=dict(t=64, r=24))
    ui.show(fig)
    dropped = [r.ticker for r in rows if r not in pts]
    if dropped:
        st.caption(f"Not plotted (margin or growth NM): {', '.join(dropped)}.")

# ---------------------------------------------------------------- sources -----------
with st.expander("Inputs and sources: where every number came from", icon=":material/source:"):
    qmap = {q.ticker: q for q in quotes}
    src = []
    for f in fundamentals:
        q = qmap[f.ticker]
        src.append({"Ticker": f.ticker, "Concept": "price", "Basis": f"spot {q.as_of}",
                    "Value": f"${q.price:,.2f}", "Source": q.source})
        src.append({"Ticker": f.ticker, "Concept": "shares outstanding", "Basis": "spot",
                    "Value": f"{q.shares_outstanding:,.0f}", "Source": q.shares_source or q.source})
        for concept in ("revenue", "cogs", "operating_income", "da", "net_income"):
            v = f.ltm(concept)
            src.append({"Ticker": f.ticker, "Concept": concept, "Basis": f"LTM {f.ltm_period}",
                        "Value": ui.fmt_usd(v), "Source": f.sources.get(concept, "missing"),
                        "LTM arithmetic": f.basis.get(concept, "")})
        for concept in ("cash", "total_debt"):
            src.append({"Ticker": f.ticker, "Concept": concept, "Basis": f"as of {f.balance_date}",
                        "Value": ui.fmt_usd(f.latest(concept)), "Source": f.sources.get(concept, "missing")})
    st.dataframe(pd.DataFrame(src).fillna(""), hide_index=True, width="stretch")
    st.caption("LTM basis per ticker: " + "; ".join(f"{f.ticker} {f.ltm_period}" for f in fundamentals))

# ---------------------------------------------------------------- download ----------
data, name = ui.build_file(write_comps_workbook, rows, fundamentals, quotes, peerset)
st.download_button("Download Excel workbook", data, file_name=name, mime=ui.XLSX_MIME,
                   icon=":material/download:", type="primary")
st.caption("Same workbook as `python -m projects.comps run`: Comps, Inputs (blue, sourced), Notes.")
