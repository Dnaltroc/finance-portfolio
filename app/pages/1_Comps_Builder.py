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

ui.page_setup("Comparable companies",
              "Tickers in, a comps table out: EV multiples, margins and growth built from what each company "
              "tagged in its 10-K and 10-Q, with the source of every input one click away.",
              number="01", section="Comps builder")

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
    ui.live_badge("Fundamentals from SEC companyfacts (XBRL); price per the source column in Exhibit 4.")

summary = {s["company"]: s for s in summary_rows(rows)}
med = summary["Median"]

# ---------------------------------------------------------------- headline ----------
with st.container(horizontal=True):
    st.metric("Median EV / EBITDA", ui.fmt_x(med["ev_ebitda"]), f"mean {ui.fmt_x(summary['Mean']['ev_ebitda'])}",
              delta_color="off", delta_arrow="off", border=True, height="stretch")
    st.metric("Median EV / Revenue", ui.fmt_x(med["ev_revenue"]), f"mean {ui.fmt_x(summary['Mean']['ev_revenue'])}",
              delta_color="off", delta_arrow="off", border=True, height="stretch")
    st.metric("Median P / E", ui.fmt_x(med["pe"]), f"mean {ui.fmt_x(summary['Mean']['pe'])}",
              delta_color="off", delta_arrow="off", border=True, height="stretch")
    st.metric("Median EBITDA margin", ui.fmt_pct(med["ebitda_margin"]), f"mean {ui.fmt_pct(summary['Mean']['ebitda_margin'])}",
              delta_color="off", delta_arrow="off", border=True, height="stretch")
    st.metric("Peers", f"{len(rows)}", ", ".join(r.ticker for r in rows), delta_color="off",
              delta_arrow="off", border=True, height="stretch")

# ---------------------------------------------------------------- table -------------
ui.exhibit(1, "Trading comparables, last twelve months")
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
st.dataframe(table, hide_index=True, width="stretch",
             column_config={"Ticker": st.column_config.TextColumn(pinned=True),
                            "Company": st.column_config.TextColumn(pinned=True)})
ui.source("$ in millions except per share. NM = not meaningful (zero or negative base), excluded from the "
          "median and mean. " + ("Bundled fixtures." if offline else "SEC companyfacts; prices per Exhibit 4."))

# ---------------------------------------------------------------- charts ------------
left, right = st.columns(2, gap="large")
with left:
    ui.exhibit(2, "EV / EBITDA against the peer median")
    valid = [r for r in rows if r.ev_ebitda is not None]
    nm = [r.ticker for r in rows if r.ev_ebitda is None]
    fig = go.Figure(go.Bar(
        x=[r.ticker for r in valid], y=[r.ev_ebitda for r in valid], marker_color=ui.BLUE,
        marker_cornerradius=2, text=[ui.fmt_x(r.ev_ebitda) for r in valid], textposition="outside",
        textfont=dict(color=ui.INK, size=12), cliponaxis=False,
        customdata=[r.company for r in valid], hovertemplate="%{customdata}<br>EV / EBITDA %{y:.1f}x<extra></extra>",
    ))
    if med["ev_ebitda"] is not None:
        fig.add_hline(y=med["ev_ebitda"], line=dict(color=ui.INK, width=1, dash="dot"),
                      annotation_text=f"Median {ui.fmt_x(med['ev_ebitda'])}", annotation_position="top right",
                      annotation_font=dict(color=ui.INK_2, size=11))
    fig.update_layout(bargap=0.55)
    ui.show(ui.style(fig, y_title="EV / EBITDA (x)", y_format=".0f"))
    ui.source("company filings via the comps engine" + (f". Not shown (NM): {', '.join(nm)}." if nm else "."))

with right:
    ui.exhibit(3, "Profitability against growth")
    pts = [r for r in rows if r.ebitda_margin is not None and r.revenue_growth is not None]
    fig = go.Figure(go.Scatter(
        x=[r.revenue_growth for r in pts], y=[r.ebitda_margin for r in pts], mode="markers+text",
        marker=dict(size=13, color=ui.BLUE, line=dict(color=ui.PAPER, width=2)),
        text=[r.ticker for r in pts], textposition="top center", textfont=dict(color=ui.INK, size=12),
        cliponaxis=False,
        customdata=[[r.company, ui.fmt_pct(r.revenue_growth), ui.fmt_pct(r.ebitda_margin)] for r in pts],
        hovertemplate="%{customdata[0]}<br>Revenue growth %{customdata[1]}<br>"
                      "EBITDA margin %{customdata[2]}<extra></extra>",
    ))
    fig.update_xaxes(title="Revenue growth, LTM vs prior LTM", tickformat=".0%", title_font=dict(size=11))
    fig = ui.style(fig, y_title="EBITDA margin", y_format=".0%")
    fig.update_layout(margin=dict(t=24, r=24))
    ui.show(fig)
    dropped = [r.ticker for r in rows if r not in pts]
    ui.source("LTM vs prior-year LTM" + (f". Not plotted (NM): {', '.join(dropped)}." if dropped else "."))

# ---------------------------------------------------------------- sources -----------
ui.exhibit(4, "Inputs and sources")
with st.expander("Every pulled value, its basis and its XBRL tag", icon=":material/source:"):
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
st.space("small")
data, name = ui.build_file(write_comps_workbook, rows, fundamentals, quotes, peerset)
st.download_button("Download Excel workbook", data, file_name=name, mime=ui.XLSX_MIME,
                   icon=":material/download:", type="primary")
st.caption("Same workbook as `python -m projects.comps run`: Comps, Inputs (blue, sourced), Notes.")
