"""KPI Dashboard: monthly portfolio KPIs -> tiles, small multiples, RAG table, drill-down."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import fp_ui as ui
from projects.dashboard.cli import HERE, payload_from_frame, render_page

ui.page_setup("Portfolio companies, month by month",
              "Monthly KPIs for every company in the portfolio: revenue, margin, burn and runway, with a red / "
              "amber / green flag on anyone running out of cash and a drill-down for each company.",
              number="05", section="Portfolio KPIs")

STATUS = {"critical": ("▼ critical", ui.CRITICAL), "warning": ("● warning", ui.WARNING_TEXT),
          "good": ("▲ good", ui.GOOD)}

# ---------------------------------------------------------------- inputs ------------
sb = st.sidebar
sb.header("Inputs")
up = sb.file_uploader("kpis.csv", type="csv",
                      help="One row per company-month: company, month (YYYY-MM), revenue, cogs, opex, cash, headcount")
title = sb.text_input("Dashboard title", "Gulf Coast Portfolio KPIs")
raw = pd.read_csv(up) if up is not None else pd.read_csv(HERE / "sample_data" / "kpis.csv")
try:
    p = payload_from_frame(raw, title)
except Exception as e:
    st.error(f"Input problem:\n\n```\n{e}\n```")
    st.stop()

t = p["totals"]
if up is not None:
    st.markdown(f":blue-badge[:material/upload_file: Your data]&nbsp; <small>{up.name}, through {t['month']}.</small>",
                unsafe_allow_html=True)
else:
    ui.sample_badge(f"{p['title']}: 8 companies, 24 months, with planted stories (two near the end of their "
                    "runway, one flat, three profitable, one post-Series-B).")

companies = p["companies"]
snap = p["snapshot"]

# ---------------------------------------------------------------- tiles -------------
with st.container(horizontal=True):
    st.metric(f"Portfolio revenue, {t['month']}", ui.fmt_usd(t["revenue"]),
              None if t["revenue_yoy"] is None else f"{t['revenue_yoy']:+.1%} YoY", border=True, height="stretch",
              help="Total revenue of the companies reporting this month versus the total 12 months earlier. "
                   "Not same-store: a company added during the year raises it.")
    st.metric("Blended gross margin", ui.fmt_pct(t["gross_margin"]), "revenue-weighted", delta_color="off",
              delta_arrow="off", border=True, height="stretch")
    st.metric("Net burn this month", ui.fmt_usd(t["net_burn"]), "across burning companies", delta_color="off",
              delta_arrow="off", border=True, height="stretch")
    st.metric("Cash on hand", ui.fmt_usd(t["cash"]), f"{t['headcount']:,} headcount", delta_color="off",
              delta_arrow="off", border=True, height="stretch")
    st.metric("Needs attention", f"{t['critical'] + t['warning']} of {t['companies']}",
              f"{t['critical']} critical, {t['warning']} warning", delta_color="off", delta_arrow="off", border=True, height="stretch")

# ---------------------------------------------------------------- small multiples ---
ui.exhibit(1, "Monthly revenue by company, largest first")
cols = 4
nrows = -(-len(companies) // cols)
fig = make_subplots(rows=nrows, cols=cols, subplot_titles=[
    f"<b>{co}</b>  {'' if snap[co]['yoy_growth'] is None else format(snap[co]['yoy_growth'], '+.0%') + ' YoY'}"
    for co in companies], vertical_spacing=0.2, horizontal_spacing=0.06)
for i, co in enumerate(companies):
    s = p["series"][co]
    rr, cc = i // cols + 1, i % cols + 1
    fig.add_trace(go.Scatter(x=s["month"], y=s["revenue"], mode="lines", line=dict(color=ui.BLUE, width=1.75),
                             fill="tozeroy", fillcolor=ui.BLUE_SOFT, name=co,
                             hovertemplate=f"{co}<br>%{{x}}: $%{{y:,.0f}}<extra></extra>"), row=rr, col=cc)
    fig.add_trace(go.Scatter(x=[s["month"][-1]], y=[s["revenue"][-1]], mode="markers", hoverinfo="skip",
                             marker=dict(color=ui.BLUE, size=7, line=dict(color=ui.PAPER, width=2))), row=rr, col=cc)
fig = ui.style(fig, height=190 * nrows + 30)
fig.update_annotations(font=dict(size=12, color=ui.INK, family=ui.FONT), xanchor="left")
for k, ann in enumerate(fig.layout.annotations):  # left-align each panel title over its panel
    ann.x = fig.layout[f"xaxis{'' if k == 0 else k + 1}"].domain[0]
fig.update_xaxes(nticks=3, tickfont=dict(size=10))
fig.update_yaxes(tickprefix="$", tickformat="~s", nticks=3, tickfont=dict(size=10))
fig.update_layout(margin=dict(t=30))
ui.show(fig)
ui.source("company monthly reporting. Each panel has its own scale; compare shapes, not heights.")

# ---------------------------------------------------------------- RAG table ---------
ui.exhibit(2, f"Latest month, {t['month']}: runway and unit economics")
rows = []
for co in companies:
    s = snap[co]
    label, _ = STATUS[s["runway_status"]]
    if s["runway_months"] is None:
        label = "▲ profitable"
    rows.append({"Company": co, "Revenue": s["revenue"], "YoY": s["yoy_growth"], "Gross margin": s["gross_margin"],
                 "Net burn": s["net_burn"] or None, "Runway (mo)": s["runway_months"], "Status": label,
                 "Headcount": s["headcount"], "Rev / head": s["rev_per_head"]})
df = pd.DataFrame(rows)


def _status_color(v: str) -> str:
    for key, (_, color) in STATUS.items():
        if key in v or (key == "good" and "profitable" in v):
            return f"color: {color}; font-weight: 600"
    return ""


styled = (df.style.map(_status_color, subset=["Status"])
          .format(na_rep="–", formatter={"Revenue": "${:,.0f}", "Net burn": lambda v: "–" if pd.isna(v) else f"${v:,.0f}",
                   "YoY": lambda v: ui.fmt_pct(None if pd.isna(v) else v),
                   "Gross margin": lambda v: ui.fmt_pct(None if pd.isna(v) else v),
                   "Runway (mo)": lambda v: "–" if pd.isna(v) else f"{v:.1f}", "Rev / head": "${:,.0f}"}))
event = st.dataframe(styled, hide_index=True, width="stretch", on_select="rerun", placeholder="–",
                     selection_mode="single-row", key="kpi_table")
ui.source("runway = cash ÷ monthly net burn. Critical under 6 months, warning 6 to 12, good 12 or more or "
          "profitable. Click a row to open it below; click a header to sort.")

# ---------------------------------------------------------------- drill-down --------
critical = [co for co in companies if snap[co]["runway_status"] == "critical"]
chosen = df.iloc[event.selection.rows[0]]["Company"] if event.selection.rows else (critical[0] if critical else companies[0])
ui.exhibit(3, "Company detail")
pick = st.selectbox("Company", companies, index=companies.index(chosen), label_visibility="collapsed")
s, d = snap[pick], p["series"][pick]
label, _ = STATUS[s["runway_status"]]
color = {"critical": "red", "warning": "orange", "good": "green"}[s["runway_status"]]
status_text = label if s["runway_months"] is not None else "▲ profitable"
st.markdown(f"### {pick} &nbsp;:{color}-badge[{status_text}]")
with st.container(horizontal=True):
    st.metric(f"Revenue, {s['month']}", ui.fmt_usd(s["revenue"]),
              None if s["yoy_growth"] is None else f"{s['yoy_growth']:+.1%} YoY", border=True, height="stretch",
              chart_data=d["revenue"], chart_type="area")
    st.metric("Cash", ui.fmt_usd(s["cash"]), border=True, height="stretch", chart_data=d["cash"], chart_type="line")
    st.metric("Net burn", "–" if not s["net_burn"] else ui.fmt_usd(s["net_burn"]), border=True, height="stretch",
              chart_data=d["net_burn"], chart_type="bar")
    st.metric("Runway", "profitable" if s["runway_months"] is None else f"{s['runway_months']:.1f} mo",
              border=True, height="stretch")
with st.container(horizontal=True):
    st.metric("3-month growth", ui.fmt_pct(s["t3m_growth"]), border=True, height="stretch")
    st.metric("Gross margin", ui.fmt_pct(s["gross_margin"]), border=True, height="stretch")
    st.metric("Headcount", f"{s['headcount']:,}", border=True, height="stretch")
    st.metric("Revenue per head", ui.fmt_usd(s["rev_per_head"]), border=True, height="stretch")

a, b, c3 = st.columns(3, gap="large")
with a:
    st.markdown("**Revenue and gross profit**")
    fig = go.Figure([
        go.Bar(x=d["month"], y=d["revenue"], name="Revenue", marker_color=ui.SERIES[0],
               hovertemplate="Revenue $%{y:,.0f}<extra></extra>"),
        go.Scatter(x=d["month"], y=d["gross_profit"], name="Gross profit", mode="lines",
                   line=dict(color=ui.SERIES[1], width=2), hovertemplate="Gross profit $%{y:,.0f}<extra></extra>"),
    ])
    fig.update_layout(bargap=0.35, hovermode="x unified")
    ui.show(ui.style(fig, height=280, y_format="$~s", legend=True))
with b:
    st.markdown("**Cash balance**")
    fig = go.Figure(go.Scatter(x=d["month"], y=d["cash"], mode="lines", line=dict(color=ui.BLUE, width=2),
                               fill="tozeroy", fillcolor=ui.BLUE_SOFT,
                               hovertemplate="%{x}<br>Cash $%{y:,.0f}<extra></extra>"))
    ui.show(ui.style(fig, height=280, y_format="$~s"))
with c3:
    st.markdown("**Monthly net burn**")
    fig = go.Figure(go.Bar(x=d["month"], y=d["net_burn"], marker_color=ui.BLUE,
                           hovertemplate="%{x}<br>Net burn $%{y:,.0f}<extra></extra>"))
    fig.update_layout(bargap=0.35)
    ui.show(ui.style(fig, height=280, y_format="$~s"))
ui.source(f"{pick} monthly reporting, {d['month'][0]} to {d['month'][-1]}.")

st.space("small")
st.download_button("Download the static dashboard (.html)", render_page(p).encode("utf-8"),
                   file_name="index.html", mime="text/html", icon=":material/download:", type="primary")
st.caption("Same self-contained page as `python -m projects.dashboard build`: no server, host it anywhere.")
