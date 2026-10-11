"""KPI Dashboard: monthly portfolio KPIs -> tiles, small multiples, RAG table, drill-down."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import fp_ui as ui
from projects.dashboard.cli import HERE, payload_from_frame, render_page

ui.page_setup("KPI Dashboard",
              "Monthly KPIs for a portfolio of companies: revenue, margin, burn and runway, with red / amber / "
              "green runway status and a per-company drill-down.", ":material/monitoring:")

STATUS = {"critical": ("✖ critical", ui.CRITICAL), "warning": ("▲ warning", ui.WARNING_TEXT),
          "good": ("✔ good", ui.GOOD)}

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
    st.success(f"**Your data.** {up.name}, through {t['month']}.", icon=":material/upload_file:")
else:
    ui.sample_badge("8 companies, 24 months, with planted stories: two near the end of their runway, one flat, "
                    "three profitable, one post-Series-B.")

# ---------------------------------------------------------------- tiles -------------
st.subheader(f"{p['title']}, through {t['month']}")
c = st.columns(5)
c[0].metric("Portfolio revenue", ui.fmt_usd(t["revenue"]),
            None if t["revenue_yoy"] is None else f"{t['revenue_yoy']:+.1%} YoY",
            help="Total revenue of the companies reporting this month versus the total 12 months earlier. "
                 "Not same-store: a company added during the year raises it.")
c[1].metric("Blended gross margin", ui.fmt_pct(t["gross_margin"]), help="Revenue-weighted.")
c[2].metric("Net burn this month", ui.fmt_usd(t["net_burn"]), help="Across the companies burning cash.")
c[3].metric("Cash on hand", ui.fmt_usd(t["cash"]), f"{t['headcount']:,} headcount", delta_color="off", delta_arrow="off")
c[4].metric("Needs attention", f"{t['critical'] + t['warning']} of {t['companies']}",
            f"{t['critical']} critical, {t['warning']} warning", delta_color="off", delta_arrow="off")

# ---------------------------------------------------------------- small multiples ---
companies = p["companies"]
snap = p["snapshot"]
st.subheader("Monthly revenue by company")
cols = 4
nrows = -(-len(companies) // cols)
fig = make_subplots(rows=nrows, cols=cols, subplot_titles=[
    f"{co}  {'' if snap[co]['yoy_growth'] is None else format(snap[co]['yoy_growth'], '+.0%') + ' YoY'}"
    for co in companies], vertical_spacing=0.16, horizontal_spacing=0.05)
for i, co in enumerate(companies):
    s = p["series"][co]
    rr, cc = i // cols + 1, i % cols + 1
    fig.add_trace(go.Scatter(x=s["month"], y=s["revenue"], mode="lines", line=dict(color=ui.BLUE, width=2),
                             fill="tozeroy", fillcolor=ui.BLUE_SOFT, name=co,
                             hovertemplate=f"{co}<br>%{{x}}: $%{{y:,.0f}}<extra></extra>"), row=rr, col=cc)
    fig.add_trace(go.Scatter(x=[s["month"][-1]], y=[s["revenue"][-1]], mode="markers", hoverinfo="skip",
                             marker=dict(color=ui.BLUE, size=8, line=dict(color="white", width=2))), row=rr, col=cc)
fig.update_annotations(font_size=12, font_color=ui.INK)
fig.update_xaxes(nticks=3, tickfont_size=10)
fig.update_yaxes(tickprefix="$", tickformat="~s", nticks=3, tickfont_size=10)
ui.show(ui.style(fig, height=200 * nrows + 40))

# ---------------------------------------------------------------- RAG table ---------
st.subheader("Latest month")
rows = []
for co in companies:
    s = snap[co]
    label, _ = STATUS[s["runway_status"]]
    if s["runway_months"] is None:
        label = "✔ profitable"
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
st.caption("Click a row to open it below; click a column header to sort. Runway = cash ÷ monthly net burn. Critical under 6 months, warning 6 to 12, "
           "good 12 or more or profitable.")

# ---------------------------------------------------------------- drill-down --------
st.subheader("Company detail")
critical = [co for co in companies if snap[co]["runway_status"] == "critical"]
chosen = df.iloc[event.selection.rows[0]]["Company"] if event.selection.rows else (critical[0] if critical else companies[0])
pick = st.selectbox("Company", companies, index=companies.index(chosen))
s, d = snap[pick], p["series"][pick]
label, _ = STATUS[s["runway_status"]]
st.markdown(f"**{pick}**: {label if s['runway_months'] is not None else '✔ profitable'}")
f = st.columns(4)
f[0].metric(f"Revenue, {s['month']}", ui.fmt_usd(s["revenue"]),
            None if s["yoy_growth"] is None else f"{s['yoy_growth']:+.1%} YoY")
f[1].metric("3-month growth", ui.fmt_pct(s["t3m_growth"]))
f[2].metric("Gross margin", ui.fmt_pct(s["gross_margin"]))
f[3].metric("Runway", "profitable" if s["runway_months"] is None else f"{s['runway_months']:.1f} months")
g = st.columns(4)
g[0].metric("Cash", ui.fmt_usd(s["cash"]))
g[1].metric("Net burn", "–" if not s["net_burn"] else ui.fmt_usd(s["net_burn"]))
g[2].metric("Headcount", f"{s['headcount']:,}")
g[3].metric("Revenue per head", ui.fmt_usd(s["rev_per_head"]))

a, b, c3 = st.columns(3)
with a:
    fig = go.Figure([
        go.Bar(x=d["month"], y=d["revenue"], name="Revenue", marker_color=ui.SERIES[0],
               hovertemplate="Revenue $%{y:,.0f}<extra></extra>"),
        go.Scatter(x=d["month"], y=d["gross_profit"], name="Gross profit", mode="lines",
                   line=dict(color=ui.SERIES[1], width=2), hovertemplate="Gross profit $%{y:,.0f}<extra></extra>"),
    ])
    fig.update_layout(title=dict(text="Revenue and gross profit", font_size=14), bargap=0.35, hovermode="x unified")
    ui.show(ui.style(fig, height=300, y_format="$~s", legend=True))
with b:
    fig = go.Figure(go.Scatter(x=d["month"], y=d["cash"], mode="lines", line=dict(color=ui.BLUE, width=2),
                               fill="tozeroy", fillcolor=ui.BLUE_SOFT, hovertemplate="%{x}<br>Cash $%{y:,.0f}<extra></extra>"))
    fig.update_layout(title=dict(text="Cash balance", font_size=14))
    ui.show(ui.style(fig, height=300, y_format="$~s"))
with c3:
    fig = go.Figure(go.Bar(x=d["month"], y=d["net_burn"], marker_color=ui.BLUE,
                           hovertemplate="%{x}<br>Net burn $%{y:,.0f}<extra></extra>"))
    fig.update_layout(title=dict(text="Monthly net burn", font_size=14), bargap=0.35)
    ui.show(ui.style(fig, height=300, y_format="$~s"))

st.download_button("Download the static dashboard (.html)", render_page(p).encode("utf-8"),
                   file_name="index.html", mime="text/html", icon=":material/download:", type="primary")
st.caption("Same self-contained page as `python -m projects.dashboard build`: no server, host it anywhere.")
