"""LBO Engine: deal inputs -> operating build, debt schedule, returns, IRR sensitivity."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from pydantic import ValidationError

import fp_ui as ui
from finlib.excel_writer.lbo_workbook import write_lbo_workbook
from finlib.models.lbo import LBOAssumptions, run_lbo, sensitivity
from projects.lbo.cli import HERE

ui.page_setup("Leveraged buyout returns",
              "A sponsor-style LBO. Change the deal in the sidebar and watch the debt pay down and the returns "
              "move; the export is a workbook where every computed cell is a live Excel formula.",
              number="03", section="LBO engine")

DEAL = yaml.safe_load((HERE / "deal.yaml").read_text())

# ---------------------------------------------------------------- inputs ------------
sb = st.sidebar
sb.header("Deal")
deal_name = sb.text_input("Deal name", DEAL["deal_name"])
scenario = sb.selectbox("Scenario", list(DEAL["scenarios"]),
                        index=list(DEAL["scenarios"]).index(DEAL["active_scenario"]))
c1, c2 = sb.columns(2)
entry_revenue = c1.number_input("Entry revenue ($M)", 1.0, 100_000.0, float(DEAL["entry_revenue"]), 10.0)
entry_ebitda = c2.number_input("Entry EBITDA ($M)", 0.1, 50_000.0, float(DEAL["entry_ebitda"]), 5.0)
entry_multiple = sb.slider("Entry EV / EBITDA", 4.0, 16.0, float(DEAL["entry_multiple"]), 0.5, format="%.1fx")
with sb.expander("Fees, cash, tax and reinvestment"):
    fees = st.number_input("Fees (% of EV)", 0.0, 10.0, DEAL["fees_pct_ev"] * 100, 0.25, format="%.2f")
    min_cash = st.number_input("Minimum cash ($M)", 0.0, 10_000.0, float(DEAL["min_cash"]), 1.0)
    tax = st.number_input("Tax rate (%)", 0.0, 60.0, DEAL["tax_rate"] * 100, 1.0)
    da = st.number_input("D&A (% of revenue)", 0.0, 30.0, DEAL["da_pct_revenue"] * 100, 0.5)
    capex = st.number_input("Capex (% of revenue)", 0.0, 30.0, DEAL["capex_pct_revenue"] * 100, 0.5)
    nwc = st.number_input("NWC (% of change in revenue)", 0.0, 50.0, DEAL["nwc_pct_revenue"] * 100, 1.0)

sb.subheader("Debt tranches")
sb.caption("Raised as a multiple of entry EBITDA. Sweep tranches are repaid senior-first from excess cash.")
tr = sb.data_editor(
    pd.DataFrame([{"Tranche": t["name"], "x EBITDA": t["ebitda_multiple"], "Rate %": t["rate"] * 100,
                   "Amort %": t["amort_pct"] * 100, "Sweep": t["sweep"]} for t in DEAL["tranches"]]),
    num_rows="dynamic", hide_index=True, key="tranches",
    column_config={"Tranche": st.column_config.TextColumn(width="small"), "Sweep": st.column_config.CheckboxColumn(width="small"),
                   "x EBITDA": st.column_config.NumberColumn(width="small", min_value=0.0, step=0.25, format="%.2fx"),
                   "Rate %": st.column_config.NumberColumn(width="small", min_value=0.0, step=0.25, format="%.2f%%"),
                   "Amort %": st.column_config.NumberColumn(width="small", min_value=0.0, max_value=100.0, step=0.5,
                                                            format="%.1f%%")},
)

sb.subheader(f"{scenario} case path")
sc = DEAL["scenarios"][scenario]
path = sb.data_editor(
    pd.DataFrame({"Year": range(1, len(sc["revenue_growth"]) + 1),
                  "Revenue growth %": [g * 100 for g in sc["revenue_growth"]],
                  "EBITDA margin %": [m * 100 for m in sc["ebitda_margin"]]}),
    hide_index=True, disabled=["Year"], key=f"path_{scenario}",
    column_config={"Year": st.column_config.NumberColumn(width="small"),
                   "Revenue growth %": st.column_config.NumberColumn("Growth %", width="small", step=0.5, format="%.1f%%"),
                   "EBITDA margin %": st.column_config.NumberColumn("Margin %", width="small", step=0.5, format="%.1f%%")},
)
exit_multiple = sb.slider("Exit EV / EBITDA", 4.0, 16.0, float(sc["exit_multiple"]), 0.5, format="%.1fx",
                          key=f"exit_{scenario}")
sb.caption("Edits apply to the selected scenario; the others keep their deal.yaml paths.")

# ---------------------------------------------------------------- build -------------
scenarios = {k: dict(v) for k, v in DEAL["scenarios"].items()}
scenarios[scenario] = {"revenue_growth": [float(g) / 100 for g in path["Revenue growth %"]],
                       "ebitda_margin": [float(m) / 100 for m in path["EBITDA margin %"]],
                       "exit_multiple": exit_multiple}
tranches = [{"name": str(r["Tranche"] or f"Tranche {i + 1}"), "ebitda_multiple": float(r["x EBITDA"] or 0),
             "rate": float(r["Rate %"] or 0) / 100, "amort_pct": float(r["Amort %"] or 0) / 100,
             "sweep": bool(r["Sweep"])} for i, r in tr.iterrows()]
try:
    a = LBOAssumptions(
        deal_name=deal_name or "Target Co", entry_revenue=entry_revenue, entry_ebitda=entry_ebitda,
        entry_multiple=entry_multiple, fees_pct_ev=fees / 100, min_cash=min_cash, tax_rate=tax / 100,
        da_pct_revenue=da / 100, capex_pct_revenue=capex / 100, nwc_pct_revenue=nwc / 100,
        tranches=tranches, scenarios=scenarios, active_scenario=scenario,
    )
except ValidationError as e:
    st.error("Check the inputs: " + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()))
    st.stop()
if not tranches:
    st.warning("Add at least one debt tranche.")
    st.stop()

r = run_lbo(a)
years = r.years
equity_ok = a.sponsor_equity > 0

ui.sample_badge(f"Illustrative deal from `projects/lbo/deal.yaml`, {scenario} case, $ in millions.")

# ---------------------------------------------------------------- returns tile ------
with st.container(horizontal=True):
    st.metric("Entry EV", ui.fmt_musd(a.entry_ev), f"{a.entry_multiple:.1f}x EBITDA", delta_color="off",
              delta_arrow="off", border=True, height="stretch")
    st.metric("Total debt", ui.fmt_musd(a.total_debt), f"{a.total_debt / a.entry_ebitda:.1f}x EBITDA",
              delta_color="off", delta_arrow="off", border=True, height="stretch")
    st.metric("Sponsor equity", ui.fmt_musd(a.sponsor_equity), "cheque at close", delta_color="off",
              delta_arrow="off", border=True, height="stretch")
    if not equity_ok:
        st.metric("MOIC", "n/a", border=True, height="stretch")
        st.metric("IRR", "n/a", border=True, height="stretch")
    else:
        st.metric("MOIC", ui.fmt_x(r.moic, 2), f"exit equity {ui.fmt_musd(r.exit_equity)}", delta_color="off",
                  delta_arrow="off", border=True, height="stretch")
        st.metric("IRR", ui.fmt_pct(r.irr) if r.irr_defined else "n/a: equity wiped out",
                  f"{a.hold_years}-year hold", delta_color="off", delta_arrow="off", border=True, height="stretch")
if not equity_ok:
    st.error("**n/a: debt exceeds the purchase price.** Debt is larger than EV + fees + minimum cash, so the "
             "sponsor puts in no equity and MOIC and IRR have no meaning. Lower leverage or raise the entry "
             "multiple.", icon=":material/block:")
if r.cash_shortfall_years:
    st.warning(f"Cash falls below the {ui.fmt_musd(a.min_cash)} minimum in year(s) "
               f"{', '.join(map(str, r.cash_shortfall_years))}. There is no revolver in this model, so the deal "
               "needs less debt or more cash flow; returns in this case overstate what is achievable.",
               icon=":material/warning:")

# ---------------------------------------------------------------- sources & uses ----
left, right = st.columns([1, 2], gap="large")
with left:
    ui.exhibit(1, "Sources and uses at close")
    src = [(name, r.tranche_end[name][0]) for name in r.tranche_end] + [("Sponsor equity", a.sponsor_equity),
                                                                          ("Total sources", a.total_debt + a.sponsor_equity)]
    st.dataframe(pd.DataFrame({"Sources": [k for k, _ in src], "$M": [ui.fmt_musd(v) for _, v in src]}),
                 hide_index=True, width="stretch")
    use = [("Purchase enterprise value", a.entry_ev), ("Transaction fees", a.fees),
           ("Cash to balance sheet", a.min_cash), ("Total uses", a.entry_ev + a.fees + a.min_cash)]
    st.dataframe(pd.DataFrame({"Uses": [k for k, _ in use], "$M": [ui.fmt_musd(v) for _, v in use]}),
                 hide_index=True, width="stretch")

with right:
    ui.exhibit(2, "Operating build and cash flow")
    lines = [("Revenue", r.revenue), ("EBITDA", r.ebitda), ("D&A", r.da), ("EBIT", r.ebit),
             ("Cash interest", r.interest), ("Taxes", r.taxes), ("Capex", r.capex), ("Change in NWC", r.delta_nwc),
             ("FCF before debt service", r.fcf_pre_debt), ("Cash available for sweep", r.cash_available_for_debt),
             ("Cash balance", r.cash_balance)]
    op = pd.DataFrame({"$M": [k for k, _ in lines],
                       **{f"Y{y}": [ui.fmt_musd(v[y]) for _, v in lines] for y in years}})
    st.dataframe(op, hide_index=True, width="stretch", height=(len(op) + 1) * 35 + 3,
                 column_config={"$M": st.column_config.TextColumn(width="medium", pinned=True)})
    ui.source("LBO engine. Year 0 = entry; interest on beginning balances; taxes floored at zero.")

# ---------------------------------------------------------------- charts ------------
c1, c2 = st.columns(2, gap="large")
palette = ui.SERIES + ui.SERIES_EXTRA
with c1:
    ui.exhibit(3, "Debt outstanding by tranche, end of year")
    fig = go.Figure()
    for i, name in enumerate(r.tranche_end):
        fig.add_trace(go.Scatter(
            x=years, y=r.tranche_end[name], name=name, stackgroup="debt", mode="lines",
            line=dict(width=1.5, color=palette[i % len(palette)]),
            hovertemplate=f"{name}<br>Year %{{x}}: $%{{y:,.1f}}M<extra></extra>",
        ))
    fig.update_xaxes(tickvals=years, ticktext=[f"Y{y}" for y in years])
    fig = ui.style(fig, y_title="$M", y_format=",.0f", legend=True)
    fig.update_layout(legend_traceorder="normal")
    ui.show(fig)
    ui.source("debt schedule. Sweep tranches repaid senior-first after mandatory amortization.")

with c2:
    ui.exhibit(4, "Cash balance against the minimum")
    fig = go.Figure()
    for y in r.cash_shortfall_years:
        fig.add_vrect(x0=y - 0.5, x1=y + 0.5, fillcolor=ui.CRITICAL, opacity=0.10, line_width=0)
    fig.add_trace(go.Scatter(x=years, y=r.cash_balance, mode="lines+markers", line=dict(color=ui.BLUE, width=2),
                             marker=dict(size=8, color=ui.BLUE, line=dict(color=ui.PAPER, width=2)),
                             hovertemplate="Year %{x}<br>Cash $%{y:,.1f}M<extra></extra>"))
    fig.add_hline(y=a.min_cash, line=dict(color=ui.INK, width=1, dash="dot"),
                  annotation_text=f"Minimum {ui.fmt_musd(a.min_cash)}", annotation_position="bottom right",
                  annotation_font=dict(color=ui.INK_2, size=11))
    fig.update_xaxes(tickvals=years, ticktext=[f"Y{y}" for y in years])
    fig = ui.style(fig, y_title="$M", y_format=",.0f")
    fig.update_yaxes(rangemode="tozero")
    ui.show(fig)
    ui.source("cash roll. " + ("Red bands mark years below the minimum." if r.cash_shortfall_years
                               else "Cash holds at the minimum; everything above it sweeps to debt."))

# ---------------------------------------------------------------- sensitivity -------
ui.exhibit(5, f"IRR by entry and exit multiple, {scenario} case")
entry_grid = [round(a.entry_multiple + d, 2) for d in (-2, -1, 0, 1, 2) if a.entry_multiple + d > 0]
exit_grid = [round(exit_multiple + d, 2) for d in (-2, -1, 0, 1, 2) if exit_multiple + d > 0]
grid = sensitivity(a, entry_grid, exit_grid, scenario)
text = [[ui.fmt_pct(v) if v == v else "n/a" for v in row] for row in grid]
fig = go.Figure(go.Heatmap(
    z=grid, x=[f"{x:.1f}x" for x in exit_grid], y=[f"{e:.1f}x" for e in entry_grid], text=text,
    texttemplate="%{text}", textfont=dict(color=ui.INK, size=13, family=ui.MONO),
    # light half of the blue ramp so dark text stays readable in every cell
    colorscale=[[i / 3, c] for i, c in enumerate(ui.SEQ_BLUE[:4])], showscale=False,
    xgap=3, ygap=3, hovertemplate="Entry %{y}, exit %{x}<br>IRR %{text}<extra></extra>",
))
if a.entry_multiple in entry_grid and exit_multiple in exit_grid:  # outline the current deal
    i, j = entry_grid.index(a.entry_multiple), exit_grid.index(exit_multiple)
    fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5, line=dict(color=ui.INK, width=2))
fig = ui.style(fig, height=330)  # house style first; the heatmap's own axis settings override it
fig.update_xaxes(title="Exit EV / EBITDA", side="top", showline=False, ticks="", title_font=dict(color=ui.MUTED, size=11))
fig.update_yaxes(title="Entry EV / EBITDA", autorange="reversed", showgrid=False, zeroline=False, ticks="")
ui.show(fig)
ui.source("LBO engine, all other inputs as set. Outlined cell = current deal. n/a = equity wiped out or none invested.")

# ---------------------------------------------------------------- download ----------
st.space("small")
data, name = ui.build_file(write_lbo_workbook, a, entry_grid=entry_grid, exit_grid=exit_grid)
st.download_button("Download live-formula Excel model", data, file_name=name, mime=ui.XLSX_MIME,
                   icon=":material/download:", type="primary")
st.caption("Same workbook as `python -m projects.lbo run`: Assumptions (scenario dropdown), Operating, Debt, "
           "Returns with `=IRR()`, and this sensitivity grid. Interest is on beginning balances, so the "
           "workbook opens with no circular reference.")
