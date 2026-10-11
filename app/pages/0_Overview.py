"""Overview: what the five tools are, today's headline figure from each, how they are verified."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st
import yaml

import fp_ui as ui

ui.page_setup(
    "Finance tools that tie to the filing.",
    "Five working tools for banking, private equity and FP&A, built on one tested Python library. "
    "Each page opens on sample data, runs the same engine as its command-line twin, and exports the "
    "Excel or markdown file an analyst would actually hand over.",
    section="finance-portfolio · working papers",
)

TOOLS = [
    ("01", "comps", "pages/1_Comps_Builder.py", "Comps builder",
     "Tickers in; a comparable-company table out, with EV multiples, margins, growth and a source "
     "for every input."),
    ("02", "variance", "pages/2_Variance_Report.py", "Variance report",
     "Two GL exports in; a sign-correct budget-vs-actual pack and draft close commentary out."),
    ("03", "lbo", "pages/3_LBO_Engine.py", "LBO engine",
     "A sponsor-style buyout with a senior-first cash sweep, scenarios, and a live-formula Excel model."),
    ("04", "sourcing", "pages/4_Deal_Sourcing.py", "Deal sourcing",
     "A day of SEC Form D filings in; funds filtered out, operating-company raises scored and ranked."),
    ("05", "dashboard", "pages/5_KPI_Dashboard.py", "Portfolio KPIs",
     "Monthly company KPIs in; revenue, margin, burn and runway with red / amber / green status."),
]


def resume_line(project: str) -> str:
    text = (ui.ROOT / "projects" / project / "README.md").read_text(encoding="utf-8")
    m = re.search(r"## Resume line\s+> (.+)", text)
    return m.group(1).strip() if m else ""


@st.cache_data(show_spinner=False)
def tape() -> dict:
    """One headline figure per tool, each from the same builder its page uses (sample data)."""
    from datetime import date

    from finlib.models.lbo import LBOAssumptions, run_lbo
    from finlib.models.ratios import summary_rows
    from finlib.models.sourcing import funnel, rank
    from finlib.providers.formd import FIXTURE_DIR, FormDProvider
    from projects.comps.cli import load_all
    from projects.dashboard.cli import HERE as DASH, payload_from_frame
    from projects.sourcing.cli import HERE as SRC, load_config
    from projects.variance.cli import HERE as VAR, build_variance, load_thresholds

    with ui.finlib_mode(True):
        _, _, rows = load_all(["AAPL", "MSFT", "GOOGL"])
        cfg, _ = load_config(SRC / "config.yaml")
        day = max(date.fromisoformat(p.name) for p in FIXTURE_DIR.iterdir() if p.is_dir())  # as the page defaults
        stages = funnel(rank(FormDProvider().filings_for(day), cfg, set()))
    actuals, budget = pd.read_csv(VAR / "sample_data/actuals.csv"), pd.read_csv(VAR / "sample_data/budget.csv")
    period = max(set(actuals["period"].astype(str)) & set(budget["period"].astype(str)))  # the page's default
    pack = build_variance(actuals, budget, period, load_thresholds(VAR / "config.yaml"))
    net = pack.summary.set_index("line").loc["Net"]
    deal = LBOAssumptions(**yaml.safe_load((ui.ROOT / "projects/lbo/deal.yaml").read_text()))
    lbo = run_lbo(deal)
    totals = payload_from_frame(pd.read_csv(DASH / "sample_data/kpis.csv"), "")["totals"]
    return {
        "comps": (ui.fmt_x(summary_rows(rows)[0]["ev_ebitda"]), "median EV / EBITDA"),
        "variance": (ui.fmt_usd(net["variance_amt"]), f"net income vs budget, {pack.period}"),
        "lbo": (ui.fmt_pct(lbo.irr), f"base-case IRR, {ui.fmt_x(lbo.moic, 2)} MOIC"),
        "sourcing": (f"{stages[-1][1]} of {stages[0][1]}", "Form D raises kept"),
        "dashboard": ("NM" if totals["revenue_yoy"] is None else f"{totals['revenue_yoy']:+.0%}",
                      f"portfolio revenue, YoY to {totals['month']}"),
    }


t = tape()
st.html('<p class="fp-eyebrow" style="margin-top:18px">Headline figures, sample data</p>')
with st.container(horizontal=True):
    for num, key, page, name, _ in TOOLS:
        value, label = t[key]
        st.metric(f"{num} · {name}", value, label, delta_color="off", delta_arrow="off", border=True, height="stretch")

st.space("medium")
st.subheader("Contents")
for num, key, page, name, desc in TOOLS:
    ui.rule()
    c0, c1, c2, c3 = st.columns([0.45, 3.2, 4, 1.1], vertical_alignment="top")
    c0.html(f'<div class="fp-num">{num}</div>')
    with c1:
        st.markdown(f"#### {name}")
        st.write(desc)
    c2.caption(f"Resume line: *{resume_line(key)}*")
    c3.page_link(page, label="Open", icon=":material/arrow_forward:")
ui.rule()

st.space("medium")
st.subheader("How the numbers are verified")
left, right = st.columns(2, gap="large")
with left:
    st.markdown(
        """
**1. Comps tie to the filing.** In live mode, for AAPL, MSFT and GOOGL, every component of every LTM
figure (fiscal year, year-to-date, prior year-to-date), plus cash and debt, matched the companies'
own earnings releases on Oct 10, 2026. LTM is *prior FY + YTD − prior YTD*, because companies
don't tag Q4 as a three-month value, and the arithmetic for each number is printed next to it.

**2. Missing data stays visible.** A balance-sheet tag counts only if dated within 45 days of the
LTM end. A multiple on a zero or negative base reads **NM** and drops out of the median. Every
pulled value carries its XBRL tag or fixture name.
"""
    )
with right:
    st.markdown(
        """
**3. Excel agrees with Python.** A test recalculates the exported LBO workbook in LibreOffice and
asserts Excel's `IRR()` matches the engine within 0.1%, for the base case and after switching the
scenario dropdown. Sources equal uses, and the sweep never over-repays.

**4. The app is a view, not a copy.** Every figure on these pages comes from the same `finlib`
functions the CLIs use. `tests/test_app.py` checks each page's headline numbers against a direct
call, and checks that downloads match the CLI files byte for byte.
"""
    )
ui.source("bundled fixtures, labeled sample data (synthetic). The comps fixtures are illustrative, the Form D "
          "day is synthetic, and the variance and KPI files are generated. Live comps and Form D need a free "
          "SEC identity string.")
