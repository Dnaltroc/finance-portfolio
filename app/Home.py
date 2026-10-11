"""finance-portfolio: five finance tools on one verified Python library.

Run:  streamlit run app/Home.py
"""

from __future__ import annotations

import re

import streamlit as st

import fp_ui as ui

ui.page_setup("finance-portfolio",
              "Five finance tools a team would actually use, built on one shared Python library. Each page "
              "opens on sample data, runs the same engine as the command-line tool, and downloads the same "
              "Excel or markdown file.", ":material/finance:")

TOOLS = [
    ("comps", "pages/1_Comps_Builder.py", "Comps Builder", ":material/table_chart:",
     "Ticker list in, a comparable-company table out, with EV multiples, margins, growth, and a source for every input."),
    ("variance", "pages/2_Variance_Report.py", "Variance Report", ":material/difference:",
     "Two GL exports in, a sign-correct budget-vs-actual pack and draft close commentary out."),
    ("lbo", "pages/3_LBO_Engine.py", "LBO Engine", ":material/account_balance:",
     "A sponsor-style buyout model with a cash sweep, scenarios, and a live-formula Excel export."),
    ("sourcing", "pages/4_Deal_Sourcing.py", "Deal Sourcing", ":material/travel_explore:",
     "One day of SEC Form D filings in, funds filtered out, operating-company raises scored and ranked."),
    ("dashboard", "pages/5_KPI_Dashboard.py", "KPI Dashboard", ":material/monitoring:",
     "Monthly portfolio KPIs in: revenue, margin, burn, runway, with red / amber / green status."),
]


def resume_line(project: str) -> str:
    text = (ui.ROOT / "projects" / project / "README.md").read_text(encoding="utf-8")
    m = re.search(r"## Resume line\s+> (.+)", text)
    return m.group(1).strip() if m else ""


cols = st.columns(3)
for i, (key, page, name, icon, desc) in enumerate(TOOLS):
    with cols[i % 3].container(border=True, height="stretch"):
        st.markdown(f"#### {name}")
        st.write(desc)
        st.caption(f"Resume line: *{resume_line(key)}*")
        st.page_link(page, label=f"Open {name}", icon=icon)

st.divider()
st.subheader("How the numbers are verified")
st.markdown(
    """
- **Comps, live mode, tied to filings.** For AAPL, MSFT and GOOGL, every component of every LTM figure
  (fiscal year, year-to-date, prior year-to-date), plus cash and debt, matched the companies' own earnings
  releases on Oct 10, 2026, and share counts matched the market feed to the share. LTM is built as
  *prior fiscal year + year-to-date − prior year-to-date*, because companies do not tag Q4 as a
  three-month value, and the arithmetic for each number is printed next to it.
- **Stale and missing data are visible, not hidden.** A balance-sheet tag is used only if dated within 45
  days of the LTM end, a multiple on a zero or negative base shows **NM** and drops out of the median, and
  every pulled value carries its XBRL tag or fixture name.
- **LBO: Excel agrees with Python.** Tests recalculate the exported workbook in LibreOffice and assert
  Excel's `IRR()` matches the Python engine within 0.1%, for the base case and after switching the
  scenario dropdown. Sources equal uses, the debt roll-forward ties, and the sweep never over-repays.
- **Variance: sign logic and tie-outs.** Revenue over budget is favorable and expenses over budget are
  unfavorable; department rollups tie to the detail, and the bridge on the Variance page sums exactly to
  the net variance.
- **This app is a view, not a second implementation.** Every figure on these pages comes from the same
  `finlib` functions the CLIs and the test suite use, and `tests/test_app.py` checks each page's
  headline numbers against a direct `finlib` call.
"""
)
st.caption("Offline pages run on bundled fixtures labeled *Sample data (synthetic)*. The comps fixtures are "
           "illustrative, the Form D day is synthetic, and the variance and KPI files are generated. "
           "Live comps and Form D need a free SEC identity string; see the README.")
