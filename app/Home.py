"""finance-portfolio: six finance tools on one verified Python library.

Run:  streamlit run app/Home.py
"""

from __future__ import annotations

import streamlit as st

import fp_ui  # noqa: F401  (puts the repo root on sys.path for every page)

st.set_page_config(page_title="finance-portfolio", page_icon=":material/request_quote:", layout="wide")

PAGES = [
    st.Page("pages/0_Overview.py", title="Overview", default=True),
    st.Page("pages/1_Comps_Builder.py", title="Comps", url_path="comps"),
    st.Page("pages/2_Variance_Report.py", title="Variance", url_path="variance"),
    st.Page("pages/3_LBO_Engine.py", title="LBO", url_path="lbo"),
    st.Page("pages/4_Deal_Sourcing.py", title="Deal sourcing", url_path="sourcing"),
    st.Page("pages/5_KPI_Dashboard.py", title="Portfolio KPIs", url_path="kpis"),
    st.Page("pages/6_Filing_Tone.py", title="Filing tone", url_path="tone"),
    st.Page("pages/7_AEHR_Live.py", title="AEHR live", url_path="live"),
]

st.navigation(PAGES, position="top").run()
