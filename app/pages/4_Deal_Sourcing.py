"""Deal Sourcing: one day of SEC Form D filings -> fund filter -> scored, ranked list."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fp_ui as ui
from finlib.models.sourcing import ScoringConfig, digest_markdown, funnel, rank
from finlib.providers.formd import FIXTURE_DIR, FormDProvider
from finlib.store.sourcing_db import SourcingDB
from projects.sourcing.cli import HERE, load_config

ui.page_setup("Deal Sourcing",
              "Every private raise under Reg D files a Form D with the SEC. This reads one day of filings, "
              "drops the funds, and scores the operating-company raises against your targets.",
              ":material/travel_explore:")

BASE_CFG, BASE_TOP_N = load_config(HERE / "config.yaml")
FIXTURE_DAYS = sorted(date.fromisoformat(p.name) for p in FIXTURE_DIR.iterdir() if p.is_dir())
LOCAL_DB = Path("sourcing.sqlite")

# ---------------------------------------------------------------- inputs ------------
sb = st.sidebar
sb.header("Filings")
offline = ui.mode_picker("sourcing_mode", "Live reads the SEC daily index (a few hundred requests, "
                                          "about a minute per day).")
if offline:
    day = sb.selectbox("Day", FIXTURE_DAYS, index=len(FIXTURE_DAYS) - 1, format_func=date.isoformat,
                       help="Offline mode has one synthetic fixture day.")
else:
    day = sb.date_input("Day", date.today() - timedelta(days=1), max_value=date.today(),
                        help="Weekends and SEC holidays have no index.")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch(day_iso: str, offline: bool):
    with ui.finlib_mode(offline):
        return FormDProvider().filings_for(date.fromisoformat(day_iso))


with st.spinner("Reading the fixture day..." if offline else f"Fetching Form D filings for {day}..."):
    try:
        filings = fetch(day.isoformat(), offline)
    except Exception as e:
        st.error(f"Could not load filings for {day}: {e}")
        st.stop()
if not filings:
    st.warning(f"No Form D index for {day}. That usually means a weekend or SEC holiday; pick another day.")
    st.stop()

states = sorted({f.state for f in filings if f.state} | set(BASE_CFG.target_states))
industries = sorted({f.industry_group for f in filings if f.industry_group} | set(BASE_CFG.target_industries))
sb.header("Targets")
target_states = sb.multiselect("Target states", states, default=BASE_CFG.target_states)
target_industries = sb.multiselect("Target industries", industries, default=BASE_CFG.target_industries)
min_amount = sb.number_input("Minimum amount sold ($)", 0, 100_000_000, int(BASE_CFG.min_amount_sold), 50_000)
include_amend = sb.toggle("Include amendments (D/A)", BASE_CFG.include_amendments)
sb.header("Score weights")
sb.caption("Re-scores instantly; nothing is re-fetched.")
w = BASE_CFG.weights
weights = {
    "amount": sb.slider("Amount sold (log-scaled)", 0, 100, w["amount"]),
    "investors": sb.slider("Investor count (full at 25+)", 0, 100, w["investors"]),
    "first_time": sb.slider("First-time filer", 0, 100, w["first_time"]),
    "state": sb.slider("Target state", 0, 100, w["state"]),
    "industry": sb.slider("Target industry", 0, 100, w["industry"]),
}
top_n = sb.slider("Rows in the digest", 5, 100, BASE_TOP_N, 5)
if sum(weights.values()) != 100:
    sb.caption(f"Weights sum to {sum(weights.values())}, so the top score is {sum(weights.values())}, not 100.")

cfg = ScoringConfig(target_states=target_states, target_industries=target_industries,
                    min_amount_sold=float(min_amount), weights=weights,
                    amount_log_ceiling=BASE_CFG.amount_log_ceiling, include_amendments=include_amend)
# First-time = CIK not in the local database before this day. Read-only here: the page never writes it.
seen = SourcingDB(LOCAL_DB).seen_ciks(before=day.isoformat()) if LOCAL_DB.exists() else set()
rows = rank(filings, cfg, seen)
kept = [r for r in rows if r["kept"]]

if offline:
    ui.sample_badge("36 synthetic filings shaped like real `edgarSubmission` XML; links point at "
                    "accession numbers that do not exist on EDGAR.")
else:
    ui.live_badge(f"SEC EDGAR daily index for {day}; amounts are as reported by each issuer.")
if not LOCAL_DB.exists():
    st.caption("First-time filer = CIK not in a local `sourcing.sqlite`. None exists here, so every issuer "
               "counts as first-time.")

# ---------------------------------------------------------------- headline ----------
m = st.columns(3)
m[0].metric("Form D filings", len(rows))
m[1].metric("Operating-company raises kept", len(kept))
m[2].metric("Top score", f"{kept[0]['score']:.0f}" if kept else "–")

left, right = st.columns(2)
with left:
    stages = funnel(rows)
    fig = go.Figure(go.Funnel(y=[s for s, _ in stages], x=[n for _, n in stages], marker_color=ui.BLUE,
                              textinfo="value+percent initial", connector=dict(fillcolor=ui.BLUE_SOFT),
                              hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_layout(title=dict(text="Filter funnel", font_size=14))
    ui.show(ui.style(fig, height=320))
with right:
    by_state = pd.Series([r["state"] or "?" for r in kept]).value_counts()
    fig = go.Figure(go.Bar(x=by_state.index, y=by_state.values, marker_color=ui.BLUE, marker_cornerradius=4,
                           text=by_state.values, textposition="outside",
                           hovertemplate="%{x}: %{y} kept<extra></extra>"))
    fig.update_layout(title=dict(text="Kept raises by issuer state", font_size=14), bargap=0.4)
    ui.show(ui.style(fig, height=320, y_title="Filings"))

# ---------------------------------------------------------------- ranked table ------
st.subheader(f"Ranked raises (top {min(top_n, len(kept))} of {len(kept)})")
table = pd.DataFrame([{
    "#": i, "Score": round(r["score"]), "Issuer": r["issuer"], "State": r["state"],
    "Industry": r["industry_group"], "Sold": ui.fmt_usd(r["total_sold"]) if r["total_sold"] else "–",
    "Offering": ui.fmt_usd(r["total_offering"]) if r["total_offering"] else "indefinite",
    "Investors": r["investors"], "First-time": "yes" if r["first_time"] else "", "Filing": r["url"],
} for i, r in enumerate(kept[:top_n], start=1)])
if table.empty:
    st.info("No filings pass the current filters.")
else:
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=max(sum(weights.values()), 1),
                                                 format="%d"),
        "Filing": st.column_config.LinkColumn("SEC filing", display_text="primary_doc.xml"),
    })

    st.markdown("**Score breakdown**")
    labels = {"amount": "Amount", "investors": "Investors", "first_time": "First-time", "state": "State",
              "industry": "Industry"}
    for i, r in enumerate(kept[:top_n], start=1):
        with st.expander(f"{i}. {r['issuer']}: {r['score']:.0f} points"):
            parts = pd.DataFrame([{"Component": labels[k], "Points": round(v, 1), "Max": weights[k]}
                                  for k, v in r["score_parts"].items()])
            st.dataframe(parts, hide_index=True)
            st.caption(f"{r['entity_type']}, {r['state']}, incorporated {r['year_of_inc'] or 'n/a'}; "
                       f"exemption {r['exemption'] or 'n/a'}; first sale {r['date_of_first_sale'] or 'n/a'}.")

with st.expander(f"Dropped filings ({len(rows) - len(kept)})"):
    st.dataframe(pd.DataFrame([{"Issuer": r["issuer"], "Reason": r["drop_reason"], "Industry": r["industry_group"],
                                "Entity": r["entity_type"], "Sold": ui.fmt_usd(r["total_sold"])
                                if r["total_sold"] else "–"} for r in rows if not r["kept"]]),
                 hide_index=True, width="stretch")

md = digest_markdown(rows, day.isoformat(), top_n, cfg)
st.download_button("Download digest (.md)", md.encode("utf-8"), file_name=f"digest_{day.isoformat()}.md",
                   mime="text/markdown", icon=":material/download:", type="primary")
st.caption("Same digest as `python -m projects.sourcing run`, scored with the targets and weights above. "
           "Form D carries no business description; industry is the issuer's own coarse category.")
