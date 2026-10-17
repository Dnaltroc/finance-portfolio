"""AEHR live: the stock, the headlines and the SEC filings since the memo was priced.

A view only. Every number comes from projects/pitch/tracker.build_tracker (live feeds) and the
moves from finlib/models/tracker. The page re-runs itself on a timer; the cache below decides
how often the network is actually asked.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fp_ui as ui
from projects.pitch.tracker import build_tracker, memo_numbers

REFRESH_SECONDS = 60
INSIDER_FORMS = {"3", "4", "5", "144", "3/A", "4/A", "5/A", "144/A"}

ui.page_setup("AEHR since the memo, live",
              "The equity memo rated Aehr Test Systems a Sell with a $52 target. This page follows what the stock, "
              "the headlines and the company's SEC filings have done since, and refreshes itself every minute.",
              number="07", section="AEHR live")

default_mode = "Offline" if os.getenv("FINLIB_OFFLINE") == "1" else "Live"
mode = st.sidebar.segmented_control("Data source", ["Live", "Offline"], default=default_mode, key="live_mode",
                                    help="Live reads Yahoo Finance, Yahoo and Google News RSS, and SEC EDGAR. "
                                         "Offline reads a synthetic fixture.")
offline = mode != "Live"
auto = st.sidebar.toggle(f"Refresh every {REFRESH_SECONDS} s", value=not offline, key="live_auto")


@st.cache_data(show_spinner=False)
def _memo() -> dict:
    return memo_numbers("aehr")


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner="Fetching live data…")
def _tracker(offline: bool) -> dict:
    with ui.finlib_mode(offline):
        return build_tracker("aehr", memo=_memo(), filings_limit=100)


def _money(v: float) -> str:
    return ui._paren(f"${abs(v):,.2f}", v < 0)


def _when(iso: str) -> str:
    if not iso:
        return ""
    d = datetime.fromisoformat(iso)
    return f"{d:%b} {d.day}, {d:%H:%M} UTC"


def _age(iso: str, now: datetime) -> str:
    if not iso:
        return ""
    mins = int((now - datetime.fromisoformat(iso)).total_seconds() // 60)
    if mins < 60:
        return f"{max(mins, 0)}m ago"
    return f"{mins // 60}h ago" if mins < 48 * 60 else f"{mins // 1440}d ago"


@st.fragment(run_every=REFRESH_SECONDS if auto else None)
def live_view() -> None:
    t = _tracker(offline)
    memo, mk, mv = t["memo"], t["market"], t["moves"]
    q = mk["quote"]
    any_sample = mk["sample"] or t["news"]["sample"] or t["filings"]["sample"]

    last = datetime.fromisoformat(q["as_of"])
    fetched = datetime.fromisoformat(t["fetched_at"])
    if any_sample:
        badge = f":gray-badge[:material/science: {ui.SAMPLE_LABEL}]"
    else:
        badge = ":gray-badge[:material/cloud_done: Live · Yahoo Finance, news RSS, SEC EDGAR]"
    st.markdown(f"{badge}&nbsp; <small>Last trade {last:%I:%M %p} exchange time, {last:%b} {last.day}. "
                f"Fetched {fetched:%H:%M:%S} UTC"
                f"{f', next refresh within {REFRESH_SECONDS} s' if auto else ''}.</small>",
                unsafe_allow_html=True)
    for name, sec in (("Prices", mk), ("News", t["news"]), ("Filings", t["filings"])):
        if sec["error"]:
            st.warning(f"{name} feed unavailable ({ui.escape_text(sec['error'])}). "
                       f"Showing {ui.SAMPLE_LABEL.lower()} for that section.", icon=":material/cloud_off:")

    # ------------------------------------------------------------ KPI strip ----------
    with st.container(horizontal=True):
        st.metric("Last price", _money(q["price"]),
                  # Streamlit colors the delta by its leading sign, so this one uses a minus, not parentheses.
                  None if mv["day_change"] is None else f"{mv['day_change']:+,.2f} ({mv['day_pct']:+.1%}) today",
                  border=True, height="stretch")
        st.metric("Since the memo", ui.fmt_pct(mv["since_memo_pct"]),
                  f"memo priced at {_money(memo['price'])} on {memo['as_of']}",
                  delta_color="off", delta_arrow="off", border=True, height="stretch")
        st.metric(f"To the ${memo['target']} target", ui.fmt_pct(mv["to_target_pct"]),
                  "move still needed from today's price", delta_color="off", delta_arrow="off", border=True,
                  height="stretch")
        st.metric("Day range", f"{q['day_low']:,.2f} to {q['day_high']:,.2f}",
                  f"volume {q['volume'] / 1e6:,.1f}M shares", delta_color="off", delta_arrow="off", border=True,
                  height="stretch")

    src_px = "synthetic fixture" if mk["sample"] else "Yahoo Finance via yfinance"
    left, right = st.columns(2, gap="large")
    with left:
        ui.exhibit(1, "Latest session, 5-minute closes")
        xs = [datetime.fromisoformat(x) for x, _ in mk["intraday"]]
        fig = go.Figure(go.Scatter(x=xs, y=[v for _, v in mk["intraday"]], mode="lines",
                                   line=dict(color=ui.BLUE, width=2),
                                   hovertemplate="%{x|%H:%M}  $%{y:.2f}<extra></extra>"))
        fig.add_hline(y=q["previous_close"], line=dict(color=ui.MUTED, width=1, dash="dot"),
                      annotation_text=f"prior close {_money(q['previous_close'])}", annotation_position="top left",
                      annotation_font=dict(color=ui.MUTED, size=11))
        fig = ui.style(fig, height=300, y_title="$ per share", y_format="$,.0f")
        fig.update_yaxes(zeroline=False)
        fig.update_xaxes(tickformat="%H:%M")
        ui.show(fig, key="live_intraday")
        ui.source(f"{src_px}; exchange time. Dotted line: prior close.")
    with right:
        ui.exhibit(2, "One year of daily closes against the memo")
        xs = [x for x, _ in mk["daily"]]
        fig = go.Figure(go.Scatter(x=xs, y=[v for _, v in mk["daily"]], mode="lines",
                                   line=dict(color=ui.BLUE, width=2),
                                   hovertemplate="%{x|%b %d, %Y}  $%{y:.2f}<extra></extra>"))
        fig.add_hline(y=memo["price"], line=dict(color=ui.SERIES[1], width=1.25, dash="dot"),
                      annotation_text=f"memo price {_money(memo['price'])}", annotation_position="top left",
                      annotation_font=dict(color=ui.SERIES[1], size=11))
        fig.add_hline(y=memo["target"], line=dict(color=ui.INK_2, width=1.25, dash="dash"),
                      annotation_text=f"target ${memo['target']}", annotation_position="bottom left",
                      annotation_font=dict(color=ui.INK_2, size=11))
        fig = ui.style(fig, height=300, y_title="$ per share", y_format="$,.0f")
        fig.update_yaxes(zeroline=False, rangemode="tozero")
        ui.show(fig, key="live_daily")
        ui.source(f"{src_px}. Memo price and target from projects/pitch (memo dated {memo['as_of']}).")

    # ------------------------------------------------------------ news ---------------
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.exhibit(3, "Latest headlines")
        now = datetime.now(timezone.utc)
        items = t["news"]["items"]
        if not items:
            st.caption("No headlines returned.")
        for n in items:
            title = ui.escape_text(n["title"])
            link = f"[{title}]({n['url']})" if n["url"].startswith(("https://", "http://")) else title
            st.markdown(f"{link}  \n<small>{ui.escape_text(n['source'])} · {_when(n['published'])} · "
                        f"{_age(n['published'], now)}</small>", unsafe_allow_html=True)
        ui.source("synthetic fixture" if t["news"]["sample"] else
                  "Yahoo Finance headline RSS for AEHR and a Google News search for \"Aehr Test Systems\", "
                  "de-duplicated, newest first. Headlines are third-party and unverified.")

    # ------------------------------------------------------------ filings ------------
    with right:
        ui.exhibit(4, "SEC filings, newest first")
        kind = st.segmented_control("Show", ["All", "Company", "Insider"], default="Company", key="live_forms",
                                    help="Insider: Forms 3, 4, 5 and 144. Company: everything else.")
        fl = t["filings"]["items"]
        if kind == "Company":
            fl = [f for f in fl if f["form"] not in INSIDER_FORMS]
        elif kind == "Insider":
            fl = [f for f in fl if f["form"] in INSIDER_FORMS]
        insiders_30d = sum(1 for f in t["filings"]["items"] if f["form"] in INSIDER_FORMS
                           and (now.date() - datetime.fromisoformat(f["filed"]).date()).days <= 30)
        if fl:
            st.dataframe(pd.DataFrame([{"Filed": f["filed"], "What it is": f["description"], "Link": f["url"]}
                                       for f in fl[:15]]),
                         hide_index=True, width="stretch",
                         column_config={"What it is": st.column_config.TextColumn(width="medium"),
                                        "Link": st.column_config.LinkColumn("Link", display_text="open",
                                                                            width="small")})
        else:
            st.caption("None in the latest filings.")
        st.caption(f"{insiders_30d} insider filings in the last 30 days.")
        ui.source("synthetic fixture" if t["filings"]["sample"] else
                  "SEC EDGAR submissions API, latest 100 filings.")


live_view()
ui.source("Student project, not investment advice. Prices can be delayed by the data source.")
