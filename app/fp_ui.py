"""Shared view helpers for the Streamlit app: page anatomy, formatting, mode switching, charts.

Nothing here computes a financial figure. Every number shown comes from finlib or the
project builders; this module only formats it, labels it, and packages downloads.
Visual tokens mirror DESIGN.md and .streamlit/config.toml.
"""

from __future__ import annotations

import html
import os
import re
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

# ---------------------------------------------------------------- palette ------------
# Paper and ink, one blue. Series 2-3 only when a chart needs more than one series
# (validated on the paper surface; see DESIGN.md). Status colors mean gain / loss only.
PAPER, PAPER_2, RULE = "#FBFAF7", "#F3F1EB", "#E2DED4"
INK, INK_2, MUTED = "#16181D", "#4B4F58", "#8A8D94"
NAVY = "#1C3F73"
BLUE = "#2A5EA8"
BLUE_SOFT = "rgba(42,94,168,0.10)"
SERIES = ["#2A5EA8", "#D4692C", "#1B9A78"]
SERIES_EXTRA = ["#8A8D94", "#6B4E9B", "#9C6B1F", "#3C7A89", "#A23B56"]  # rarely needed
GOOD, CRITICAL = "#1A7F4F", "#B42318"
WARNING, WARNING_TEXT = "#F5B544", "#B54708"
GRID, AXIS = RULE, "#CFCABD"
SEQ_BLUE = ["#F1F4F9", "#DCE5F2", "#C3D3EA", "#A6BEE0", "#86A5D3", "#678CC5", "#4B74B5"]
FONT = '"IBM Plex Sans", system-ui, sans-serif'
MONO = '"IBM Plex Mono", ui-monospace, monospace'

SAMPLE_LABEL = "Sample data (synthetic)"

_CSS = f"""
<style>
.fp-eyebrow {{ font-family: {MONO}; font-size: 11px; letter-spacing: .12em; text-transform: uppercase;
  color: {MUTED}; margin: 0 0 -6px 0; }}
.fp-eyebrow b {{ color: {NAVY}; font-weight: 500; }}
.fp-dek {{ font-size: 17px; line-height: 1.5; color: {INK_2}; max-width: 62rem; margin: -4px 0 4px 0; }}
.fp-exhibit {{ border-top: 1.5px solid {INK}; padding-top: 8px; margin-top: 6px; font-weight: 600;
  font-size: 14px; color: {INK}; }}
.fp-exhibit span {{ font-family: {MONO}; font-weight: 500; font-size: 11px; letter-spacing: .08em;
  text-transform: uppercase; color: {MUTED}; margin-right: 10px; }}
.fp-rule {{ border: 0; border-top: 1px solid {RULE}; margin: 4px 0; }}
.fp-num {{ font-family: {MONO}; font-size: 13px; color: {MUTED}; padding-top: 6px; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
[data-testid="stMetricLabel"] p {{ font-size: 12px; color: {INK_2}; }}
</style>
"""


# ---------------------------------------------------------------- formatting ---------
# Same conventions as finlib/excel_writer/styles.py: negatives in parentheses, NM where
# the engine returns None.
def _paren(s: str, neg: bool) -> str:
    return f"({s})" if neg else s


def fmt_musd(v: float | None, decimals: int = 1) -> str:
    """Value already in $ millions -> "$1,234.5M"; negatives "($12.3M)"."""
    if v is None or v != v:
        return "NM"
    return _paren(f"${abs(v):,.{decimals}f}M", v < 0)


def fmt_usd(v: float | None) -> str:
    """Whole dollars -> "$1,234"; negatives "($1,234)"."""
    if v is None or v != v:
        return "NM"
    return _paren(f"${abs(v):,.0f}", v < 0)


def fmt_x(v: float | None, decimals: int = 1) -> str:
    if v is None or v != v:
        return "NM"
    return _paren(f"{abs(v):.{decimals}f}x", v < 0)


def fmt_pct(v: float | None, decimals: int = 1) -> str:
    if v is None or v != v:
        return "NM"
    return _paren(f"{abs(v):.{decimals}%}", v < 0)


# ---------------------------------------------------------------- page anatomy -------
def page_setup(title: str, dek: str, number: str = "", section: str = "") -> None:
    """Eyebrow, serif title and one-line dek: the masthead of every page."""
    load_secrets()
    st.html(_CSS)
    if number or section:
        num = f"<b>{html.escape(number)}</b> · " if number else ""
        st.html(f'<p class="fp-eyebrow">{num}{html.escape(section)}</p>')
    st.title(title)
    st.html(f'<p class="fp-dek">{html.escape(dek)}</p>')


def provenance(sample: bool, note: str) -> None:
    """One quiet line saying where the numbers on this page came from. Always a gray badge:
    provenance is not a status, so it never borrows gain / loss colors."""
    badge = (f":gray-badge[:material/science: {SAMPLE_LABEL}]" if sample
             else ":gray-badge[:material/cloud_done: Live · SEC EDGAR]")
    st.markdown(f"{badge}&nbsp; <small>{note}</small>", unsafe_allow_html=True)


_MD_SPECIALS = re.compile(r"([\\`*_{}\[\]()#+\-.!|~:$])")


def escape_text(text: str) -> str:
    """User-supplied text (an upload's file name) made inert for st.markdown with HTML on:
    markdown punctuation backslash-escaped, then <, > and & as entities."""
    return html.escape(_MD_SPECIALS.sub(lambda m: "\\" + m.group(1), text), quote=False)


def upload_badge(names: list[str], note: str) -> None:
    """Provenance line for data the user uploaded; file names are escaped."""
    files = " and ".join(escape_text(n) for n in names)
    st.markdown(f":gray-badge[:material/upload_file: Your data]&nbsp; <small>{files}, {note}</small>",
                unsafe_allow_html=True)


def sample_badge(note: str = "") -> None:
    provenance(True, note)


def live_badge(note: str) -> None:
    provenance(False, note)


def exhibit(n: int, title: str) -> None:
    """Research-note exhibit label: a heavy rule, a mono number, a sentence-case title."""
    st.html(f'<div class="fp-exhibit"><span>Exhibit {n}</span>{html.escape(title)}</div>')


def source(text: str) -> None:
    st.caption(f"Source: {text}")


def rule() -> None:
    st.html('<hr class="fp-rule">')


# ---------------------------------------------------------------- secrets / mode -----
def load_secrets() -> None:
    """Copy SEC_IDENTITY / FINNHUB_KEY from st.secrets or .env into the environment,
    where the finlib providers read them. Never overwrites a value already set."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    for key in ("SEC_IDENTITY", "FINNHUB_KEY"):
        if os.getenv(key):
            continue
        try:
            val = st.secrets.get(key)
        except Exception:  # no secrets.toml at all
            val = None
        if val:
            os.environ[key] = str(val)


def has_sec_identity() -> bool:
    return bool(os.getenv("SEC_IDENTITY", "").strip())


_MODE_LOCK = threading.Lock()


@contextmanager
def finlib_mode(offline: bool):
    """Run a block with FINLIB_OFFLINE set, then restore it.

    finlib reads the flag from the process environment on every call, and Streamlit
    serves sessions on threads of one process, so the lock keeps one session's Live
    run from reading another session's Offline flag (which would mislabel data).
    """
    with _MODE_LOCK:
        prev = os.environ.get("FINLIB_OFFLINE")
        os.environ["FINLIB_OFFLINE"] = "1" if offline else "0"
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop("FINLIB_OFFLINE", None)
            else:
                os.environ["FINLIB_OFFLINE"] = prev


def mode_picker(key: str, live_help: str) -> bool:
    """Sidebar Live/Offline toggle. Returns True for offline. Live needs SEC_IDENTITY."""
    choice = st.sidebar.segmented_control(
        "Data source", ["Offline", "Live"], default="Offline", key=key,
        help="Offline reads the bundled fixtures. " + live_help,
    )
    if choice == "Live" and not has_sec_identity():
        st.sidebar.warning(
            "Live mode needs `SEC_IDENTITY` (\"Your Name you@example.com\") in `.env` locally "
            "or in the app's Secrets on Streamlit Cloud. Showing sample data instead.",
            icon=":material/key_off:",
        )
        return True
    return choice != "Live"


# ---------------------------------------------------------------- downloads ----------
def build_file(writer, *args, **kwargs) -> tuple[bytes, str]:
    """Run a finlib/project writer that saves into a directory; return (bytes, filename)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(writer(*args, Path(tmp), **kwargs))
        return path.read_bytes(), path.name


XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ---------------------------------------------------------------- charts -------------
def style(fig: go.Figure, height: int = 320, y_title: str | None = None,
          y_format: str | None = None, legend: bool = False) -> go.Figure:
    """House chart style: one y-axis, hairline grid, Plex Sans, no in-plot title (the page
    carries the exhibit label), hover on."""
    fig.update_layout(
        height=height, margin=dict(l=4, r=12, t=12, b=4), title=None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK_2),
        hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor="white", bordercolor=RULE),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, title=None, font=dict(size=12)),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS, ticklen=3,
                     tickfont=dict(color=MUTED), automargin=True, title_standoff=8, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=True, zerolinecolor=AXIS, title=y_title,
                     tickformat=y_format, tickfont=dict(color=MUTED), title_font=dict(color=MUTED, size=11),
                     automargin=True, title_standoff=8)
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key, theme=None)
