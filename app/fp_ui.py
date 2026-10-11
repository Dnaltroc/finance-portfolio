"""Shared view helpers for the Streamlit app: formatting, mode switching, chart styling.

Nothing here computes a financial figure. Every number shown comes from finlib or the
project builders; this module only formats it, labels it, and packages downloads.
"""

from __future__ import annotations

import os
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
# Reference data-viz palette: blue is the primary series; categorical slots 2-3 only
# when a chart needs more than one series; status colors are reserved for state.
BLUE = "#2a78d6"
BLUE_SOFT = "rgba(42,120,214,0.12)"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SERIES_EXTRA = ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # slots 4-8, in order
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
WARNING_TEXT = "#c98500"  # amber that clears contrast on white for text
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SAMPLE_LABEL = "Sample data (synthetic)"


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


# ---------------------------------------------------------------- page chrome --------
def page_setup(title: str, one_liner: str, icon: str = ":material/insights:") -> None:
    st.set_page_config(page_title=f"{title} | finance-portfolio", page_icon=icon, layout="wide")
    load_secrets()
    st.title(title)
    st.caption(one_liner)


def sample_badge(note: str = "") -> None:
    st.info(f"**{SAMPLE_LABEL}.** {note}".strip(), icon=":material/science:")


def live_badge(note: str) -> None:
    st.success(f"**Live data.** {note}", icon=":material/cloud_done:")


def fu_pill(fu: str) -> str:
    """Markdown badge for a favorable / unfavorable judgment (status color + label)."""
    if fu == "F":
        return ":green-badge[:material/check: Favorable]"
    if fu == "U":
        return ":red-badge[:material/close: Unfavorable]"
    return ":gray-badge[On budget]"


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
def style(fig: go.Figure, height: int = 340, y_title: str | None = None,
          y_format: str | None = None, legend: bool = False) -> go.Figure:
    """House chart style: one y-axis, recessive grid, hover on, system sans."""
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=40, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK_2),
        hoverlabel=dict(font_size=12), showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, title=None),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, ticks="")
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=AXIS, title=y_title,
                     tickformat=y_format)
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)
