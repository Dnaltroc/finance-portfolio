"""Equity research pitch: live data snapshot -> DCF, reverse DCF and comps -> two-page memo.

    python -m projects.pitch snapshot --name aehr     # live: SEC + market data, writes aehr_snapshot.json
    python -m projects.pitch build --name aehr        # offline: memo HTML + PDF from the snapshot
    python -m projects.pitch summary --name aehr      # print the headline numbers

The snapshot is committed so the memo is reproducible: every figure in it comes from the
snapshot plus <name>.yaml, through finlib. Nothing in the memo is typed by hand.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median

import typer
import yaml

from finlib.models.dcf import (DCFAssumptions, capm, implied_exit_multiple, implied_revenue_scale, multiple_value,
                               run_dcf, sensitivity, wacc)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOCS = ROOT / "docs" / "pitch"
app = typer.Typer(add_completion=False, help="Equity research pitch: DCF, reverse DCF, comps, memo.")

# Research-note palette (DESIGN.md)
INK, INK_2, MUTED, RULE, PAPER = "#16181D", "#4B4F58", "#8A8D94", "#E2DED4", "#FBFAF7"
NAVY, BLUE, COPPER, TEAL, LOSS = "#1C3F73", "#2A5EA8", "#D4692C", "#1B9A78", "#B42318"


def load_config(name: str) -> dict:
    return yaml.safe_load((HERE / f"{name}.yaml").read_text(encoding="utf-8"))


def snapshot_path(name: str) -> Path:
    return HERE / f"{name}_snapshot.json"


# ---------------------------------------------------------------- snapshot (live) ----
ANNUAL_FLOWS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"],
    "gross_profit": ["GrossProfit"], "operating_income": ["OperatingIncomeLoss"], "net_income": ["NetIncomeLoss"],
    "depreciation": ["DepreciationDepletionAndAmortization", "Depreciation"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"], "sbc": ["ShareBasedCompensation"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
}
ANNUAL_INSTANTS = {
    "inventory": ["InventoryNet"], "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "current_assets": ["AssetsCurrent"], "current_liabilities": ["LiabilitiesCurrent"],
}


def _annual_history(ticker: str, years: int = 6) -> dict:
    """Fiscal-year values from 10-K facts, $M, keyed by fiscal year end. Data access only."""
    from finlib.providers.edgar import COMPANYFACTS, EdgarProvider

    e = EdgarProvider()
    cik, _ = e.cik_for(ticker)
    gaap = e._get_json(COMPANYFACTS.format(cik=cik))["facts"]["us-gaap"]
    out: dict[str, dict] = {}
    sources = {}

    def rows(tag):
        return [x for x in gaap.get(tag, {}).get("units", {}).get("USD", []) if x.get("form") == "10-K"]

    for key, tags in ANNUAL_FLOWS.items():
        for tag in tags:
            vals = {x["end"]: x["val"] / 1e6 for x in rows(tag) if x.get("start")
                    and 350 <= (date.fromisoformat(x["end"]) - date.fromisoformat(x["start"])).days <= 380}
            if vals:
                sources[key] = f"xbrl:{tag}"
                for end, v in vals.items():
                    out.setdefault(end, {})[key] = v
                break
    for key, tags in ANNUAL_INSTANTS.items():
        for tag in tags:
            vals = {x["end"]: x["val"] / 1e6 for x in rows(tag) if not x.get("start")}
            if vals:
                sources[key] = f"xbrl:{tag}"
                for end, v in vals.items():
                    out.setdefault(end, {})[key] = v
                break
    keep = sorted(k for k in out if "revenue" in out[k])[-years:]
    return {"years": {k: out[k] for k in keep}, "sources": sources}


def take_snapshot(name: str) -> dict:
    import yfinance as yf

    from projects.comps.cli import load_all

    cfg = load_config(name)
    t = cfg["ticker"]
    fundamentals, quotes, rows = load_all([t] + cfg["peers"])
    comps = []
    for f, q, r in zip(fundamentals, quotes, rows):
        info = yf.Ticker(r.ticker).info
        comps.append({**asdict(r), "ltm_period": f.ltm_period, "debt_source": f.sources.get("total_debt", ""),
                      "price_source": q.source, "price_as_of": q.as_of, "beta": info.get("beta")})
    tk = yf.Ticker(t)
    info = tk.info
    hist = tk.history(period="1y", interval="1wk")["Close"]
    last = tk.history(period="5d")["Close"]
    tnx = yf.Ticker("^TNX").history(period="5d")["Close"]
    return {
        "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quote": {"price": float(last.iloc[-1]), "as_of": last.index[-1].date().isoformat(),
                  "source": "Yahoo Finance close", "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                  "fifty_two_week_high": info.get("fiftyTwoWeekHigh"), "beta": info.get("beta"),
                  "short_pct_float": info.get("shortPercentOfFloat"), "target_mean": info.get("targetMeanPrice"),
                  "analysts": info.get("numberOfAnalystOpinions")},
        "price_history": [[d.date().isoformat(), round(float(v), 2)] for d, v in hist.items()],
        "risk_free": {"value": float(tnx.iloc[-1]) / 100, "as_of": tnx.index[-1].date().isoformat(),
                      "source": "10-year US Treasury yield (^TNX, Yahoo Finance)"},
        "history": _annual_history(t),
        "comps": comps,
    }


# ---------------------------------------------------------------- numbers (offline) ---
def assumptions(cfg: dict, snap: dict, scenario: str = "base", **override) -> DCFAssumptions:
    n = compute_inputs(cfg, snap)
    s, op = cfg["scenarios"][scenario], cfg["operating"]
    a = dict(base_revenue=n["fy26_revenue"], growth=s["growth"], ebit_margin=s["ebit_margin"],
             tax_rate=op["tax_rate"], da_pct=op["da_pct"], capex_pct=op["capex_pct"], nwc_pct=op["nwc_pct"],
             wacc=n["wacc"], terminal_growth=op["terminal_growth"], net_cash=n["net_cash"],
             shares=cfg["guidance"]["diluted_shares"], mid_year=op["mid_year"], years_elapsed=n["years_elapsed"])
    a.update(override)
    return DCFAssumptions(**a)


def compute_inputs(cfg: dict, snap: dict) -> dict:
    """Market and filing inputs shared by every scenario."""
    subject = next(c for c in snap["comps"] if c["ticker"] == cfg["ticker"])
    peers = [c for c in snap["comps"] if c["ticker"] != cfg["ticker"]]
    betas = [c["beta"] for c in peers if c["beta"] is not None]
    beta = median(betas)
    rf = snap["risk_free"]["value"]
    ke = capm(rf, beta, cfg["cost_of_capital"]["equity_risk_premium"])
    w = wacc(ke, subject["market_cap_m"], 0.0, cfg["cost_of_capital"]["debt"], cfg["operating"]["tax_rate"])
    as_of = date.fromisoformat(snap["quote"]["as_of"])
    fy_end = cfg["base_fiscal_year_end"]
    fy_end = fy_end if isinstance(fy_end, date) else date.fromisoformat(str(fy_end))
    fy26 = snap["history"]["years"][max(snap["history"]["years"])]
    return {"subject": subject, "peers": peers, "beta": beta, "betas": betas, "rf": rf, "ke": ke, "wacc": w,
            "net_cash": subject["market_cap_m"] - subject["ev_m"], "fy26_revenue": fy26["revenue"],
            "years_elapsed": (as_of - fy_end).days / 365, "as_of": as_of}


def compute(cfg: dict, snap: dict) -> dict:
    """Every number the memo prints."""
    n = compute_inputs(cfg, snap)
    q = snap["quote"]
    price = q["price"]
    runs = {k: run_dcf(assumptions(cfg, snap, k)) for k in cfg["scenarios"]}
    base_a = assumptions(cfg, snap, "base")
    sens = cfg["sensitivity"]
    waccs = wacc_rows(cfg, n["wacc"])
    grid_g = sensitivity(base_a, waccs, sens["terminal_growth"])
    grid_x = [[run_dcf(base_a.model_copy(update=dict(wacc=wv, terminal_method="exit_multiple",
                                                     exit_multiple=float(x)))).per_share
               for x in sens["exit_multiple"]] for wv in waccs]
    scale = implied_revenue_scale(base_a, price)
    implied_run = None
    if scale is not None:
        g = [(1 + base_a.growth[0]) * scale - 1] + list(base_a.growth[1:])
        implied_run = run_dcf(base_a.model_copy(update=dict(growth=g)))
    exit_x = implied_exit_multiple(base_a, price)
    peer_ev_rev = [c["ev_revenue"] for c in n["peers"] if c["ev_revenue"] is not None]
    fy28_rev = runs["base"].years[1]["revenue"]
    relative = multiple_value(fy28_rev, median(peer_ev_rev), n["net_cash"], base_a.shares)
    relative_range = (multiple_value(fy28_rev, min(peer_ev_rev), n["net_cash"], base_a.shares),
                      multiple_value(fy28_rev, max(peer_ev_rev), n["net_cash"], base_a.shares))
    target = round(relative)
    fy27_mid = (cfg["guidance"]["fy27_revenue_low"] + cfg["guidance"]["fy27_revenue_high"]) / 2
    return {**n, "price": price, "runs": runs, "grid_g": grid_g, "grid_x": grid_x, "rev_scale": scale,
            "implied_run": implied_run, "implied_exit": exit_x, "peer_ev_rev": peer_ev_rev,
            "peer_ev_rev_median": median(peer_ev_rev), "fy28_revenue": fy28_rev, "relative": relative,
            "relative_range": relative_range, "target": target, "downside": target / price - 1,
            "ev_rev_fy27": n["subject"]["ev_m"] / fy27_mid, "fy27_mid": fy27_mid,
            "waccs": waccs, "exit_range": (min(grid_x[waccs.index(n["wacc"])]), max(grid_x[waccs.index(n["wacc"])])),
            "history_stats": history_stats(snap)}


def wacc_rows(cfg: dict, w: float) -> list[float]:
    """Sensitivity rows: the model's own WACC plus the configured offsets, so the base case
    sits on the grid exactly."""
    return [w + d for d in cfg["sensitivity"]["wacc_offsets"]]


def history_stats(snap: dict) -> dict:
    """Ratios from the reported history that the memo quotes next to its assumptions."""
    years = snap["history"]["years"]
    keys = sorted(years)
    label = {k: f"FY{k[2:4]}" for k in keys}
    margins = {k: years[k]["operating_income"] / years[k]["revenue"] for k in keys}
    peak = max(margins, key=margins.get)
    nwc = {k: (years[k]["current_assets"] - years[k]["cash"] - years[k]["current_liabilities"]) / years[k]["revenue"]
           for k in keys[-4:]}
    last5 = keys[-5:]
    return {"peak_margin": margins[peak], "peak_year": label[peak],
            "nwc_low": min(nwc.values()), "nwc_high": max(nwc.values()),
            "nwc_span": f"{label[keys[-4]]} to {label[keys[-1]]}",
            "capex_pct": sum(years[k]["capex"] for k in last5) / sum(years[k]["revenue"] for k in last5),
            "da_pct_last": years[keys[-1]]["depreciation"] / years[keys[-1]]["revenue"],
            "capex_span": f"{label[last5[0]]} to {label[last5[-1]]}"}


# ---------------------------------------------------------------- formatting ---------
def usd(v: float, d: int = 2) -> str:
    return f"(${abs(v):,.{d}f})" if v < 0 else f"${v:,.{d}f}"


def musd(v: float | None, d: int = 0) -> str:
    if v is None:
        return "NM"
    return f"(${abs(v):,.{d}f}M)" if v < 0 else f"${v:,.{d}f}M"


def bn_or_m(v: float) -> str:
    return f"${v / 1000:,.1f} billion" if abs(v) >= 1000 else f"${v:,.0f} million"


def pct(v: float | None, d: int = 1) -> str:
    if v is None:
        return "NM"
    return f"({abs(v):.{d}%})" if v < 0 else f"{v:.{d}%}"


def x(v: float | None, d: int = 1) -> str:
    return "NM" if v is None else f"{v:.{d}f}x"


def placeholders(cfg: dict, c: dict) -> dict:
    g, h = cfg["guidance"], c["history_stats"]
    return {
        "fy27_growth": pct(c["fy27_mid"] / c["fy26_revenue"] - 1, 0),
        "guide_low": f"${g['fy27_revenue_low']:,.0f}", "guide_high": f"${g['fy27_revenue_high']:,.0f}",
        "peak_margin": pct(h["peak_margin"]), "peak_year": h["peak_year"],
        "fy26_revenue": bn_or_m(c["fy26_revenue"]), "effective_backlog": bn_or_m(g["effective_backlog"]),
        "price": usd(c["price"]), "mcap": bn_or_m(c["subject"]["market_cap_m"]),
        "ev_rev_fy27": f"{c['ev_rev_fy27']:.0f}", "wacc": pct(c["wacc"]),
        "rev_scale": f"{c['rev_scale']:.1f}" if c["rev_scale"] else "more than 100",
        "implied_fy31_rev": bn_or_m(c["implied_run"].years[4]["revenue"]) if c["implied_run"] else "n/a",
        "implied_exit": f"{c['implied_exit']:.0f}" if c["implied_exit"] else "n/a",
        "base_value": usd(c["runs"]["base"].per_share), "bull_value": usd(c["runs"]["bull"].per_share),
        "bear_value": usd(c["runs"]["bear"].per_share), "peer_ev_rev": x(c["peer_ev_rev_median"]),
        "relative_value": usd(c["relative"]), "target": usd(c["target"], 0),
        "downside": pct(c["downside"], 0).strip("()") + (" below" if c["downside"] < 0 else " above"),
        "short_float": pct(c["quote_short"], 0) if c.get("quote_short") is not None else "n/a",
    }


# ---------------------------------------------------------------- charts -------------
def _fig_style(fig, height: int, y_title: str = "", y_format: str = ",.0f", width: int = 500):
    fig.update_layout(height=height, width=width, autosize=False, margin=dict(l=8, r=12, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="IBM Plex Sans, sans-serif", size=11, color=INK_2),
                      legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=10)), showlegend=True)
    fig.update_xaxes(showgrid=False, linecolor="#CFCABD", tickfont=dict(color=MUTED), automargin=True)
    fig.update_yaxes(gridcolor=RULE, zeroline=False, title=y_title, tickformat=y_format, tickfont=dict(color=MUTED),
                     title_font=dict(size=10, color=MUTED), automargin=True)
    return fig


def revenue_chart(cfg: dict, snap: dict, c: dict):
    import plotly.graph_objects as go

    years = snap["history"]["years"]
    hist_x = [f"FY{k[2:4]}" for k in years]
    fig = go.Figure(go.Bar(x=hist_x, y=[v["revenue"] for v in years.values()], name="Reported", marker_color=INK_2,
                           marker_cornerradius=2, hovertemplate="%{x}: $%{y:,.0f}M<extra></extra>"))
    fx = [hist_x[-1]] + cfg["forecast_labels"][:5]
    for key, color, dash in (("bull", TEAL, "solid"), ("base", BLUE, "solid"), ("bear", COPPER, "solid")):
        ys = [c["fy26_revenue"]] + [y["revenue"] for y in c["runs"][key].years[:5]]
        fig.add_trace(go.Scatter(x=fx, y=ys, name=cfg["scenarios"][key]["label"], mode="lines",
                                 line=dict(color=color, width=2, dash=dash)))
    if c["implied_run"]:
        ys = [c["fy26_revenue"]] + [y["revenue"] for y in c["implied_run"].years[:5]]
        fig.add_trace(go.Scatter(x=fx, y=ys, name=f"Implied by {usd(c['price'])}", mode="lines+markers",
                                 line=dict(color=LOSS, width=2, dash="dot"), marker=dict(size=5)))
        fig.add_annotation(x=fx[-1], y=ys[-1], text=musd(ys[-1]), showarrow=False, yshift=12,
                           font=dict(size=10, color=LOSS))
    end = c["runs"]["base"].years[4]["revenue"]
    fig.add_annotation(x=fx[-1], y=end, text=f"Base {musd(end)}", showarrow=False, yshift=-12, xanchor="right",
                       font=dict(size=10, color=BLUE))
    return _fig_style(fig, 232, "Revenue ($M)", width=505)


def football_chart(cfg: dict, snap: dict, c: dict):
    import plotly.graph_objects as go

    q = snap["quote"]
    rows = [
        ("DCF, perpetuity growth (bear to bull)", c["runs"]["bear"].per_share, c["runs"]["bull"].per_share,
         c["runs"]["base"].per_share),
        ("DCF, exit 10x to 30x EBITDA (base)", c["exit_range"][0], c["exit_range"][1], None),
        ("Peer EV / revenue on FY28 revenue", c["relative_range"][0], c["relative_range"][1], c["relative"]),
        ("52-week trading range", q["fifty_two_week_low"], q["fifty_two_week_high"], None),
    ]
    rows = list(reversed(rows))
    fig = go.Figure(go.Bar(y=[r[0] for r in rows], x=[r[2] - r[1] for r in rows], base=[r[1] for r in rows],
                           orientation="h", marker_color=[MUTED, BLUE, BLUE, BLUE], marker_cornerradius=2,
                           width=0.5, showlegend=False,
                           customdata=[[usd(r[1]), usd(r[2])] for r in rows],
                           hovertemplate="%{y}: %{customdata[0]} to %{customdata[1]}<extra></extra>",
                           text=[f"{usd(r[1], 0)} to {usd(r[2], 0)}" for r in rows], textposition="outside",
                           textfont=dict(size=10, color=INK), cliponaxis=False))
    marks = [(r[0], r[3]) for r in rows if r[3] is not None]
    fig.add_trace(go.Scatter(y=[m[0] for m in marks], x=[m[1] for m in marks], mode="markers", showlegend=False,
                             marker=dict(symbol="line-ns", size=16, line=dict(width=2.5, color=INK)),
                             hovertemplate="%{y}: $%{x:.2f}<extra></extra>"))
    fig.add_vline(x=c["price"], line=dict(color=LOSS, width=1.5, dash="dot"),
                  annotation_text=f"Price {usd(c['price'])}", annotation_position="top",
                  annotation_font=dict(size=10, color=LOSS))
    fig = _fig_style(fig, 178, "", "$,.0f", width=720)
    fig.update_xaxes(tickprefix="$", gridcolor=RULE, showgrid=True, range=[0, max(r[2] for r in rows) * 1.18])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK, size=10))
    fig.update_layout(margin=dict(t=20, r=60, b=4), bargap=0.35)
    return fig


# ---------------------------------------------------------------- memo ---------------
CSS = f"""
@page {{ size: Letter; margin: 0.45in 0.5in; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "IBM Plex Sans", sans-serif; color: {INK}; background: white; font-size: 8.9pt;
  line-height: 1.34; margin: 0; }}
.mast {{ display: flex; justify-content: space-between; align-items: baseline; border-bottom: 2px solid {INK};
  padding-bottom: 4px; font-family: "IBM Plex Mono", monospace; font-size: 8pt; letter-spacing: .08em;
  text-transform: uppercase; color: {INK_2}; }}
h1 {{ font-family: Newsreader, serif; font-weight: 500; font-size: 18pt; line-height: 1.12; margin: 8px 0 6px;
  text-wrap: balance; }}
.grid {{ display: grid; grid-template-columns: 1fr 2.05in; gap: 18px; }}
.box {{ border-top: 1.5px solid {INK}; padding-top: 4px; }}
.box h3, .ex {{ font-family: "IBM Plex Mono", monospace; font-size: 7.5pt; letter-spacing: .1em;
  text-transform: uppercase; color: {MUTED}; margin: 0 0 4px; font-weight: 500; }}
.ex b {{ color: {INK}; font-family: "IBM Plex Sans", sans-serif; letter-spacing: 0; text-transform: none;
  font-size: 9pt; font-weight: 600; margin-left: 6px; }}
.rating {{ font-family: Newsreader, serif; font-size: 20pt; color: {LOSS}; line-height: 1; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
td, th {{ padding: 2px 4px; border-bottom: 1px solid {RULE}; text-align: right; font-size: 7.8pt; }}
th {{ color: {INK_2}; font-weight: 500; }}
td:first-child, th:first-child {{ text-align: left; }}
.kv td {{ font-size: 8.2pt; }} .kv td:last-child {{ font-weight: 600; }}
ul {{ margin: 0; padding-left: 16px; }} li {{ margin-bottom: 4px; }}
p {{ margin: 0 0 6px; }}
.src {{ font-size: 7pt; color: {MUTED}; margin: 2px 0 8px; }}
.hl td {{ background: #F1F4F9; font-weight: 600; }}
.cur {{ outline: 1.5px solid {INK}; outline-offset: -1.5px; font-weight: 600; }}
.page2 {{ break-before: page; }}
.two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
.foot {{ font-size: 7.2pt; color: {MUTED}; border-top: 1px solid {RULE}; padding-top: 4px; margin-top: 8px; }}
"""


def _table(head: list[str], body: list[list[str]], classes: list[str] | None = None) -> str:
    h = "".join(f"<th>{html.escape(c)}</th>" for c in head)
    rows = []
    for i, r in enumerate(body):
        cls = f' class="{classes[i]}"' if classes and classes[i] else ""
        rows.append(f"<tr{cls}>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>")
    return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _grid_table(row_vals, col_vals, grid, row_fmt, col_fmt, cur_row, cur_col) -> str:
    h = "<th>WACC</th>" + "".join(f"<th>{col_fmt(v)}</th>" for v in col_vals)
    out = []
    for rv, row in zip(row_vals, grid):
        cells = []
        for cv, val in zip(col_vals, row):
            cls = ' class="cur"' if rv == cur_row and cv == cur_col else ""
            cells.append(f"<td{cls}>{'n/a' if val != val else usd(val)}</td>")
        out.append(f"<tr><td>{row_fmt(rv)}</td>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(out)}</tbody></table>"


def render_memo(cfg: dict, snap: dict, c: dict | None = None) -> str:
    import plotly.io as pio

    c = c or compute(cfg, snap)
    c["quote_short"] = snap["quote"].get("short_pct_float")
    ph = placeholders(cfg, c)
    q, g, memo = snap["quote"], cfg["guidance"], cfg["memo"]
    subject = c["subject"]
    base = c["runs"]["base"]
    hs = c["history_stats"]

    def para(t):
        return html.escape(" ".join(t.split()).format(**ph))

    kv = [("Price", usd(c["price"])), ("12-month target", usd(c["target"], 0)),
          ("Implied return", pct(c["downside"], 0)), ("Market cap", musd(subject["market_cap_m"])),
          ("Net cash", musd(c["net_cash"])), ("EV / FY27 guided rev", x(c["ev_rev_fy27"])),
          ("52-week range", f"{usd(q['fifty_two_week_low'], 0)} to {usd(q['fifty_two_week_high'], 0)}"),
          ("Short interest", pct(q.get("short_pct_float"), 0)),
          (f"Consensus target ({q['analysts']})", usd(q["target_mean"], 0))]
    kv_html = "<table class='kv'>" + "".join(f"<tr><td>{a}</td><td>{html.escape(b)}</td></tr>" for a, b in kv) + "</table>"

    scen_rows = []
    for key in ("bull", "base", "bear"):
        r = c["runs"][key]
        scen_rows.append([cfg["scenarios"][key]["label"], musd(r.years[0]["revenue"]), musd(r.years[4]["revenue"]),
                          pct(r.years[4]["ebit_margin"], 0), usd(r.per_share), pct(r.per_share / c["price"] - 1, 0),
                          pct(r.tv_share, 0)])
    scen_html = _table(["Case", "FY27 rev", "FY31 rev", "FY31 EBIT mgn", "Value / share", "vs price", "TV % of EV"],
                       scen_rows, ["", "hl", ""])

    fy = cfg["forecast_labels"]
    dcf_rows = [["Revenue"] + [musd(y["revenue"]) for y in base.years[:5]],
                ["Growth"] + [pct(y["growth"], 0) for y in base.years[:5]],
                ["EBIT"] + [musd(y["ebit"]) for y in base.years[:5]],
                ["EBIT margin"] + [pct(y["ebit_margin"], 0) for y in base.years[:5]],
                ["Less: change in NWC"] + [musd(-y["dnwc"]) for y in base.years[:5]],
                ["Unlevered FCF"] + [musd(y["ufcf"]) for y in base.years[:5]]]
    dcf_html = _table(["$M, base case"] + fy[:5], dcf_rows)

    sens = cfg["sensitivity"]
    cur_w = c["wacc"]
    sens_g = _grid_table(c["waccs"], sens["terminal_growth"], c["grid_g"], lambda v: pct(v), lambda v: pct(v),
                         cur_w, cfg["operating"]["terminal_growth"])
    sens_x = _grid_table(c["waccs"], sens["exit_multiple"], c["grid_x"], lambda v: pct(v), lambda v: f"{v}x",
                         cur_w, None)

    comps_rows = []
    for row in [subject] + c["peers"]:
        comps_rows.append([row["ticker"], musd(row["market_cap_m"]), musd(row["ev_m"]), musd(row["revenue_m"]),
                           x(row["ev_revenue"]), x(row["ev_ebitda"]), pct(row["ebitda_margin"], 0),
                           pct(row["revenue_growth"], 0)])
    comps_rows.append(["Peer median", "", "", "", x(c["peer_ev_rev_median"]),
                       x(median([p["ev_ebitda"] for p in c["peers"] if p["ev_ebitda"] is not None])), "", ""])
    comps_html = _table(["Ticker", "Mkt cap", "EV", "LTM rev", "EV/Rev", "EV/EBITDA", "EBITDA mgn", "Growth"],
                        comps_rows, ["hl"] + [""] * len(c["peers"]) + ["hl"])
    nm = [p["ticker"] for p in c["peers"] if p["ev_m"] is None]

    rev_fig = pio.to_html(revenue_chart(cfg, snap, c), full_html=False, include_plotlyjs="inline",
                          config={"displayModeBar": False, "staticPlot": True})
    ff_fig = pio.to_html(football_chart(cfg, snap, c), full_html=False, include_plotlyjs=False,
                         config={"displayModeBar": False, "staticPlot": True})
    sec = snap["history"]["sources"]

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(cfg['company'])}: {cfg['rating']}</title>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<div class="mast"><span>Equity research · {cfg['exchange']}: {cfg['ticker']} · Semiconductor test equipment</span><span>{c['as_of']:%B} {c['as_of'].day}, {c['as_of'].year} · {html.escape(cfg['author'])}</span></div>
<h1>{html.escape(memo['headline'])}</h1>
<div class="grid">
<div>
  <div class="box"><h3>Thesis</h3><ul>{''.join(f'<li>{para(t)}</li>' for t in memo['thesis'])}</ul></div>
  <div class="ex" style="margin-top:8px">Exhibit 1<b>Revenue: reported, our cases, and what the price implies</b></div>
  {rev_fig}
  <p class="src">Source: Aehr 10-K filings via SEC XBRL ({sec.get('revenue', '')}); FY27 cases start from guidance of {musd(g['fy27_revenue_low'])} to {musd(g['fy27_revenue_high'])} ({html.escape(g['source'])}). The dotted path is our base case scaled until the DCF equals the share price, other inputs held.</p>
</div>
<div>
  <div class="box"><h3>Rating</h3><div class="rating">{html.escape(cfg['rating'])}</div>
  <p style="margin-top:4px">Target {usd(c['target'], 0)} · {pct(c['downside'], 0)}</p></div>
  <div class="box" style="margin-top:6px"><h3>Key data</h3>{kv_html}</div>
  <p class="src">Price (close), range, short interest and consensus of {q['analysts']} analysts: Yahoo Finance, {q['as_of']}. Market cap and net cash: SEC filings at {subject['ltm_period'][-10:]}.</p>
  <div class="box"><h3>Catalysts</h3><ul>{''.join(f'<li>{para(t)}</li>' for t in memo['catalysts'])}</ul></div>
</div>
</div>
<div class="ex">Exhibit 2<b>Value per share by method against the price</b></div>
{ff_fig}
<p class="src">Source: this model. DCF at {pct(c['wacc'])} WACC ({pct(c['rf'], 2)} 10-year Treasury + peer median beta {c['beta']:.2f} x {pct(cfg['cost_of_capital']['equity_risk_premium'])} equity risk premium). Ticks mark the base case and the peer median.</p>

<div class="page2">
<div class="mast"><span>{cfg['ticker']} · Valuation detail</span><span>Page 2</span></div>
<div class="two" style="margin-top:8px">
<div>
  <div class="ex">Exhibit 3<b>Scenarios, perpetuity-growth DCF</b></div>{scen_html}
  <p class="src">{'; '.join(html.escape(cfg['scenarios'][k]['label'] + ': ' + cfg['scenarios'][k]['note'].format(**ph)) for k in ('bull', 'base', 'bear'))}.</p>
  <div class="ex">Exhibit 4<b>Base case build, first five years</b></div>{dcf_html}
  <p class="src">Tax {pct(cfg['operating']['tax_rate'], 0)}. D&amp;A {pct(cfg['operating']['da_pct'])} of revenue (FY26: {pct(hs['da_pct_last'])}); capex {pct(cfg['operating']['capex_pct'])} ({hs['capex_span']}: {pct(hs['capex_pct'])}). Working capital {pct(cfg['operating']['nwc_pct'], 0)} of new revenue, generous against Aehr's own {pct(hs['nwc_low'], 0)} to {pct(hs['nwc_high'], 0)} of revenue ({hs['nwc_span']}). Mid-year convention; valuation date {q['as_of']}.</p>
</div>
<div>
  <div class="ex">Exhibit 5<b>Value per share: WACC by terminal growth</b></div>{sens_g}
  <p class="src">Outlined: our base case. Terminal value is {pct(base.tv_share, 0)} of enterprise value and implies {x(base.implied_exit_ev_ebitda)} FY36 EBITDA.</p>
  <div class="ex">Exhibit 6<b>Value per share: WACC by exit EV / EBITDA</b></div>{sens_x}
  <p class="src">Exit multiple on FY36 EBITDA, discounted from the end of FY36. The price implies {x(c['implied_exit'], 0)}.</p>
</div>
</div>
<div class="ex" style="margin-top:4px">Exhibit 7<b>Trading comparables, last twelve months</b></div>{comps_html}
<p class="src">Source: SEC companyfacts (XBRL) through each company's latest 10-Q or 10-K, prices from Yahoo Finance on {q['as_of']}, built with this repo's comps engine. NM = not meaningful (negative or missing base){(': ' + ', '.join(nm) + ' carry convertible notes or other borrowings the engine does not sum, so EV is left blank rather than understated') if nm else ''}. Relative value applies the peer median EV / revenue to base-case FY28 revenue of {musd(c['fy28_revenue'])}.</p>
<div class="two">
<div><div class="box"><h3>What would make us wrong</h3><ul>{''.join(f'<li>{para(t)}</li>' for t in memo['risks'])}</ul></div></div>
<div><div class="box"><h3>Method</h3><p>Two lenses. Intrinsic: a ten-year DCF of unlevered free cash flow with a perpetuity-growth terminal value, cross-checked with exit multiples. Relative: what peers pay per dollar of revenue, applied a year past guidance. Then a reverse DCF: hold the price fixed and solve for the revenue path or exit multiple it requires.</p>
<p>Every figure is computed from a dated snapshot by the open-source engine at github.com/Dnaltroc/finance-portfolio (projects/pitch). Change an assumption in aehr.yaml and the memo rebuilds.</p></div></div>
</div>
<p class="foot">Student research, prepared for a portfolio. Not investment advice. Data as of {q['as_of']}; snapshot pulled {snap['pulled_at']}. Fiscal years end in late May. $ in millions except per share.</p>
</div>
</body></html>"""


def write_pdf(html_path: Path, pdf_path: Path) -> Path:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(html_path.resolve().as_uri())
        pg.wait_for_load_state("networkidle")
        pg.evaluate("document.fonts.ready")
        pg.pdf(path=str(pdf_path), format="Letter", print_background=True, prefer_css_page_size=True)
        b.close()
    return pdf_path


# ---------------------------------------------------------------- commands -----------
@app.command()
def snapshot(name: str = typer.Option("aehr")):
    """Pull live SEC and market data into <name>_snapshot.json (needs SEC_IDENTITY)."""
    from finlib import offline

    if offline():
        raise typer.BadParameter("snapshot pulls live data; unset FINLIB_OFFLINE")
    snap = take_snapshot(name)
    snapshot_path(name).write_text(json.dumps(snap, indent=1), encoding="utf-8")
    typer.echo(f"Wrote {snapshot_path(name)} (price {snap['quote']['price']:.2f} as of {snap['quote']['as_of']})")


@app.command()
def build(name: str = typer.Option("aehr"), out: Path = typer.Option(DOCS), pdf: bool = typer.Option(True)):
    """Render the memo from the snapshot: HTML always, PDF when Playwright is installed."""
    cfg, snap = load_config(name), json.loads(snapshot_path(name).read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / f"{name}_memo.html"
    html_path.write_text(render_memo(cfg, snap), encoding="utf-8")
    typer.echo(f"Wrote {html_path}")
    if pdf:
        typer.echo(f"Wrote {write_pdf(html_path, out / f'{name}_memo.pdf')}")


@app.command()
def summary(name: str = typer.Option("aehr")):
    cfg, snap = load_config(name), json.loads(snapshot_path(name).read_text(encoding="utf-8"))
    c = compute(cfg, snap)
    typer.echo(f"{cfg['ticker']} {usd(c['price'])} as of {snap['quote']['as_of']}  WACC {pct(c['wacc'])}")
    for k, r in c["runs"].items():
        typer.echo(f"  DCF {k:5} {usd(r.per_share)}  TV {pct(r.tv_share, 0)} of EV")
    typer.echo(f"  Relative {usd(c['relative'])}  target {usd(c['target'], 0)} ({pct(c['downside'], 0)})")
    typer.echo(f"  Price implies revenue x{c['rev_scale']:.2f} of base, or {x(c['implied_exit'])} FY36 EBITDA")
