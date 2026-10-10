"""Comps Builder CLI.

    python -m projects.comps run --tickers AAPL,MSFT,GOOGL --peerset megacap
    FINLIB_OFFLINE=1 python -m projects.comps run --tickers AAPL,MSFT,GOOGL   # fixtures
"""

from __future__ import annotations

from pathlib import Path

import typer

from finlib import offline
from finlib.excel_writer import write_comps_workbook
from finlib.models.ratios import build_comps_row, summary_rows
from finlib.providers import EdgarProvider, MarketProvider

app = typer.Typer(add_completion=False, help="Build a comparable-company table in Excel.")


def run_comps(tickers: list[str], peerset: str, out_dir: Path) -> Path:
    edgar, market = EdgarProvider(), MarketProvider()
    fundamentals, quotes, rows = [], [], []
    for t in tickers:
        f = edgar.fundamentals(t)
        q = market.quote(t)
        fundamentals.append(f)
        quotes.append(q)
        rows.append(build_comps_row(f, q))
    return write_comps_workbook(rows, fundamentals, quotes, peerset, out_dir)


@app.command()
def run(
    tickers: str = typer.Option(..., help="Comma-separated tickers, e.g. AAPL,MSFT"),
    peerset: str = typer.Option("peerset", help="Label used in the file name and title"),
    out: Path = typer.Option(Path("output"), help="Output directory"),
):
    """Pull financials and prices, compute multiples, write the workbook."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    tlist = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    mode = "offline (fixtures)" if offline() else "live"
    typer.echo(f"Comps for {', '.join(tlist)} [{mode}]")
    path = run_comps(tlist, peerset, out)
    typer.echo(f"Wrote {path}")


@app.command()
def show(tickers: str = typer.Option(...)):
    """Print the comps table to the terminal without writing a file."""
    edgar, market = EdgarProvider(), MarketProvider()
    rows = [build_comps_row(edgar.fundamentals(t), market.quote(t)) for t in tickers.split(",")]
    fmt = lambda v, p: "NM" if v is None else (f"{v:.1f}x" if p == "x" else f"{v:.1%}")
    typer.echo(f"{'Ticker':<7}{'EV/Rev':>8}{'EV/EBITDA':>11}{'P/E':>7}{'EBITDA mgn':>12}{'Growth':>9}")
    for r in rows:
        typer.echo(
            f"{r.ticker:<7}{fmt(r.ev_revenue,'x'):>8}{fmt(r.ev_ebitda,'x'):>11}{fmt(r.pe,'x'):>7}"
            f"{fmt(r.ebitda_margin,'%'):>12}{fmt(r.revenue_growth,'%'):>9}"
        )
    for s in summary_rows(rows):
        typer.echo(
            f"{s['company']:<7}{fmt(s['ev_revenue'],'x'):>8}{fmt(s['ev_ebitda'],'x'):>11}"
            f"{fmt(s['pe'],'x'):>7}{fmt(s['ebitda_margin'],'%'):>12}{fmt(s['revenue_growth'],'%'):>9}"
        )
