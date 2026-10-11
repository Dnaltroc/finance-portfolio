"""Portfolio KPI Dashboard CLI.

    python -m projects.dashboard build                       # sample data -> dist/index.html
    python -m projects.dashboard build --kpis my_kpis.csv --title "Fund II Portfolio"
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import typer

from finlib.models.kpi import compute_kpis, latest_snapshot, portfolio_totals, validate_kpis

HERE = Path(__file__).resolve().parent
app = typer.Typer(add_completion=False, help="Build a self-contained portfolio KPI dashboard page.")


@app.callback()
def _root():
    """Portfolio KPI dashboard."""


def build_payload(kpis_csv: Path, title: str) -> dict:
    return payload_from_frame(pd.read_csv(kpis_csv), title)


def payload_from_frame(raw: pd.DataFrame, title: str) -> dict:
    """Validate and compute every KPI, then shape the page payload. No I/O."""
    k = compute_kpis(validate_kpis(raw))
    snap = latest_snapshot(k)
    companies = snap.sort_values("revenue", ascending=False)["company"].tolist()
    series = {}
    for c, g in k.groupby("company"):
        g = g.sort_values("month")
        series[c] = {col: [None if pd.isna(v) else (float(v) if isinstance(v, float) else v) for v in g[col]]
                     for col in ("month", "revenue", "gross_profit", "cash", "net_burn", "gross_margin", "headcount")}
    snapshot = {}
    for r in snap.to_dict("records"):
        snapshot[r["company"]] = {k2: (None if isinstance(v, float) and pd.isna(v) else v)
                                  for k2, v in r.items() if k2 != "company"}
    return {"title": title, "companies": companies, "series": series, "snapshot": snapshot,
            "totals": portfolio_totals(k)}


def save_sqlite(kpis_csv: Path, db: Path) -> None:
    k = compute_kpis(validate_kpis(pd.read_csv(kpis_csv)))
    with sqlite3.connect(db) as con:
        k.to_sql("kpis", con, if_exists="replace", index=False)
        latest_snapshot(k).to_sql("snapshot", con, if_exists="replace", index=False)


def build_page(kpis_csv: Path, title: str, out: Path) -> Path:
    html = render_page(build_payload(kpis_csv, title))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


def render_page(payload: dict) -> str:
    """The self-contained dashboard HTML for a payload built by payload_from_frame."""
    import html as html_mod

    data = json.dumps(payload).replace("</", "<\\/")  # never let a value close the script tag
    html = (HERE / "template.html").read_text().replace("__DATA__", data)
    title = html_mod.escape(payload["title"])
    return html.replace("<title>Gulf Coast Portfolio KPIs</title>", f"<title>{title}</title>")


@app.command()
def build(
    kpis: Path = typer.Option(HERE / "sample_data" / "kpis.csv", exists=True),
    title: str = typer.Option("Gulf Coast Portfolio KPIs"),
    out: Path = typer.Option(Path("dist") / "index.html"),
    db: Path = typer.Option(None, help="Also write computed KPIs to this SQLite file"),
):
    """Compute KPIs and write the dashboard page."""
    path = build_page(kpis, title, out)
    if db:
        save_sqlite(kpis, db)
        typer.echo(f"Wrote {db}")
    typer.echo(f"Wrote {path} ({path.stat().st_size // 1024} KB)")
