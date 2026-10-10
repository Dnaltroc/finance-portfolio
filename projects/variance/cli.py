"""Variance Report Generator CLI.

    python -m projects.variance run --actuals a.csv --budget b.csv --period 2026-09
    python -m projects.variance demo          # runs on sample_data/
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
import yaml

from finlib.excel_writer.variance_pack import write_variance_workbook
from finlib.models.variance import (
    Thresholds, commentary, compute_variance, pnl_summary, rollup, top_flags, validate_ledger,
)

HERE = Path(__file__).resolve().parent
app = typer.Typer(add_completion=False, help="Budget vs actual variance pack from two CSV exports.")


def load_thresholds(config: Path) -> Thresholds:
    if not config.exists():
        return Thresholds()
    raw = yaml.safe_load(config.read_text()) or {}
    return Thresholds(**raw.get("thresholds", {}))


def run_variance(actuals_csv: Path, budget_csv: Path, period: str, config: Path, out_dir: Path) -> tuple[Path, Path]:
    th = load_thresholds(config)
    actuals = validate_ledger(pd.read_csv(actuals_csv), "actuals")
    budget = validate_ledger(pd.read_csv(budget_csv), "budget")
    actuals = actuals[actuals["period"] == period]
    budget = budget[budget["period"] == period]
    if actuals.empty or budget.empty:
        raise typer.BadParameter(f"No rows for period {period} in both files")

    detail = compute_variance(actuals, budget, th)
    summary = pnl_summary(detail)
    by_dept = rollup(detail, ["department"])
    flags = top_flags(detail, th.top_n)
    text = f"|variance| >= ${th.abs_amount:,.0f} or >= {th.pct:.0%}"

    xlsx = write_variance_workbook(period, summary, by_dept, detail, flags, actuals, budget, text, out_dir)
    md = out_dir / f"commentary_{period}.md"
    md.write_text(commentary(flags, period))
    return xlsx, md


@app.command()
def run(
    actuals: Path = typer.Option(..., exists=True),
    budget: Path = typer.Option(..., exists=True),
    period: str = typer.Option(..., help="YYYY-MM"),
    config: Path = typer.Option(HERE / "config.yaml"),
    out: Path = typer.Option(Path("output")),
):
    """Validate, compute, and write the Excel pack plus draft commentary."""
    xlsx, md = run_variance(actuals, budget, period, config, out)
    typer.echo(f"Wrote {xlsx}\nWrote {md}")


@app.command()
def demo(period: str = typer.Option("2026-09"), out: Path = typer.Option(Path("output"))):
    """Run on the bundled sample data."""
    sd = HERE / "sample_data"
    xlsx, md = run_variance(sd / "actuals.csv", sd / "budget.csv", period, HERE / "config.yaml", out)
    typer.echo(f"Wrote {xlsx}\nWrote {md}")
