"""Variance Report Generator CLI.

    python -m projects.variance run --actuals a.csv --budget b.csv --period 2026-09
    python -m projects.variance demo          # runs on sample_data/
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class VariancePack:
    """Everything the Excel pack and the commentary are built from, for one period."""
    period: str
    thresholds: Thresholds
    actuals: pd.DataFrame
    budget: pd.DataFrame
    detail: pd.DataFrame
    summary: pd.DataFrame
    by_dept: pd.DataFrame
    flags: pd.DataFrame

    @property
    def thresholds_text(self) -> str:
        th = self.thresholds
        return f"|variance| >= ${th.abs_amount:,.0f} or >= {th.pct:.0%}"


def build_variance(actuals_raw: pd.DataFrame, budget_raw: pd.DataFrame, period: str,
                   th: Thresholds) -> VariancePack:
    """Validate both ledgers, keep one period, and compute every table. No I/O."""
    actuals = validate_ledger(actuals_raw, "actuals")
    budget = validate_ledger(budget_raw, "budget")
    actuals = actuals[actuals["period"] == period]
    budget = budget[budget["period"] == period]
    if actuals.empty or budget.empty:
        raise typer.BadParameter(f"No rows for period {period} in both files")
    detail = compute_variance(actuals, budget, th)
    return VariancePack(period, th, actuals, budget, detail, pnl_summary(detail),
                        rollup(detail, ["department"]), top_flags(detail, th.top_n))


def write_variance(pack: VariancePack, out_dir: Path) -> tuple[Path, Path]:
    xlsx = write_variance_workbook(pack.period, pack.summary, pack.by_dept, pack.detail, pack.flags,
                                   pack.actuals, pack.budget, pack.thresholds_text, out_dir)
    md = out_dir / f"commentary_{pack.period}.md"
    md.write_text(commentary(pack.flags, pack.period))
    return xlsx, md


def run_variance(actuals_csv: Path, budget_csv: Path, period: str, config: Path, out_dir: Path) -> tuple[Path, Path]:
    pack = build_variance(pd.read_csv(actuals_csv), pd.read_csv(budget_csv), period, load_thresholds(config))
    return write_variance(pack, out_dir)


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
