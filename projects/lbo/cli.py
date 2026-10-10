"""LBO Returns Engine CLI.

    python -m projects.lbo run --deal projects/lbo/deal.yaml
    python -m projects.lbo summary --deal projects/lbo/deal.yaml
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from finlib.excel_writer.lbo_workbook import write_lbo_workbook
from finlib.models.lbo import LBOAssumptions, run_lbo, sensitivity

HERE = Path(__file__).resolve().parent
app = typer.Typer(add_completion=False, help="LBO returns model with live-formula Excel export.")


def load_deal(path: Path) -> LBOAssumptions:
    return LBOAssumptions(**yaml.safe_load(path.read_text()))


def run_lbo_export(deal: Path, out_dir: Path) -> Path:
    return write_lbo_workbook(load_deal(deal), out_dir)


@app.command()
def run(deal: Path = typer.Option(HERE / "deal.yaml", exists=True), out: Path = typer.Option(Path("output"))):
    """Export the workbook and print the headline returns."""
    a = load_deal(deal)
    path = run_lbo_export(deal, out)
    r = run_lbo(a)
    typer.echo(f"{a.deal_name}: {a.active_scenario} case  MOIC {r.moic:.2f}x  IRR {_irr(r)}")
    _warn(r)
    typer.echo(f"Wrote {path}")


def _irr(r) -> str:
    return f"{r.irr:.1%}" if r.irr_defined else "n/a (equity wiped out)"


def _warn(r) -> None:
    if r.cash_shortfall_years:
        typer.secho(f"  WARNING: cash falls below minimum in year(s) {r.cash_shortfall_years}; "
                    f"the deal needs a revolver or less debt.", fg="yellow")


@app.command()
def summary(deal: Path = typer.Option(HERE / "deal.yaml", exists=True)):
    """Print every scenario plus the entry/exit sensitivity grid."""
    a = load_deal(deal)
    typer.echo(f"{a.deal_name}  EV {a.entry_ev:,.0f}  debt {a.total_debt:,.0f} "
               f"({a.total_debt / a.entry_ebitda:.1f}x)  equity {a.sponsor_equity:,.0f}")
    for sc in a.scenarios:
        r = run_lbo(a, sc)
        typer.echo(f"  {sc:<9} exit EV {r.exit_ev:8,.1f}  net debt {r.exit_net_debt:7,.1f}  "
                   f"equity {r.exit_equity:8,.1f}  MOIC {r.moic:.2f}x  IRR {_irr(r):>6}")
        _warn(r)
    ems, xms = [7.0, 8.0, 9.0, 10.0, 11.0], [7.0, 8.0, 9.0, 10.0, 11.0]
    g = sensitivity(a, ems, xms)
    typer.echo("\nIRR: entry (rows) x exit (cols)")
    typer.echo("        " + "".join(f"{x:>7.1f}x" for x in xms))
    for em, row in zip(ems, g):
        typer.echo(f"  {em:4.1f}x " + "".join(f"{v:>8.1%}" if v == v else "     n/a" for v in row))
