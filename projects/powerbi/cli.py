"""Power BI export CLI.

    python -m projects.powerbi export          # -> projects/powerbi/data/*.csv

Writes a star schema of CSV tables for Power BI Desktop: variance and KPI facts, their
dimensions, one shared monthly calendar, and the AEHR pitch comps. Every number comes
from a finlib call; this module only reshapes. Each fact row carries its data source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from finlib.models.kpi import compute_kpis, latest_snapshot, portfolio_totals, validate_kpis
from finlib.models.variance import REVENUE_TYPES, compute_variance, pnl_summary, validate_ledger
from projects.variance.cli import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
VARIANCE_DIR = ROOT / "variance"
KPI_CSV = ROOT / "dashboard" / "sample_data" / "kpis.csv"
SNAPSHOT = ROOT / "pitch" / "aehr_snapshot.json"
SYNTHETIC = "Sample data (synthetic)"

app = typer.Typer(add_completion=False, help="Export finlib outputs as CSV tables for Power BI.")


@app.callback()
def _root():
    """Power BI export."""


def _calendar(months: pd.Series) -> pd.DataFrame:
    """One row per month from the first to the last month seen, with no gaps."""
    p = pd.period_range(min(months), max(months), freq="M")
    return pd.DataFrame({
        "month": p.strftime("%Y-%m"),
        "month_start": p.to_timestamp().strftime("%Y-%m-%d"),
        "year": p.year,
        "month_num": p.month,
        "month_name": p.strftime("%b %Y"),
        "month_index": p.year * 12 + p.month - 1,  # consecutive integer, for DAX "12 months back"
    })


def _variance_detail() -> pd.DataFrame:
    sd = VARIANCE_DIR / "sample_data"
    actuals = validate_ledger(pd.read_csv(sd / "actuals.csv"), "actuals")
    budget = validate_ledger(pd.read_csv(sd / "budget.csv"), "budget")
    return compute_variance(actuals, budget, load_thresholds(VARIANCE_DIR / "config.yaml"))


def _kpis() -> pd.DataFrame:
    return compute_kpis(validate_kpis(pd.read_csv(KPI_CSV)))


def build_tables() -> dict[str, pd.DataFrame]:
    """Every table Power BI loads, keyed by file name. No I/O beyond reading the inputs."""
    detail = _variance_detail()
    variance = detail.drop(columns="driver").rename(columns={"period": "month"})
    variance["is_revenue"] = variance["account_type"].isin(REVENUE_TYPES)
    variance["data_source"] = SYNTHETIC

    accounts = (detail[["account", "account_type"]].drop_duplicates().sort_values("account"))
    if accounts["account"].duplicated().any():
        raise ValueError("an account name maps to two account types; dim_account needs a composite key")
    accounts["is_revenue"] = accounts["account_type"].isin(REVENUE_TYPES)

    k = _kpis()
    kpi = k.copy()
    kpi["data_source"] = SYNTHETIC
    company = latest_snapshot(k)[["company", "month", "runway_months", "runway_status", "t3m_growth"]]
    company = company.rename(columns={"month": "latest_month", "runway_months": "latest_runway_months"})

    snap = json.loads(SNAPSHOT.read_text())
    comps = pd.DataFrame(snap["comps"]).drop(columns=["flags"])
    comps.insert(2, "role", ["Subject" if t == "AEHR" else "Peer" for t in comps["ticker"]])
    comps["snapshot_pulled_at"] = snap["pulled_at"]
    prices = pd.DataFrame(snap["price_history"], columns=["date", "close"])
    prices.insert(0, "ticker", "AEHR")
    prices["price_source"] = snap["quote"]["source"]

    return {
        "variance_fact": variance,
        "dim_department": pd.DataFrame({"department": sorted(detail["department"].unique())}),
        "dim_account": accounts.reset_index(drop=True),
        "kpi_fact": kpi,
        "dim_company": company.sort_values("company").reset_index(drop=True),
        "calendar": _calendar(pd.concat([variance["month"], kpi["month"]])),
        "comps": comps,
        "aehr_price": prices,
    }


def expected_values() -> dict:
    """Headline numbers straight from finlib, to check the Power BI cards against."""
    detail = _variance_detail()
    out = {}
    for m in sorted(detail["period"].unique()):
        s = pnl_summary(detail[detail["period"] == m]).set_index("line")
        out[f"variance {m}"] = {line: {"actual": float(s.loc[line, "actual"]), "budget": float(s.loc[line, "budget"]),
                                       "favorable": float(s.loc[line, "favorable_amt"])} for line in s.index}
        out[f"variance {m}"]["flagged_lines"] = int(detail[(detail["period"] == m) & detail["flag"]].shape[0])
    out["kpi portfolio"] = portfolio_totals(_kpis())
    comps = json.loads(SNAPSHOT.read_text())["comps"]
    peers = pd.DataFrame([c for c in comps if c["ticker"] != "AEHR"])
    out["comps peer median"] = {"ev_revenue": float(peers["ev_revenue"].median()),
                                "ev_ebitda": float(peers["ev_ebitda"].median())}
    return out


def write_tables(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in build_tables().items():
        p = out_dir / f"{name}.csv"
        df.to_csv(p, index=False, lineterminator="\n")
        paths.append(p)
    return paths


@app.command()
def export(out: Path = typer.Option(HERE / "data", help="Folder Power BI reads from")):
    """Write the CSV tables and print the numbers the report should show."""
    for p in write_tables(out):
        typer.echo(f"Wrote {p} ({sum(1 for _ in p.open()) - 1} rows)")
    typer.echo("\nCheck values (your Power BI cards should match):")
    typer.echo(json.dumps(expected_values(), indent=2, default=str))
