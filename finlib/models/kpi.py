"""Portfolio KPI math. Pure pandas on a long-format monthly table.

Input columns: company, month (YYYY-MM), revenue, cogs, opex, cash, headcount.
Derived per company-month:
    gross_margin   = (revenue - cogs) / revenue
    net_income     = revenue - cogs - opex
    net_burn       = max(0, -net_income)                 # cash consumed per month
    runway_months  = cash / net_burn, or None when net_burn == 0 (profitable)
    mom_growth     = revenue / prior month - 1
    yoy_growth     = revenue / same month last year - 1
    rev_per_head   = revenue / headcount
Runway status: good >= 12 months or profitable; warning 6-12; critical < 6.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

REQUIRED = ["company", "month", "revenue", "cogs", "opex", "cash", "headcount"]


class KpiRow(BaseModel):
    company: str
    month: str = Field(pattern=r"^\d{4}-\d{2}$")
    revenue: float = Field(ge=0)
    cogs: float = Field(ge=0)
    opex: float = Field(ge=0)
    cash: float
    headcount: int = Field(ge=0)


def validate_kpis(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"kpis: missing columns {sorted(missing)}")
    errors = []
    for i, rec in enumerate(df[REQUIRED].to_dict("records"), start=2):
        try:
            KpiRow(**rec)
        except ValidationError as e:
            errors.append(f"  row {i}: {e.errors()[0]['msg']} ({rec})")
    if errors:
        raise ValueError(f"kpis: {len(errors)} invalid row(s)\n" + "\n".join(errors[:20]))
    out = df[REQUIRED].copy()
    out["month"] = out["month"].astype(str)
    return out.sort_values(["company", "month"]).reset_index(drop=True)


def runway_status(months: float | None) -> str:
    if months is None or months != months:  # None or NaN = profitable
        return "good"
    if months < 6:
        return "critical"
    if months < 12:
        return "warning"
    return "good"


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["gross_profit"] = d["revenue"] - d["cogs"]
    d["gross_margin"] = (d["gross_profit"] / d["revenue"]).where(d["revenue"] > 0)
    d["net_income"] = d["revenue"] - d["cogs"] - d["opex"]
    d["net_burn"] = (-d["net_income"]).clip(lower=0)
    d["runway_months"] = (d["cash"].clip(lower=0) / d["net_burn"]).where(d["net_burn"] > 0)
    d["rev_per_head"] = (d["revenue"] / d["headcount"]).where(d["headcount"] > 0)
    g = d.groupby("company")["revenue"]
    d["mom_growth"] = g.pct_change(1)
    d["yoy_growth"] = g.pct_change(12)
    d["runway_status"] = d["runway_months"].apply(runway_status)
    return d


def latest_snapshot(k: pd.DataFrame) -> pd.DataFrame:
    """One row per company at its most recent month, plus trailing-3-month revenue growth."""
    last = k.sort_values("month").groupby("company").tail(1).set_index("company")
    t3 = (k.sort_values("month").groupby("company")["revenue"]
          .apply(lambda s: s.iloc[-1] / s.iloc[-4] - 1 if len(s) >= 4 and s.iloc[-4] > 0 else None))
    last["t3m_growth"] = t3
    return last.reset_index()


def portfolio_totals(k: pd.DataFrame) -> dict:
    snap = latest_snapshot(k)
    month = snap["month"].max()
    cur = k[k["month"] == month]
    prior_m = sorted(k["month"].unique())
    yoy_month = prior_m[prior_m.index(month) - 12] if prior_m.index(month) >= 12 else None
    prev = k[k["month"] == yoy_month] if yoy_month else None
    rev = cur["revenue"].sum()
    return {
        "month": month,
        "companies": int(cur["company"].nunique()),
        "revenue": float(rev),
        "revenue_yoy": float(rev / prev["revenue"].sum() - 1) if prev is not None and prev["revenue"].sum() > 0 else None,
        "gross_margin": float(cur["gross_profit"].sum() / rev) if rev > 0 else None,
        "net_burn": float(cur["net_burn"].sum()),
        "cash": float(cur["cash"].sum()),
        "headcount": int(cur["headcount"].sum()),
        "critical": int((snap["runway_status"] == "critical").sum()),
        "warning": int((snap["runway_status"] == "warning").sum()),
    }
