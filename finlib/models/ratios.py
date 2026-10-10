"""Valuation ratios. Pure functions on numbers, no I/O.

Conventions
- Multiples on a zero or negative denominator return None and render as "NM".
- Enterprise value = market cap + total debt - cash (no minority interest or preferred;
  add them if the provider supplies them).
- EBITDA = operating income + D&A. If D&A is missing, EBITDA is None, not EBIT.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import mean, median

from finlib.providers.edgar import Fundamentals
from finlib.providers.market import Quote


def safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return num / den


def growth(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior <= 0:
        return None
    return current / prior - 1


@dataclass
class CompsRow:
    ticker: str
    company: str
    ltm_period: str
    price: float
    shares_m: float
    market_cap_m: float
    total_debt_m: float | None
    cash_m: float | None
    ev_m: float | None
    revenue_m: float | None
    ebitda_m: float | None
    ebit_m: float | None
    net_income_m: float | None
    ev_revenue: float | None
    ev_ebitda: float | None
    pe: float | None
    gross_margin: float | None
    ebitda_margin: float | None
    revenue_growth: float | None
    flags: str

    def as_dict(self) -> dict:
        return asdict(self)


def build_comps_row(f: Fundamentals, q: Quote) -> CompsRow:
    m = 1_000_000
    revenue = f.ltm("revenue")
    cogs = f.ltm("cogs")
    ebit = f.ltm("operating_income")
    da = f.ltm("da")
    ni = f.ltm("net_income")
    cash = f.latest("cash")
    debt = f.latest("total_debt")

    ebitda = ebit + da if (ebit is not None and da is not None) else None
    mcap = q.market_cap
    ev = mcap + (debt or 0) - (cash or 0) if (debt is not None and cash is not None) else None

    flags = []
    if f.missing:
        flags.append("missing: " + ", ".join(f.missing))
    if ebitda is not None and ebitda <= 0:
        flags.append("negative EBITDA")
    if ni is not None and ni <= 0:
        flags.append("negative NI")
    if ev is None:
        flags.append("EV needs debt and cash")

    def mm(x):
        return None if x is None else x / m

    return CompsRow(
        ticker=f.ticker,
        company=f.company,
        ltm_period=f.ltm_period,
        price=q.price,
        shares_m=q.shares_outstanding / m,
        market_cap_m=mcap / m,
        total_debt_m=mm(debt),
        cash_m=mm(cash),
        ev_m=mm(ev),
        revenue_m=mm(revenue),
        ebitda_m=mm(ebitda),
        ebit_m=mm(ebit),
        net_income_m=mm(ni),
        ev_revenue=safe_div(ev, revenue),
        ev_ebitda=safe_div(ev, ebitda),
        pe=safe_div(mcap, ni),
        gross_margin=safe_div(revenue - cogs, revenue) if (revenue and cogs is not None) else None,
        ebitda_margin=safe_div(ebitda, revenue),
        revenue_growth=growth(revenue, f.prior_ltm("revenue")),
        flags="; ".join(flags),
    )


SUMMARY_COLS = ["ev_revenue", "ev_ebitda", "pe", "gross_margin", "ebitda_margin", "revenue_growth"]


def summary_rows(rows: list[CompsRow]) -> list[dict]:
    """Median and mean over the peer set, skipping NM values per column."""
    out = []
    for label, fn in (("Median", median), ("Mean", mean)):
        d = {"company": label}
        for col in SUMMARY_COLS:
            vals = [getattr(r, col) for r in rows if getattr(r, col) is not None]
            d[col] = fn(vals) if vals else None
        out.append(d)
    return out
