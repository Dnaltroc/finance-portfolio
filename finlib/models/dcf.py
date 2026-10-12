"""Discounted cash flow engine. Pure Python, $ in millions, shares in millions.

Modeling choices (be ready to defend these in an interview):
- Unlevered free cash flow = EBIT x (1 - tax) + D&A - capex - change in NWC.
  Taxes = max(0, EBIT) x tax_rate: no tax credit on a loss and no NOL carryforward,
  which understates value slightly for a company coming off losses.
- Change in NWC = nwc_pct x (revenue_t - revenue_{t-1}), as in the LBO engine.
- Terminal value, two methods. Perpetuity growth on the final year's UFCF:
  UFCF_N x (1 + g) / (WACC - g), reported with the exit EV / EBITDA it implies. Or an exit
  multiple of final-year EBITDA, reported with the perpetuity growth it implies. Each is the
  sanity check on the other.
- Mid-year convention discounts each year's flow at t - 0.5. A perpetuity-growth terminal value
  is discounted with the final year's flow; an exit-multiple terminal value at the end of year N,
  since a sale happens at year end (Rosenbaum & Pearl).
- years_elapsed moves the valuation date past the base fiscal year end, so every discount
  period shortens by that much. The cash flows themselves are not prorated.
- Equity value = enterprise value + net cash (cash - debt). Per share uses diluted shares.
- Reverse DCF: the price is held fixed and one input is solved for. implied_exit_multiple is
  closed-form; implied_revenue_scale scales the whole revenue path (bisection). A single growth
  rate for every year is deliberately not offered: with working capital tied to revenue growth,
  value is not monotonic in growth, so such a solve can fail or mislead.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DCFAssumptions(BaseModel):
    base_revenue: float = Field(description="revenue of the last reported fiscal year, $M")
    growth: list[float] = Field(description="revenue growth for each forecast year")
    ebit_margin: list[float] = Field(description="EBIT margin for each forecast year")
    tax_rate: float = Field(ge=0, le=1)
    da_pct: float = Field(ge=0, description="D&A as % of revenue")
    capex_pct: float = Field(ge=0, description="capex as % of revenue")
    nwc_pct: float = Field(description="net working capital as % of incremental revenue")
    wacc: float
    terminal_growth: float = 0.03
    terminal_method: Literal["perpetuity", "exit_multiple"] = "perpetuity"
    exit_multiple: float | None = Field(None, gt=0, description="EV / final-year EBITDA, exit_multiple method")
    net_cash: float = Field(description="cash minus debt at the valuation date, $M")
    shares: float = Field(gt=0, description="diluted shares, millions")
    mid_year: bool = True
    years_elapsed: float = Field(0.0, ge=0, lt=1, description="valuation date minus base fiscal year end, years")

    @model_validator(mode="after")
    def _check(self):
        if len(self.growth) != len(self.ebit_margin) or not self.growth:
            raise ValueError("growth and ebit_margin must be non-empty and the same length")
        if self.terminal_method == "perpetuity" and self.wacc <= self.terminal_growth:
            raise ValueError("WACC must exceed terminal growth")
        if self.terminal_method == "exit_multiple" and self.exit_multiple is None:
            raise ValueError("terminal_method exit_multiple needs exit_multiple")
        return self


@dataclass
class DCFResult:
    years: list[dict]
    pv_fcf: float
    terminal_value: float
    pv_terminal: float
    enterprise_value: float
    equity_value: float
    per_share: float
    tv_share: float                      # PV of terminal value / enterprise value
    implied_exit_ev_ebitda: float | None  # terminal value / final-year EBITDA
    implied_terminal_growth: float        # perpetuity growth consistent with the terminal value
    final_ebitda: float
    terminal_discount: float              # factor applied to the terminal value


def capm(risk_free: float, beta: float, equity_risk_premium: float) -> float:
    return risk_free + beta * equity_risk_premium


def wacc(cost_of_equity: float, equity_value: float, pre_tax_cost_of_debt: float, debt: float,
         tax_rate: float) -> float:
    total = equity_value + debt
    return (equity_value * cost_of_equity + debt * pre_tax_cost_of_debt * (1 - tax_rate)) / total


def run_dcf(a: DCFAssumptions) -> DCFResult:
    years, revenue, pv_fcf = [], a.base_revenue, 0.0
    shift = (0.5 if a.mid_year else 0.0) + a.years_elapsed
    for t, (g, margin) in enumerate(zip(a.growth, a.ebit_margin), start=1):
        prior, revenue = revenue, revenue * (1 + g)
        ebit = revenue * margin
        nopat = ebit - max(0.0, ebit) * a.tax_rate
        da, capex = revenue * a.da_pct, revenue * a.capex_pct
        dnwc = a.nwc_pct * (revenue - prior)
        ufcf = nopat + da - capex - dnwc
        discount = 1 / (1 + a.wacc) ** (t - shift)
        pv_fcf += ufcf * discount
        years.append(dict(year=t, revenue=revenue, growth=g, ebit=ebit, ebit_margin=margin, nopat=nopat, da=da,
                          capex=capex, dnwc=dnwc, ufcf=ufcf, discount=discount, pv=ufcf * discount))
    last, n = years[-1], len(years)
    ebitda_n = last["ebit"] + last["da"]
    if a.terminal_method == "perpetuity":
        tv = last["ufcf"] * (1 + a.terminal_growth) / (a.wacc - a.terminal_growth)
        tv_discount = last["discount"]
    else:
        tv = a.exit_multiple * ebitda_n
        tv_discount = 1 / (1 + a.wacc) ** (n - a.years_elapsed)
    pv_tv = tv * tv_discount
    ev = pv_fcf + pv_tv
    equity = ev + a.net_cash
    g_implied = (tv * a.wacc - last["ufcf"]) / (tv + last["ufcf"]) if tv + last["ufcf"] else float("nan")
    return DCFResult(years=years, pv_fcf=pv_fcf, terminal_value=tv, pv_terminal=pv_tv, enterprise_value=ev,
                     equity_value=equity, per_share=equity / a.shares, tv_share=pv_tv / ev if ev else float("nan"),
                     implied_exit_ev_ebitda=tv / ebitda_n if ebitda_n > 0 else None,
                     implied_terminal_growth=g_implied, final_ebitda=ebitda_n, terminal_discount=tv_discount)


def sensitivity(a: DCFAssumptions, waccs: list[float], terminal_growths: list[float]) -> list[list[float]]:
    """Per-share value for each WACC (rows) and terminal growth (columns). NaN where WACC <= g."""
    return [[run_dcf(a.model_copy(update=dict(wacc=w, terminal_growth=g))).per_share if w > g else float("nan")
             for g in terminal_growths] for w in waccs]


def implied_exit_multiple(a: DCFAssumptions, price: float) -> float | None:
    """Exit EV / EBITDA on the final forecast year at which per-share value equals `price`,
    every other assumption held. Closed form. None if final-year EBITDA is not positive."""
    r = run_dcf(a.model_copy(update=dict(terminal_method="exit_multiple", exit_multiple=1.0)))
    if r.final_ebitda <= 0:
        return None
    tv_needed = (price * a.shares - a.net_cash - r.pv_fcf) / r.terminal_discount
    return tv_needed / r.final_ebitda


def implied_revenue_scale(a: DCFAssumptions, price: float, lo: float = 0.05, hi: float = 100.0,
                          tol: float = 1e-9) -> float | None:
    """Factor k on the whole revenue path (every year's revenue x k; margins and later growth
    rates held) at which per-share value equals `price`. Applied through year-1 growth:
    (1 + g1) x k - 1. None if no k in [lo, hi] reaches the price."""
    def value(k: float) -> float:
        g = [(1 + a.growth[0]) * k - 1] + list(a.growth[1:])
        return run_dcf(a.model_copy(update=dict(growth=g))).per_share - price

    if value(lo) > 0 or value(hi) < 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if value(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def multiple_value(metric: float, multiple: float, net_cash: float, shares: float) -> float | None:
    """Relative valuation: EV = multiple x metric (revenue or EBITDA, $M); per share after net
    cash. None when the metric is not positive, where a multiple has no meaning."""
    if metric is None or metric <= 0:
        return None
    return (metric * multiple + net_cash) / shares
