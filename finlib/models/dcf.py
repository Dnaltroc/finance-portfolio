"""Discounted cash flow engine. Pure Python, $ in millions, shares in millions.

Modeling choices (be ready to defend these in an interview):
- Unlevered free cash flow = EBIT x (1 - tax) + D&A - capex - change in NWC.
  Taxes = max(0, EBIT) x tax_rate: no tax credit on a loss and no NOL carryforward,
  which understates value slightly for a company coming off losses.
- Change in NWC = nwc_pct x (revenue_t - revenue_{t-1}), as in the LBO engine.
- Terminal value by perpetuity growth on the final year's UFCF: UFCF_N x (1 + g) / (WACC - g).
  The implied exit EV / EBITDA is reported so the growth assumption can be sanity-checked
  against trading multiples.
- Mid-year convention discounts each year's flow at t - 0.5, and the perpetuity-growth
  terminal value with the final year's flow (Rosenbaum & Pearl).
- years_elapsed moves the valuation date past the base fiscal year end, so every discount
  period shortens by that much. The cash flows themselves are not prorated.
- Equity value = enterprise value + net cash (cash - debt). Per share uses diluted shares.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    terminal_growth: float
    net_cash: float = Field(description="cash minus debt at the valuation date, $M")
    shares: float = Field(gt=0, description="diluted shares, millions")
    mid_year: bool = True
    years_elapsed: float = Field(0.0, ge=0, lt=1, description="valuation date minus base fiscal year end, years")

    @model_validator(mode="after")
    def _check(self):
        if len(self.growth) != len(self.ebit_margin) or not self.growth:
            raise ValueError("growth and ebit_margin must be non-empty and the same length")
        if self.wacc <= self.terminal_growth:
            raise ValueError("WACC must exceed terminal growth")
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
    last = years[-1]
    tv = last["ufcf"] * (1 + a.terminal_growth) / (a.wacc - a.terminal_growth)
    pv_tv = tv * last["discount"]
    ev = pv_fcf + pv_tv
    equity = ev + a.net_cash
    ebitda_n = last["ebit"] + last["da"]
    return DCFResult(years=years, pv_fcf=pv_fcf, terminal_value=tv, pv_terminal=pv_tv, enterprise_value=ev,
                     equity_value=equity, per_share=equity / a.shares, tv_share=pv_tv / ev if ev else float("nan"),
                     implied_exit_ev_ebitda=tv / ebitda_n if ebitda_n > 0 else None)


def sensitivity(a: DCFAssumptions, waccs: list[float], terminal_growths: list[float]) -> list[list[float]]:
    """Per-share value for each WACC (rows) and terminal growth (columns). NaN where WACC <= g."""
    return [[run_dcf(a.model_copy(update=dict(wacc=w, terminal_growth=g))).per_share if w > g else float("nan")
             for g in terminal_growths] for w in waccs]


def implied_cagr(a: DCFAssumptions, price: float, lo: float = -0.5, hi: float = 2.0, tol: float = 1e-9) -> float | None:
    """The single revenue growth rate, applied to every forecast year with all other assumptions
    held, at which per-share value equals `price` (a reverse DCF). None if no rate in [lo, hi]
    gets there. Bisection: value rises with growth while margins are positive."""
    def value(g: float) -> float:
        return run_dcf(a.model_copy(update=dict(growth=[g] * len(a.growth)))).per_share - price

    f_lo, f_hi = value(lo), value(hi)
    if f_lo > 0 or f_hi < 0:
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
