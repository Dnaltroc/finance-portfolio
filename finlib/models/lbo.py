"""LBO returns engine. Pure Python; the Excel export mirrors every line as formulas.

Modeling choices (be ready to defend these in an interview):
- Interest is charged on the BEGINNING balance of each year. This avoids a circular
  reference in Excel. Average-balance interest is more precise but needs iteration.
- Cash sweep: after mandatory amortization and minimum cash, all remaining free cash
  flow repays the sweep tranche(s) in order, senior first.
- Change in NWC = nwc_pct_revenue * (revenue_t - revenue_{t-1}).
- Taxes = max(0, EBIT - interest) * tax_rate. No NOL carryforward.
- Exit at end of hold_years at exit_multiple * EBITDA of that year; equity proceeds =
  exit EV - net debt. No management option pool, no dividends.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.optimize import brentq


class Tranche(BaseModel):
    name: str
    ebitda_multiple: float = Field(ge=0, description="debt raised as a multiple of entry EBITDA")
    rate: float = Field(ge=0, description="annual cash interest rate")
    amort_pct: float = Field(ge=0, le=1, default=0.0, description="mandatory amortization per year, % of original")
    sweep: bool = False


class Scenario(BaseModel):
    revenue_growth: list[float]
    ebitda_margin: list[float]
    exit_multiple: float

    @model_validator(mode="after")
    def _same_length(self):
        if len(self.revenue_growth) != len(self.ebitda_margin):
            raise ValueError("revenue_growth and ebitda_margin must have the same length")
        return self


class LBOAssumptions(BaseModel):
    deal_name: str = "Target Co"
    entry_revenue: float = Field(gt=0)
    entry_ebitda: float = Field(gt=0)
    entry_multiple: float = Field(gt=0)
    fees_pct_ev: float = Field(ge=0, default=0.02)
    min_cash: float = Field(ge=0, default=0.0)
    tax_rate: float = Field(ge=0, le=1, default=0.25)
    da_pct_revenue: float = Field(ge=0, default=0.03)
    capex_pct_revenue: float = Field(ge=0, default=0.03)
    nwc_pct_revenue: float = Field(ge=0, default=0.10)
    tranches: list[Tranche]
    scenarios: dict[str, Scenario]
    active_scenario: str = "Base"

    @model_validator(mode="after")
    def _checks(self):
        if self.active_scenario not in self.scenarios:
            raise ValueError(f"active_scenario {self.active_scenario!r} not in scenarios")
        n = {len(s.revenue_growth) for s in self.scenarios.values()}
        if len(n) != 1:
            raise ValueError("every scenario must cover the same number of years")
        return self

    @property
    def hold_years(self) -> int:
        return len(next(iter(self.scenarios.values())).revenue_growth)

    @property
    def entry_ev(self) -> float:
        return self.entry_ebitda * self.entry_multiple

    @property
    def fees(self) -> float:
        return self.entry_ev * self.fees_pct_ev

    @property
    def total_debt(self) -> float:
        return sum(t.ebitda_multiple for t in self.tranches) * self.entry_ebitda

    @property
    def sponsor_equity(self) -> float:
        return self.entry_ev + self.fees + self.min_cash - self.total_debt


@dataclass
class LBOResult:
    years: list[int]
    revenue: np.ndarray
    ebitda: np.ndarray
    da: np.ndarray
    ebit: np.ndarray
    interest: np.ndarray
    taxes: np.ndarray
    capex: np.ndarray
    delta_nwc: np.ndarray
    fcf_pre_debt: np.ndarray        # EBITDA - taxes - capex - dNWC (before interest and amort)
    cash_available_for_debt: np.ndarray
    tranche_begin: dict[str, np.ndarray]
    tranche_mandatory: dict[str, np.ndarray]
    tranche_sweep: dict[str, np.ndarray]
    tranche_end: dict[str, np.ndarray]
    tranche_interest: dict[str, np.ndarray]
    cash_balance: np.ndarray
    exit_ev: float
    exit_net_debt: float
    exit_equity: float
    sponsor_equity: float
    moic: float
    irr: float
    equity_cashflows: list[float] = field(default_factory=list)


def irr(cashflows: list[float]) -> float:
    """Annual IRR by root-finding on NPV. Raises if no sign change."""
    cf = np.asarray(cashflows, dtype=float)
    if not (cf.min() < 0 < cf.max()):
        raise ValueError("IRR needs at least one negative and one positive cash flow")
    npv = lambda r: float(np.sum(cf / (1 + r) ** np.arange(len(cf))))
    return brentq(npv, -0.99, 10.0, maxiter=500)


def run_lbo(a: LBOAssumptions, scenario: str | None = None) -> LBOResult:
    s = a.scenarios[scenario or a.active_scenario]
    n = a.hold_years
    years = list(range(0, n + 1))

    # Operating build (index 0 = entry year, 1..n = projection)
    revenue = np.zeros(n + 1)
    revenue[0] = a.entry_revenue
    for t in range(1, n + 1):
        revenue[t] = revenue[t - 1] * (1 + s.revenue_growth[t - 1])
    margin = np.array([a.entry_ebitda / a.entry_revenue] + list(s.ebitda_margin))
    ebitda = revenue * margin
    da = revenue * a.da_pct_revenue
    ebit = ebitda - da
    capex = revenue * a.capex_pct_revenue
    delta_nwc = np.zeros(n + 1)
    delta_nwc[1:] = a.nwc_pct_revenue * np.diff(revenue)

    # Debt schedule
    names = [t.name for t in a.tranches]
    begin = {k: np.zeros(n + 1) for k in names}
    mand = {k: np.zeros(n + 1) for k in names}
    sweep = {k: np.zeros(n + 1) for k in names}
    end = {k: np.zeros(n + 1) for k in names}
    intr = {k: np.zeros(n + 1) for k in names}
    original = {t.name: t.ebitda_multiple * a.entry_ebitda for t in a.tranches}
    for t in a.tranches:
        end[t.name][0] = original[t.name]

    interest = np.zeros(n + 1)
    taxes = np.zeros(n + 1)
    fcf_pre_debt = np.zeros(n + 1)
    cash_avail = np.zeros(n + 1)
    cash = np.zeros(n + 1)
    cash[0] = a.min_cash

    for y in range(1, n + 1):
        for t in a.tranches:
            begin[t.name][y] = end[t.name][y - 1]
            intr[t.name][y] = begin[t.name][y] * t.rate
        interest[y] = sum(intr[k][y] for k in names)
        taxes[y] = max(0.0, ebit[y] - interest[y]) * a.tax_rate
        fcf_pre_debt[y] = ebitda[y] - taxes[y] - capex[y] - delta_nwc[y]
        fcf_after_interest = fcf_pre_debt[y] - interest[y]
        # Mandatory amortization
        for t in a.tranches:
            mand[t.name][y] = min(begin[t.name][y], original[t.name] * t.amort_pct)
        after_mand = cash[y - 1] + fcf_after_interest - sum(mand[k][y] for k in names)
        cash_avail[y] = max(0.0, after_mand - a.min_cash)
        # Cash sweep, senior first
        remaining = cash_avail[y]
        for t in a.tranches:
            if t.sweep and remaining > 0:
                pay = min(remaining, begin[t.name][y] - mand[t.name][y])
                sweep[t.name][y] = pay
                remaining -= pay
        for t in a.tranches:
            end[t.name][y] = begin[t.name][y] - mand[t.name][y] - sweep[t.name][y]
        cash[y] = after_mand - sum(sweep[k][y] for k in names)

    exit_ev = s.exit_multiple * ebitda[n]
    exit_debt = sum(end[k][n] for k in names)
    exit_net_debt = exit_debt - cash[n]
    exit_equity = exit_ev - exit_net_debt
    eq0 = a.sponsor_equity
    flows = [-eq0] + [0.0] * (n - 1) + [exit_equity]
    return LBOResult(
        years=years, revenue=revenue, ebitda=ebitda, da=da, ebit=ebit, interest=interest,
        taxes=taxes, capex=capex, delta_nwc=delta_nwc, fcf_pre_debt=fcf_pre_debt,
        cash_available_for_debt=cash_avail, tranche_begin=begin, tranche_mandatory=mand,
        tranche_sweep=sweep, tranche_end=end, tranche_interest=intr, cash_balance=cash,
        exit_ev=exit_ev, exit_net_debt=exit_net_debt, exit_equity=exit_equity,
        sponsor_equity=eq0, moic=exit_equity / eq0, irr=irr(flows), equity_cashflows=flows,
    )


def sensitivity(a: LBOAssumptions, entry_multiples: list[float], exit_multiples: list[float],
                scenario: str | None = None) -> np.ndarray:
    """IRR grid: rows = entry multiple, cols = exit multiple. NaN where IRR is undefined."""
    grid = np.full((len(entry_multiples), len(exit_multiples)), np.nan)
    sc = scenario or a.active_scenario
    for i, em in enumerate(entry_multiples):
        for j, xm in enumerate(exit_multiples):
            b = a.model_copy(deep=True)
            b.entry_multiple = em
            b.scenarios[sc].exit_multiple = xm
            try:
                grid[i, j] = run_lbo(b, sc).irr
            except ValueError:
                pass
    return grid
