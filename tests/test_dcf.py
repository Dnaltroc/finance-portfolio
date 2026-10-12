"""DCF engine: every expected value below is recomputed by hand in the test, not copied
from the engine's output."""

from __future__ import annotations

import math

import pytest

from finlib.models.dcf import DCFAssumptions, capm, implied_cagr, run_dcf, sensitivity, wacc


def _a(**kw) -> DCFAssumptions:
    base = dict(base_revenue=100.0, growth=[0.10, 0.10], ebit_margin=[0.20, 0.20], tax_rate=0.25,
                da_pct=0.05, capex_pct=0.05, nwc_pct=0.10, wacc=0.10, terminal_growth=0.03,
                net_cash=50.0, shares=10.0, mid_year=False, years_elapsed=0.0)
    base.update(kw)
    return DCFAssumptions(**base)


def test_two_year_dcf_matches_hand_calculation():
    r = run_dcf(_a())
    # Year 1: revenue 110, EBIT 22, NOPAT 16.5, D&A 5.5, capex 5.5, change in NWC 10% x 10 = 1.0
    y1 = r.years[0]
    assert (y1["revenue"], y1["ebit"], y1["nopat"]) == pytest.approx((110.0, 22.0, 16.5))
    assert y1["ufcf"] == pytest.approx(16.5 + 5.5 - 5.5 - 1.0)
    # Year 2: revenue 121, NOPAT 18.15, change in NWC 1.1 -> UFCF 17.05
    assert r.years[1]["ufcf"] == pytest.approx(17.05)
    pv = 15.5 / 1.1 + 17.05 / 1.1 ** 2
    tv = 17.05 * 1.03 / (0.10 - 0.03)
    assert r.pv_fcf == pytest.approx(pv)
    assert r.terminal_value == pytest.approx(tv)
    assert r.pv_terminal == pytest.approx(tv / 1.1 ** 2)
    assert r.enterprise_value == pytest.approx(pv + tv / 1.1 ** 2)
    assert r.equity_value == pytest.approx(r.enterprise_value + 50.0)
    assert r.per_share == pytest.approx(r.equity_value / 10.0)
    assert r.tv_share == pytest.approx(r.pv_terminal / r.enterprise_value)
    # Implied exit multiple: terminal value over final-year EBITDA (EBIT + D&A = 24.2 + 6.05)
    assert r.implied_exit_ev_ebitda == pytest.approx(tv / (24.2 + 6.05))


def test_mid_year_convention_pulls_every_flow_half_a_year_earlier():
    end, mid = run_dcf(_a()), run_dcf(_a(mid_year=True))
    assert mid.pv_fcf == pytest.approx(15.5 / 1.1 ** 0.5 + 17.05 / 1.1 ** 1.5)
    # Perpetuity-growth terminal value is discounted with the final year's flow (Rosenbaum & Pearl).
    assert mid.pv_terminal == pytest.approx(end.terminal_value / 1.1 ** 1.5)


def test_years_elapsed_since_base_year_shortens_every_discount_period():
    r = run_dcf(_a(years_elapsed=0.25))
    assert r.pv_fcf == pytest.approx(15.5 / 1.1 ** 0.75 + 17.05 / 1.1 ** 1.75)


def test_wacc_at_or_below_terminal_growth_is_rejected():
    with pytest.raises(ValueError, match="WACC must exceed terminal growth"):
        run_dcf(_a(wacc=0.03))


def test_growth_and_margin_paths_must_match_in_length():
    with pytest.raises(ValueError, match="same length"):
        run_dcf(_a(ebit_margin=[0.2]))


def test_capm_and_wacc():
    assert capm(0.05, 1.5, 0.045) == pytest.approx(0.1175)
    # 80/20 equity/debt, 6% pre-tax debt, 25% tax: 0.8 x 11.75% + 0.2 x 4.5%
    assert wacc(0.1175, 800.0, 0.06, 200.0, 0.25) == pytest.approx(0.8 * 0.1175 + 0.2 * 0.045)
    assert wacc(0.1175, 800.0, 0.06, 0.0, 0.25) == pytest.approx(0.1175)


def test_sensitivity_grid_rises_as_wacc_falls_and_blanks_invalid_cells():
    grid = sensitivity(_a(), [0.08, 0.10, 0.12], [0.02, 0.03, 0.09])
    assert grid[0][0] > grid[1][0] > grid[2][0]
    assert grid[1][1] == pytest.approx(run_dcf(_a()).per_share)
    assert math.isnan(grid[0][2])  # 8% WACC vs 9% growth


def test_implied_cagr_recovers_the_growth_that_produced_a_price():
    target = run_dcf(_a(growth=[0.25, 0.25])).per_share
    assert implied_cagr(_a(), target) == pytest.approx(0.25, abs=1e-6)


def test_implied_cagr_returns_none_when_no_growth_rate_reaches_the_price():
    assert implied_cagr(_a(), 1e9) is None
