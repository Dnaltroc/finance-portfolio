"""Equity pitch: every headline figure in the memo equals a direct finlib call on the committed
snapshot, and the reverse-DCF answers really do reproduce the share price. Offline: the
snapshot is a file, nothing here touches the network."""

from __future__ import annotations

import json
from statistics import median

import pytest

from finlib.models.dcf import DCFAssumptions, capm, run_dcf
from projects.pitch import cli


@pytest.fixture(scope="module")
def pitch():
    cfg = cli.load_config("aehr")
    snap = json.loads(cli.snapshot_path("aehr").read_text(encoding="utf-8"))
    return cfg, snap, cli.compute(cfg, snap)


def test_wacc_is_capm_on_the_peer_median_beta(pitch):
    cfg, snap, c = pitch
    betas = [p["beta"] for p in snap["comps"] if p["ticker"] != "AEHR"]
    expected = snap["risk_free"]["value"] + median(betas) * cfg["cost_of_capital"]["equity_risk_premium"]
    assert c["wacc"] == pytest.approx(expected)  # no debt, so WACC = cost of equity
    assert c["ke"] == pytest.approx(capm(snap["risk_free"]["value"], median(betas), 0.045))


def test_base_case_value_matches_an_independently_built_dcf(pitch):
    cfg, snap, c = pitch
    aehr = next(p for p in snap["comps"] if p["ticker"] == "AEHR")
    s, op = cfg["scenarios"]["base"], cfg["operating"]
    a = DCFAssumptions(base_revenue=50.0, growth=s["growth"], ebit_margin=s["ebit_margin"], tax_rate=op["tax_rate"],
                       da_pct=op["da_pct"], capex_pct=op["capex_pct"], nwc_pct=op["nwc_pct"], wacc=c["wacc"],
                       terminal_growth=op["terminal_growth"], net_cash=aehr["cash_m"] - aehr["total_debt_m"],
                       shares=33.17, mid_year=True, years_elapsed=(133 / 365))  # May 29 to Oct 9, 2026
    assert c["fy26_revenue"] == pytest.approx(50.0, abs=0.05)
    assert c["runs"]["base"].per_share == pytest.approx(run_dcf(a).per_share, rel=2e-3)


def test_reverse_dcf_answers_reproduce_the_price(pitch):
    cfg, snap, c = pitch
    assert c["implied_run"].per_share == pytest.approx(c["price"], rel=1e-6)
    a = cli.assumptions(cfg, snap, "base", terminal_method="exit_multiple", exit_multiple=c["implied_exit"])
    assert run_dcf(a).per_share == pytest.approx(c["price"], rel=1e-9)


def test_relative_value_is_peer_median_ev_revenue_on_fy28_revenue(pitch):
    cfg, snap, c = pitch
    multiples = [p["ev_revenue"] for p in snap["comps"] if p["ticker"] != "AEHR" and p["ev_revenue"] is not None]
    fy28 = c["runs"]["base"].years[1]["revenue"]
    assert c["relative"] == pytest.approx((fy28 * median(multiples) + c["net_cash"]) / 33.17)
    assert c["target"] == round(c["relative"])


def test_sensitivity_grid_puts_the_base_case_on_the_outlined_cell(pitch):
    cfg, _, c = pitch
    row = c["waccs"].index(c["wacc"])
    col = cfg["sensitivity"]["terminal_growth"].index(cfg["operating"]["terminal_growth"])
    assert c["grid_g"][row][col] == pytest.approx(c["runs"]["base"].per_share)


def test_history_ratios_quoted_in_the_memo(pitch):
    _, snap, c = pitch
    h, fy26 = c["history_stats"], snap["history"]["years"][max(snap["history"]["years"])]
    assert h["nwc_high"] == pytest.approx((fy26["current_assets"] - fy26["cash"] - fy26["current_liabilities"])
                                          / fy26["revenue"])
    assert h["peak_year"] == "FY23" and h["peak_margin"] == pytest.approx(13.4 / 65.0, abs=1e-3)


def test_memo_prints_the_computed_figures_and_no_unfilled_placeholders(pitch, tmp_path):
    cfg, snap, c = pitch
    html = cli.render_memo(cfg, snap)
    for figure in (cli.usd(c["runs"]["base"].per_share), cli.usd(c["runs"]["bull"].per_share),
                   cli.usd(c["runs"]["bear"].per_share), cli.usd(c["target"], 0), cli.usd(c["price"]),
                   cli.pct(c["wacc"]), f"{c['rev_scale']:.1f} times"):
        assert figure in html, figure
    assert "{" not in html.split("<style>")[0] and "}" not in html.split("</style>")[1].split("<script")[0]
