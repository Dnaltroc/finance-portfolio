from finlib.models.ratios import growth, safe_div, summary_rows, build_comps_row
from finlib.providers import EdgarProvider, MarketProvider


def test_safe_div_handles_zero_and_negative():
    assert safe_div(10, 0) is None
    assert safe_div(10, -5) is None
    assert safe_div(None, 5) is None
    assert safe_div(10, 4) == 2.5


def test_growth_requires_positive_prior():
    assert abs(growth(110, 100) - 0.10) < 1e-12
    assert growth(110, 0) is None
    assert growth(110, None) is None


def test_ltm_is_sum_of_four_quarters():
    f = EdgarProvider().fundamentals("MSFT")
    assert f.ltm("revenue") == sum(f.quarterly["revenue"][:4])
    assert f.prior_ltm("revenue") == sum(f.quarterly["revenue"][4:8])


def test_comps_row_math_ties_to_inputs():
    f = EdgarProvider().fundamentals("AAPL")
    q = MarketProvider().quote("AAPL")
    r = build_comps_row(f, q)
    m = 1_000_000
    ev = q.market_cap + f.latest("total_debt") - f.latest("cash")
    ebitda = f.ltm("operating_income") + f.ltm("da")
    assert abs(r.ev_m - ev / m) < 1e-6
    assert abs(r.ev_ebitda - ev / ebitda) < 1e-9
    assert abs(r.pe - q.market_cap / f.ltm("net_income")) < 1e-9
    assert 0 < r.gross_margin < 1
    assert r.revenue_growth is not None and r.revenue_growth > 0


def test_negative_ebitda_renders_nm_and_flags():
    f = EdgarProvider().fundamentals("LOSS")
    q = MarketProvider().quote("LOSS")
    r = build_comps_row(f, q)
    assert "da" in f.missing
    assert r.ebitda_m is None           # missing D&A -> no EBITDA, never EBIT in disguise
    assert r.ev_ebitda is None
    assert r.pe is None                 # negative net income
    assert "negative NI" in r.flags and "missing: da" in r.flags


def test_summary_skips_nm_values():
    tickers = ["AAPL", "MSFT", "LOSS"]
    e, m = EdgarProvider(), MarketProvider()
    rows = [build_comps_row(e.fundamentals(t), m.quote(t)) for t in tickers]
    med, mean = summary_rows(rows)
    valid = sorted(r.ev_ebitda for r in rows if r.ev_ebitda is not None)
    assert len(valid) == 2
    assert med["ev_ebitda"] == sum(valid) / 2
    assert mean["ev_ebitda"] == sum(valid) / 2
