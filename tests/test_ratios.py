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
    import json
    from finlib.providers.edgar import FIXTURE_DIR

    f = EdgarProvider().fundamentals("MSFT")
    q = json.loads((FIXTURE_DIR / "MSFT.json").read_text())["quarterly"]
    assert f.ltm("revenue") == sum(q["revenue"][:4])
    assert f.prior_ltm("revenue") == sum(q["revenue"][4:8])


def test_live_ltm_math_fy_plus_ytd_minus_prior_ytd():
    """The exact method used on real filings, on a hand-built fact list.

    FY ending 2025-09-27 = 400; YTD to 2026-06-27 = 330; prior YTD to 2025-06-28 = 300.
    LTM ending 2026-06-27 must be 400 + 330 - 300 = 430.
    """
    from datetime import date
    from finlib.providers.edgar import ltm_ending

    ent = [
        dict(start="2024-09-29", end="2025-09-27", val=400, form="10-K", filed="2025-10-31"),
        dict(start="2025-09-28", end="2026-06-27", val=330, form="10-Q", filed="2026-07-31"),
        dict(start="2024-09-29", end="2025-06-28", val=300, form="10-Q", filed="2025-08-01"),
        dict(start="2026-03-29", end="2026-06-27", val=110, form="10-Q", filed="2026-07-31"),  # 3-month, ignored
    ]
    v, basis = ltm_ending(ent, date(2026, 6, 27))
    assert v == 430 and basis.startswith("FY ")
    # A 12-month entry ending on the date wins outright
    v2, basis2 = ltm_ending(ent, date(2025, 9, 27))
    assert v2 == 400 and basis2.startswith("12 months")
    # Restated value (later filed) replaces the original for the same period
    from finlib.providers.edgar import _dedupe
    d = _dedupe(ent + [dict(start="2024-09-29", end="2025-09-27", val=401, form="10-K/A", filed="2025-12-01")])
    assert next(e["val"] for e in d if e["end"] == "2025-09-27" and e["start"] == "2024-09-29") == 401


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
