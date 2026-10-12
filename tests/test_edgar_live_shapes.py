"""Live companyfacts shapes the provider mishandles, reproduced offline with synthetic facts.

Found 2026-10-11 running live comps on AEHR, COHU, FORM, ONTO, KLIC, CAMT:
- COHU tags revenue as RevenueFromContractWithCustomerIncludingAssessedTax, which is not in
  the revenue tag list, so the LTM anchors on a 2017 SalesRevenueNet fact and the comps row
  silently shows 2017 numbers.
- Debt-free issuers (AEHR, ONTO, KLIC, CAMT) tag no borrowing line at all, so total_debt is
  "missing" and EV is NM, though the filing supports EV = market cap - cash.
"""

from __future__ import annotations

from finlib.providers.edgar import EdgarProvider


def _flow(tag_val: dict[str, list[tuple[str, str, float]]]) -> dict:
    return {tag: {"units": {"USD": [{"start": s, "end": e, "val": v, "form": "10-K", "filed": e, "fy": 0, "fp": "FY"}
                                    for s, e, v in rows]}}
            for tag, rows in tag_val.items()}


def _inst(tag_val: dict[str, list[tuple[str, float]]]) -> dict:
    return {tag: {"units": {"USD": [{"end": e, "val": v, "form": "10-K", "filed": e, "fy": 0, "fp": "FY"}
                                    for e, v in rows]}}
            for tag, rows in tag_val.items()}


def _provider(monkeypatch, gaap: dict) -> EdgarProvider:
    p = EdgarProvider(identity="test test@example.com")
    monkeypatch.setattr(p, "cik_for", lambda t: ("0000000001", "TEST CO"))
    monkeypatch.setattr(p, "_get_json", lambda url: {"facts": {"us-gaap": gaap}})
    monkeypatch.setenv("FINLIB_OFFLINE", "0")
    return p


FY = [("2024-07-01", "2025-06-28", 400.0), ("2025-06-29", "2026-06-27", 500.0)]


def test_revenue_tagged_including_assessed_tax_anchors_on_latest_period(monkeypatch):
    gaap = {
        **_flow({"SalesRevenueNet": [("2017-01-01", "2017-12-30", 350.0)],
                 "RevenueFromContractWithCustomerIncludingAssessedTax": FY,
                 "OperatingIncomeLoss": [("2025-06-29", "2026-06-27", 50.0)]}),
        **_inst({"CashAndCashEquivalentsAtCarryingValue": [("2026-06-27", 100.0)],
                 "LongTermDebt": [("2026-06-27", 20.0)]}),
    }
    f = _provider(monkeypatch, gaap).fundamentals("TEST")
    assert f.ltm_period.endswith("2026-06-27")
    assert f.ltm("revenue") == 500.0


def test_issuer_with_no_borrowing_tags_has_zero_debt_not_missing(monkeypatch):
    gaap = {
        **_flow({"Revenues": FY, "OperatingIncomeLoss": [("2025-06-29", "2026-06-27", 50.0)]}),
        **_inst({"CashAndCashEquivalentsAtCarryingValue": [("2026-06-27", 100.0)],
                 "Liabilities": [("2026-06-27", 80.0)]}),
    }
    f = _provider(monkeypatch, gaap).fundamentals("TEST")
    assert f.latest("total_debt") == 0.0
    assert "total_debt" not in f.missing
    assert f.sources["total_debt"] == "xbrl:no borrowing tags at balance date (debt = 0)"


def test_unlisted_borrowing_tag_keeps_debt_missing(monkeypatch):
    """A convertible note or credit line we don't sum must not be read as zero debt."""
    gaap = {
        **_flow({"Revenues": FY, "OperatingIncomeLoss": [("2025-06-29", "2026-06-27", 50.0)]}),
        **_inst({"CashAndCashEquivalentsAtCarryingValue": [("2026-06-27", 100.0)],
                 "Liabilities": [("2026-06-27", 80.0)],
                 "ConvertibleNotesPayable": [("2026-06-27", 60.0)]}),
    }
    f = _provider(monkeypatch, gaap).fundamentals("TEST")
    assert f.latest("total_debt") is None
    assert "total_debt" in f.missing


def test_no_balance_sheet_at_ltm_end_keeps_debt_missing(monkeypatch):
    gaap = {
        **_flow({"Revenues": FY, "OperatingIncomeLoss": [("2025-06-29", "2026-06-27", 50.0)]}),
        **_inst({"CashAndCashEquivalentsAtCarryingValue": [("2026-06-27", 100.0)]}),
    }
    f = _provider(monkeypatch, gaap).fundamentals("TEST")
    assert "total_debt" in f.missing


def test_ltm_far_behind_newest_filing_is_flagged(monkeypatch):
    """Revenue under a tag we don't know: the anchor falls back to an old tag. Say so."""
    from finlib.models.ratios import build_comps_row
    from finlib.providers.market import Quote

    gaap = {
        **_flow({"SalesRevenueNet": [("2017-01-01", "2017-12-30", 350.0)],
                 "SomeNewRevenueTag": FY}),
        **_inst({"CashAndCashEquivalentsAtCarryingValue": [("2026-06-27", 100.0)]}),
    }
    f = _provider(monkeypatch, gaap).fundamentals("TEST")
    assert f.warnings == ["stale LTM: ends 2017-12-30, filings run to 2026-06-27"]
    row = build_comps_row(f, Quote("TEST", 10.0, 1e6, "2026-10-11", "manual price"))
    assert "stale LTM: ends 2017-12-30" in row.flags


def test_any_borrowing_shaped_tag_keeps_debt_missing(monkeypatch):
    """ONTO tags $1.47B of converts as ConvertibleLongTermNotesPayable, a name no fixed list
    anticipated. Any liability-side borrowing tag at the balance date must block debt = 0."""
    gaap = {
        **_flow({"Revenues": FY, "OperatingIncomeLoss": [("2025-06-29", "2026-06-27", 50.0)]}),
        **_inst({"CashAndCashEquivalentsAtCarryingValue": [("2026-06-27", 100.0)],
                 "Liabilities": [("2026-06-27", 80.0)],
                 "ConvertibleLongTermNotesPayable": [("2026-06-27", 1471.7)]}),
    }
    f = _provider(monkeypatch, gaap).fundamentals("TEST")
    assert "total_debt" in f.missing


def test_asset_side_debt_words_do_not_block_zero_debt(monkeypatch):
    """Debt securities held and notes receivable are assets, not borrowings."""
    gaap = {
        **_flow({"Revenues": FY, "OperatingIncomeLoss": [("2025-06-29", "2026-06-27", 50.0)]}),
        **_inst({"CashAndCashEquivalentsAtCarryingValue": [("2026-06-27", 100.0)],
                 "Liabilities": [("2026-06-27", 80.0)],
                 "AvailableForSaleSecuritiesDebtSecurities": [("2026-06-27", 30.0)],
                 "AccountsNotesAndLoansReceivableNetCurrent": [("2026-06-27", 12.0)]}),
    }
    f = _provider(monkeypatch, gaap).fundamentals("TEST")
    assert f.latest("total_debt") == 0.0
