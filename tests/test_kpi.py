import json
from pathlib import Path

import pandas as pd
import pytest

from finlib.models.kpi import compute_kpis, latest_snapshot, portfolio_totals, runway_status, validate_kpis
from projects.dashboard import build_page, build_payload

SAMPLE = Path(__file__).resolve().parents[1] / "projects" / "dashboard" / "sample_data" / "kpis.csv"


def _two_months():
    return pd.DataFrame([
        dict(company="A", month="2026-08", revenue=100, cogs=40, opex=90, cash=600, headcount=10),
        dict(company="A", month="2026-09", revenue=110, cogs=44, opex=90, cash=576, headcount=10),
        dict(company="B", month="2026-08", revenue=200, cogs=50, opex=100, cash=1000, headcount=20),
        dict(company="B", month="2026-09", revenue=190, cogs=50, opex=100, cash=1040, headcount=19),
    ])


def test_kpi_math():
    k = compute_kpis(validate_kpis(_two_months())).set_index(["company", "month"])
    a = k.loc[("A", "2026-09")]
    assert abs(a["gross_margin"] - 0.6) < 1e-9
    assert a["net_income"] == 110 - 44 - 90 == -24
    assert a["net_burn"] == 24 and a["runway_months"] == 576 / 24 == 24
    assert abs(a["mom_growth"] - 0.10) < 1e-9
    b = k.loc[("B", "2026-09")]
    assert b["net_burn"] == 0 and pd.isna(b["runway_months"])  # profitable
    assert b["runway_status"] == "good"
    assert abs(b["rev_per_head"] - 10) < 1e-9


def test_runway_status_bands():
    assert runway_status(None) == "good"
    assert runway_status(5.9) == "critical"
    assert runway_status(6) == "warning"
    assert runway_status(12) == "good"


def test_negative_cash_is_zero_runway():
    df = _two_months()
    df.loc[1, "cash"] = -50
    k = compute_kpis(validate_kpis(df)).set_index(["company", "month"])
    assert k.loc[("A", "2026-09"), "runway_months"] == 0
    assert k.loc[("A", "2026-09"), "runway_status"] == "critical"


def test_validation_messages():
    bad = _two_months()
    bad.loc[0, "month"] = "Aug-26"
    with pytest.raises(ValueError, match="row 2"):
        validate_kpis(bad)


def test_totals_and_snapshot_on_sample():
    k = compute_kpis(validate_kpis(pd.read_csv(SAMPLE)))
    snap = latest_snapshot(k)
    t = portfolio_totals(k)
    assert len(snap) == 8 and t["companies"] == 8
    assert abs(t["revenue"] - snap["revenue"].sum()) < 1e-6
    assert t["critical"] + t["warning"] == (snap["runway_status"] != "good").sum()
    assert t["revenue_yoy"] is not None and -1 < t["revenue_yoy"] < 5


def test_page_builds_and_embeds_data(tmp_path: Path):
    out = build_page(SAMPLE, "Test Portfolio", tmp_path / "index.html")
    html = out.read_text()
    assert "<title>Test Portfolio</title>" in html and "__DATA__" not in html
    payload = build_payload(SAMPLE, "Test Portfolio")
    assert json.dumps(payload) in html
    assert set(payload["series"]) == set(payload["companies"]) == set(payload["snapshot"])
    assert all(len(payload["series"][c]["month"]) == 24 for c in payload["companies"])
