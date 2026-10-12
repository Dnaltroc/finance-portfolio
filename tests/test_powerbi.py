from pathlib import Path

import pandas as pd

from finlib.models.kpi import compute_kpis, portfolio_totals, validate_kpis
from projects.powerbi import build_tables, expected_values, write_tables
from projects.powerbi.cli import KPI_CSV

COMMITTED = Path(__file__).resolve().parents[1] / "projects" / "powerbi" / "data"


def test_committed_csvs_match_a_fresh_export(tmp_path):
    """The CSVs Power BI reads are exactly what the export writes today (no stale data)."""
    for p in write_tables(tmp_path):
        assert p.read_text() == (COMMITTED / p.name).read_text(), f"{p.name} is stale: run make export-powerbi"


def test_star_schema_keys_resolve():
    t = build_tables()
    cal = set(t["calendar"]["month"])
    assert set(t["variance_fact"]["month"]) <= cal and set(t["kpi_fact"]["month"]) <= cal
    assert set(t["variance_fact"]["department"]) == set(t["dim_department"]["department"])
    assert set(t["variance_fact"]["account"]) == set(t["dim_account"]["account"])
    assert set(t["kpi_fact"]["company"]) == set(t["dim_company"]["company"])
    for dim, key in [("calendar", "month"), ("dim_department", "department"), ("dim_account", "account"),
                     ("dim_company", "company")]:
        assert t[dim][key].is_unique, dim
    # The calendar has no gaps, so "12 months back" in DAX is month_index - 12.
    assert (t["calendar"]["month_index"].diff().dropna() == 1).all()


def test_every_fact_row_carries_its_source():
    t = build_tables()
    assert (t["variance_fact"]["data_source"] == "Sample data (synthetic)").all()
    assert (t["kpi_fact"]["data_source"] == "Sample data (synthetic)").all()
    assert t["comps"]["price_source"].notna().all() and t["comps"]["debt_source"].notna().all()
    assert t["aehr_price"]["price_source"].notna().all()


def test_variance_totals_equal_finlib():
    """What a Power BI SUM over the fact table shows equals pnl_summary."""
    v, ev = build_tables()["variance_fact"], expected_values()
    for m in v["month"].unique():
        rows = v[v["month"] == m]
        sign = rows["is_revenue"].map({True: 1.0, False: -1.0})
        net_actual = (rows["actual"] * sign).sum()
        assert abs(net_actual - ev[f"variance {m}"]["Net"]["actual"]) < 1e-6
        assert abs(rows["favorable_amt"].sum() - ev[f"variance {m}"]["Net"]["favorable"]) < 1e-6
        assert int(rows["flag"].sum()) == ev[f"variance {m}"]["flagged_lines"]


def test_kpi_fact_equals_finlib():
    k = build_tables()["kpi_fact"]
    direct = compute_kpis(validate_kpis(pd.read_csv(KPI_CSV)))
    pd.testing.assert_frame_equal(k.drop(columns="data_source"), direct)
    tot = portfolio_totals(direct)
    latest = k[k["month"] == tot["month"]]
    assert latest["revenue"].sum() == tot["revenue"] and latest["cash"].sum() == tot["cash"]


def test_comps_peer_median_matches_the_memo():
    t = build_tables()["comps"]
    assert list(t.loc[t["role"] == "Subject", "ticker"]) == ["AEHR"]
    assert round(expected_values()["comps peer median"]["ev_revenue"], 1) == 8.8  # memo: 8.8x
    assert t.loc[t["ticker"].isin(["ONTO", "CAMT"]), "ev_revenue"].isna().all()  # NM stays blank
