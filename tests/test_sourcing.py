from datetime import date
from pathlib import Path

from finlib.models.sourcing import ScoringConfig, digest_markdown, is_fund, keep, rank, score
from finlib.providers.formd import FormD, FormDProvider, parse_daily_index, parse_form_d
from finlib.store.sourcing_db import SourcingDB
from projects.sourcing import run_day

DAY = date(2026, 10, 9)
CFG = Path(__file__).resolve().parents[1] / "projects" / "sourcing" / "config.yaml"


def _fd(**kw) -> FormD:
    base = dict(accession="0000000001-26-000001", cik="0000000001", issuer="Acme Robotics Inc",
                entity_type="Corporation", state="TX", year_of_inc="2021", industry_group="Technology",
                is_amendment=False, date_of_first_sale="2026-10-01", exemption="06b", equity=True, debt=False,
                total_offering=5e6, total_sold=3e6, total_remaining=2e6, investors=10, min_investment=0,
                filing_date="2026-10-09", url="https://example.invalid")
    base.update(kw)
    return FormD(**base)


def test_daily_index_keeps_only_form_d():
    text = (FormDProvider().fixture_dir / "2026-10-09" / "form.idx").read_text()
    rows = parse_daily_index(text)
    assert {r["form"] for r in rows} == {"D", "D/A"}
    assert len(rows) == 36


def test_parser_reads_core_fields():
    folder = FormDProvider().fixture_dir / "2026-10-09"
    xml_path = sorted(folder.glob("*.xml"))[0]
    f = parse_form_d(xml_path.read_text(), xml_path.stem, "2026-10-09", "u")
    assert f.issuer and f.cik.isdigit() and len(f.cik) == 10
    assert f.industry_group and f.total_sold is not None
    assert f.exemption == "06b"


def test_parser_handles_indefinite_offering():
    xml = """<edgarSubmission xmlns="http://www.sec.gov/edgar/formd"><primaryIssuer><cik>7</cik>
    <entityName>X</entityName></primaryIssuer><offeringData><offeringSalesAmounts>
    <totalOfferingAmount>Indefinite</totalOfferingAmount><totalAmountSold>1000000</totalAmountSold>
    </offeringSalesAmounts></offeringData></edgarSubmission>"""
    f = parse_form_d(xml, "a", "2026-10-09", "u")
    assert f.total_offering is None and f.total_sold == 1_000_000 and f.cik == "0000000007"


def test_fund_filter_by_industry_and_name():
    assert is_fund(_fd(issuer="Meridian Growth Partners III LP", industry_group="Pooled Investment Fund"))
    assert is_fund(_fd(issuer="Keel Ventures SPV 7 LLC", industry_group="Other"))
    assert not is_fund(_fd(issuer="Acme Robotics Inc", industry_group="Technology"))
    assert not is_fund(_fd(issuer="Pelican Fintech Holdings Inc", industry_group="Other Banking and Financial Services"))


def test_keep_rules():
    cfg = ScoringConfig()
    assert keep(_fd(), cfg) == (True, "")
    assert keep(_fd(is_amendment=True), cfg)[1] == "amendment"
    assert keep(_fd(total_sold=100_000), cfg)[1] == "below min amount"
    assert keep(_fd(industry_group="Pooled Investment Fund"), cfg)[1] == "fund"


def test_score_bounded_and_monotonic_in_amount():
    cfg = ScoringConfig()
    lo, _ = score(_fd(total_sold=500_000, investors=0, state="ZZ", industry_group="Other"), cfg, False)
    hi, parts = score(_fd(total_sold=60_000_000, investors=100), cfg, True)
    assert 0 <= lo < hi <= 100
    assert parts["amount"] == cfg.weights["amount"]  # capped at the ceiling
    assert parts["first_time"] == cfg.weights["first_time"]


def test_rank_puts_kept_first_and_digest_has_one_row_per_kept(tmp_path):
    filings = FormDProvider().filings_for(DAY)
    cfg = ScoringConfig()
    rows = rank(filings, cfg, seen_ciks=set())
    kept = [r for r in rows if r["kept"]]
    assert rows[: len(kept)] == kept
    assert all(not r["kept"] for r in rows[len(kept):])
    assert not any("Pooled Investment Fund" == r["industry_group"] for r in kept)
    md = digest_markdown(rows, DAY.isoformat(), 25, cfg)
    assert md.count("\n| ") - 2 == min(25, len(kept))  # minus header and separator rows


def test_db_dedupes_across_runs_and_marks_first_time(tmp_path):
    db = tmp_path / "s.sqlite"
    _, s1 = run_day(DAY, CFG, db, tmp_path)
    _, s2 = run_day(DAY, CFG, db, tmp_path)
    assert s1["new"] == s1["filings"] > 0
    assert s2["new"] == 0 and s2["db_total"] == s1["db_total"]
    # Re-running the same day must NOT strip the first-time bonus: only filings dated
    # before the day count as "seen". Scores are identical across the two runs.
    store = SourcingDB(db)
    assert all(r[0] == 1 for r in store.db.execute("SELECT first_time FROM filings WHERE kept=1"))
    d1 = (tmp_path / "digest_2026-10-09.md").read_text()
    _, _ = run_day(DAY, CFG, db, tmp_path)
    assert (tmp_path / "digest_2026-10-09.md").read_text() == d1


def test_holdings_inc_is_not_a_fund_but_feeder_lp_is():
    assert not is_fund(_fd(issuer="Pelican Fintech Holdings Inc", industry_group="Other", entity_type="Corporation"))
    assert is_fund(_fd(issuer="Cedar Hollow Feeder Fund LP", industry_group="", entity_type="Limited Partnership"))
    assert is_fund(_fd(issuer="Summit Coast Investors LLC", industry_group="Other", entity_type="Limited Partnership"))
    assert not is_fund(_fd(issuer="Summit Coast Investors LLC", industry_group="Other", entity_type="Limited Liability Company"))


def test_digest_file_written(tmp_path):
    path, stats = run_day(DAY, CFG, tmp_path / "s.sqlite", tmp_path)
    text = path.read_text()
    assert text.startswith("# Form D sourcing digest: 2026-10-09")
    assert "sec.gov/Archives/edgar/data" in text
    assert stats["kept"] >= 15


def test_funnel_counts_follow_drop_order():
    from finlib.models.sourcing import funnel

    rows = rank([
        _fd(accession="1", cik="1"),
        _fd(accession="2", cik="2", industry_group="Pooled Investment Fund"),
        _fd(accession="3", cik="3", is_amendment=True),
        _fd(accession="4", cik="4", total_sold=1_000),
        _fd(accession="5", cik="5"),
    ], ScoringConfig(), set())
    assert funnel(rows) == [("Form D filings", 5), ("After fund filter", 4),
                            ("After amendment filter", 3), ("Kept (at or above min amount)", 2)]


def test_kept_in_states_counts_only_kept_rows():
    from finlib.models.sourcing import kept_in_states

    rows = rank([
        _fd(accession="1", cik="1", state="TX"),
        _fd(accession="2", cik="2", state="TX", industry_group="Pooled Investment Fund"),
        _fd(accession="3", cik="3", state="LA"),
        _fd(accession="4", cik="4", state="CA"),
    ], ScoringConfig(), set())
    assert kept_in_states(rows, ["TX", "LA"]) == 2  # the TX fund is dropped, CA is not a target
    assert kept_in_states(rows, []) == 0
