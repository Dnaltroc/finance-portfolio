"""Streamlit app tests: every page renders on sample data, shows the same headline numbers as a
direct finlib call, and its download buttons produce the same non-empty files the CLIs write."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from openpyxl import load_workbook

pytest.importorskip("streamlit")
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))  # pages import fp_ui, as they do under `streamlit run app/Home.py`

import fp_ui as ui  # noqa: E402


@pytest.fixture
def downloads(monkeypatch):
    """Capture every file handed to st.download_button: {filename: bytes}."""
    files: dict[str, bytes] = {}
    original = MemoryMediaFileStorage.load_and_get_id

    def spy(self, path_or_data, mimetype, kind, filename=None):
        data = path_or_data if isinstance(path_or_data, bytes) else Path(path_or_data).read_bytes()
        files[filename or f"unnamed.{mimetype}"] = data
        return original(self, path_or_data, mimetype, kind, filename)

    monkeypatch.setattr(MemoryMediaFileStorage, "load_and_get_id", spy)
    return files


def run_page(name: str) -> AppTest:
    at = AppTest.from_file(str(APP / name), default_timeout=120).run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


def metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def sample_labeled(at: AppTest) -> bool:
    return any(ui.SAMPLE_LABEL in m.value for m in at.markdown)


# ---------------------------------------------------------------- formatting ---------
def test_formatters_follow_excel_conventions():
    assert ui.fmt_musd(1234.56) == "$1,234.6M"
    assert ui.fmt_musd(-12.3) == "($12.3M)"
    assert ui.fmt_x(12.34) == "12.3x" and ui.fmt_x(None) == "NM" and ui.fmt_x(float("nan")) == "NM"
    assert ui.fmt_pct(-0.0534) == "(5.3%)"
    assert ui.fmt_usd(-190288.48) == "($190,288)"


def test_upload_names_are_escaped_before_markdown():
    out = ui.escape_text("a</small><img src=x onerror=alert(1)>_*b*:material/x:.csv")
    assert "<" not in out and ">" not in out
    assert r"\_\*b\*\:material/x\:" in out


# ---------------------------------------------------------------- home ---------------
def test_router_opens_on_overview():
    at = run_page("Home.py")
    assert "Finance tools that tie to the filing." in [t.value for t in at.title]


def test_overview_lists_five_tools_with_resume_lines_and_live_tape():
    from datetime import date

    from finlib.models.lbo import LBOAssumptions, run_lbo
    from finlib.models.ratios import summary_rows
    from finlib.models.sourcing import funnel, rank
    from finlib.providers.formd import FormDProvider
    from projects.comps.cli import load_all
    from projects.dashboard.cli import HERE as DASH, payload_from_frame
    from projects.sourcing.cli import HERE as SRC, load_config
    from projects.variance.cli import HERE as VAR, build_variance, load_thresholds

    at = run_page("Home.py")  # page links resolve only through the router
    captions = " ".join(c.value for c in at.caption)
    assert captions.count("Resume line:") == 6
    assert "Resume line: **" not in captions  # every README has its line
    assert "How the numbers are verified" in [s.value for s in at.subheader]
    tape = metrics(at)
    lbo = run_lbo(LBOAssumptions(**yaml.safe_load((ROOT / "projects/lbo/deal.yaml").read_text())))
    assert tape["03 · LBO engine"] == ui.fmt_pct(lbo.irr) == "20.6%"
    _, _, rows = load_all(["AAPL", "MSFT", "GOOGL"])
    assert tape["01 · Comps builder"] == ui.fmt_x(summary_rows(rows)[0]["ev_ebitda"]) == "20.8x"
    stages = funnel(rank(FormDProvider().filings_for(date(2026, 10, 9)), load_config(SRC / "config.yaml")[0], set()))
    assert tape["04 · Deal sourcing"] == f"{stages[-1][1]} of {stages[0][1]}" == "25 of 36"
    pack = build_variance(pd.read_csv(VAR / "sample_data/actuals.csv"), pd.read_csv(VAR / "sample_data/budget.csv"),
                          "2026-09", load_thresholds(VAR / "config.yaml"))
    net = pack.summary.set_index("line").loc["Net"]
    assert tape["02 · Variance report"] == ui.fmt_usd(net["variance_amt"]) == "($190,288)"
    totals = payload_from_frame(pd.read_csv(DASH / "sample_data/kpis.csv"), "")["totals"]
    assert tape["05 · Portfolio KPIs"] == f"{totals['revenue_yoy']:+.0%}"
    from projects.filing_tone.cli import load_snapshot
    tone = load_snapshot()["companies"]["AEHR"]["sections"]["mdna"]["net_tone"]
    assert tape["06 · Filing tone"] == f"{tone:+.2f}"


# ---------------------------------------------------------------- comps --------------
def test_comps_matches_finlib_and_downloads_workbook(downloads):
    from finlib.models.ratios import summary_rows
    from projects.comps.cli import load_all

    at = run_page("pages/1_Comps_Builder.py")
    assert sample_labeled(at)
    _, _, rows = load_all(["AAPL", "MSFT", "GOOGL"])
    med = summary_rows(rows)[0]
    m = metrics(at)
    assert m["Median EV / EBITDA"] == ui.fmt_x(med["ev_ebitda"]) == "20.8x"
    assert m["Median P / E"] == ui.fmt_x(med["pe"])
    assert m["Median EV / Revenue"] == ui.fmt_x(med["ev_revenue"])
    table = at.dataframe[0].value
    assert list(table["EV / EBITDA"][:3]) == [ui.fmt_x(r.ev_ebitda) for r in rows]

    name = next(n for n in downloads if n.endswith(".xlsx"))
    wb = load_workbook(io.BytesIO(downloads[name]))
    assert wb.sheetnames == ["Comps", "Inputs", "Notes"]
    assert wb["Comps"]["A5"].value == "AAPL"


def test_comps_offline_rejects_unknown_ticker_with_message():
    at = AppTest.from_file(str(APP / "pages/1_Comps_Builder.py"), default_timeout=120).run()
    at.sidebar.text_input[0].set_value("AAPL,ZZZZ").run()
    assert any("No offline fixture for ZZZZ" in e.value for e in at.error)


def test_live_mode_without_identity_falls_back_to_labeled_sample(monkeypatch):
    monkeypatch.delenv("SEC_IDENTITY", raising=False)
    monkeypatch.setattr(ui, "load_secrets", lambda: None)  # ignore any local .env
    at = AppTest.from_file(str(APP / "pages/1_Comps_Builder.py"), default_timeout=120).run()
    at.sidebar.segmented_control[0].set_value("Live").run()
    assert any("SEC_IDENTITY" in w.value for w in at.sidebar.warning)
    assert sample_labeled(at)
    assert metrics(at)["Median EV / EBITDA"] == "20.8x"


# ---------------------------------------------------------------- variance -----------
def test_variance_matches_finlib_and_downloads_pack(downloads):
    from finlib.models.variance import commentary
    from projects.variance.cli import HERE, build_variance, load_thresholds

    at = run_page("pages/2_Variance_Report.py")
    assert sample_labeled(at)
    sd = HERE / "sample_data"
    pack = build_variance(pd.read_csv(sd / "actuals.csv"), pd.read_csv(sd / "budget.csv"), "2026-09",
                          load_thresholds(HERE / "config.yaml"))
    shown = [m.value for m in at.metric]
    assert shown == [ui.fmt_usd(v) for v in pack.summary["actual"]]
    assert shown[2] == "($362,288)"

    assert downloads["commentary_2026-09.md"].decode() == commentary(pack.flags, "2026-09")
    wb = load_workbook(io.BytesIO(downloads["variance_2026-09.xlsx"]))
    assert wb.sheetnames == ["Summary", "Flags", "Detail", "Inputs"]
    assert wb["Flags"].max_row == 3 + len(pack.flags)


def test_variance_threshold_slider_changes_flag_count():
    at = run_page("pages/2_Variance_Report.py")
    def flagged(at):
        return next(c.value for c in at.caption if "lines flagged" in c.value).split(" lines")[0]

    before = flagged(at)
    at.sidebar.slider[0].set_value(5_000).run()
    assert before == "4" and int(flagged(at)) > 4


# ---------------------------------------------------------------- LBO ----------------
def test_lbo_matches_engine_and_downloads_model(downloads):
    from finlib.models.lbo import LBOAssumptions, run_lbo

    at = run_page("pages/3_LBO_Engine.py")
    a = LBOAssumptions(**yaml.safe_load((ROOT / "projects/lbo/deal.yaml").read_text()))
    r = run_lbo(a)
    m = metrics(at)
    assert m["MOIC"] == ui.fmt_x(r.moic, 2) == "2.55x"
    assert m["IRR"] == ui.fmt_pct(r.irr) == "20.6%"
    assert m["Sponsor equity"] == ui.fmt_musd(a.sponsor_equity)

    name = next(n for n in downloads if n.endswith(".xlsx"))
    wb = load_workbook(io.BytesIO(downloads[name]))
    assert wb.sheetnames == ["Assumptions", "Operating", "Debt", "Returns", "Sensitivity"]
    assert str(wb["Returns"]["C16"].value).startswith("=IRR(")


def test_lbo_blocks_returns_when_debt_exceeds_price():
    at = run_page("pages/3_LBO_Engine.py")
    at.sidebar.slider[0].set_value(4.0).run()  # 4x entry vs 5.5x debt
    m = metrics(at)
    assert m["MOIC"] == "n/a" and m["IRR"] == "n/a"
    assert any("debt exceeds the purchase price" in e.value for e in at.error)


def test_lbo_downside_scenario_switch():
    from finlib.models.lbo import LBOAssumptions, run_lbo

    at = run_page("pages/3_LBO_Engine.py")
    at.sidebar.selectbox[0].set_value("Downside").run()
    a = LBOAssumptions(**yaml.safe_load((ROOT / "projects/lbo/deal.yaml").read_text()))
    assert metrics(at)["IRR"] == ui.fmt_pct(run_lbo(a, "Downside").irr) == "0.4%"


# ---------------------------------------------------------------- sourcing -----------
def test_sourcing_matches_rank_and_downloads_digest(downloads):
    from datetime import date

    from finlib.models.sourcing import digest_markdown, funnel, kept_in_states, rank
    from finlib.providers.formd import FormDProvider
    from projects.sourcing.cli import HERE, load_config

    at = run_page("pages/4_Deal_Sourcing.py")
    assert sample_labeled(at)
    cfg, top_n = load_config(HERE / "config.yaml")
    rows = rank(FormDProvider().filings_for(date(2026, 10, 9)), cfg, set())
    m = metrics(at)
    assert m["Form D filings"] == str(len(rows)) == "36"
    assert m["Operating-company raises kept"] == str(funnel(rows)[-1][1]) == "25"
    assert m["In target states"] == str(kept_in_states(rows, cfg.target_states))
    assert "top 10" in " ".join(h.value for h in at.get("html") if "Exhibit 4" in str(h.value))
    assert downloads["digest_2026-10-09.md"].decode() == digest_markdown(rows, "2026-10-09", top_n, cfg)


def test_sourcing_weights_rescore_matches_engine():
    from dataclasses import replace
    from datetime import date

    from finlib.models.sourcing import rank
    from finlib.providers.formd import FormDProvider
    from projects.sourcing.cli import HERE, load_config

    at = run_page("pages/4_Deal_Sourcing.py")
    for s in at.sidebar.slider:
        if s.label.startswith("Target state"):
            s.set_value(0)
        if s.label.startswith("Amount sold"):
            s.set_value(100)
    at.run()
    assert not at.exception
    cfg, _ = load_config(HERE / "config.yaml")
    cfg = replace(cfg, weights={**cfg.weights, "state": 0, "amount": 100})
    kept = [r for r in rank(FormDProvider().filings_for(date(2026, 10, 9)), cfg, set()) if r["kept"]]
    table = at.dataframe[0].value
    assert list(table["Issuer"][:5]) == [r["issuer"] for r in kept[:5]]
    assert list(table["Score"][:5]) == [round(r["score"]) for r in kept[:5]]


# ---------------------------------------------------------------- dashboard ----------
def test_dashboard_matches_totals_and_downloads_page(downloads):
    from projects.dashboard.cli import HERE, payload_from_frame, render_page

    at = run_page("pages/5_KPI_Dashboard.py")
    assert sample_labeled(at)
    p = payload_from_frame(pd.read_csv(HERE / "sample_data" / "kpis.csv"), "Gulf Coast Portfolio KPIs")
    t = p["totals"]
    m = metrics(at)
    assert m[f"Portfolio revenue, {t['month']}"] == ui.fmt_usd(t["revenue"])
    assert m["Blended gross margin"] == ui.fmt_pct(t["gross_margin"])
    assert m["Needs attention"] == f"{t['critical'] + t['warning']} of {t['companies']}" == "3 of 8"
    assert downloads["index.html"].decode() == render_page(p)


# ---------------------------------------------------------------- filing tone --------
def test_filing_tone_page_shows_snapshot_numbers_without_torch():
    from projects.filing_tone import cli as ft

    snap = ft.load_snapshot()
    aehr = snap["companies"]["AEHR"]["sections"]
    at = run_page("pages/6_Filing_Tone.py")
    m = metrics(at)
    assert m["AEHR net tone, MD&A"] == f"{aehr['mdna']['net_tone']:+.2f}"
    assert m["AEHR net tone, risk factors"] == f"{aehr['risk_factors']['net_tone']:+.2f}"
    assert m["Peer median, MD&A"] == f"{ft.peer_median(snap, 'mdna'):+.2f}"
    assert m["Model macro-F1, test split"] == f"{ft.load_results()['distilbert']['macro_f1']:.3f}"
    assert not any(mod == "torch" or mod.startswith("torch.") for mod in sys.modules)
    assert any("Snapshot" in x.value for x in at.markdown)  # provenance badge
