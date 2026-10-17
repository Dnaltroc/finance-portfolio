"""AEHR live tracker: feed parsers, the move math, and the offline builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finlib.live.feeds import LiveFeeds, merge_news, parse_rss, parse_submissions
from finlib.models.tracker import pct_change, tracker_moves

FX = Path(__file__).parent / "fixtures" / "live"


# ---------------------------------------------------------------- math ---------------
def test_pct_change_by_hand():
    assert pct_change(110.0, 100.0) == pytest.approx(0.10)
    assert pct_change(52.0, 84.64) == pytest.approx(52 / 84.64 - 1)


def test_pct_change_without_a_reference_is_none():
    assert pct_change(10.0, None) is None
    assert pct_change(10.0, 0.0) is None
    assert pct_change(None, 10.0) is None


def test_tracker_moves_by_hand():
    m = tracker_moves(price=60.0, previous_close=64.0, memo_price=80.0, target=52.0)
    assert m["day_change"] == pytest.approx(-4.0)
    assert m["day_pct"] == pytest.approx(-0.0625)
    assert m["since_memo_pct"] == pytest.approx(-0.25)
    assert m["to_target_pct"] == pytest.approx(52 / 60 - 1)  # what is left to the target from here


# ---------------------------------------------------------------- parsers ------------
def test_parse_yahoo_rss_skips_untitled_items_and_reads_utc_times():
    items = parse_rss((FX / "yahoo_rss.xml").read_bytes(), feed="Yahoo Finance")
    assert [i.title for i in items] == ["Example Co. receives follow-on production orders",
                                        "Implied volatility rises for Example Co. options"]
    assert items[0].published == "2026-10-07T11:30:00+00:00"
    assert items[0].source == "Yahoo Finance"  # no <source> tag: the feed is the source


def test_parse_google_rss_strips_the_publisher_suffix_from_titles():
    items = parse_rss((FX / "google_rss.xml").read_bytes(), feed="Google News")
    assert items[0].title == "Example Co. Receives Follow-On Production Orders"
    assert items[0].source == "Sample Wire"
    assert items[1].published == "2026-10-16T07:59:08+00:00"


def test_merge_news_dedupes_across_feeds_and_sorts_newest_first():
    yahoo = parse_rss((FX / "yahoo_rss.xml").read_bytes(), feed="Yahoo Finance")
    google = parse_rss((FX / "google_rss.xml").read_bytes(), feed="Google News")
    merged = merge_news([yahoo, google], limit=10)
    titles = [i.title.lower() for i in merged]
    assert titles.count("example co. receives follow-on production orders") == 1  # same story, two feeds
    assert [i.published for i in merged] == sorted((i.published for i in merged), reverse=True)
    assert len(merge_news([yahoo, google], limit=2)) == 2


def test_parse_submissions_builds_archive_links_and_labels_forms():
    js = json.loads((FX / "submissions.json").read_text(encoding="utf-8"))
    f = parse_submissions(js, limit=10)
    assert [x.form for x in f] == ["4", "8-K", "10-K"]
    assert f[0].url == ("https://www.sec.gov/Archives/edgar/data/1040470/000126000003/"
                        "xslF345X06/primary_01.xml")
    assert f[0].description == "Insider transaction (Form 4)"
    assert f[1].description == "Current report (8-K), items 7.01, 9.01"
    assert len(parse_submissions(js, limit=1)) == 1


# ---------------------------------------------------------------- offline feeds ------
def test_offline_feeds_read_the_labeled_fixture(monkeypatch):
    monkeypatch.setenv("FINLIB_OFFLINE", "1")
    feeds = LiveFeeds()
    m = feeds.market("AEHR")
    assert m["quote"]["source"] == "fixture"
    assert len(m["intraday"]) > 10 and len(m["daily"]) > 200
    assert all("Sample headline" in n.title for n in feeds.news("AEHR", "Aehr Test Systems"))
    assert feeds.filings("AEHR")[0].form == "4"


# ---------------------------------------------------------------- builder ------------
def test_build_tracker_offline_ties_live_price_to_the_memo(monkeypatch):
    monkeypatch.setenv("FINLIB_OFFLINE", "1")
    from projects.pitch import cli as pitch
    from projects.pitch.tracker import build_tracker

    cfg = pitch.load_config("aehr")
    snap = json.loads(pitch.snapshot_path("aehr").read_text(encoding="utf-8"))
    c = pitch.compute(cfg, snap)
    t = build_tracker("aehr")
    q = t["market"]["quote"]
    assert t["memo"]["price"] == pytest.approx(c["price"])
    assert t["memo"]["target"] == c["target"]
    assert t["moves"] == tracker_moves(q["price"], q["previous_close"], c["price"], c["target"])
    assert t["market"]["sample"] and t["news"]["sample"] and t["filings"]["sample"]


def test_build_tracker_falls_back_per_section_and_says_why(monkeypatch):
    monkeypatch.setenv("FINLIB_OFFLINE", "0")
    from projects.pitch.tracker import build_tracker

    class Broken(LiveFeeds):
        def news(self, ticker, company, limit=12):
            raise RuntimeError("feed down")

        def market(self, ticker):
            return LiveFeeds._fixture("AEHR")["market"] | {}

        def filings(self, ticker, limit=12):
            return []

    t = build_tracker("aehr", feeds=Broken())
    assert t["news"]["sample"] is True
    assert "feed down" in t["news"]["error"]
    assert all("Sample headline" in n["title"] for n in t["news"]["items"])
    assert t["market"]["sample"] is False  # this section succeeded, so it is not labeled sample
