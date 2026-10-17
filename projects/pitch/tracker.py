"""AEHR live tracker: today's price, news and filings set against the memo's price and target.

The memo was priced on one day; this asks what has happened since. Each section (market,
news, filings) is fetched on its own. If one fails, that section falls back to the synthetic
fixture, is labeled sample, and carries the error, while the others stay live.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from finlib import offline
from finlib.live.feeds import LiveFeeds
from finlib.models.tracker import tracker_moves
from projects.pitch.cli import compute, load_config, snapshot_path


def memo_numbers(name: str = "aehr") -> dict:
    """The memo's price, pricing date and 12-month target, computed from the committed snapshot."""
    cfg = load_config(name)
    snap = json.loads(snapshot_path(name).read_text(encoding="utf-8"))
    c = compute(cfg, snap)
    return {"ticker": cfg["ticker"], "company": cfg.get("company", cfg["ticker"]), "price": c["price"],
            "as_of": snap["quote"]["as_of"], "target": c["target"]}


def _section(fetch, fallback) -> dict:
    if offline():
        return {"data": fallback(), "sample": True, "error": ""}
    try:
        return {"data": fetch(), "sample": False, "error": ""}
    except Exception as e:  # say what failed, show labeled sample data instead
        return {"data": fallback(), "sample": True, "error": f"{type(e).__name__}: {e}"}


def build_tracker(name: str = "aehr", feeds: LiveFeeds | None = None, memo: dict | None = None,
                  news_limit: int = 12, filings_limit: int = 12) -> dict:
    feeds = feeds or LiveFeeds()
    memo = memo or memo_numbers(name)
    t = memo["ticker"]
    fx = LiveFeeds._fixture(t)
    m = _section(lambda: feeds.market(t), lambda: fx["market"])
    n = _section(lambda: [asdict(x) for x in feeds.news(t, memo["company"], news_limit)], lambda: fx["news"])
    f = _section(lambda: [asdict(x) for x in feeds.filings(t, filings_limit)], lambda: fx["filings"])
    q = m["data"]["quote"]
    return {
        "ticker": t, "company": memo["company"], "memo": memo,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "market": {**m["data"], "sample": m["sample"], "error": m["error"]},
        "moves": tracker_moves(q["price"], q["previous_close"], memo["price"], memo["target"]),
        "news": {"items": n["data"], "sample": n["sample"], "error": n["error"]},
        "filings": {"items": f["data"], "sample": f["sample"], "error": f["error"]},
    }
