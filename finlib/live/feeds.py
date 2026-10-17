"""Free, keyless live feeds for one ticker, with a labeled offline fixture.

    market    Yahoo Finance through yfinance: last price, previous close, day range, volume,
              5-minute bars for the latest session, and a year of daily closes.
    news      Yahoo Finance's headline RSS for the ticker plus a Google News search for the
              company name, de-duplicated and sorted newest first.
    filings   SEC EDGAR submissions API (needs SEC_IDENTITY), newest first, linked to the archive.

FINLIB_OFFLINE=1 reads tests/fixtures/live/<TICKER>.json, which is synthetic and says so.
Nothing here caches: the page decides how often to refresh.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus

from finlib import offline

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "live"
YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GOOGLE_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (finance-portfolio tracker)"}

FORM_LABELS = {
    "4": "Insider transaction (Form 4)", "3": "Initial insider ownership (Form 3)",
    "144": "Proposed insider sale (Form 144)", "8-K": "Current report (8-K)",
    "10-Q": "Quarterly report (10-Q)", "10-K": "Annual report (10-K)", "DEF 14A": "Proxy statement",
    "S-3": "Shelf registration (S-3)", "S-3ASR": "Automatic shelf registration (S-3ASR)",
    "424B5": "Prospectus supplement (424B5)", "SC 13G": "5%+ holder (13G)",
    "SC 13G/A": "5%+ holder (13G/A)", "SCHEDULE 13G": "5%+ holder (13G)",
    "SCHEDULE 13G/A": "5%+ holder (13G/A)", "SCHEDULE 13D": "Activist 5%+ holder (13D)",
    "SCHEDULE 13D/A": "Activist 5%+ holder (13D/A)", "S-8": "Employee stock plan (S-8)",
    "ARS": "Annual report to shareholders", "DEFA14A": "Additional proxy material", "10-K/A": "Amended 10-K",
    "8-K/A": "Amended 8-K", "4/A": "Amended insider transaction (Form 4/A)", "SD": "Specialized disclosure (conflict minerals)",
}


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    published: str  # ISO 8601, UTC
    feed: str


@dataclass
class FilingItem:
    form: str
    filed: str
    description: str
    url: str


# ---------------------------------------------------------------- parsers (pure) -----
def parse_rss(xml: bytes, feed: str) -> list[NewsItem]:
    """RSS 2.0 items -> NewsItem. Google News appends " - Publisher" to titles; strip it."""
    out = []
    for it in ET.fromstring(xml).findall("./channel/item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        src = (it.findtext("source") or "").strip()
        if src and title.endswith(f" - {src}"):
            title = title[: -len(src) - 3].strip()
        try:
            when = parsedate_to_datetime(it.findtext("pubDate") or "").astimezone(timezone.utc)
            published = when.isoformat(timespec="seconds")
        except (TypeError, ValueError):
            published = ""
        out.append(NewsItem(title=title, url=(it.findtext("link") or "").strip(), source=src or feed,
                            published=published, feed=feed))
    return out


def _key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())[:60]


def merge_news(feeds: list[list[NewsItem]], limit: int = 12) -> list[NewsItem]:
    """One list, newest first, the same story kept once (first feed wins)."""
    seen, merged = set(), []
    for items in feeds:
        for n in items:
            k = _key(n.title)
            if k not in seen:
                seen.add(k)
                merged.append(n)
    merged.sort(key=lambda n: n.published, reverse=True)
    return merged[:limit]


def parse_submissions(js: dict, limit: int = 12) -> list[FilingItem]:
    """SEC submissions JSON -> newest filings with archive links and a plain-English form label."""
    r, cik = js["filings"]["recent"], int(js["cik"])
    items = r.get("items") or [""] * len(r["form"])
    out = []
    for i in range(min(limit, len(r["form"]))):
        form = r["form"][i]
        desc = FORM_LABELS.get(form, form)
        if form == "8-K" and items[i]:
            desc += ", items " + ", ".join(x.strip() for x in items[i].split(","))
        out.append(FilingItem(form=form, filed=r["filingDate"][i], description=desc,
                              url=ARCHIVE.format(cik=cik, acc=r["accessionNumber"][i].replace("-", ""),
                                                 doc=r["primaryDocument"][i])))
    return out


# ---------------------------------------------------------------- fetchers -----------
class LiveFeeds:
    def __init__(self, identity: str | None = None, timeout: int = 15):
        self.identity = identity or os.getenv("SEC_IDENTITY", "")
        self.timeout = timeout

    @staticmethod
    def _fixture(ticker: str) -> dict:
        return json.loads((FIXTURE_DIR / f"{ticker.upper()}.json").read_text(encoding="utf-8"))

    def market(self, ticker: str) -> dict:
        """{"quote": {...}, "intraday": [[iso, close], ...], "daily": [[date, close], ...]}"""
        if offline():
            return self._fixture(ticker)["market"]
        import yfinance as yf  # in the [live] and [app] extras

        t = yf.Ticker(ticker)
        bars = t.history(period="1d", interval="5m")["Close"].dropna()
        daily = t.history(period="1y", interval="1d")["Close"].dropna()
        if bars.empty or daily.empty:
            raise RuntimeError(f"Yahoo Finance returned no prices for {ticker}")
        fi = t.fast_info
        quote = {"ticker": ticker.upper(), "price": float(fi["last_price"]),
                 "previous_close": float(fi["previous_close"]), "day_high": float(fi["day_high"]),
                 "day_low": float(fi["day_low"]), "volume": float(fi["last_volume"]),
                 "as_of": bars.index[-1].isoformat(), "source": "Yahoo Finance"}
        return {"quote": quote,
                "intraday": [[i.isoformat(), round(float(v), 2)] for i, v in bars.items()],
                "daily": [[i.date().isoformat(), round(float(v), 2)] for i, v in daily.items()]}

    def news(self, ticker: str, company: str, limit: int = 12) -> list[NewsItem]:
        if offline():
            return [NewsItem(**n) for n in self._fixture(ticker)["news"]][:limit]
        import requests

        feeds, errors = [], []
        query = quote_plus('"' + company + '"')
        for name, url in (("Yahoo Finance", YAHOO_RSS.format(ticker=ticker.upper())),
                          ("Google News", GOOGLE_RSS.format(query=query))):
            try:
                r = requests.get(url, headers=BROWSER_UA, timeout=self.timeout)
                r.raise_for_status()
                feeds.append(parse_rss(r.content, feed=name))
            except Exception as e:  # one feed down is fine; both down is an error
                errors.append(f"{name}: {e}")
        if not feeds:
            raise RuntimeError("; ".join(errors))
        return merge_news(feeds, limit)

    def filings(self, ticker: str, limit: int = 12) -> list[FilingItem]:
        if offline():
            return [FilingItem(**f) for f in self._fixture(ticker)["filings"]][:limit]
        if not self.identity.strip():
            raise RuntimeError("SEC_IDENTITY is not set; EDGAR requires a name and email")
        import requests

        h = {"User-Agent": self.identity}
        tickers = requests.get(TICKERS_URL, headers=h, timeout=self.timeout).json()
        cik = next((int(v["cik_str"]) for v in tickers.values() if v["ticker"].upper() == ticker.upper()),
                   None)
        if cik is None:
            raise LookupError(f"{ticker} not in SEC company_tickers.json")
        js = requests.get(SUBMISSIONS.format(cik=cik), headers=h, timeout=self.timeout).json()
        return parse_submissions(js, limit)
