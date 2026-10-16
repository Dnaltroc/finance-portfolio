"""Real 10-K text in, a tone number out.

A 10-K is an HTML document. The steps:
    html_to_text       flatten the HTML to plain text, one block element per line
    extract_sections   cut out Item 1A (Risk Factors) and Item 7 (MD&A, management's own
                       narrative of the year). Every heading appears at least twice, once in
                       the table of contents and once in the body, so we keep the longest span.
    split_sentences    sentences of 6 to 120 words that are mostly letters: prose, not tables
    filing_tone        score every sentence with the classifier, then summarize:
                       net tone = share of sentences positive - share negative, in [-1, 1],
                       with the sentences the model was most sure about kept as evidence.

TenKFetcher finds the newest 10-K through the SEC submissions API with the same etiquette as
finlib.providers.edgar (identity header, under 10 requests a second, cached). Offline it
reads the synthetic fixture.
"""

from __future__ import annotations

import html as html_lib
import json
import os
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

from finlib import offline
from finlib.store.cache import JsonCache

# ---------------------------------------------------------------- html -> text --------
BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "td", "th"}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html)
    p.close()
    text = html_lib.unescape("".join(p.parts)).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


# ---------------------------------------------------------------- sections ------------
_DASH = r"[\-–—:]?"
# End headings must start a line: html_to_text puts every block element on its own line, while an
# in-text cross-reference ("see Part II, Item 8 - Financial Statements") sits mid-paragraph.
SECTIONS = {
    "risk_factors": (
        re.compile(rf"item\s*1a\.?\s*{_DASH}\s*risk\s+factors", re.I),
        re.compile(rf"^\s*(?:item\s*1b\.?|item\s*2\.?\s*{_DASH}\s*propert)", re.I | re.M),
        "Item 1A",
    ),
    "mdna": (
        re.compile(rf"item\s*7\.?\s*{_DASH}\s*management['’]?s\s+discussion", re.I),
        re.compile(rf"^\s*(?:item\s*7a\.?|item\s*8\.?\s*{_DASH}\s*financial\s+statements)", re.I | re.M),
        "Item 7",
    ),
}
MIN_SECTION_CHARS = 200


def extract_sections(text: str) -> tuple[dict[str, str], list[str]]:
    """Among every start-heading -> next end-heading span, keep the longest: the body, not the
    table of contents. A section that is missing or tiny is returned empty with a warning,
    never guessed."""
    out, warnings = {}, []
    for key, (start_re, end_re, label) in SECTIONS.items():
        best = ""
        for m in start_re.finditer(text):
            e = end_re.search(text, m.end())
            span = text[m.end(): e.start() if e else len(text)].strip()
            if len(span) > len(best):
                best = span
        if len(best) < MIN_SECTION_CHARS:
            warnings.append(f"{label} not found or under {MIN_SECTION_CHARS} characters; section left empty")
            best = ""
        out[key] = best
    return out, warnings


# ---------------------------------------------------------------- sentences -----------
_ABBR = ["Inc.", "Corp.", "Ltd.", "Co.", "LLC.", "U.S.", "U.K.", "No.", "vs.", "Mr.", "Ms.", "Dr.", "e.g.", "i.e.",
         "approx.", "Jan.", "Feb.", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."]
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
# Page breaks leave "<page number> Table of Contents" glued to the next paragraph.
_FURNITURE = re.compile(r"(?:^|\s)\d{1,3}\s*Table of Contents\s+")


def split_sentences(text: str, min_words: int = 6, max_words: int = 120) -> list[str]:
    """Split on sentence-ending punctuation followed by a capital, after hiding the periods
    inside common abbreviations. Decimals never split because no space follows their period."""
    protected = _FURNITURE.sub(" ", text)
    for i, a in enumerate(_ABBR):
        protected = protected.replace(a, a[:-1] + f"\x00{i}\x00")
    out = []
    for raw in _SPLIT.split(re.sub(r"\s+", " ", protected)):
        s = raw
        for i, a in enumerate(_ABBR):
            s = s.replace(a[:-1] + f"\x00{i}\x00", a)
        s = s.strip()
        words = s.split()
        if not (min_words <= len(words) <= max_words):
            continue
        compact = s.replace(" ", "")
        if sum(ch.isalpha() for ch in compact) < 0.5 * len(compact):  # tables and page furniture
            continue
        out.append(s)
    return out


# ---------------------------------------------------------------- tone ----------------
def filing_tone(probs: np.ndarray, sentences: list[str], top_k: int = 5) -> dict:
    n = len(sentences)
    if n == 0:
        return {"n": 0, "share_positive": 0.0, "share_negative": 0.0, "share_neutral": 0.0, "net_tone": 0.0,
                "top_positive": [], "top_negative": []}
    pred = probs.argmax(1)
    share = {c: float((pred == c).mean()) for c in range(3)}

    def top(col: int) -> list[dict]:
        idx = np.argsort(-probs[:, col])[:top_k]
        return [{"sentence": sentences[i], "p": round(float(probs[i, col]), 4)} for i in idx]

    return {"n": n, "share_positive": share[2], "share_negative": share[0], "share_neutral": share[1],
            "net_tone": share[2] - share[0], "top_positive": top(2), "top_negative": top(0)}


# ---------------------------------------------------------------- fetching ------------
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "filing_tone"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}"


@dataclass
class Filing:
    ticker: str
    company: str
    cik: str
    accession: str
    form: str
    report_date: str
    filed: str
    url: str
    html: str


class TenKFetcher:
    def __init__(self, identity: str | None = None, fixture_dir: Path = FIXTURE_DIR,
                 cache_dir: Path = Path("output/filing_tone/cache")):
        self.identity = identity or os.getenv("SEC_IDENTITY", "")
        self.fixture_dir = fixture_dir
        self.cache_dir = cache_dir
        self.cache = JsonCache("filing_tone", ttl_seconds=24 * 3600)
        self._last = 0.0

    def latest_10k(self, ticker: str) -> Filing:
        ticker = ticker.upper()
        return self._fixture(ticker) if offline() else self._live(ticker)

    def _fixture(self, ticker: str) -> Filing:
        table = json.loads((self.fixture_dir / "filings.json").read_text(encoding="utf-8"))
        if ticker not in table:
            raise LookupError(f"No fixture filing for {ticker}")
        r = table[ticker]
        return Filing(ticker, r["company"], r["cik"], r["accession"], r["form"], r["report_date"], r["filed"],
                      f"fixture:{r['document']}", (self.fixture_dir / r["document"]).read_text(encoding="utf-8"))

    def _get(self, url: str, as_json: bool):
        import requests

        if not self.identity:
            raise RuntimeError("SEC_IDENTITY is required for live EDGAR calls (see .env.example)")
        if as_json and (cached := self.cache.get(url)) is not None:
            return cached
        wait = self._last + 0.125 - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        r = requests.get(url, headers={"User-Agent": self.identity, "Accept-Encoding": "gzip, deflate"}, timeout=60)
        self._last = time.monotonic()
        r.raise_for_status()
        if as_json:
            data = r.json()
            self.cache.set(url, data)
            return data
        return r.text

    def _live(self, ticker: str) -> Filing:
        table = self._get(TICKERS_URL, True)
        row = next((v for v in table.values() if v["ticker"].upper() == ticker), None)
        if row is None:
            raise LookupError(f"{ticker} not in SEC company_tickers.json")
        cik = str(row["cik_str"]).zfill(10)
        rec = self._get(SUBMISSIONS.format(cik=cik), True)["filings"]["recent"]
        for i, form in enumerate(rec["form"]):
            if form == "10-K":
                acc, doc = rec["accessionNumber"][i], rec["primaryDocument"][i]
                url = ARCHIVE.format(cik_int=int(cik), acc=acc.replace("-", ""), doc=doc)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cached = self.cache_dir / f"{acc}.html"
                if cached.exists():
                    html = cached.read_text(encoding="utf-8")
                else:
                    html = self._get(url, False)
                    cached.write_text(html, encoding="utf-8")
                return Filing(ticker, row["title"], cik, acc, form, rec["reportDate"][i], rec["filingDate"][i], url, html)
        raise LookupError(f"{ticker} ({row['title']}) has no 10-K in recent filings (foreign filers use 20-F)")
