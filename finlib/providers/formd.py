"""SEC Form D provider: daily index -> each filing's primary_doc.xml -> parsed record.

Live mode respects the SEC's access rules: a descriptive User-Agent with contact email
(SEC_IDENTITY) and a request rate under 10 per second (we cap at 8). Offline mode reads
tests/fixtures/formd/<YYYY-MM-DD>/ which mirrors the live layout: form.idx plus one
primary_doc.xml per accession.

Form D fields parsed (edgarSubmission schema): issuer name, CIK, entity type, state,
year of incorporation, industry group, offering amounts, investor count, exemption,
amendment flag, date of first sale, security types. No related-person (officer/director)
data is parsed or stored.
"""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from finlib import offline

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "formd"
SEC_ARCHIVE = "https://www.sec.gov/Archives"
RATE_PER_SEC = 8


@dataclass
class FormD:
    accession: str
    cik: str
    issuer: str
    entity_type: str
    state: str
    year_of_inc: str
    industry_group: str
    is_amendment: bool
    date_of_first_sale: str
    exemption: str
    equity: bool
    debt: bool
    total_offering: float | None
    total_sold: float | None
    total_remaining: float | None
    investors: int | None
    min_investment: float | None
    filing_date: str
    url: str

    def as_dict(self) -> dict:
        return asdict(self)


def _text(root: ET.Element, path: str) -> str | None:
    """Namespace-agnostic find by local tag names, e.g. 'primaryIssuer/entityName'."""
    node = root
    for part in path.split("/"):
        node = next((c for c in node if c.tag.split("}")[-1] == part), None)
        if node is None:
            return None
    return (node.text or "").strip() if node is not None else None


def _num(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    if v.lower() in ("indefinite", "n/a"):
        return None
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return None


def parse_form_d(xml_text: str, accession: str, filing_date: str, url: str) -> FormD:
    root = ET.fromstring(xml_text)
    exemptions = [c.text.strip() for e in root.iter() if e.tag.split("}")[-1] == "federalExemptionsExclusions"
                  for c in e if c.text]
    inv = _text(root, "offeringData/investors/totalNumberAlreadyInvested")
    return FormD(
        accession=accession,
        cik=(_text(root, "primaryIssuer/cik") or "").zfill(10),
        issuer=_text(root, "primaryIssuer/entityName") or "",
        entity_type=_text(root, "primaryIssuer/entityType") or "",
        state=_text(root, "primaryIssuer/issuerAddress/stateOrCountry") or "",
        year_of_inc=_text(root, "primaryIssuer/yearOfInc/value") or "",
        industry_group=_text(root, "offeringData/industryGroup/industryGroupType") or "",
        is_amendment=(_text(root, "offeringData/typeOfFiling/newOrAmendment/isAmendment") or "false").lower() == "true",
        date_of_first_sale=_text(root, "offeringData/typeOfFiling/dateOfFirstSale/value") or "",
        exemption=",".join(exemptions),
        equity=(_text(root, "offeringData/typesOfSecuritiesOffered/isEquityType") or "false").lower() == "true",
        debt=(_text(root, "offeringData/typesOfSecuritiesOffered/isDebtType") or "false").lower() == "true",
        total_offering=_num(_text(root, "offeringData/offeringSalesAmounts/totalOfferingAmount")),
        total_sold=_num(_text(root, "offeringData/offeringSalesAmounts/totalAmountSold")),
        total_remaining=_num(_text(root, "offeringData/offeringSalesAmounts/totalRemaining")),
        investors=int(inv) if inv and inv.isdigit() else None,
        min_investment=_num(_text(root, "offeringData/minimumInvestmentAccepted")),
        filing_date=filing_date,
        url=url,
    )


IDX_LINE = re.compile(r"^(?P<form>D(?:/A)?)\s{2,}(?P<name>.+?)\s{2,}(?P<cik>\d+)\s+(?P<date>\d{8})\s+(?P<path>edgar/data/\S+)$")


def parse_daily_index(text: str) -> list[dict]:
    """Rows from an EDGAR daily form index that are Form D or D/A."""
    rows = []
    for line in text.splitlines():
        m = IDX_LINE.match(line.strip())
        if m:
            rows.append(m.groupdict())
    return rows


def accession_from_path(path: str) -> str:
    # edgar/data/1234567/0001234567-26-000123.txt -> 0001234567-26-000123
    return Path(path).stem


class FormDProvider:
    def __init__(self, identity: str | None = None, fixture_dir: Path = FIXTURE_DIR):
        self.identity = identity or os.getenv("SEC_IDENTITY", "")
        self.fixture_dir = fixture_dir
        self._last = 0.0

    # ---- public ----------------------------------------------------------------------
    def filings_for(self, day: date) -> list[FormD]:
        if offline():
            return self._offline(day)
        return self._live(day)

    # ---- offline ---------------------------------------------------------------------
    def _offline(self, day: date) -> list[FormD]:
        folder = self.fixture_dir / day.isoformat()
        if not folder.exists():
            raise FileNotFoundError(f"No Form D fixture for {day} at {folder}")
        rows = parse_daily_index((folder / "form.idx").read_text())
        out = []
        for r in rows:
            acc = accession_from_path(r["path"])
            xml_path = folder / f"{acc}.xml"
            if not xml_path.exists():
                continue
            url = f"{SEC_ARCHIVE}/edgar/data/{int(r['cik'])}/{acc.replace('-', '')}/primary_doc.xml"
            out.append(parse_form_d(xml_path.read_text(), acc, _fmt_date(r["date"]), url))
        return out

    # ---- live ------------------------------------------------------------------------
    def _live(self, day: date) -> list[FormD]:
        import requests

        if not self.identity:
            raise RuntimeError("SEC_IDENTITY is required for live EDGAR calls (see .env.example)")
        headers = {"User-Agent": self.identity, "Accept-Encoding": "gzip, deflate"}
        q = (day.month - 1) // 3 + 1
        idx_url = f"{SEC_ARCHIVE}/edgar/daily-index/{day.year}/QTR{q}/form.{day.strftime('%Y%m%d')}.idx"
        resp = self._get(requests, idx_url, headers)
        if resp.status_code == 404:
            return []  # weekend or holiday
        resp.raise_for_status()
        out = []
        for r in parse_daily_index(resp.text):
            acc = accession_from_path(r["path"])
            url = f"{SEC_ARCHIVE}/edgar/data/{int(r['cik'])}/{acc.replace('-', '')}/primary_doc.xml"
            x = self._get(requests, url, headers)
            if x.status_code != 200:
                continue
            try:
                out.append(parse_form_d(x.text, acc, _fmt_date(r["date"]), url))
            except ET.ParseError:
                continue
        return out

    def _get(self, requests, url: str, headers: dict):
        gap = 1.0 / RATE_PER_SEC
        wait = self._last + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            resp = requests.get(url, headers=headers, timeout=30)
            self._last = time.monotonic()
            if resp.status_code != 429:
                return resp
            time.sleep(2 ** attempt)
        return resp


def _fmt_date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
