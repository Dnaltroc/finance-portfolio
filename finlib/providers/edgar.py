"""SEC EDGAR fundamentals provider built on the XBRL companyfacts API.

Live:    https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json (free, no key).
         SEC requires a descriptive User-Agent with contact info (SEC_IDENTITY) and
         fewer than 10 requests per second.
Offline: tests/fixtures/comps/<TICKER>.json with eight quarters per concept.

LTM (last twelve months) for flow concepts, computed per concept from the filed facts:
    1. If a 12-month value ends on the latest period end, use it (a 10-K).
    2. Otherwise (a 10-Q): LTM = prior fiscal year + year-to-date this year
                             - year-to-date prior year.
    Companies do not tag Q4 as a 3-month period, so summing four quarters fails for
    most tickers; this method works for every filer because every 10-Q carries YTD.
Prior-year LTM (for growth) is the same calculation one year earlier.
Balance-sheet concepts are instants: the latest reported value.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from finlib import offline
from finlib.store.cache import JsonCache

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "comps"
COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

FLOW_CONCEPTS = ["revenue", "cogs", "operating_income", "da", "net_income"]
STOCK_CONCEPTS = ["cash", "total_debt"]

# XBRL concept resolution. For each concept, try the `total` tags in order; if none
# is present, add up the `components` (each component = first tag found in its list).
# Components marked optional may be absent (a company with no commercial paper).
XBRL_TAGS: dict[str, dict] = {
    "revenue": {"total": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"]},
    "cogs": {"total": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"]},
    "operating_income": {"total": ["OperatingIncomeLoss"]},
    "da": {"total": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                     "DepreciationAmortizationAndAccretionNet", "DepreciationAmortizationAndOther"],
           "components": [(["Depreciation"], False), (["AmortizationOfIntangibleAssets"], True)]},
    "net_income": {"total": ["NetIncomeLoss", "ProfitLoss"]},
    # Cash = cash and equivalents plus short-term investments (street convention for EV).
    "cash": {"total": ["CashCashEquivalentsAndShortTermInvestments"],
             "components": [(["CashAndCashEquivalentsAtCarryingValue",
                              "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"], False),
                            (["MarketableSecuritiesCurrent", "ShortTermInvestments",
                              "AvailableForSaleSecuritiesCurrent"], True)]},
    # Debt = every borrowing line: long-term, current portion, commercial paper, short-term.
    "total_debt": {"components": [(["LongTermDebtNoncurrent", "LongTermDebt"], False),
                                  (["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent",
                                    "DebtCurrent"], True),
                                  (["CommercialPaper"], True), (["ShortTermBorrowings"], True)]},
}


@dataclass
class Fundamentals:
    ticker: str
    company: str
    cik: str
    ltm_values: dict[str, float | None]
    prior_ltm_values: dict[str, float | None]
    latest_values: dict[str, float | None]
    ltm_period: str                       # e.g. "2025-06-28 to 2026-06-27"
    balance_date: str                     # instant date for cash and debt
    basis: dict[str, str] = field(default_factory=dict)    # concept -> how LTM was built
    sources: dict[str, str] = field(default_factory=dict)  # concept -> tags used
    missing: list[str] = field(default_factory=list)

    def ltm(self, concept: str) -> float | None:
        return self.ltm_values.get(concept)

    def prior_ltm(self, concept: str) -> float | None:
        return self.prior_ltm_values.get(concept)

    def latest(self, concept: str) -> float | None:
        return self.latest_values.get(concept)


# ---------------------------------------------------------------------- LTM math ------
def _d(s: str) -> date:
    return date.fromisoformat(s)


def _dedupe(entries: list[dict]) -> list[dict]:
    """One entry per (start, end): the most recently filed (restatements win)."""
    best: dict[tuple, dict] = {}
    for e in entries:
        if e.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"):
            continue
        k = (e.get("start"), e["end"])
        if k not in best or e["filed"] > best[k]["filed"]:
            best[k] = e
    return list(best.values())


def _near(a: date, b: date, days: int) -> bool:
    return abs((a - b).days) <= days


def ltm_ending(entries: list[dict], end: date, tol: int = 10) -> tuple[float | None, str]:
    """Twelve-month value ending at `end` (±tol days). Returns (value, basis text)."""
    durs = [e for e in entries if e.get("start")]
    ending = [e for e in durs if _near(_d(e["end"]), end, tol)]
    if not ending:
        return None, "no period ends near " + end.isoformat()
    full = [e for e in ending if 350 <= (_d(e["end"]) - _d(e["start"])).days <= 380]
    if full:
        e = max(full, key=lambda x: x["end"])
        return float(e["val"]), f"12 months {e['start']} to {e['end']} ({e['form']})"
    ytd = max(ending, key=lambda e: (_d(e["end"]) - _d(e["start"])).days)
    ys, ye = _d(ytd["start"]), _d(ytd["end"])
    ytd_days = (ye - ys).days
    fy = [e for e in durs if 350 <= (_d(e["end"]) - _d(e["start"])).days <= 380
          and _near(_d(e["end"]), ys - timedelta(days=1), tol)]
    prior = [e for e in durs if abs((_d(e["end"]) - _d(e["start"])).days - ytd_days) <= tol
             and _near(_d(e["end"]), ye - timedelta(days=365), tol)]
    if not fy or not prior:
        return None, f"YTD {ytd['start']} to {ytd['end']} found but prior FY or prior YTD missing"
    f, p = max(fy, key=lambda e: e["end"]), max(prior, key=lambda e: e["end"])
    val = float(f["val"]) + float(ytd["val"]) - float(p["val"])
    return val, (f"FY {f['start']}..{f['end']} {f['val']:,.0f} + YTD {ytd['start']}..{ytd['end']} "
                 f"{ytd['val']:,.0f} - prior YTD {p['start']}..{p['end']} {p['val']:,.0f}")


def latest_instant(entries: list[dict]) -> tuple[float | None, str]:
    inst = [e for e in entries if not e.get("start")]
    if not inst:
        return None, ""
    e = max(inst, key=lambda x: (x["end"], x["filed"]))
    return float(e["val"]), e["end"]


# ---------------------------------------------------------------------- provider ------
class EdgarProvider:
    def __init__(self, identity: str | None = None, fixture_dir: Path = FIXTURE_DIR,
                 cache: JsonCache | None = None):
        self.identity = identity or os.getenv("SEC_IDENTITY", "")
        self.fixture_dir = fixture_dir
        self.cache = cache or JsonCache("edgar", ttl_seconds=24 * 3600)
        self._last = 0.0

    def fundamentals(self, ticker: str) -> Fundamentals:
        ticker = ticker.upper()
        return self._from_fixture(ticker) if offline() else self._from_companyfacts(ticker)

    # ---- offline: fixtures hold 8 quarters per concept, newest first -------------------
    def _from_fixture(self, ticker: str) -> Fundamentals:
        path = self.fixture_dir / f"{ticker}.json"
        if not path.exists():
            raise FileNotFoundError(f"No fixture for {ticker} at {path}. Add one or run with FINLIB_OFFLINE=0.")
        raw = json.loads(path.read_text())
        q, periods = raw["quarterly"], raw["periods"]

        def s(c, lo, hi):
            v = q.get(c, [])[lo:hi]
            return float(sum(v)) if len(v) == hi - lo and all(x is not None for x in v) else None

        return Fundamentals(
            ticker=ticker, company=raw["company"], cik=raw["cik"],
            ltm_values={c: s(c, 0, 4) for c in FLOW_CONCEPTS},
            prior_ltm_values={c: s(c, 4, 8) for c in FLOW_CONCEPTS},
            latest_values={c: (float(q[c][0]) if c in q and q[c] else None) for c in STOCK_CONCEPTS},
            ltm_period=f"{periods[3]} to {periods[0]}" if len(periods) >= 4 else "insufficient quarters",
            balance_date=periods[0] if periods else "",
            basis={c: "sum of 4 fixture quarters" for c in FLOW_CONCEPTS},
            sources={c: f"fixture:{path.name}" for c in q},
            missing=[c for c in FLOW_CONCEPTS + STOCK_CONCEPTS if c not in q],
        )

    # ---- live ------------------------------------------------------------------------
    def _get_json(self, url: str) -> dict:
        import requests

        if not self.identity:
            raise RuntimeError("SEC_IDENTITY is required for live EDGAR calls (see .env.example)")
        cached = self.cache.get(url)
        if cached:
            return cached
        wait = self._last + 0.125 - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        r = requests.get(url, headers={"User-Agent": self.identity, "Accept-Encoding": "gzip, deflate"}, timeout=60)
        self._last = time.monotonic()
        r.raise_for_status()
        data = r.json()
        self.cache.set(url, data)
        return data

    def cik_for(self, ticker: str) -> tuple[str, str]:
        table = self._get_json(TICKERS_URL)
        for row in table.values():
            if row["ticker"].upper() == ticker:
                return str(row["cik_str"]).zfill(10), row["title"]
        raise KeyError(f"{ticker} not in SEC company_tickers.json")

    def _from_companyfacts(self, ticker: str) -> Fundamentals:
        cik, title = self.cik_for(ticker)
        facts = self._get_json(COMPANYFACTS.format(cik=cik))
        gaap = facts["facts"].get("us-gaap", {})

        def entries(tag: str) -> list[dict]:
            units = gaap.get(tag, {}).get("units", {})
            return _dedupe(units.get("USD", []))

        # Anchor every flow on the latest reported period end of the revenue concept.
        # Companies switch revenue tags over time (Alphabet moved to "Revenues" in 2025),
        # so the anchor is the latest period end across every candidate revenue tag.
        ends = [_d(e["end"]) for t in XBRL_TAGS["revenue"]["total"] for e in entries(t) if e.get("start")]
        if not ends:
            raise ValueError(f"{ticker}: no revenue facts in companyfacts")
        end = max(ends)
        start_guess = (end - timedelta(days=364)).isoformat()

        ltm_v, prior_v, latest_v, basis, sources, missing = {}, {}, {}, {}, {}, []

        def flow(tag: str):
            ent = entries(tag)
            if not ent:
                return None
            v, b = ltm_ending(ent, end)
            if v is None:
                return None
            pv, _ = ltm_ending(ent, end - timedelta(days=365))
            return v, pv, b

        def instant(tag: str):
            v, dt = latest_instant(entries(tag))
            # Only accept a balance-sheet value dated with the LTM end; stale tags
            # (a company that stopped using one years ago) are ignored.
            return (v, dt) if v is not None and _near(_d(dt), end, 45) else None

        def resolve(concept: str, getter, combine):
            spec = XBRL_TAGS[concept]
            for tag in spec.get("total", []):
                r = getter(tag)
                if r is not None:
                    return r, f"xbrl:{tag}"
            parts, used = [], []
            for tags, optional in spec.get("components", []):
                hit = next(((getter(t), t) for t in tags if getter(t) is not None), None)
                if hit is None:
                    if optional:
                        continue
                    return None, ""
                parts.append(hit[0]); used.append(hit[1])
            return (combine(parts), "xbrl:" + "+".join(used)) if parts else (None, "")

        balance_date = ""
        for concept in FLOW_CONCEPTS:
            def comb(parts):
                pv = sum(p[1] for p in parts) if all(p[1] is not None for p in parts) else None
                return sum(p[0] for p in parts), pv, " + ".join(p[2] for p in parts)
            r, src = resolve(concept, flow, comb)
            if r is None:
                ltm_v[concept], prior_v[concept] = None, None
                missing.append(concept)
            else:
                ltm_v[concept], prior_v[concept], basis[concept], sources[concept] = r[0], r[1], r[2], src
        for concept in STOCK_CONCEPTS:
            r, src = resolve(concept, instant, lambda parts: (sum(p[0] for p in parts), parts[0][1]))
            if r is None:
                latest_v[concept] = None
                missing.append(concept)
            else:
                latest_v[concept], sources[concept] = r[0], src
                balance_date = balance_date or r[1]

        return Fundamentals(
            ticker=ticker, company=title, cik=cik, ltm_values=ltm_v, prior_ltm_values=prior_v,
            latest_values=latest_v, ltm_period=f"{start_guess} to {end.isoformat()}",
            balance_date=balance_date, basis=basis, sources=sources, missing=missing,
        )

    def shares_outstanding(self, ticker: str) -> tuple[float, str, str] | None:
        """Share count from the filings: the dei cover-page count, else the balance-sheet
        us-gaap CommonStockSharesOutstanding (multi-class issuers such as Alphabet have no
        single dei figure in companyfacts). Returns (shares, as_of, tag) or None."""
        cik, _ = self.cik_for(ticker.upper())
        facts = self._get_json(COMPANYFACTS.format(cik=cik))
        for taxonomy, tag in (("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")):
            ent = facts["facts"].get(taxonomy, {}).get(tag, {}).get("units", {}).get("shares", [])
            ent = [e for e in ent if e.get("form") in ("10-K", "10-Q", "10-K/A", "10-Q/A")]
            if ent:
                latest = max(e["end"] for e in ent)
                val = max(float(e["val"]) for e in ent if e["end"] == latest)
                return val, latest, f"xbrl:{taxonomy}:{tag}"
        return None
