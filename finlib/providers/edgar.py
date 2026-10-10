"""SEC EDGAR fundamentals provider.

Live mode uses `edgartools` (pip install edgartools) and requires SEC_IDENTITY in the
environment. Offline mode reads tests/fixtures/comps/<TICKER>.json, which holds the last
five quarters of the concepts we need. Both paths return the same `Fundamentals` object.

LTM (last twelve months) is built from quarterly income statement values:
    LTM = sum of the four most recent quarters.
Balance sheet items are taken from the most recent quarter.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from finlib import offline

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "comps"

# Flow concepts: summed over four quarters for LTM.
FLOW_CONCEPTS = ["revenue", "cogs", "operating_income", "da", "net_income"]
# Stock concepts: most recent quarter only.
STOCK_CONCEPTS = ["cash", "total_debt"]

# XBRL tag fallbacks, in preference order. Companies tag the same idea differently.
XBRL_TAGS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "operating_income": ["OperatingIncomeLoss"],
    "da": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "total_debt": ["LongTermDebt", "DebtInstrumentCarryingAmount", "LongTermDebtNoncurrent"],
}


@dataclass
class Fundamentals:
    ticker: str
    company: str
    cik: str
    periods: list[str]  # quarter end dates, most recent first
    quarterly: dict[str, list[float | None]]  # concept -> values aligned with periods
    sources: dict[str, str] = field(default_factory=dict)  # concept -> tag or URL used
    missing: list[str] = field(default_factory=list)

    def ltm(self, concept: str) -> float | None:
        vals = self.quarterly.get(concept, [])[:4]
        if len(vals) < 4 or any(v is None for v in vals):
            return None
        return float(sum(vals))

    def prior_ltm(self, concept: str) -> float | None:
        """LTM one year earlier (needs 8 quarters); falls back to None."""
        vals = self.quarterly.get(concept, [])[4:8]
        if len(vals) < 4 or any(v is None for v in vals):
            return None
        return float(sum(vals))

    def latest(self, concept: str) -> float | None:
        vals = self.quarterly.get(concept, [])
        return float(vals[0]) if vals and vals[0] is not None else None

    @property
    def ltm_period(self) -> str:
        if len(self.periods) >= 4:
            return f"{self.periods[3]} to {self.periods[0]}"
        return "insufficient quarters"


class EdgarProvider:
    def __init__(self, identity: str | None = None, fixture_dir: Path = FIXTURE_DIR):
        self.identity = identity or os.getenv("SEC_IDENTITY", "")
        self.fixture_dir = fixture_dir

    def fundamentals(self, ticker: str) -> Fundamentals:
        ticker = ticker.upper()
        if offline():
            return self._from_fixture(ticker)
        return self._from_edgartools(ticker)

    # ---- offline -------------------------------------------------------------------
    def _from_fixture(self, ticker: str) -> Fundamentals:
        path = self.fixture_dir / f"{ticker}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No fixture for {ticker} at {path}. Add one or run with FINLIB_OFFLINE=0."
            )
        raw = json.loads(path.read_text())
        q = raw["quarterly"]
        missing = [c for c in FLOW_CONCEPTS + STOCK_CONCEPTS if c not in q]
        return Fundamentals(
            ticker=ticker,
            company=raw["company"],
            cik=raw["cik"],
            periods=raw["periods"],
            quarterly=q,
            sources={c: f"fixture:{path.name}" for c in q},
            missing=missing,
        )

    # ---- live ------------------------------------------------------------------------
    def _from_edgartools(self, ticker: str) -> Fundamentals:
        if not self.identity:
            raise RuntimeError("SEC_IDENTITY is required for live EDGAR calls (see .env.example)")
        try:
            from edgar import Company, set_identity  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install edgartools for live mode") from e

        set_identity(self.identity)
        co = Company(ticker)
        facts = co.get_facts()
        periods: list[str] = []
        quarterly: dict[str, list[float | None]] = {}
        sources: dict[str, str] = {}
        missing: list[str] = []

        for concept, tags in XBRL_TAGS.items():
            series = None
            used = None
            for tag in tags:
                try:
                    series = self._quarterly_series(facts, tag)
                except Exception:
                    series = None
                if series:
                    used = tag
                    break
            if not series:
                missing.append(concept)
                quarterly[concept] = []
                continue
            sources[concept] = f"xbrl:{used}"
            dates = [d for d, _ in series]
            if len(dates) > len(periods):
                periods = dates
            quarterly[concept] = [v for _, v in series]

        # Align every concept to the master period list.
        for concept, vals in quarterly.items():
            if len(vals) < len(periods):
                quarterly[concept] = vals + [None] * (len(periods) - len(vals))

        return Fundamentals(
            ticker=ticker,
            company=co.name,
            cik=str(co.cik).zfill(10),
            periods=periods[:8],
            quarterly={k: v[:8] for k, v in quarterly.items()},
            sources=sources,
            missing=missing,
        )

    @staticmethod
    def _quarterly_series(facts, tag: str) -> list[tuple[str, float]]:
        """Return [(period_end, value)] for quarterly (3-month) durations, newest first.

        edgartools exposes facts as a DataFrame via facts.to_dataframe(); the exact column
        names have shifted between versions, so this is deliberately defensive. Validate
        against your installed version before relying on it in a demo.
        """
        df = facts.to_dataframe()
        sub = df[df["concept"] == tag] if "concept" in df.columns else df[df["fact"] == tag]
        if sub.empty:
            return []
        if "period_type" in sub.columns:
            sub = sub[sub["period_type"].str.lower().str.contains("duration|instant", na=False)]
        # Keep ~3-month durations for flows; instants pass through untouched for stocks.
        if {"period_start", "period_end"} <= set(sub.columns):
            import pandas as pd

            start = pd.to_datetime(sub["period_start"], errors="coerce")
            end = pd.to_datetime(sub["period_end"], errors="coerce")
            days = (end - start).dt.days
            sub = sub[(days.isna()) | ((days > 80) & (days < 100))]
        sub = sub.sort_values("period_end", ascending=False).drop_duplicates("period_end")
        return [(str(r["period_end"])[:10], float(r["value"])) for _, r in sub.iterrows()]
