"""Market data provider: price and shares outstanding.

Order of preference in live mode: Finnhub (free key, 60 calls/min), then yfinance as a
best-effort fallback. Offline mode reads the `market` block of the comps fixture.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from finlib import offline
from finlib.store.cache import JsonCache

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "comps"


@dataclass
class Quote:
    ticker: str
    price: float
    shares_outstanding: float  # absolute share count
    as_of: str
    source: str
    shares_source: str = ""  # filled in when shares come from the SEC filing instead

    @property
    def market_cap(self) -> float:
        return self.price * self.shares_outstanding


class MarketProvider:
    def __init__(self, finnhub_key: str | None = None, cache: JsonCache | None = None,
                 price_overrides: dict[str, float] | None = None):
        self.key = finnhub_key or os.getenv("FINNHUB_KEY", "")
        self.cache = cache or JsonCache("market", ttl_seconds=6 * 3600)
        self.overrides = {k.upper(): float(v) for k, v in (price_overrides or {}).items()}

    def quote(self, ticker: str) -> Quote:
        ticker = ticker.upper()
        if offline():
            raw = json.loads((FIXTURE_DIR / f"{ticker}.json").read_text())["market"]
            return Quote(ticker=ticker, source="fixture", **raw)
        if ticker in self.overrides:
            from datetime import date

            return Quote(ticker=ticker, price=self.overrides[ticker], shares_outstanding=0.0,
                         as_of=date.today().isoformat(), source="manual price")

        cached = self.cache.get(ticker)
        if cached:
            return Quote(**cached)

        q = self._finnhub(ticker) if self.key else None
        if q is None:
            q = self._yfinance(ticker)
        if q is None:
            raise RuntimeError(f"No market data for {ticker}: set FINNHUB_KEY or install yfinance")
        self.cache.set(ticker, q.__dict__)
        return q

    def _finnhub(self, ticker: str) -> Quote | None:
        import requests

        base = "https://finnhub.io/api/v1"
        try:
            px = requests.get(f"{base}/quote", params={"symbol": ticker, "token": self.key}, timeout=15).json()
            prof = requests.get(
                f"{base}/stock/profile2", params={"symbol": ticker, "token": self.key}, timeout=15
            ).json()
        except Exception:
            return None
        price = px.get("c")
        shares_m = prof.get("shareOutstanding")  # Finnhub reports millions
        if not price or not shares_m:
            return None
        from datetime import date

        return Quote(
            ticker=ticker,
            price=float(price),
            shares_outstanding=float(shares_m) * 1_000_000,
            as_of=date.today().isoformat(),
            source="finnhub",
        )

    def _yfinance(self, ticker: str) -> Quote | None:
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            return None
        try:
            info = yf.Ticker(ticker).fast_info
            price = float(info["last_price"])
            shares = float(info["shares"])
        except Exception:
            return None
        from datetime import date

        return Quote(
            ticker=ticker,
            price=price,
            shares_outstanding=shares,
            as_of=date.today().isoformat(),
            source="yfinance",
        )
