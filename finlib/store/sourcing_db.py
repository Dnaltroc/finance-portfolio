"""SQLite store for Form D filings. Upsert on accession; tracks which CIKs we have seen."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    accession TEXT PRIMARY KEY,
    cik TEXT, issuer TEXT, entity_type TEXT, state TEXT, year_of_inc TEXT, industry_group TEXT,
    is_amendment INTEGER, date_of_first_sale TEXT, exemption TEXT, equity INTEGER, debt INTEGER,
    total_offering REAL, total_sold REAL, total_remaining REAL, investors INTEGER, min_investment REAL,
    filing_date TEXT, url TEXT,
    kept INTEGER, drop_reason TEXT, first_time INTEGER, score REAL, score_parts TEXT,
    inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_filings_date ON filings(filing_date);
CREATE INDEX IF NOT EXISTS idx_filings_cik ON filings(cik);
"""

COLS = ["accession", "cik", "issuer", "entity_type", "state", "year_of_inc", "industry_group", "is_amendment",
        "date_of_first_sale", "exemption", "equity", "debt", "total_offering", "total_sold", "total_remaining",
        "investors", "min_investment", "filing_date", "url", "kept", "drop_reason", "first_time", "score",
        "score_parts"]


class SourcingDB:
    def __init__(self, path: Path = Path("sourcing.sqlite")):
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def seen_ciks(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT DISTINCT cik FROM filings")}

    def upsert(self, rows: list[dict]) -> int:
        """Insert or replace rows; returns how many were new."""
        existing = {r[0] for r in self.db.execute("SELECT accession FROM filings")}
        new = 0
        for r in rows:
            vals = [json.dumps(r[c]) if c == "score_parts" else r[c] for c in COLS]
            self.db.execute(
                f"INSERT OR REPLACE INTO filings ({', '.join(COLS)}) VALUES ({', '.join('?' * len(COLS))})", vals
            )
            new += r["accession"] not in existing
        self.db.commit()
        return new

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM filings").fetchone()[0]

    def top(self, day: str, n: int) -> list[tuple]:
        return self.db.execute(
            "SELECT score, issuer, state, industry_group, total_sold FROM filings "
            "WHERE filing_date=? AND kept=1 ORDER BY score DESC LIMIT ?", (day, n)
        ).fetchall()
