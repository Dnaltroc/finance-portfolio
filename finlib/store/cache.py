"""Tiny SQLite-backed JSON cache so demos never hit a rate limit twice."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DEFAULT_DB = Path(".finlib_cache.sqlite")


class JsonCache:
    def __init__(self, namespace: str, ttl_seconds: int = 3600, db_path: Path = DEFAULT_DB):
        self.ns = namespace
        self.ttl = ttl_seconds
        self.db = sqlite3.connect(db_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS cache (ns TEXT, key TEXT, ts REAL, payload TEXT, "
            "PRIMARY KEY (ns, key))"
        )

    def get(self, key: str):
        row = self.db.execute(
            "SELECT ts, payload FROM cache WHERE ns=? AND key=?", (self.ns, key)
        ).fetchone()
        if not row or time.time() - row[0] > self.ttl:
            return None
        return json.loads(row[1])

    def set(self, key: str, payload) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO cache (ns, key, ts, payload) VALUES (?, ?, ?, ?)",
            (self.ns, key, time.time(), json.dumps(payload)),
        )
        self.db.commit()
