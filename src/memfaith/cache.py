"""SQLite cache for deterministic MemFaith predictions."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional
import json
import sqlite3


class SQLitePredictionCache:
    """Small wrapper around sqlite3 for prediction reuse and resuming."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def build_key(payload: Dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT payload FROM predictions WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def put(self, cache_key: str, payload: Dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO predictions(cache_key, payload) VALUES(?, ?)",
            (cache_key, json.dumps(payload)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
