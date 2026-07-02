"""
SQLite cache manager for API responses.

Provides a simple key-value cache backed by SQLite, used by all data fetchers
(PokeAPI, Smogon, Bulbapedia) to avoid redundant network requests.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


CACHE_DIR = Path(__file__).parent.parent / "cache"


class CacheDB:
    """SQLite-backed cache for API responses."""

    def __init__(self, db_name: str):
        """
        Initialize cache database.

        Args:
            db_name: Name of the SQLite database file (e.g. 'pokeapi.db')
        """
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = CACHE_DIR / db_name
        self._init_db()

    def _init_db(self):
        """Create the cache table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    source TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_source
                ON cache(source)
            """)
            conn.commit()

    def get(self, key: str, max_age_hours: Optional[float] = None) -> Optional[Any]:
        """
        Retrieve a cached value by key.

        Args:
            key: Cache key (typically the API URL)
            max_age_hours: If set, only return cache entries newer than this many hours

        Returns:
            Parsed JSON value if found and not expired, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value, timestamp FROM cache WHERE key = ?", (key,)
            ).fetchone()

        if row is None:
            return None

        value_str, timestamp = row

        if max_age_hours is not None:
            age_hours = (time.time() - timestamp) / 3600
            if age_hours > max_age_hours:
                return None

        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            return value_str

    def set(self, key: str, value: Any, source: str = "") -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key (typically the API URL)
            value: Value to cache (will be JSON-serialized)
            source: Source identifier (e.g. 'pokeapi', 'smogon', 'bulbapedia')
        """
        value_str = json.dumps(value) if not isinstance(value, str) else value
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, timestamp, source)
                VALUES (?, ?, ?, ?)
                """,
                (key, value_str, time.time(), source),
            )
            conn.commit()

    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM cache WHERE key = ?", (key,)
            ).fetchone()
        return row is not None

    def count(self) -> int:
        """Return the number of entries in the cache."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        return row[0] if row else 0

    def clear(self) -> None:
        """Clear all cache entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()

    def keys_by_source(self, source: str) -> list[str]:
        """Return all cache keys for a given source."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT key FROM cache WHERE source = ?", (source,)
            ).fetchall()
        return [row[0] for row in rows]
