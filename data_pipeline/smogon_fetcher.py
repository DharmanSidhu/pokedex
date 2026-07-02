"""
Smogon competitive data fetcher.

Pulls usage stats, movesets, items, EV spreads, and team archetypes from
Smogon's public stats files (chaos JSON format), with SQLite caching.
"""

import json
import logging
import re
import time
from typing import Any, Optional

import requests

from data_pipeline.db import CacheDB

logger = logging.getLogger(__name__)

SMOGON_STATS_BASE = "https://www.smogon.com/stats"

# Default tiers to fetch (Gen 9 competitive formats)
DEFAULT_TIERS = ["gen9ou", "gen9uu", "gen9vgc2026regi"]

# Rate limiting: 1 request/second (respectful to Smogon)
MIN_REQUEST_INTERVAL = 1.0
MAX_RETRIES = 3
BACKOFF_BASE = 2.0


def decrement_month(month_str: str) -> str:
    """Decrement a month string of format 'YYYY-MM' by one month."""
    parts = month_str.split("-")
    if len(parts) == 2:
        try:
            year = int(parts[0])
            month = int(parts[1])
            if month == 1:
                year -= 1
                month = 12
            else:
                month -= 1
            return f"{year:04d}-{month:02d}"
        except ValueError:
            pass
    return month_str


class SmogonFetcher:
    """Client for Smogon usage statistics with SQLite caching."""

    def __init__(self, cache_max_age_hours: Optional[float] = 168.0):
        """
        Args:
            cache_max_age_hours: Cache TTL in hours (default 1 week).
        """
        self.cache = CacheDB("smogon.db")
        self.cache_max_age = cache_max_age_hours
        self._last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Pokedex/1.0 (local research project)",
        })
        self._chaos_cache = {}

    def _rate_limit(self):
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch a URL with caching, rate limiting, and retries. Returns raw text."""
        cached = self.cache.get(url, max_age_hours=self.cache_max_age)
        if cached is not None:
            return cached

        for attempt in range(MAX_RETRIES):
            self._rate_limit()
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    self.cache.set(url, resp.text, source="smogon")
                    return resp.text
                elif resp.status_code == 404:
                    logger.info(f"Smogon 404: {url}")
                    return None
                elif resp.status_code == 429:
                    wait = BACKOFF_BASE ** (attempt + 1)
                    logger.warning(f"Smogon rate limited, waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.warning(
                        f"Smogon {resp.status_code} for {url} "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
            except requests.RequestException as e:
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"Smogon request error: {e}, retrying in {wait}s")
                time.sleep(wait)

        logger.error(f"Smogon failed after {MAX_RETRIES} retries: {url}")
        return None

    def get_latest_month(self) -> Optional[str]:
        """
        Discover the latest available stats month from Smogon.

        Returns a string like '2026-05' or None if unavailable.
        """
        cached = self.cache.get("__latest_month__", max_age_hours=24.0)
        if cached is not None:
            return cached

        html = self._fetch_url(f"{SMOGON_STATS_BASE}/")
        if html is None:
            return None

        # Parse directory listing for month folders (e.g., '2026-05/')
        months = re.findall(r'href="(\d{4}-\d{2})/"', html)
        if not months:
            logger.warning("Could not find any month directories on Smogon stats page")
            return None

        latest = sorted(months)[-1]
        self.cache.set("__latest_month__", latest, source="smogon")
        logger.info(f"Latest Smogon stats month: {latest}")
        return latest

    def fetch_chaos_json(
        self, month: str, tier: str, rating: int = 1500
    ) -> Optional[dict]:
        """
        Fetch the chaos JSON file for a given month/tier/rating.

        Args:
            month: Stats month (e.g. '2026-05')
            tier: Tier name (e.g. 'gen9ou')
            rating: Minimum ELO rating filter (e.g. 1500, 1695, 1760)

        Returns:
            Parsed JSON dict with full moveset data, or None
        """
        url = f"{SMOGON_STATS_BASE}/{month}/chaos/{tier}-{rating}.json"
        if url in self._chaos_cache:
            return self._chaos_cache[url]

        raw = self._fetch_url(url)
        if raw is None:
            # Try without rating
            url_no_rating = f"{SMOGON_STATS_BASE}/{month}/chaos/{tier}.json"
            if url_no_rating in self._chaos_cache:
                return self._chaos_cache[url_no_rating]
            raw = self._fetch_url(url_no_rating)
            if raw is None:
                return None
            url = url_no_rating

        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            self._chaos_cache[url] = parsed
            return parsed
        except json.JSONDecodeError:
            logger.error(f"Failed to parse chaos JSON from {url}")
            return None

    def extract_pokemon_competitive_data(
        self, chaos_data: dict, pokemon_name: str
    ) -> Optional[dict]:
        """
        Extract competitive data for a single Pokemon from chaos JSON.

        Args:
            chaos_data: Full chaos JSON data
            pokemon_name: Pokemon name to look up (case-insensitive)

        Returns:
            Dict with usage %, top moves, items, abilities, spreads, teammates
        """
        data_section = chaos_data.get("data", {})

        # Case-insensitive lookup
        target = None
        for name in data_section:
            if name.lower() == pokemon_name.lower():
                target = data_section[name]
                break

        if target is None:
            return None

        raw_count = target.get("Raw count", 0)
        total_battles = chaos_data.get("info", {}).get("number of battles", 1)
        usage_pct = (raw_count / (total_battles * 2)) * 100 if total_battles > 0 else 0

        def top_n(d: dict, n: int = 5) -> list[dict]:
            """Get top N entries from a frequency dict, sorted by value."""
            if not d:
                return []
            sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]
            total = sum(d.values())
            return [
                {"name": k, "usage_pct": round(v / total * 100, 1) if total > 0 else 0}
                for k, v in sorted_items
            ]

        def parse_spreads(spreads: dict, n: int = 3) -> list[dict]:
            """Parse EV spread strings like 'Modest:0/0/4/252/0/252'."""
            if not spreads:
                return []
            sorted_spreads = sorted(spreads.items(), key=lambda x: x[1], reverse=True)[:n]
            total = sum(spreads.values())
            results = []
            for spread_str, count in sorted_spreads:
                parts = spread_str.split(":")
                if len(parts) == 2:
                    nature = parts[0]
                    evs_raw = parts[1].split("/")
                    if len(evs_raw) == 6:
                        stat_names = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
                        evs = {
                            stat_names[i]: int(evs_raw[i])
                            for i in range(6)
                            if int(evs_raw[i]) > 0
                        }
                        results.append({
                            "nature": nature,
                            "evs": evs,
                            "usage_pct": round(count / total * 100, 1) if total > 0 else 0,
                        })
            return results

        return {
            "usage_pct": round(usage_pct, 2),
            "raw_count": raw_count,
            "top_moves": top_n(target.get("Moves", {}), 10),
            "top_items": top_n(target.get("Items", {}), 5),
            "top_abilities": top_n(target.get("Abilities", {}), 3),
            "top_spreads": parse_spreads(target.get("Spreads", {}), 3),
            "top_teammates": top_n(target.get("Teammates", {}), 6),
            "checks_and_counters": list(target.get("Checks and Counters", {}).keys())[:5],
        }

    def fetch_pokemon_competitive(
        self,
        pokemon_name: str,
        tiers: Optional[list[str]] = None,
        month: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Fetch competitive data for a Pokemon across multiple tiers.

        Args:
            pokemon_name: Pokemon name
            tiers: List of tier names to check (defaults to DEFAULT_TIERS)
            month: Stats month (defaults to latest)

        Returns:
            Dict mapping tier names to competitive data dicts
        """
        if tiers is None:
            tiers = DEFAULT_TIERS
        if month is None:
            month = self.get_latest_month()
            if month is None:
                logger.warning("Could not determine latest Smogon stats month")
                return {}

        results = {}
        for tier in tiers:
            current_month = month
            # Try up to 3 months back for each tier individually
            for _ in range(3):
                chaos = self.fetch_chaos_json(current_month, tier)
                if chaos is not None:
                    comp_data = self.extract_pokemon_competitive_data(chaos, pokemon_name)
                    if comp_data is not None:
                        comp_data["tier"] = tier
                        comp_data["month"] = current_month
                        results[tier] = comp_data
                        break  # Successfully found data, proceed to next tier
                
                # Decrement month if stats for current month aren't loaded yet
                old_month = current_month
                current_month = decrement_month(current_month)
                if current_month == old_month:
                    break

        return results

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "total_entries": self.cache.count(),
            "db_path": str(self.cache.db_path),
        }
