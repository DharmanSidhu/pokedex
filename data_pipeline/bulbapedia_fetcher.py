"""
Bulbapedia data fetcher.

Pulls lore, trivia, biology, and origin text from Bulbapedia using the
MediaWiki API, with SQLite caching and respectful rate limiting.
"""

import logging
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from data_pipeline.db import CacheDB

logger = logging.getLogger(__name__)

BULBAPEDIA_API = "https://bulbapedia.bulbagarden.net/w/api.php"

# Rate limiting: 1 request/second (respectful to wiki servers)
MIN_REQUEST_INTERVAL = 1.0
MAX_RETRIES = 3
BACKOFF_BASE = 2.0


class BulbapediaFetcher:
    """Client for Bulbapedia MediaWiki API with SQLite caching."""

    def __init__(self, cache_max_age_hours: Optional[float] = 168.0):
        """
        Args:
            cache_max_age_hours: Cache TTL in hours (default 1 week).
        """
        self.cache = CacheDB("bulbapedia.db")
        self.cache_max_age = cache_max_age_hours
        self._last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Pokedex/1.0 (local research project; respectful single-threaded access)",
        })

    def _rate_limit(self):
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _fetch_api(self, params: dict) -> Optional[dict]:
        """Make a MediaWiki API request with caching and rate limiting."""
        # Build cache key from params
        cache_key = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        cached = self.cache.get(cache_key, max_age_hours=self.cache_max_age)
        if cached is not None:
            return cached

        for attempt in range(MAX_RETRIES):
            self._rate_limit()
            try:
                resp = self.session.get(BULBAPEDIA_API, params=params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    self.cache.set(cache_key, data, source="bulbapedia")
                    return data
                elif resp.status_code == 429:
                    wait = BACKOFF_BASE ** (attempt + 1)
                    logger.warning(f"Bulbapedia rate limited, waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.warning(
                        f"Bulbapedia {resp.status_code} "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
            except requests.RequestException as e:
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"Bulbapedia request error: {e}, retrying in {wait}s")
                time.sleep(wait)

        logger.error(f"Bulbapedia failed after {MAX_RETRIES} retries")
        return None

    def fetch_page_wikitext(self, title: str) -> Optional[str]:
        """
        Fetch the raw wikitext content of a Bulbapedia page.

        Args:
            title: Page title (e.g. 'Pikachu_(Pokémon)')

        Returns:
            Raw wikitext string, or None
        """
        data = self._fetch_api({
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
        })

        if data is None:
            return None

        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                logger.info(f"Bulbapedia page not found: {title}")
                return None
            revisions = page_data.get("revisions", [])
            if revisions:
                return revisions[0].get("slots", {}).get("main", {}).get("*", "")

        return None

    def fetch_page_html(self, title: str) -> Optional[str]:
        """
        Fetch the parsed HTML content of a Bulbapedia page.

        Args:
            title: Page title

        Returns:
            Parsed HTML string, or None
        """
        data = self._fetch_api({
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json",
        })

        if data is None:
            return None

        return data.get("parse", {}).get("text", {}).get("*", "")

    def _clean_wikitext(self, text: str) -> str:
        """Clean wikitext markup into plain text."""
        # Remove templates like {{...}}
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
        # Remove remaining nested templates (one more pass)
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
        # Remove category links
        text = re.sub(r'\[\[Category:[^\]]*\]\]', '', text)
        # Convert wiki links [[Target|Display]] to Display, [[Target]] to Target
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
        # Remove file/image links
        text = re.sub(r'\[\[File:[^\]]*\]\]', '', text)
        # Remove HTML tags
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*/>', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        # Remove bold/italic markup
        text = re.sub(r"'{2,5}", '', text)
        # Remove bullet point markers
        text = re.sub(r'^\*+\s*', '', text, flags=re.MULTILINE)
        # Clean whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def _extract_section(self, wikitext: str, section_name: str) -> Optional[str]:
        """
        Extract a specific section from wikitext by heading name.

        Handles == Section == and === Subsection === levels.
        """
        if not wikitext:
            return None
            
        # Match section header (any level) - use concatenation to avoid f-string backreference bug
        pattern = r'(={2,})\s*' + re.escape(section_name) + r'\s*\1'
        match = re.search(pattern, wikitext, re.IGNORECASE)
        if not match:
            return None

        # Find the header level
        header_level = len(match.group(1))
        start = match.end()

        # Find the next section at the same or higher level
        next_pattern = r'(={2,' + str(header_level) + r'})\s*[^=]'
        next_section = re.search(next_pattern, wikitext[start:])
        end = start + next_section.start() if next_section else len(wikitext)

        section_text = wikitext[start:end]
        return self._clean_wikitext(section_text)

    def fetch_pokemon_lore(self, pokemon_name: str) -> Optional[dict]:
        """
        Fetch lore and trivia for a Pokemon from Bulbapedia.

        Extracts: Biology, Trivia, In the anime, Origin, Name origin sections.

        Args:
            pokemon_name: Pokemon name (e.g. 'Pikachu')

        Returns:
            Dict with section keys and clean text values, or None
        """
        # Bulbapedia article naming convention
        title = f"{pokemon_name.title()} (Pokémon)"

        wikitext = self.fetch_page_wikitext(title)
        if wikitext is None:
            # Try without the (Pokémon) suffix
            wikitext = self.fetch_page_wikitext(pokemon_name.title())
            if wikitext is None:
                return None

        sections_to_extract = [
            "Biology",
            "Trivia",
            "In the anime",
            "In the manga",
            "Origin",
            "Name origin",
            "In other generations",
            "Game data",
        ]

        result = {}
        for section in sections_to_extract:
            content = self._extract_section(wikitext, section)
            if content and len(content.strip()) > 20:
                result[section.lower().replace(" ", "_")] = content.strip()

        if not result:
            logger.info(f"No lore sections found for {pokemon_name}")
            return None

        return result

    def fetch_pokemon_lore_html(self, pokemon_name: str) -> Optional[dict]:
        """
        Alternative: fetch lore via parsed HTML (cleaner text, slower).

        Uses BeautifulSoup to extract text from rendered HTML sections.
        """
        title = f"{pokemon_name.title()} (Pokémon)"
        html = self.fetch_page_html(title)
        if html is None:
            return None

        soup = BeautifulSoup(html, "lxml")
        result = {}

        target_sections = {
            "Biology": "biology",
            "Trivia": "trivia",
            "In the anime": "in_the_anime",
            "Origin": "origin",
            "Name origin": "name_origin",
        }

        for heading in soup.find_all(["h2", "h3"]):
            heading_text = heading.get_text(strip=True)
            for section_name, key in target_sections.items():
                if section_name.lower() in heading_text.lower():
                    # Collect text until next heading of same or higher level
                    content_parts = []
                    sibling = heading.find_next_sibling()
                    while sibling and sibling.name not in ["h2", "h3"]:
                        text = sibling.get_text(separator=" ", strip=True)
                        if text and len(text) > 10:
                            content_parts.append(text)
                        sibling = sibling.find_next_sibling()
                    if content_parts:
                        result[key] = " ".join(content_parts)

        return result if result else None

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "total_entries": self.cache.count(),
            "db_path": str(self.cache.db_path),
        }
