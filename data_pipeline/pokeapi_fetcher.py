"""
PokeAPI data fetcher.

Pulls structured Pokemon data (species, types, stats, abilities, moves,
evolution chains, sprites, cries) from PokeAPI v2, with rate limiting
and SQLite caching.
"""

import logging
import time
from typing import Any, Optional

import requests

from data_pipeline.db import CacheDB

logger = logging.getLogger(__name__)

POKEAPI_BASE = "https://pokeapi.co/api/v2"

# Rate limiting: max 100 requests/minute = ~1.7 req/sec
MIN_REQUEST_INTERVAL = 0.6  # seconds between requests
MAX_RETRIES = 3
BACKOFF_BASE = 2.0


class PokeAPIFetcher:
    """Client for PokeAPI v2 with rate limiting and SQLite caching."""

    def __init__(self, cache_max_age_hours: Optional[float] = None):
        """
        Args:
            cache_max_age_hours: If set, cached responses older than this are re-fetched.
                                 None means cache never expires.
        """
        self.cache = CacheDB("pokeapi.db")
        self.cache_max_age = cache_max_age_hours
        self._last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Pokedex/1.0 (local research project)",
            "Accept": "application/json",
        })

    def _rate_limit(self):
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _fetch_url(self, url: str) -> Optional[dict]:
        """
        Fetch a URL with caching, rate limiting, and retries.

        Returns parsed JSON or None on failure.
        """
        # Check cache first
        cached = self.cache.get(url, max_age_hours=self.cache_max_age)
        if cached is not None:
            return cached

        # Fetch from API with retries
        for attempt in range(MAX_RETRIES):
            self._rate_limit()
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    self.cache.set(url, data, source="pokeapi")
                    return data
                elif resp.status_code == 404:
                    logger.warning(f"PokeAPI 404: {url}")
                    return None
                elif resp.status_code == 429:
                    wait_time = BACKOFF_BASE ** (attempt + 1)
                    logger.warning(f"PokeAPI rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(
                        f"PokeAPI {resp.status_code} for {url} "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
            except requests.RequestException as e:
                wait_time = BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"PokeAPI request error: {e}, retrying in {wait_time}s")
                time.sleep(wait_time)

        logger.error(f"PokeAPI failed after {MAX_RETRIES} retries: {url}")
        return None

    def _fetch_endpoint(self, endpoint: str, id_or_name: Any) -> Optional[dict]:
        """Fetch a specific PokeAPI endpoint."""
        url = f"{POKEAPI_BASE}/{endpoint}/{id_or_name}"
        return self._fetch_url(url)

    # === High-level fetch methods ===

    def fetch_pokemon(self, id_or_name: Any) -> Optional[dict]:
        """
        Fetch core Pokemon data (types, stats, abilities, moves, sprites, cries).

        Returns a normalized dict with key fields extracted.
        """
        raw = self._fetch_endpoint("pokemon", id_or_name)
        if raw is None:
            return None

        return {
            "id": raw["id"],
            "name": raw["name"],
            "types": [t["type"]["name"] for t in raw["types"]],
            "base_stats": {
                s["stat"]["name"]: s["base_stat"] for s in raw["stats"]
            },
            "abilities": [
                {
                    "name": a["ability"]["name"],
                    "is_hidden": a["is_hidden"],
                }
                for a in raw["abilities"]
            ],
            "moves": [m["move"]["name"] for m in raw["moves"]],
            "height": raw["height"],  # decimetres
            "weight": raw["weight"],  # hectograms
            "sprites": {
                "front_default": raw["sprites"].get("front_default"),
                "official_artwork": (
                    raw["sprites"]
                    .get("other", {})
                    .get("official-artwork", {})
                    .get("front_default")
                ),
            },
            "cries": {
                "latest": raw.get("cries", {}).get("latest"),
                "legacy": raw.get("cries", {}).get("legacy"),
            },
            "base_experience": raw.get("base_experience"),
            "forms": [f["name"] for f in raw.get("forms", [])],
        }

    def fetch_species(self, id_or_name: Any) -> Optional[dict]:
        """
        Fetch species data (flavor text, evolution chain, generation, habitat, etc).
        """
        raw = self._fetch_endpoint("pokemon-species", id_or_name)
        if raw is None:
            return None

        # Extract English flavor text entries
        flavor_texts = []
        for entry in raw.get("flavor_text_entries", []):
            if entry["language"]["name"] == "en":
                flavor_texts.append({
                    "text": entry["flavor_text"].replace("\n", " ").replace("\f", " "),
                    "version": entry["version"]["name"],
                })

        # Extract English genus
        genus = ""
        for g in raw.get("genera", []):
            if g["language"]["name"] == "en":
                genus = g["genus"]
                break

        return {
            "id": raw["id"],
            "name": raw["name"],
            "genus": genus,
            "generation": raw["generation"]["name"] if raw.get("generation") else None,
            "flavor_texts": flavor_texts,
            "evolution_chain_url": (
                raw["evolution_chain"]["url"] if raw.get("evolution_chain") else None
            ),
            "habitat": raw["habitat"]["name"] if raw.get("habitat") else None,
            "is_legendary": raw.get("is_legendary", False),
            "is_mythical": raw.get("is_mythical", False),
            "color": raw["color"]["name"] if raw.get("color") else None,
            "shape": raw["shape"]["name"] if raw.get("shape") else None,
            "varieties": [
                {
                    "name": v["pokemon"]["name"],
                    "is_default": v["is_default"],
                }
                for v in raw.get("varieties", [])
            ],
        }

    def fetch_ability(self, id_or_name: Any) -> Optional[dict]:
        """Fetch ability description."""
        raw = self._fetch_endpoint("ability", id_or_name)
        if raw is None:
            return None

        # Get English effect
        effect = ""
        short_effect = ""
        for entry in raw.get("effect_entries", []):
            if entry["language"]["name"] == "en":
                effect = entry["effect"]
                short_effect = entry["short_effect"]
                break

        # Fallback to flavor text if no effect entry
        if not effect:
            for entry in raw.get("flavor_text_entries", []):
                if entry["language"]["name"] == "en":
                    effect = entry["flavor_text"].replace("\n", " ")
                    break

        return {
            "name": raw["name"],
            "effect": effect,
            "short_effect": short_effect,
            "generation": raw["generation"]["name"] if raw.get("generation") else None,
        }

    def fetch_move(self, id_or_name: Any) -> Optional[dict]:
        """Fetch move details."""
        raw = self._fetch_endpoint("move", id_or_name)
        if raw is None:
            return None

        # Get English effect
        effect = ""
        short_effect = ""
        for entry in raw.get("effect_entries", []):
            if entry["language"]["name"] == "en":
                effect = entry["effect"].replace("$effect_chance", str(raw.get("effect_chance", "")))
                short_effect = entry["short_effect"].replace(
                    "$effect_chance", str(raw.get("effect_chance", ""))
                )
                break

        return {
            "name": raw["name"],
            "type": raw["type"]["name"] if raw.get("type") else None,
            "power": raw.get("power"),
            "accuracy": raw.get("accuracy"),
            "pp": raw.get("pp"),
            "damage_class": (
                raw["damage_class"]["name"] if raw.get("damage_class") else None
            ),
            "effect": effect,
            "short_effect": short_effect,
            "priority": raw.get("priority", 0),
            "generation": raw["generation"]["name"] if raw.get("generation") else None,
        }

    def fetch_type(self, id_or_name: Any) -> Optional[dict]:
        """Fetch type data including damage relations."""
        raw = self._fetch_endpoint("type", id_or_name)
        if raw is None:
            return None

        dr = raw.get("damage_relations", {})
        return {
            "name": raw["name"],
            "damage_relations": {
                "double_damage_to": [t["name"] for t in dr.get("double_damage_to", [])],
                "half_damage_to": [t["name"] for t in dr.get("half_damage_to", [])],
                "no_damage_to": [t["name"] for t in dr.get("no_damage_to", [])],
                "double_damage_from": [t["name"] for t in dr.get("double_damage_from", [])],
                "half_damage_from": [t["name"] for t in dr.get("half_damage_from", [])],
                "no_damage_from": [t["name"] for t in dr.get("no_damage_from", [])],
            },
        }

    def fetch_evolution_chain(self, chain_url: str) -> Optional[list[dict]]:
        """
        Fetch and flatten an evolution chain from its URL.

        Returns a list of dicts representing each stage in the chain.
        """
        raw = self._fetch_url(chain_url)
        if raw is None:
            return None

        chain = []
        self._walk_chain(raw.get("chain", {}), chain, stage=1)
        return chain

    def _walk_chain(self, node: dict, result: list, stage: int):
        """Recursively walk an evolution chain node."""
        species_name = node.get("species", {}).get("name", "")
        if not species_name:
            return

        # Extract evolution trigger details
        details = []
        for d in node.get("evolution_details", []):
            trigger = d.get("trigger", {}).get("name", "")
            detail = {"trigger": trigger}
            if d.get("min_level"):
                detail["min_level"] = d["min_level"]
            if d.get("item"):
                detail["item"] = d["item"]["name"]
            if d.get("held_item"):
                detail["held_item"] = d["held_item"]["name"]
            if d.get("known_move"):
                detail["known_move"] = d["known_move"]["name"]
            if d.get("min_happiness"):
                detail["min_happiness"] = d["min_happiness"]
            if d.get("time_of_day"):
                detail["time_of_day"] = d["time_of_day"]
            if d.get("location"):
                detail["location"] = d["location"]["name"]
            details.append(detail)

        result.append({
            "species": species_name,
            "stage": stage,
            "evolution_details": details if details else None,
        })

        for child in node.get("evolves_to", []):
            self._walk_chain(child, result, stage + 1)

    def fetch_all_pokemon_names(self, limit: int = 1025) -> list[dict]:
        """
        Fetch a list of all Pokemon names and IDs.

        Args:
            limit: Max number of Pokemon to list

        Returns:
            List of dicts with 'name' and 'url' keys
        """
        url = f"{POKEAPI_BASE}/pokemon?limit={limit}&offset=0"
        data = self._fetch_url(url)
        if data is None:
            return []
        return data.get("results", [])

    def fetch_full_pokemon(self, id_or_name: Any) -> Optional[dict]:
        """
        Fetch complete Pokemon data: core + species + ability details + evolution.

        This is the main method used by the ETL pipeline.
        """
        pokemon = self.fetch_pokemon(id_or_name)
        if pokemon is None:
            return None

        species = self.fetch_species(id_or_name)

        # Fetch ability details
        abilities_detailed = []
        for ability_info in pokemon["abilities"]:
            ability_data = self.fetch_ability(ability_info["name"])
            if ability_data:
                ability_data["is_hidden"] = ability_info["is_hidden"]
                abilities_detailed.append(ability_data)

        # Fetch evolution chain
        evolution_chain = None
        if species and species.get("evolution_chain_url"):
            evolution_chain = self.fetch_evolution_chain(species["evolution_chain_url"])

        return {
            **pokemon,
            "species": species,
            "abilities_detailed": abilities_detailed,
            "evolution_chain": evolution_chain,
        }

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "total_entries": self.cache.count(),
            "db_path": str(self.cache.db_path),
        }
