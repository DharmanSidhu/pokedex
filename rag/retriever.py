"""
Hybrid retriever for Pokemon knowledge.

Combines fuzzy name matching, metadata filtering, and semantic vector search
with a reranking step for optimal retrieval quality.
"""

import logging
import re
from typing import Optional

from rapidfuzz import fuzz, process

from rag.vector_store import PokemonVectorStore

logger = logging.getLogger(__name__)

# All Pokemon names for fuzzy matching (populated at runtime from PokeAPI or fallback)
_pokemon_names_cache: Optional[list[str]] = None


def get_pokemon_names(store: PokemonVectorStore) -> list[str]:
    """Get list of all 1025 Pokemon names for fuzzy matching."""
    global _pokemon_names_cache
    if _pokemon_names_cache is None:
        try:
            from data_pipeline.pokeapi_fetcher import PokeAPIFetcher
            fetcher = PokeAPIFetcher()
            all_names_raw = fetcher.fetch_all_pokemon_names(limit=1025)
            if all_names_raw:
                _pokemon_names_cache = [p["name"] for p in all_names_raw]
                logger.info(f"Loaded {len(_pokemon_names_cache)} Pokemon names from PokeAPI cache/server for fuzzy matching")
            else:
                _pokemon_names_cache = store.get_all_pokemon_names()
                logger.warning("PokeAPI returned empty names list. Falling back to vector store names.")
        except Exception as e:
            logger.error(f"Error fetching names from PokeAPI: {e}. Falling back to vector store names.")
            _pokemon_names_cache = store.get_all_pokemon_names()
    return _pokemon_names_cache


def fuzzy_match_pokemon(
    query: str,
    pokemon_names: list[str],
    threshold: int = 65,
    max_matches: int = 3,
) -> list[dict]:
    """
    Find Pokemon names mentioned in a query using fuzzy matching.

    Handles typos (e.g., "pikachy" → "pikachu") and partial matches.

    Args:
        query: User query string
        pokemon_names: List of known Pokemon names
        threshold: Minimum fuzzy match score (0-100)
        max_matches: Maximum number of matches to return

    Returns:
        List of dicts with 'name' and 'score' keys
    """
    # Extract potential Pokemon name tokens from the query
    # Split on non-alphanumeric, keep hyphenated names (e.g., "mr-mime")
    tokens = re.findall(r'[a-zA-Z][\w-]*', query.lower())

    matches = []
    seen = set()

    for token in tokens:
        if len(token) < 3:
            continue

        # Try exact match first
        if token in pokemon_names:
            if token not in seen:
                matches.append({"name": token, "score": 100})
                seen.add(token)
            continue

        # Fuzzy match against all Pokemon names
        results = process.extract(
            token,
            pokemon_names,
            scorer=fuzz.ratio,
            limit=2,
        )
        for name, score, _ in results:
            if score >= threshold and name not in seen:
                matches.append({"name": name, "score": score})
                seen.add(name)

    # Also try matching multi-word Pokemon names (e.g., "Mr. Mime")
    query_lower = query.lower()
    for name in pokemon_names:
        if "-" in name:
            # Handle names like "mr-mime" matching "mr mime" or "mr. mime"
            readable = name.replace("-", " ")
            if readable in query_lower or name in query_lower:
                if name not in seen:
                    matches.append({"name": name, "score": 95})
                    seen.add(name)

    # Sort by score descending
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:max_matches]


def detect_query_category(query: str) -> Optional[str]:
    """
    Detect what category of information the query is asking about.

    Used to boost relevant document categories during reranking.
    """
    query_lower = query.lower()

    category_keywords = {
        "base_info": [
            "stats", "stat", "base stat", "hp", "attack", "defense",
            "speed", "special", "bst", "total", "type", "typing",
            "height", "weight", "generation", "gen",
        ],
        "abilities": [
            "ability", "abilities", "hidden ability", "ha",
        ],
        "evolution": [
            "evolve", "evolution", "evolves", "pre-evolution",
            "mega", "form", "variant", "regional",
        ],
        "competitive": [
            "competitive", "smogon", "vgc", "set", "moveset", "ev",
            "spread", "team", "counter", "check", "usage", "tier",
            "ou", "uu", "uber", "strategy", "build",
        ],
        "lore": [
            "lore", "trivia", "anime", "origin", "biology",
            "manga", "story", "history", "based on", "inspired",
        ],
        "flavor_text": [
            "pokedex", "pokédex", "dex entry", "description",
            "flavor text", "flavor",
        ],
    }

    best_category = None
    best_score = 0

    for category, keywords in category_keywords.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > best_score:
            best_score = score
            best_category = category

    return best_category if best_score > 0 else None


def rerank_results(
    results: list[dict],
    matched_pokemon: list[dict],
    query_category: Optional[str],
) -> list[dict]:
    """
    Rerank search results based on:
    1. Semantic similarity (from vector search distance)
    2. Pokemon name match bonus
    3. Category relevance bonus

    Args:
        results: Raw vector search results
        matched_pokemon: Fuzzy-matched Pokemon names from the query
        query_category: Detected query category

    Returns:
        Reranked list of results
    """
    matched_names = {m["name"] for m in matched_pokemon}

    for result in results:
        # Base score: inverse of distance (lower distance = higher score)
        base_score = 1.0 - result.get("distance", 0.5)

        # Pokemon name match bonus
        name_bonus = 0.0
        result_pokemon = result.get("metadata", {}).get("pokemon_name", "")
        if result_pokemon in matched_names:
            name_bonus = 0.3

        # Category match bonus
        category_bonus = 0.0
        result_category = result.get("metadata", {}).get("category", "")
        if query_category and result_category == query_category:
            category_bonus = 0.15

        result["rerank_score"] = base_score + name_bonus + category_bonus

    # Sort by rerank score descending
    results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return results


class HybridRetriever:
    """
    Hybrid retriever combining keyword matching, semantic search, and reranking.

    Pipeline:
    1. Fuzzy-match Pokemon names in the query
    2. If Pokemon found → metadata-filtered vector search
    3. Also run unfiltered semantic search
    4. Merge and rerank results
    5. Return top-k with source attribution
    """

    def __init__(self, store: Optional[PokemonVectorStore] = None):
        self.store = store or PokemonVectorStore()
        self._pokemon_names = None

    @property
    def pokemon_names(self) -> list[str]:
        """Lazy-load Pokemon names list."""
        if self._pokemon_names is None:
            self._pokemon_names = get_pokemon_names(self.store)
        return self._pokemon_names

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_sources: bool = True,
    ) -> dict:
        """
        Run the full hybrid retrieval pipeline.

        Args:
            query: Natural language query
            top_k: Number of final results to return
            include_sources: Whether to include source attribution

        Returns:
            Dict with:
                - 'chunks': List of top-k document chunks
                - 'matched_pokemon': Fuzzy-matched Pokemon names
                - 'query_category': Detected query category
                - 'sources': Set of data sources used
        """
        # Step 1: Fuzzy match Pokemon names
        matched_pokemon = fuzzy_match_pokemon(query, self.pokemon_names)
        logger.info(
            f"Query: '{query}' → Matched Pokemon: "
            f"{[m['name'] for m in matched_pokemon]}"
        )

        # Step 2: Detect query category
        query_category = detect_query_category(query)
        logger.info(f"Detected category: {query_category}")

        all_results = []

        # Step 3a: Pokemon-filtered search (if Pokemon detected)
        if matched_pokemon:
            for match in matched_pokemon[:2]:  # Top 2 Pokemon matches
                filtered_results = self.store.query(
                    query_text=query,
                    n_results=top_k,
                    where={"pokemon_name": match["name"]},
                )
                all_results.extend(filtered_results)

        # Step 3b: Unfiltered semantic search (always)
        semantic_results = self.store.query(
            query_text=query,
            n_results=top_k,
        )
        all_results.extend(semantic_results)

        # Deduplicate by ID
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                unique_results.append(r)

        # Step 4: Rerank
        reranked = rerank_results(unique_results, matched_pokemon, query_category)

        # Step 5: Take top-k
        top_results = reranked[:top_k]

        # Collect sources
        sources = set()
        if include_sources:
            for r in top_results:
                source = r.get("metadata", {}).get("source", "unknown")
                sources.add(source)

        return {
            "chunks": top_results,
            "matched_pokemon": matched_pokemon,
            "query_category": query_category,
            "sources": sources,
        }

    def retrieve_text(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve and format context text for LLM prompt injection.

        Returns a single string with all retrieved chunks, labeled by source.
        """
        result = self.retrieve(query, top_k=top_k)
        chunks = result["chunks"]

        if not chunks:
            return "No relevant Pokemon data found in the knowledge base."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", "unknown")
            category = chunk.get("metadata", {}).get("category", "")
            parts.append(
                f"[Source: {source.upper()} | {category}]\n{chunk['text']}"
            )

        return "\n\n---\n\n".join(parts)
