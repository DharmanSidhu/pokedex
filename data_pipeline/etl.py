"""
ETL pipeline: Extract, Transform, Load.

Fetches data from all three sources (PokeAPI, Smogon, Bulbapedia),
normalizes it into clean document chunks per Pokemon, and loads them
into the ChromaDB vector store for RAG retrieval.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from data_pipeline.pokeapi_fetcher import PokeAPIFetcher
from data_pipeline.smogon_fetcher import SmogonFetcher
from data_pipeline.bulbapedia_fetcher import BulbapediaFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent.parent / "cache" / "etl.log"),
    ],
)
logger = logging.getLogger(__name__)

# How many Pokemon to fetch (1025 = all Pokemon through Gen 9)
DEFAULT_POKEMON_LIMIT = 1025


def format_stat_block(stats: dict) -> str:
    """Format base stats into a readable text block."""
    stat_names = {
        "hp": "HP",
        "attack": "Attack",
        "defense": "Defense",
        "special-attack": "Sp. Atk",
        "special-defense": "Sp. Def",
        "speed": "Speed",
    }
    lines = []
    total = 0
    for key, label in stat_names.items():
        val = stats.get(key, 0)
        total += val
        bar = "█" * (val // 10) + "░" * (10 - val // 10)
        lines.append(f"  {label:<10} {val:>3} {bar}")
    lines.append(f"  {'Total':<10} {total:>3}")
    return "\n".join(lines)


def format_evolution_chain(chain: list[dict]) -> str:
    """Format evolution chain into readable text."""
    if not chain:
        return "No evolution data available."

    parts = []
    for entry in chain:
        name = entry["species"].title()
        stage = entry["stage"]
        details = entry.get("evolution_details")
        if stage == 1:
            parts.append(f"Stage 1: {name} (Base)")
        else:
            method = ""
            if details:
                d = details[0]
                trigger = d.get("trigger", "")
                if trigger == "level-up":
                    if d.get("min_level"):
                        method = f"Level {d['min_level']}"
                    elif d.get("min_happiness"):
                        method = f"Happiness ({d['min_happiness']}+)"
                    elif d.get("known_move"):
                        method = f"Knows {d['known_move'].title()}"
                    else:
                        method = "Level up"
                    if d.get("time_of_day"):
                        method += f" ({d['time_of_day']})"
                elif trigger == "use-item":
                    item = d.get("item", "unknown item").replace("-", " ").title()
                    method = f"Use {item}"
                elif trigger == "trade":
                    if d.get("held_item"):
                        item = d["held_item"].replace("-", " ").title()
                        method = f"Trade holding {item}"
                    else:
                        method = "Trade"
                else:
                    method = trigger.replace("-", " ").title()
            parts.append(f"Stage {stage}: {name} (via {method})" if method else f"Stage {stage}: {name}")

    return " → ".join(parts)


def format_competitive_data(comp_data: dict, tier: str) -> str:
    """Format Smogon competitive data into readable text."""
    lines = [f"Competitive Data ({tier.upper()}, {comp_data.get('month', 'N/A')}):"]
    lines.append(f"  Usage: {comp_data['usage_pct']}%")

    if comp_data.get("top_abilities"):
        abilities = ", ".join(
            f"{a['name'].replace('-', ' ').title()} ({a['usage_pct']}%)"
            for a in comp_data["top_abilities"]
        )
        lines.append(f"  Common Abilities: {abilities}")

    if comp_data.get("top_items"):
        items = ", ".join(
            f"{i['name'].replace('-', ' ').title()} ({i['usage_pct']}%)"
            for i in comp_data["top_items"]
        )
        lines.append(f"  Common Items: {items}")

    if comp_data.get("top_moves"):
        moves = ", ".join(
            f"{m['name'].replace('-', ' ').title()} ({m['usage_pct']}%)"
            for m in comp_data["top_moves"][:6]
        )
        lines.append(f"  Common Moves: {moves}")

    if comp_data.get("top_spreads"):
        for i, spread in enumerate(comp_data["top_spreads"]):
            evs = ", ".join(f"{v} {k}" for k, v in spread["evs"].items())
            lines.append(
                f"  Set {i + 1}: {spread['nature']} nature, "
                f"EVs: {evs} ({spread['usage_pct']}%)"
            )

    if comp_data.get("top_teammates"):
        teammates = ", ".join(
            t["name"].title() for t in comp_data["top_teammates"]
        )
        lines.append(f"  Common Teammates: {teammates}")

    if comp_data.get("checks_and_counters"):
        counters = ", ".join(
            c.title() for c in comp_data["checks_and_counters"]
        )
        lines.append(f"  Checks/Counters: {counters}")

    return "\n".join(lines)


def build_pokemon_documents(
    pokemon_data: dict,
    species_data: Optional[dict],
    competitive_data: dict,
    lore_data: Optional[dict],
) -> list[dict]:
    """
    Build document chunks for a single Pokemon.

    Returns a list of document dicts, each with:
        - id: unique document ID
        - text: document content
        - metadata: {pokemon_name, dex_id, source, category}
    """
    name = pokemon_data["name"]
    name_title = name.replace("-", " ").title()
    dex_id = pokemon_data["id"]
    documents = []

    # --- 1. Base Info ---
    types_str = "/".join(t.title() for t in pokemon_data["types"])
    height_m = pokemon_data["height"] / 10
    weight_kg = pokemon_data["weight"] / 10

    base_info = (
        f"{name_title} (#{dex_id})\n"
        f"Type: {types_str}\n"
        f"Height: {height_m}m, Weight: {weight_kg}kg\n"
    )
    if species_data:
        if species_data.get("genus"):
            base_info += f"Category: {species_data['genus']}\n"
        if species_data.get("generation"):
            gen = species_data["generation"].replace("generation-", "Gen ").upper()
            base_info += f"Generation: {gen}\n"
        if species_data.get("habitat"):
            base_info += f"Habitat: {species_data['habitat'].title()}\n"
        if species_data.get("is_legendary"):
            base_info += "Status: Legendary Pokémon\n"
        elif species_data.get("is_mythical"):
            base_info += "Status: Mythical Pokémon\n"

    base_info += f"\nBase Stats:\n{format_stat_block(pokemon_data['base_stats'])}"

    documents.append({
        "id": f"{name}__base_info",
        "text": base_info,
        "metadata": {
            "pokemon_name": name,
            "dex_id": dex_id,
            "source": "pokeapi",
            "category": "base_info",
        },
    })

    # --- 2. Abilities ---
    if pokemon_data.get("abilities_detailed"):
        ability_lines = [f"{name_title} — Abilities:\n"]
        for ab in pokemon_data["abilities_detailed"]:
            ab_name = ab["name"].replace("-", " ").title()
            hidden_tag = " (Hidden Ability)" if ab.get("is_hidden") else ""
            desc = ab.get("short_effect") or ab.get("effect") or "No description available."
            ability_lines.append(f"  • {ab_name}{hidden_tag}: {desc}")

        documents.append({
            "id": f"{name}__abilities",
            "text": "\n".join(ability_lines),
            "metadata": {
                "pokemon_name": name,
                "dex_id": dex_id,
                "source": "pokeapi",
                "category": "abilities",
            },
        })

    # --- 3. Evolution ---
    if pokemon_data.get("evolution_chain"):
        evo_text = (
            f"{name_title} — Evolution Chain:\n"
            f"{format_evolution_chain(pokemon_data['evolution_chain'])}"
        )

        # Note variant forms if they exist
        if species_data and len(species_data.get("varieties", [])) > 1:
            forms = [
                v["name"].replace("-", " ").title()
                for v in species_data["varieties"]
                if not v["is_default"]
            ]
            if forms:
                evo_text += f"\nAlternate Forms: {', '.join(forms)}"

        documents.append({
            "id": f"{name}__evolution",
            "text": evo_text,
            "metadata": {
                "pokemon_name": name,
                "dex_id": dex_id,
                "source": "pokeapi",
                "category": "evolution",
            },
        })

    # --- 4. Competitive (Smogon) ---
    for tier, comp in competitive_data.items():
        comp_text = f"{name_title} — {format_competitive_data(comp, tier)}"
        documents.append({
            "id": f"{name}__competitive_{tier}",
            "text": comp_text,
            "metadata": {
                "pokemon_name": name,
                "dex_id": dex_id,
                "source": "smogon",
                "category": "competitive",
                "tier": tier,
            },
        })

    # --- 5. Lore & Trivia (Bulbapedia) ---
    if lore_data:
        for section_key, section_text in lore_data.items():
            if len(section_text.strip()) < 30:
                continue

            # Truncate very long sections to keep chunks manageable
            if len(section_text) > 2000:
                section_text = section_text[:2000] + "..."

            section_label = section_key.replace("_", " ").title()
            lore_text = f"{name_title} — {section_label}:\n{section_text}"

            documents.append({
                "id": f"{name}__lore_{section_key}",
                "text": lore_text,
                "metadata": {
                    "pokemon_name": name,
                    "dex_id": dex_id,
                    "source": "bulbapedia",
                    "category": "lore",
                    "section": section_key,
                },
            })

    # --- 6. Flavor Text ---
    if species_data and species_data.get("flavor_texts"):
        # Take unique flavor texts (deduplicate across versions)
        seen_texts = set()
        unique_entries = []
        for ft in species_data["flavor_texts"]:
            normalized = ft["text"].strip().lower()
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_entries.append(ft)

        # Take up to 5 representative entries
        selected = unique_entries[:5]
        if selected:
            flavor_lines = [f"{name_title} — Pokédex Entries:\n"]
            for entry in selected:
                version = entry["version"].replace("-", " ").title()
                flavor_lines.append(f"  [{version}] {entry['text']}")

            documents.append({
                "id": f"{name}__flavor_text",
                "text": "\n".join(flavor_lines),
                "metadata": {
                    "pokemon_name": name,
                    "dex_id": dex_id,
                    "source": "pokeapi",
                    "category": "flavor_text",
                },
            })

    return documents


def run_etl(limit: int = DEFAULT_POKEMON_LIMIT, skip_smogon: bool = False, skip_bulbapedia: bool = False) -> list[dict]:
    """
    Run the full ETL pipeline.

    Args:
        limit: Number of Pokemon to process (default 151 = Gen 1)
        skip_smogon: Skip Smogon data fetching (faster for testing)
        skip_bulbapedia: Skip Bulbapedia data fetching (faster for testing)

    Returns:
        List of all document chunks across all Pokemon
    """
    logger.info(f"Starting ETL pipeline for {limit} Pokemon...")

    pokeapi = PokeAPIFetcher()
    smogon = SmogonFetcher() if not skip_smogon else None
    bulbapedia = BulbapediaFetcher() if not skip_bulbapedia else None

    # Get list of Pokemon
    logger.info("Fetching Pokemon list from PokeAPI...")
    pokemon_list = pokeapi.fetch_all_pokemon_names(limit=limit)
    logger.info(f"Found {len(pokemon_list)} Pokemon to process")

    all_documents = []
    errors = []

    for entry in tqdm(pokemon_list, desc="Processing Pokemon"):
        poke_name = entry["name"]
        try:
            # 1. PokeAPI data
            pokemon_data = pokeapi.fetch_full_pokemon(poke_name)
            if pokemon_data is None:
                logger.warning(f"Skipping {poke_name}: no PokeAPI data")
                errors.append(poke_name)
                continue

            species_data = pokemon_data.get("species")

            # 2. Smogon data
            competitive_data = {}
            if smogon:
                try:
                    competitive_data = smogon.fetch_pokemon_competitive(poke_name)
                except Exception as e:
                    logger.warning(f"Smogon error for {poke_name}: {e}")

            # 3. Bulbapedia data
            lore_data = None
            if bulbapedia:
                try:
                    lore_data = bulbapedia.fetch_pokemon_lore_html(poke_name)
                except Exception as e:
                    logger.warning(f"Bulbapedia error for {poke_name}: {e}")

            # 4. Build document chunks
            docs = build_pokemon_documents(
                pokemon_data, species_data, competitive_data, lore_data
            )
            all_documents.extend(docs)
            logger.debug(f"  {poke_name}: {len(docs)} chunks")

        except Exception as e:
            logger.error(f"Error processing {poke_name}: {e}")
            errors.append(poke_name)

    logger.info(
        f"ETL complete: {len(all_documents)} document chunks "
        f"from {len(pokemon_list) - len(errors)}/{len(pokemon_list)} Pokemon"
    )
    if errors:
        logger.warning(f"Failed Pokemon: {', '.join(errors)}")

    # Print cache stats
    logger.info(f"PokeAPI cache: {pokeapi.get_cache_stats()}")
    if smogon:
        logger.info(f"Smogon cache: {smogon.get_cache_stats()}")
    if bulbapedia:
        logger.info(f"Bulbapedia cache: {bulbapedia.get_cache_stats()}")

    return all_documents


def save_documents_json(documents: list[dict], output_path: Optional[str] = None):
    """Save documents to a JSON file for inspection."""
    if output_path is None:
        output_path = str(Path(__file__).parent.parent / "cache" / "documents.json")

    with open(output_path, "w") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(documents)} documents to {output_path}")


def load_into_vector_store(documents: list[dict]):
    """Load documents into the ChromaDB vector store."""
    from rag.vector_store import PokemonVectorStore

    store = PokemonVectorStore()
    logger.info("Clearing pre-existing vector store documents to ensure a clean refresh...")
    store.clear()
    store.add_documents(documents)
    logger.info(f"Loaded {len(documents)} documents into ChromaDB")


def main():
    """Main ETL entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AI Pokedex ETL Pipeline")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_POKEMON_LIMIT,
        help=f"Number of Pokemon to process (default: {DEFAULT_POKEMON_LIMIT})"
    )
    parser.add_argument(
        "--skip-smogon", action="store_true",
        help="Skip Smogon data fetching"
    )
    parser.add_argument(
        "--skip-bulbapedia", action="store_true",
        help="Skip Bulbapedia data fetching"
    )
    parser.add_argument(
        "--save-json", action="store_true",
        help="Save documents to JSON file for inspection"
    )
    parser.add_argument(
        "--no-vectorize", action="store_true",
        help="Skip loading into vector store (useful for testing pipeline only)"
    )
    args = parser.parse_args()

    # Ensure cache directory exists
    cache_dir = Path(__file__).parent.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Run ETL
    documents = run_etl(
        limit=args.limit,
        skip_smogon=args.skip_smogon,
        skip_bulbapedia=args.skip_bulbapedia,
    )

    if not documents:
        logger.error("No documents produced. Check your network connection and logs.")
        sys.exit(1)

    # Save JSON if requested
    if args.save_json:
        save_documents_json(documents)

    # Load into vector store
    if not args.no_vectorize:
        load_into_vector_store(documents)

    logger.info("ETL pipeline complete!")


if __name__ == "__main__":
    main()
