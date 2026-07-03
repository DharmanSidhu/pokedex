"""
Chat orchestration and query routing for RotoDex.

Coordinates user queries, RAG retrieval, deterministic tools,
keyword-triggered memory extraction, and live PokeAPI & Bulbapedia fallback.
"""

import logging
import json
import re
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional, Union, Iterator

from llm.client import LLMClient
from llm.prompts import SYSTEM_PROMPT_TEMPLATE, build_user_memory_context, should_extract_memory
from llm.thinking import classify_query_thinking_mode
from llm.tools import type_effectiveness, type_matchup_summary, stat_comparison, damage_calc
from rag.retriever import HybridRetriever
from memory.store import UserMemoryStore
from memory.extractor import MemoryExtractor
from data_pipeline.pokeapi_fetcher import PokeAPIFetcher
from data_pipeline.bulbapedia_fetcher import BulbapediaFetcher
from app.components import (
    render_pokemon_card,
    render_cry_player,
    render_source_footer,
    render_stat_bar,
    render_type_badges
)

logger = logging.getLogger(__name__)


def format_live_pokemon_context(poke_data: dict, lore_data: Optional[dict] = None) -> str:
    """Format live-fetched Pokemon data and Bulbapedia lore into a comprehensive context string."""
    if not poke_data:
        return ""

    name = poke_data["name"].title()
    types = ", ".join(t.title() for t in poke_data.get("types", []))
    
    # Abilities
    raw_abilities = poke_data.get("abilities_detailed", [])
    ability_strs = []
    for a in raw_abilities:
        aname = a["name"].replace("-", " ").title()
        effect = a.get("effect", "")
        if a.get("is_hidden"):
            ability_strs.append(f"{aname} (Hidden Ability): {effect}")
        else:
            ability_strs.append(f"{aname}: {effect}")
    abilities = "; ".join(ability_strs)

    # Stats
    stats = poke_data.get("base_stats", {})
    stat_str = " / ".join(f"{k.replace('-', ' ').title()}: {v}" for k, v in stats.items())
    bst = sum(stats.values())

    # Species details
    species = poke_data.get("species", {}) or {}
    flavor_text = species.get("flavor_text", "No Pokedex entries recorded.")
    genus = species.get("genus", "Unknown Category")

    # Evolution
    evolution = ""
    chain = poke_data.get("evolution_chain")
    if chain:
        evolution_steps = []
        for stage in chain:
            s_name = stage["species"].title()
            trigger = ""
            details = stage.get("evolution_details")
            if details and isinstance(details, list) and len(details) > 0:
                det = details[0]
                trig_type = det.get("trigger", "")
                if trig_type == "level-up":
                    min_lvl = det.get("min_level")
                    trigger = f" (at Level {min_lvl})" if min_lvl else " (Level-up)"
                elif trig_type == "use-item":
                    item = det.get("item", "").replace("-", " ").title()
                    trigger = f" (using {item})"
                elif trig_type == "trade":
                    trigger = " (Trade)"
            evolution_steps.append(f"{s_name}{trigger}")
        evolution = " → ".join(evolution_steps)

    # Build context
    ctx = [
        f"[Source: POKEAPI (Live Fetch)]",
        f"Pokemon: {name} (Genus: {genus})",
        f"Types: {types}",
        f"Pokedex Entry: {flavor_text}",
        f"Base Stats: {stat_str} (BST: {bst})",
        f"Abilities Details: {abilities}",
        f"Evolution Chain: {evolution or 'Does not evolve.'}"
    ]

    # Bulbapedia Lore
    if lore_data:
        ctx.append("\n[Source: BULBAPEDIA (Live Lore)]")
        if "biology" in lore_data:
            ctx.append(f"Biology: {lore_data['biology']}")
        if "origin" in lore_data:
            ctx.append(f"Origin & Inspiration: {lore_data['origin']}")
        if "trivia" in lore_data:
            ctx.append(f"Trivia: {lore_data['trivia']}")

    return "\n".join(ctx)


class ChatOrchestrator:
    """Orchestrates query parsing, tool execution, RAG, and memory updates."""

    def __init__(self):
        self.client = LLMClient()
        self.retriever = HybridRetriever()
        self.memory_store = UserMemoryStore()
        self.extractor = MemoryExtractor(store=self.memory_store, client=self.client)
        self.poke_fetcher = PokeAPIFetcher()
        self.bulbapedia_fetcher = BulbapediaFetcher()

    def detect_and_execute_tools(self, query: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Check if query asks for type matchups, stat comparisons, or damage calculation,
        and run the deterministic tool to obtain correct, non-hallucinated data.
        """
        q_lower = query.lower()

        # 1. Damage calculation
        if "damage" in q_lower and ("does" in q_lower or "calc" in q_lower or "against" in q_lower or "uses" in q_lower):
            retrieval_res = self.retriever.retrieve(query, top_k=1)
            pokemon_matches = [m["name"] for m in retrieval_res["matched_pokemon"]]

            if len(pokemon_matches) >= 2:
                attacker = pokemon_matches[0]
                defender = pokemon_matches[1]
                atk_data = self.poke_fetcher.fetch_pokemon(attacker)
                move_found = None
                if atk_data:
                    for move in atk_data.get("moves", []):
                        move_readable = move.replace("-", " ")
                        if move_readable in q_lower or move in q_lower:
                            move_found = move
                            break

                if move_found:
                    calc_res = damage_calc(attacker, move_found, defender)
                    if "error" not in calc_res:
                        res_str = (
                            f"Deterministic Damage Calculation Result:\n"
                            f"- Attacker: {calc_res['attacker']}\n"
                            f"- Defender: {calc_res['defender']}\n"
                            f"- Move: {calc_res['move']} ({calc_res['move_type']}, Power: {calc_res['power']})\n"
                            f"- Estimated Damage Range: {calc_res['damage_range'][0]} to {calc_res['damage_range'][1]} HP\n"
                            f"- Estimated HP % Dealt: {calc_res['hp_percent_range'][0]}% to {calc_res['hp_percent_range'][1]}%\n"
                            f"- Type effectiveness {calc_res['type_effectiveness']}x, STAB {calc_res['stab']}\n"
                        )
                        return res_str, {"type": "damage_calc", "data": calc_res}

        # 2. Stat comparison
        if "compare" in q_lower or "vs" in q_lower or "better stats" in q_lower:
            retrieval_res = self.retriever.retrieve(query, top_k=1)
            pokemon_matches = [m["name"] for m in retrieval_res["matched_pokemon"]]
            if len(pokemon_matches) >= 2:
                comp_res = stat_comparison(pokemon_matches[0], pokemon_matches[1])
                if "error" not in comp_res:
                    res_str = (
                        f"Deterministic Stat Comparison Result:\n"
                        f"- {comp_res['pokemon_a']} (Types: {', '.join(comp_res['types_a'])})\n"
                        f"- {comp_res['pokemon_b']} (Types: {', '.join(comp_res['types_b'])})\n"
                    )
                    for row in comp_res["comparison"]:
                        res_str += f"  * {row['stat']}: {comp_res['pokemon_a']}={row['val_a']}, {comp_res['pokemon_b']}={row['val_b']} (Winner: {row['winner']})\n"
                    return res_str, {"type": "stat_compare", "data": comp_res}

        # 3. Type matchup
        if "effective" in q_lower or "matchup" in q_lower or "weakness" in q_lower or "resist" in q_lower:
            from llm.tools import TYPE_MATCHUPS
            all_types = list(TYPE_MATCHUPS.keys())
            found_types = [t for t in all_types if t in q_lower]

            if len(found_types) >= 2:
                atk_type = found_types[0]
                def_types = found_types[1:]
                mult = type_effectiveness(atk_type, def_types)
                res_str = (
                    f"Deterministic Type Matchup Result:\n"
                    f"- Attacking: {atk_type.title()} → Defending: {', '.join(t.title() for t in def_types)}\n"
                    f"- Multiplier: {mult}x\n"
                )
                return res_str, {"type": "type_calc", "data": {"attack": atk_type, "defend": def_types, "multiplier": mult}}
            elif len(found_types) == 1:
                retrieval_res = self.retriever.retrieve(query, top_k=1)
                pokemon_matches = [m["name"] for m in retrieval_res["matched_pokemon"]]

                if pokemon_matches:
                    poke_data = self.poke_fetcher.fetch_pokemon(pokemon_matches[0])
                    if poke_data:
                        summary = type_matchup_summary(poke_data["types"])
                        res_str = (
                            f"Deterministic Type Summary for {poke_data['name'].title()} ({', '.join(poke_data['types'])}):\n"
                            f"- 4x Weak: {', '.join(summary['4x']) or 'None'}\n"
                            f"- 2x Weak: {', '.join(summary['2x']) or 'None'}\n"
                            f"- Resists: {', '.join(summary['0.5x']) or 'None'}\n"
                            f"- Immune: {', '.join(summary['0x']) or 'None'}\n"
                        )
                        return res_str, {"type": "type_summary", "data": summary}
                else:
                    target_type = found_types[0]
                    summary = type_matchup_summary([target_type])
                    res_str = (
                        f"Deterministic Type Summary for {target_type.title()} type:\n"
                        f"- Weaknesses (2x): {', '.join(summary['2x']) or 'None'}\n"
                        f"- Resistances (0.5x): {', '.join(summary['0.5x']) or 'None'}\n"
                        f"- Immunities (0x): {', '.join(summary['0x']) or 'None'}\n"
                    )
                    return res_str, {"type": "type_summary", "data": summary}

        return None, None

    def resolve_implicit_pokemon_names(self, query: str) -> List[str]:
        """Call local LLM to resolve implicit Pokemon references in query into standard species names."""
        from llm.prompts import RESOLVE_POKEMON_PROMPT
        messages = [
            {"role": "system", "content": RESOLVE_POKEMON_PROMPT},
            {"role": "user", "content": f"Input: \"{query}\""}
        ]
        try:
            # Fast, non-thinking, deterministic query resolution
            response_text = self.client.chat_completion(
                messages=messages,
                temperature=0.0,
                enable_thinking=False,
                bypass_cache=True,
                max_tokens=64
            )
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = re.sub(r'^```(?:json)?\n', '', response_text)
                response_text = re.sub(r'\n```$', '', response_text)
                response_text = response_text.strip()

            names = json.loads(response_text)
            if isinstance(names, list):
                valid_names = self.retriever.pokemon_names
                return [n.strip().lower() for n in names if n.strip().lower() in valid_names]
        except Exception as e:
            logger.warning(f"Failed to resolve implicit Pokemon names: {e}")
        return []

    def _get_live_fallback_context(self, matched_pokemon: List[Dict], rag_chunks: List[Dict]) -> Tuple[str, bool, bool]:
        """
        Fetch live data for any matched Pokemon that are missing high-quality local RAG chunks.
        """
        if not matched_pokemon:
            return "", False, False

        poke_live_used = False
        bulba_live_used = False
        fallback_contexts = []

        for match in matched_pokemon:
            name = match["name"]
            
            # Check if this Pokemon has good local RAG chunks (distance < 0.6)
            has_good_rag = any(
                c.get("metadata", {}).get("pokemon_name") == name and c.get("distance", 1.0) < 0.6
                for c in rag_chunks
            )
            
            if not has_good_rag:
                logger.info(f"RAG miss for '{name}', falling back to live fetch")
                poke_data = self.poke_fetcher.fetch_full_pokemon(name)
                lore_data = self.bulbapedia_fetcher.fetch_pokemon_lore(name)
                
                if poke_data:
                    formatted = format_live_pokemon_context(poke_data, lore_data)
                    fallback_contexts.append(formatted)
                    poke_live_used = True
                    if lore_data:
                        bulba_live_used = True

        combined_context = "\n\n---\n\n".join(fallback_contexts)
        return combined_context, poke_live_used, bulba_live_used

    def process_chat_turn(self, query: str, chat_history: List[Dict[str, str]]) -> Tuple[Union[str, Iterator[str]], Dict[str, Any]]:
        """Process a single user turn in the chat interface."""

        # Step 1: RAG retrieval
        retrieval_res = self.retriever.retrieve(query, top_k=5)
        matched_pokemon = list(retrieval_res["matched_pokemon"])

        # If no explicit matches found, resolve implicit Pokemon or carry forward active Pokemon
        if not matched_pokemon:
            # Find active Pokemon in recent chat history
            active_name = None
            if chat_history:
                for msg in reversed(chat_history):
                    if isinstance(msg, dict) and "meta" in msg:
                        meta = msg["meta"]
                        if isinstance(meta, dict) and "matched_pokemon" in meta:
                            prev_matched = meta["matched_pokemon"]
                            if prev_matched:
                                active_name = prev_matched[0] if isinstance(prev_matched[0], str) else prev_matched[0].get("name")
                                break

            # Determine if this is a follow-up query referencing the active Pokemon
            is_follow_up = False
            if active_name:
                query_words = set(re.findall(r'[a-zA-Z]+', query.lower()))
                pronoun_triggers = {"it", "its", "this", "they", "them", "he", "she", "him", "her", "his", "hers", "that", "the ability", "the moves", "more", "detail", "details"}
                if len(query_words) <= 4 or (query_words & pronoun_triggers):
                    is_follow_up = True

            if is_follow_up and active_name:
                logger.info(f"Fast-path active Pokemon context carry-forward: '{active_name}'")
                matched_pokemon.append({"name": active_name, "score": 90})
            else:
                # Run name-resolution LLM call to resolve implicit descriptors (e.g. "grass starter")
                resolved_names = self.resolve_implicit_pokemon_names(query)
                if resolved_names:
                    logger.info(f"Resolved implicit query to species: {resolved_names}")
                    for name in resolved_names:
                        matched_pokemon.append({"name": name, "score": 100})
                
                # If name resolution returned nothing, default carry-forward the active Pokemon
                if not matched_pokemon and active_name:
                    logger.info(f"Default carry-forward active Pokemon context: '{active_name}'")
                    matched_pokemon.append({"name": active_name, "score": 90})

        # Step 2: User memory facts
        user_facts = self.memory_store.get_relevant_facts(query, k=5)
        user_memory_context = build_user_memory_context(user_facts)
        user_memory_applied = len(user_facts) > 0

        # Step 3: Deterministic tool calculations
        calc_context, tool_meta = self.detect_and_execute_tools(query)

        # Step 4: Assemble context — RAG + live fallback (including Bulbapedia)
        context_parts = []
        if calc_context:
            context_parts.append(calc_context)

        # Pull exact core profile chunks (base_info, abilities, evolution) & compute type-effectiveness
        if matched_pokemon:
            for match in matched_pokemon[:2]:
                name = match["name"]
                
                # Fetch database chunks matching category base_info, abilities, evolution
                profile_results = self.retriever.store.query(
                    query_text=f"{name} base stats abilities evolution",
                    n_results=12,
                    where={"pokemon_name": name}
                )
                for r in profile_results:
                    cat = r.get("metadata", {}).get("category", "")
                    if cat in ["base_info", "abilities", "evolution"]:
                        context_parts.append(r["text"])
                
                # Calculate and inject type effectiveness summary
                poke_data = self.poke_fetcher.fetch_pokemon(name)
                if poke_data:
                    summary = type_matchup_summary(poke_data["types"])
                    weaknesses_4x = [t.title() for t in summary.get("4x", [])]
                    weaknesses_2x = [t.title() for t in summary.get("2x", [])]
                    resistances_05x = [t.title() for t in summary.get("0.5x", [])]
                    resistances_025x = [t.title() for t in summary.get("0.25x", [])]
                    immunities_0x = [t.title() for t in summary.get("0x", [])]
                    
                    type_context = (
                        f"Type Effectiveness details for {name.title()} ({'/'.join(t.title() for t in poke_data['types'])}):\n"
                        f"  - 4x Weak to: {', '.join(weaknesses_4x) or 'None'}\n"
                        f"  - 2x Weak to: {', '.join(weaknesses_2x) or 'None'}\n"
                        f"  - Resists (0.5x): {', '.join(resistances_05x) or 'None'}\n"
                        f"  - Double Resists (0.25x): {', '.join(resistances_025x) or 'None'}\n"
                        f"  - Immune to (0x): {', '.join(immunities_0x) or 'None'}"
                    )
                    context_parts.append(type_context)

        # RAG context
        rag_text = self.retriever.retrieve_text(query, top_k=5)
        context_parts.append(rag_text)

        # Live fallback (PokeAPI + Bulbapedia) for any matched/resolved Pokemon missing local chunks
        live_context, poke_live, bulba_live = self._get_live_fallback_context(matched_pokemon, retrieval_res["chunks"])
        if live_context:
            context_parts.append(live_context)

        assembled_context = "\n\n---\n\n".join(context_parts)

        # Step 5: Classify thinking mode
        enable_thinking, max_tokens = classify_query_thinking_mode(query)

        # Step 6: Build messages
        system_content = SYSTEM_PROMPT_TEMPLATE.format(
            user_memory_context=user_memory_context,
            retrieved_context=assembled_context
        )

        messages = [{"role": "system", "content": system_content}]
        
        # Slices chat_history[:-1] to exclude the current turn (which is already the last element of chat_history).
        past_history = chat_history[:-1]
        for turn in past_history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
            
        # Add current query explicitly as the final user turn
        messages.append({"role": "user", "content": query})

        # Step 7: Call LLM (streamed)
        response = self.client.chat_completion(
            messages=messages,
            temperature=0.4,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            stream=True
        )

        # Track sources
        sources = retrieval_res["sources"]
        if poke_live:
            sources.add("pokeapi (live)")
        if bulba_live:
            sources.add("bulbapedia (live)")

        rendering_meta = {
            "sources": sources,
            "user_memory_applied": user_memory_applied,
            "matched_pokemon": [m["name"] for m in matched_pokemon],
            "tool_meta": tool_meta,
            "enable_thinking": enable_thinking
        }

        return response, rendering_meta

    def finalize_turn(self, query: str, full_response: str) -> List[Dict[str, Any]]:
        """
        Post-response: only extract memory if the user used a trigger keyphrase.
        """
        if should_extract_memory(query):
            logger.info(f"Memory trigger detected in: '{query}'")
            new_facts = self.extractor.extract_and_store_facts(query)
            return new_facts

        # No trigger — skip extraction entirely
        return []
