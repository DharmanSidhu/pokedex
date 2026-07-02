"""
Deterministic tools for Pokemon calculations.

Bypasses the LLM for math, type matchups, stat comparisons, and damage calculations.
Guarantees 100% accuracy and eliminates LLM hallucinations in mechanical queries.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from data_pipeline.pokeapi_fetcher import PokeAPIFetcher

logger = logging.getLogger(__name__)

# Complete 18x18 Type Effectiveness Table from Generation VI onward.
# Multipliers represent damage dealt by Attacking Type (row) to Defending Type (col).
TYPE_MATCHUPS: Dict[str, Dict[str, float]] = {
    "normal": {
        "normal": 1.0, "fire": 1.0, "water": 1.0, "electric": 1.0, "grass": 1.0, "ice": 1.0,
        "fighting": 1.0, "poison": 1.0, "ground": 1.0, "flying": 1.0, "psychic": 1.0,
        "bug": 1.0, "rock": 0.5, "ghost": 0.0, "dragon": 1.0, "steel": 0.5, "dark": 1.0, "fairy": 1.0
    },
    "fire": {
        "normal": 1.0, "fire": 0.5, "water": 0.5, "electric": 1.0, "grass": 2.0, "ice": 2.0,
        "fighting": 1.0, "poison": 1.0, "ground": 1.0, "flying": 1.0, "psychic": 1.0,
        "bug": 2.0, "rock": 0.5, "ghost": 1.0, "dragon": 0.5, "steel": 2.0, "dark": 1.0, "fairy": 1.0
    },
    "water": {
        "normal": 1.0, "fire": 2.0, "water": 0.5, "electric": 1.0, "grass": 0.5, "ice": 1.0,
        "fighting": 1.0, "poison": 1.0, "ground": 2.0, "flying": 1.0, "psychic": 1.0,
        "bug": 1.0, "rock": 2.0, "ghost": 1.0, "dragon": 0.5, "steel": 1.0, "dark": 1.0, "fairy": 1.0
    },
    "electric": {
        "normal": 1.0, "fire": 1.0, "water": 2.0, "electric": 0.5, "grass": 0.5, "ice": 1.0,
        "fighting": 1.0, "poison": 1.0, "ground": 0.0, "flying": 2.0, "psychic": 1.0,
        "bug": 1.0, "rock": 1.0, "ghost": 1.0, "dragon": 0.5, "steel": 1.0, "dark": 1.0, "fairy": 1.0
    },
    "grass": {
        "normal": 1.0, "fire": 0.5, "water": 2.0, "electric": 1.0, "grass": 0.5, "ice": 1.0,
        "fighting": 1.0, "poison": 0.5, "ground": 2.0, "flying": 0.5, "psychic": 1.0,
        "bug": 0.5, "rock": 2.0, "ghost": 1.0, "dragon": 0.5, "steel": 0.5, "dark": 1.0, "fairy": 1.0
    },
    "ice": {
        "normal": 1.0, "fire": 0.5, "water": 0.5, "electric": 1.0, "grass": 2.0, "ice": 0.5,
        "fighting": 1.0, "poison": 1.0, "ground": 2.0, "flying": 2.0, "psychic": 1.0,
        "bug": 1.0, "rock": 1.0, "ghost": 1.0, "dragon": 2.0, "steel": 0.5, "dark": 1.0, "fairy": 1.0
    },
    "fighting": {
        "normal": 2.0, "fire": 1.0, "water": 1.0, "electric": 1.0, "grass": 1.0, "ice": 2.0,
        "fighting": 1.0, "poison": 0.5, "ground": 1.0, "flying": 0.5, "psychic": 0.5,
        "bug": 0.5, "rock": 2.0, "ghost": 0.0, "dragon": 1.0, "steel": 2.0, "dark": 2.0, "fairy": 0.5
    },
    "poison": {
        "normal": 1.0, "fire": 1.0, "water": 1.0, "electric": 1.0, "grass": 2.0, "ice": 1.0,
        "fighting": 1.0, "poison": 0.5, "ground": 0.5, "flying": 1.0, "psychic": 1.0,
        "bug": 1.0, "rock": 0.5, "ghost": 0.5, "dragon": 1.0, "steel": 0.0, "dark": 1.0, "fairy": 2.0
    },
    "ground": {
        "normal": 1.0, "fire": 2.0, "water": 1.0, "electric": 2.0, "grass": 0.5, "ice": 1.0,
        "fighting": 1.0, "poison": 2.0, "ground": 1.0, "flying": 0.0, "psychic": 1.0,
        "bug": 0.5, "rock": 2.0, "ghost": 1.0, "dragon": 1.0, "steel": 2.0, "dark": 1.0, "fairy": 1.0
    },
    "flying": {
        "normal": 1.0, "fire": 1.0, "water": 1.0, "electric": 0.5, "grass": 2.0, "ice": 1.0,
        "fighting": 2.0, "poison": 1.0, "ground": 1.0, "flying": 1.0, "psychic": 1.0,
        "bug": 2.0, "rock": 0.5, "ghost": 1.0, "dragon": 1.0, "steel": 0.5, "dark": 1.0, "fairy": 1.0
    },
    "psychic": {
        "normal": 1.0, "fire": 1.0, "water": 1.0, "electric": 1.0, "grass": 1.0, "ice": 1.0,
        "fighting": 2.0, "poison": 2.0, "ground": 1.0, "flying": 1.0, "psychic": 0.5,
        "bug": 1.0, "rock": 1.0, "ghost": 1.0, "dragon": 1.0, "steel": 0.5, "dark": 0.0, "fairy": 1.0
    },
    "bug": {
        "normal": 1.0, "fire": 0.5, "water": 1.0, "electric": 1.0, "grass": 2.0, "ice": 1.0,
        "fighting": 0.5, "poison": 0.5, "ground": 1.0, "flying": 0.5, "psychic": 2.0,
        "bug": 1.0, "rock": 1.0, "ghost": 0.5, "dragon": 1.0, "steel": 0.5, "dark": 2.0, "fairy": 0.5
    },
    "rock": {
        "normal": 1.0, "fire": 2.0, "water": 1.0, "electric": 1.0, "grass": 1.0, "ice": 2.0,
        "fighting": 0.5, "poison": 1.0, "ground": 0.5, "flying": 2.0, "psychic": 1.0,
        "bug": 2.0, "rock": 1.0, "ghost": 1.0, "dragon": 1.0, "steel": 0.5, "dark": 1.0, "fairy": 1.0
    },
    "ghost": {
        "normal": 0.0, "fire": 1.0, "water": 1.0, "electric": 1.0, "grass": 1.0, "ice": 1.0,
        "fighting": 1.0, "poison": 1.0, "ground": 1.0, "flying": 1.0, "psychic": 2.0,
        "bug": 1.0, "rock": 1.0, "ghost": 2.0, "dragon": 1.0, "steel": 1.0, "dark": 0.5, "fairy": 1.0
    },
    "dragon": {
        "normal": 1.0, "fire": 1.0, "water": 1.0, "electric": 1.0, "grass": 1.0, "ice": 1.0,
        "fighting": 1.0, "poison": 1.0, "ground": 1.0, "flying": 1.0, "psychic": 1.0,
        "bug": 1.0, "rock": 1.0, "ghost": 1.0, "dragon": 2.0, "steel": 0.5, "dark": 1.0, "fairy": 0.0
    },
    "dark": {
        "normal": 1.0, "fire": 1.0, "water": 1.0, "electric": 1.0, "grass": 1.0, "ice": 1.0,
        "fighting": 0.5, "poison": 1.0, "ground": 1.0, "flying": 1.0, "psychic": 2.0,
        "bug": 1.0, "rock": 1.0, "ghost": 2.0, "dragon": 1.0, "steel": 1.0, "dark": 0.5, "fairy": 0.5
    },
    "steel": {
        "normal": 1.0, "fire": 0.5, "water": 0.5, "electric": 0.5, "grass": 1.0, "ice": 2.0,
        "fighting": 1.0, "poison": 1.0, "ground": 1.0, "flying": 1.0, "psychic": 1.0,
        "bug": 1.0, "rock": 2.0, "ghost": 1.0, "dragon": 1.0, "steel": 0.5, "dark": 1.0, "fairy": 2.0
    },
    "fairy": {
        "normal": 1.0, "fire": 0.5, "water": 1.0, "electric": 1.0, "grass": 1.0, "ice": 1.0,
        "fighting": 2.0, "poison": 0.5, "ground": 1.0, "flying": 1.0, "psychic": 1.0,
        "bug": 1.0, "rock": 1.0, "ghost": 1.0, "dragon": 2.0, "steel": 0.5, "dark": 2.0, "fairy": 1.0
    }
}

def type_effectiveness(attack_type: str, defending_types: List[str]) -> float:
    """
    Calculate damage multiplier for an attacking type against a target's type(s).

    Args:
        attack_type: Move type (e.g. 'fire')
        defending_types: Target Pokemon's type(s) (e.g. ['grass', 'poison'])

    Returns:
        Multiplier (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    """
    atk = attack_type.lower()
    if atk not in TYPE_MATCHUPS:
        return 1.0

    multiplier = 1.0
    for def_type in defending_types:
        dt = def_type.lower()
        if dt in TYPE_MATCHUPS[atk]:
            multiplier *= TYPE_MATCHUPS[atk][dt]
            
    return multiplier

def type_matchup_summary(pokemon_types: List[str]) -> Dict[str, List[str]]:
    """
    Generate defensive summary for a typing combination: weaknesses, resistances, immunities.

    Args:
        pokemon_types: Defending types.

    Returns:
        Dict mapping multipliers (4x, 2x, 0.5x, 0.25x, 0x) to lists of attacking types.
    """
    summary = {
        "4x": [],
        "2x": [],
        "0.5x": [],
        "0.25x": [],
        "0x": []
    }
    
    # Calculate for all 18 types
    all_types = list(TYPE_MATCHUPS.keys())
    for atk_type in all_types:
        mult = type_effectiveness(atk_type, pokemon_types)
        if mult == 4.0:
            summary["4x"].append(atk_type.title())
        elif mult == 2.0:
            summary["2x"].append(atk_type.title())
        elif mult == 0.5:
            summary["0.5x"].append(atk_type.title())
        elif mult == 0.25:
            summary["0.25x"].append(atk_type.title())
        elif mult == 0.0:
            summary["0x"].append(atk_type.title())
            
    return summary

def stat_comparison(pokemon_a: str, pokemon_b: str) -> Dict[str, Any]:
    """
    Fetch and build side-by-side stat comparison between two Pokemon.

    Args:
        pokemon_a: Name of Pokemon A
        pokemon_b: Name of Pokemon B

    Returns:
        Dict with stat comparison data.
    """
    fetcher = PokeAPIFetcher()
    poke_a = fetcher.fetch_pokemon(pokemon_a.lower())
    poke_b = fetcher.fetch_pokemon(pokemon_b.lower())

    if not poke_a or not poke_b:
        return {
            "error": f"Could not fetch stats for {pokemon_a if not poke_a else pokemon_b}"
        }

    stats_list = ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]
    stat_labels = {
        "hp": "HP",
        "attack": "Atk",
        "defense": "Def",
        "special-attack": "Sp. Atk",
        "special-defense": "Sp. Def",
        "speed": "Speed"
    }

    comp = []
    total_a = 0
    total_b = 0

    for stat in stats_list:
        val_a = poke_a["base_stats"].get(stat, 0)
        val_b = poke_b["base_stats"].get(stat, 0)
        total_a += val_a
        total_b += val_b
        comp.append({
            "stat": stat_labels[stat],
            "val_a": val_a,
            "val_b": val_b,
            "winner": "A" if val_a > val_b else ("B" if val_b > val_a else "Tie")
        })

    comp.append({
        "stat": "BST (Total)",
        "val_a": total_a,
        "val_b": total_b,
        "winner": "A" if total_a > total_b else ("B" if total_b > total_a else "Tie")
    })

    return {
        "pokemon_a": poke_a["name"].title(),
        "pokemon_b": poke_b["name"].title(),
        "types_a": [t.title() for t in poke_a["types"]],
        "types_b": [t.title() for t in poke_b["types"]],
        "comparison": comp
    }

def damage_calc(
    attacker_name: str,
    move_name: str,
    defender_name: str,
    level: int = 50,
) -> Dict[str, Any]:
    """
    Run simplified Pokemon damage calculation.
    
    Formula: Damage = (((2 * Level / 5 + 2) * Power * A/D) / 50 + 2) * Modifier
    where Modifier = STAB * TypeEffectiveness * Random (0.85 to 1.00)

    Args:
        attacker_name: Attacker name
        move_name: Move name (e.g. 'earthquake')
        defender_name: Defender name
        level: Battle level (VGC defaults to 50)
    """
    fetcher = PokeAPIFetcher()
    atk_poke = fetcher.fetch_pokemon(attacker_name.lower())
    def_poke = fetcher.fetch_pokemon(defender_name.lower())
    move = fetcher.fetch_move(move_name.lower().replace(" ", "-"))

    if not atk_poke:
        return {"error": f"Attacker {attacker_name} not found."}
    if not def_poke:
        return {"error": f"Defender {defender_name} not found."}
    if not move:
        return {"error": f"Move {move_name} not found."}

    power = move.get("power")
    if not power:
        # Status move or variable power
        return {
            "error": f"Move {move['name'].replace('-', ' ').title()} has no fixed base power."
        }

    move_type = move.get("type", "normal")
    damage_class = move.get("damage_class", "physical")

    # Determine Attack / Defense stats
    if damage_class == "physical":
        a_stat = atk_poke["base_stats"].get("attack", 10)
        d_stat = def_poke["base_stats"].get("defense", 10)
        stat_desc = f"Atk ({a_stat}) vs Def ({d_stat})"
    else:
        a_stat = atk_poke["base_stats"].get("special-attack", 10)
        d_stat = def_poke["base_stats"].get("special-defense", 10)
        stat_desc = f"Sp. Atk ({a_stat}) vs Sp. Def ({d_stat})"

    # Calculate base damage
    base_damage = (((2 * level / 5 + 2) * power * (a_stat / d_stat)) / 50) + 2

    # Modifier calculations
    # 1. Type Effectiveness
    type_mult = type_effectiveness(move_type, def_poke["types"])
    
    # 2. STAB
    stab = 1.5 if move_type in atk_poke["types"] else 1.0

    # Calculate min & max ranges
    min_modifier = stab * type_mult * 0.85
    max_modifier = stab * type_mult * 1.00

    min_damage = int(base_damage * min_modifier)
    max_damage = int(base_damage * max_modifier)

    # HP percentage dealt
    def_hp = def_poke["base_stats"].get("hp", 10)
    # Estimate actual HP at Level 50 assuming 31 IVs and 0 EVs for simplicity
    # Formula: HP = (2 * Base + IV + EV/4) * Level / 100 + Level + 10
    actual_hp = int((2 * def_hp + 31) * level / 100) + level + 10

    min_pct = round((min_damage / actual_hp) * 100, 1)
    max_pct = round((max_damage / actual_hp) * 100, 1)

    return {
        "attacker": atk_poke["name"].title(),
        "defender": def_poke["name"].title(),
        "move": move["name"].replace("-", " ").title(),
        "move_type": move_type.title(),
        "damage_class": damage_class.title(),
        "power": power,
        "stat_matchup": stat_desc,
        "damage_range": (min_damage, max_damage),
        "hp_percent_range": (min_pct, max_pct),
        "actual_hp_est": actual_hp,
        "type_effectiveness": type_mult,
        "stab": stab > 1.0
    }
